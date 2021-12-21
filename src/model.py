from typing import Dict,List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from src.detector import SimpleDetector
import numpy as np


class AttentionQA(nn.Module):

    def __init__(self,config, # model.config
                 class_embs = True,
                 reasoning_use_obj = True,
                 reasoning_use_answer = True,
                 reasoning_use_question = True,
                 pool_reasoning = True,
                 pool_answer = True,
                 pool_question = False):
        super(AttentionQA, self).__init__()
        span_config = config.span_encoder
        reasoning_config = config.reasoning_encoder
        input_dropout = config.input_dropout
        hidden_dim_maxpool = config.hidden_dim_maxpool

        self.detector = SimpleDetector(pretrained=True,average_pool=True, semantic=class_embs, final_dim=512)
        ################################################################################################################
        self.rnn_input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None
        self.span_encoder = nn.LSTM(span_config.input_size, span_config.hidden_size, span_config.num_layers,bidirectional=span_config.bidirectional)

        self.reasoning_encoder = nn.LSTM(reasoning_config.input_size, reasoning_config.hidden_size, reasoning_config.num_layers, bidirectional=reasoning_config.bidirectional)
        # key : question & value : answer
        self.span_attention = BilinearMatrixAttention(span_config.hidden_size*2,span_config.hidden_size*2)
        # key: answer & value: object
        self.obj_attention = BilinearMatrixAttention(span_config.hidden_size*2, span_config.hidden_size*2)

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question

        dim = sum([d for d, to_pool in [(reasoning_config.hidden_size, self.pool_reasoning),
                                        (span_config.hidden_size, self.pool_answer),
                                        (span_config.hidden_size, self.pool_question)] if to_pool])


        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, hidden_dim_maxpool), # 512,1024
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self.get_loss = torch.nn.CrossEntropyLoss()
        self.initialize_weights()

    def initialize_weights(self):
        self.lstm_initializer(self.span_encoder)
        self.lstm_initializer(self.reasoning_encoder)

        self.final_mlp[1].weight.data.uniform_(-0.1, 0.1)
        self.final_mlp[1].bias.data.fill_(0)
        self.final_mlp[-1].weight.data.uniform_(-0.1, 0.1)
        self.final_mlp[-1].bias.data.fill_(0)

        return None


    def lstm_initializer(self, lstm):
        for layer in lstm.all_weights:
            for weight in layer:
                if weight.ndim == 2:
                    weight.data.uniform_(-0.1, 0.1)
                else:
                    weight.data.fill_(0)


    def forward(self, images, objects, segms, boxes, box_masks, question, question_tags, question_mask, answers, answer_tags,answer_mask, label):
        max_len = int(box_masks.sum(1).max().item())
        objects = objects[:, :max_len]
        box_masks = box_masks[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]

        for tag_type, tags in (("question", question_tags),('answer', answer_tags)):
            if int(tags.max()) > max_len:
                raise ValueError("tag maximum value {} is over than max length".format(tags.max()))

        obj_reps = self.detector(images=images, boxes=boxes, box_masks=box_masks,classes=objects,segms=segms)
        #[batch, 4, question, 256]
        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps["obj_reps"])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps["obj_reps"])

        # [batch, 4, question, answer]


        qa_similarity = self.span_attention(q_rep.view(q_rep.shape[0]*q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
                                            a_rep.view(a_rep.shape[0]*a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]))

        qa_similarity = qa_similarity.view(a_rep.shape[0], a_rep.shape[1], qa_similarity.shape[1], qa_similarity.shape[2])

        # question_mask : [batch, 4, question]
        # qa_attention_weight : [batch, 4, question, masked_softmax]
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[...,None], dim=2)

        attended_q = torch.einsum('bnqa, bnqd -> bnad',(qa_attention_weights, q_rep))

        #objects & answer
        #[batch, 4, answer, num_object]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1]*a_rep.shape[2], -1),
                                             obj_reps["obj_reps"]).view(a_rep.shape[0], a_rep.shape[1], a_rep.shape[2], obj_reps["obj_reps"].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_masks[:, None, None], dim=-1)
        attended_o = torch.einsum('bnao, bod -> bnad', (atoo_attention_weights, obj_reps["obj_reps"]))

        reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
                                                        (attended_o, self.reasoning_use_obj),
                                                        (attended_q, self.reasoning_use_question)] if to_pool], -1)
        if self.rnn_input_dropout is not None:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)

        B, N, A, D = reasoning_inp.shape
        reasoning_output, _ = self.reasoning_encoder(reasoning_inp.view(B*N, A, D))
        reasoning_output = reasoning_output.view(B, N, A, -1)

        things_to_pool = torch.cat([x for x, to_pool in [(reasoning_output, self.pool_reasoning),
                                                         (a_rep, self.pool_answer),
                                                         (attended_q, self.pool_question)] if to_pool], -1)

        pooled_rep = replace_masked_values(things_to_pool, answer_mask[...,None], -1e7).max(2)[0] # 2dim 에서 값이 가장 큰 텐서만 모음,(max pool) batch, 4, dim
        logits = self.final_mlp(pooled_rep).squeeze(2)

        class_probabilites = F.softmax(logits, dim=-1)

        output_dict = {'label_logits':logits, "label_probs":class_probabilites,
                       "cnn_regularization_loss": obj_reps["cnn_regularization_loss"]}

        if label is not None:
            loss = self.get_loss(logits, label.long().view(-1))
            output_dict["loss"] = loss[None]
            output_dict["accuracy"] = self.cal_accuracy(logits, label.long().view(-1))

        return output_dict

    def cal_accuracy(self, logits, labels):
        logits_ = logits.cpu().detach().numpy()
        labels_ = labels.cpu().detach().numpy()
        preds = np.argmax(logits_, axis=-1)
        return np.sum(preds==labels_)

    def _collect_obj_reps(self, span_tags, object_reps):
        span_tags_fixed = torch.clamp(span_tags, min=0) # 0보다 작은 값은 모두 0으로 변환 (패딩 (-2) 를 모두 0으로 변환)
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None] # (batch, 1) , torch.tensor([0,1,2..batch])

        leading_dims = len(span_tags.shape)-2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask , object_reps):
        features = self._collect_obj_reps(span_tags, object_reps)
        span_rep = torch.cat((span, features), -1)
        B_, N, K, D = span_rep.shape

        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)
        reps, _ = self.span_encoder(span_rep.view(B_*N, K, D))
        B, N, D = reps.shape
        return reps.view(B_,-1, N, D), features

def masked_softmax(vector, mask, dim=-1):
    while mask.dim() < vector.dim():
        mask.unsqueeze(1)
    mask_ = mask > 0
    masked_vector = vector.masked_fill_(~mask_, -10000)
    return torch.nn.functional.softmax(masked_vector, dim=dim)

def replace_masked_values(values, mask, replaced_with):
    assert values.dim() == mask.dim()
    mask = mask > 0
    return values.masked_fill_(~mask,replaced_with)

class BilinearMatrixAttention(nn.Module):
    def __init__(self, matrix_1_dim, matrix_2_dim):
        super(BilinearMatrixAttention, self).__init__()
        self._weight_matrix = nn.Linear(matrix_1_dim, matrix_2_dim)
        self.initializer()

    def initializer(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix.weight.data)


    def forward(self, matrix_1, matrix_2):
        weight = self._weight_matrix.weight
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2,3))
        return final.squeeze(1)

if __name__ == '__main__':
    import torch
    from utils.utils import *
    import sys, os
    sys.path.append("/mnt/data/user8/vision_data/r2c/r2c_pytorch")
    data = torch.load("/mnt/data/user8/vision_data/r2c/valid_0.pkl")
    config = load_config("/mnt/data/user8/vision_data/r2c/r2c_pytorch/src/config.json")
    model = AttentionQA(config.model)
    

    output = model(data['images'], data["objects"], data["segms"], data["boxes"], data["box_masks"], data["question"], data["question_tags"],\
                   data["question_mask"], data["answer"], data["answer_tags"],data["answer_mask"], data["label"])
    print(output)

