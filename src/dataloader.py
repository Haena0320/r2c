"""
Dataloaders for VCR
"""
import json
import jsonlines as jsnl
import os, sys
sys.path.append("/mnt/data/user8/vision_data/r2c/r2c_pytorch")
from path import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.box_utils import load_image, resize_image, to_tensor_and_normalize
from utils.mask_utils import make_mask
from tqdm import tqdm
import h5py
from copy import deepcopy

VCR_DATA_DIR = "/mnt/data/user8/vision_data/r2c/r2c_pytorch/vcr1data"

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

MAX_SEQUENCE_LENGTH = 50
MAX_OBJECT_NUM = 15

def select_field(features, field):
    return [feature[field] for feature in features]

def tokenization(tokenized_sent,bert_embs,old_object_to_new_ind, object_to_type, pad_id=-1):
    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = object_to_type[int_name]
                new_ind = old_object_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("invalid object index ! ")
                text_to_use = GENDER_NEUTRAL_NAMES[new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == "person" else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            new_tokenization_with_tags.append((tok, pad_id))

    tags = [tag for token, tag in new_tokenization_with_tags]

    assert bert_embs.shape[0] == len(tags)
    return bert_embs, tags


class VCR(Dataset):
    def __init__(self, split, mode="answer", vcr_dir=VCR_DATA_DIR, only_use_relevant_objects=True,add_image_as_a_box=True,embs_to_load='bert_da', conditioned_answer_choice=0):
        self.split = split
        self.mode = mode
        self.vcr_dir = Path(vcr_dir)
        self.only_use_relevant_objects=only_use_relevant_objects
        self.add_image_as_a_box = add_image_as_a_box
        self.embs_to_load=embs_to_load
        self.conditioned_answer_choice=conditioned_answer_choice

        assert split in ["train", "test", "val"]
        assert mode in ["answer","rationale"]

        vcr_annos = self.vcr_dir / "vcr1annots" / '{}.jsonl'.format(split)
        vcr_annos_reader = jsnl.Reader(vcr_annos.open("r"))
        self.items = [k for k in tqdm(vcr_annos_reader.iter())]

        coco_path = self.vcr_dir / 'coco.json'
        coco_path = Path(VCR_DATA_DIR) / "coco.json"
        coco_data = json.load(coco_path.open("r"))

        self.coco_objects = ["__background__"] + [x['name'] for k, x in sorted(coco_data.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o:i for i, o in enumerate(self.coco_objects)}

        self.h5fn = self.vcr_dir / "vcr1annots" / f'{self.embs_to_load}_{self.mode}_{self.split}.h5'

 
    @property
    def is_train(self):
        if self.split == "train":
            return True
        else:
            return False

    @classmethod
    def splits(cls, **kwargs):
        kwargs_copy = {x:y for x, y in kwargs.items()}
        if 'mode' not in kwargs:
            kwargs_copy["mode"] = "answer"
        train = cls(split="train",**kwargs_copy)
        val = cls(split="val", **kwargs_copy)
        test = cls(split="test", **kwargs_copy)
        return train, val, test

    @classmethod
    def eval_splits(cls, **kwargs):
        for forbidden_key in ["mode", "split", "conditioned_answer_choice"]:
            if forbidden_key in kwargs:
                if forbidden_key in kwargs:
                    raise ValueError(f"don't supply {forbidden_key} to eval_splits()")
        stuff_to_return = [cls(split="test", mode="answer", **kwargs)] + [cls(split='test', mode="rationale", conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return stuff_to_return

    def __len__(self):
        return len(self.items)

    def check_object_in_context(self, item):
        question = item["question"]
        answer_choices = item["{}_choices".format(self.mode)]

        if self.only_use_relevant_objects:
            object2use = np.zeros(len(item["objects"]), dtype=bool)
            people = np.array([x =="person" for x in item["objects"]], dtype=bool)
            for sent in answer_choices+[question]:
                for possible_object in sent: #[2]
                    if isinstance(possible_object, list):
                        for tag in possible_object:
                            if tag >=0 and tag < len(item['objects']):
                                object2use[tag] = True
                    elif possible_object.lower() in ("everyone", 'everyones'):
                        object2use |= people
            if not object2use.any():
                object2use |= people
        else:
            object2use = np.ones(len(item["objects"]), dtype=bool)

        object2use = np.where(object2use)[0] # 1인 위치의 인덴스만 array로 반환함 [0,1]

        old_object_to_new_ind = np.zeros(len(item["objects"]), dtype=np.int32)-1   #[-1,-1,-1,-1]
        old_object_to_new_ind[object2use] = np.arange(object2use.shape[0], dtype=np.int32) # [-1,0, 1, -1]

        if self.add_image_as_a_box:
            old_object_to_new_ind[object2use] += 1 # [0, 1, 2, -1]
        old_object_to_new_ind = old_object_to_new_ind.tolist()
        return object2use, old_object_to_new_ind

    def __getitem__(self, index):

        item = deepcopy(self.items[index])

        if self.mode == "rationale":
            conditioned_label = item['answer_label'] if self.split != "test" else self.conditioned_answer_choice
            item['question'] += item["answer_choices"][conditioned_label]

        answer_choices = item[f"{self.mode}_choices"]
        object2use, old_object_to_new_ind = self.check_object_in_context(item)

        with h5py.File(self.h5fn, 'r') as h5:
            bert_embeddings = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}
            # anwer_answer0, answer_answer1, answer_answer2, answer_answer3
            # question : ctx_answer0, ctx_answer1, ctx_answer2, ctx_answer3
        condition_key = self.conditioned_answer_choice if self.split =="test" and self.mode =="rationale" else ""
        instance_dict = {}

        # qeustion 의 버트 임베딩이 4개 선택지에 따라 4개가 존재한다.
        questions_bert_embs, question_tags = zip(*[tokenization(item["question"],
                                                                     bert_embeddings[f"ctx_{self.mode}{condition_key}{i}"],
                                                                     old_object_to_new_ind,
                                                                     item["objects"],
                                                                     pad_id=0 if self.add_image_as_a_box else -1)
                                                   for i in range(4)])
        answer_bert_embs, answer_tags = zip(*[tokenization(answer,
                                                                bert_embeddings[f"answer_{self.mode}{condition_key}{i}"],
                                                                old_object_to_new_ind,
                                                                item["objects"],
                                                                pad_id=0 if self.add_image_as_a_box else -1)
                                              for i, answer in enumerate(answer_choices)])

        ## mask
        
        instance_dict["question"] = questions_bert_embs
        instance_dict["question_mask"] = [len(q)*[1] for q in question_tags]
        instance_dict["question_len"] = len(question_tags[0])
        instance_dict["question_tags"] = question_tags

        instance_dict["answers"] = answer_bert_embs
        instance_dict["answer_mask"] = [len(a)*[1] for a in answer_tags]
        instance_dict["answer_len"] = max([len(a) for a in answer_tags])
        instance_dict['answer_tags'] = answer_tags

        assert len(questions_bert_embs)  == len(question_tags)
        assert len(answer_bert_embs) == len(answer_tags)

        if self.split != "test":
            instance_dict["label"] = item["{}_label".format(self.mode)]

        image_path = load_image(self.vcr_dir / "vcr1images" / item["img_fn"])

        #load image
        images, window, img_scale, padding = resize_image(image_path, random_pad=self.is_train) # padding
        images = to_tensor_and_normalize(images)
        c, h, w = images.shape
        #load bounding boxes, object segms, object names
        meta_path = self.vcr_dir / "vcr1images" / item["metadata_fn"]
        metadata = json.load(meta_path.open("r"))
        # [4, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata["segms"][i]) for i in object2use])
        boxes = np.array(metadata["boxes"])[object2use, :-1]

        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]

        obj_labels = [self.coco_obj_to_ind[item["objects"][i]] for i in object2use.tolist()]
        if self.add_image_as_a_box:
            boxes = np.row_stack((window, boxes))
            segms = np.concatenate((np.ones((1,14,14), dtype=np.float32), segms), 0)
            obj_labels = [self.coco_obj_to_ind["__background__"]] + obj_labels

        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))

        instance_dict["segms"] = segms
        instance_dict["objects"] = obj_labels
        instance_dict["boxes"] = boxes

        return images, instance_dict

def make_batch(data, to_gpu=False):
    batch = dict()
    images, instances = zip(*data)
    batch["images"] = torch.stack(images, 0)

    question_batch = [i["question_len"] for i in instances]
    answer_batch = [i["answer_len"] for i in instances]
    object_batch =[len(i["objects"]) for i in instances]
    answer_each_batch = [[len(i["answer_mask"][j]) for j in range(4)] for i in instances]
    max_quest_len = max(question_batch)
    max_answer_len = max(answer_batch)
    max_object_num = max(object_batch)
    batch_n = len(question_batch)

    batch["question_mask"] = torch.zeros((batch_n, 4, max_quest_len)).long()
    batch["question_tags"] = torch.ones((batch_n, 4, max_quest_len)).long() * (-2)
    batch["question"] = torch.zeros((batch_n, 4, max_quest_len, 768)).float()

    batch["answer_mask"] = torch.zeros((batch_n, 4, max_answer_len)).long()
    batch["answer_tags"] = torch.ones((batch_n, 4, max_answer_len)).long() * (-2)
    batch["answers"] = torch.zeros((batch_n, 4, max_answer_len, 768)).float()

    batch["objects"] = torch.ones(batch_n, max_object_num).long()*(-1)

    batch["boxes"] = torch.ones(batch_n, max_object_num, 4)*(-1)
    batch["segms"] = torch.zeros(batch_n, max_object_num, 14, 14)

    for i in range(batch_n):
        batch["question_mask"][i, :, :question_batch[i]] = torch.tensor(instances[i]["question_mask"]).long()
        batch["question_tags"][i, :, :question_batch[i]] = torch.tensor(instances[i]["question_tags"]).long()
        batch["question"][i, :, :question_batch[i], :] = torch.tensor(instances[i]["question"])

        batch["objects"][i, :object_batch[i]] = torch.tensor(instances[i]["objects"])
        batch["boxes"][i, :object_batch[i], :] = torch.tensor(instances[i]["boxes"])
        batch["segms"][i, :object_batch[i], :, :] = torch.tensor(instances[i]["segms"])

        a_batch = answer_each_batch[i]
        for j in range(4):
            batch["answer_mask"][i, j, :a_batch[j]] = torch.tensor(instances[i]["answer_mask"][j]).long()
            batch["answer_tags"][i, j, :a_batch[j]] = torch.tensor(instances[i]["answer_tags"][j]).long()
            batch["answers"][i, j, :a_batch[j], :] = torch.tensor(instances[i]["answers"][j])

    batch["box_masks"] = torch.all(batch["boxes"] >= 0, -1).long()
    batch["label"] = torch.tensor([i["label"] for i in instances])
    return batch

class VCRLoader(DataLoader):
    @classmethod
    def from_dataset(cls, data, batch_size=1, num_workers=1, num_gpus=1, **kwargs):

        loader = cls(dataset=data,
                     batch_size=batch_size*num_gpus,
                     shuffle=data.is_train,
                     num_workers=num_workers,
                     collate_fn=make_batch,
                     drop_last=data.is_train,
                     pin_memory=False,
                     **kwargs)
        return loader



if __name__ =="__main__":
    train, val, test = VCR.splits()
    data1 = val.__getitem__(0)
    
    items = [data1, data2, data3]
    batch = make_batch(items)

    val_loader = VCRLoader.from_dataset(val, batch_size=10)
    for i in val_loader:
        print(i)
        break
    i["segms"][0][0]
        
    torch.save(i, "valid_mine.pkl")
    data_vcr = torch.load("/mnt/data/user8/vision_data/r2c/valid_vcr.pkl")
    data_mine = i

    data_vcr["box_mask"] == data_mine["box_masks"]
    data_vcr["box_mask"][0]
    data_mine["box_masks"][0]
    torch.tensor([True, False, False]).long()









