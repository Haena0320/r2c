import logging
import os
import json
import glob
from transformers import AdamW, get_linear_schedule_with_warmup

class Dictobj(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def __getitem__(self, key):
        return getattr(self, key)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Dictobj(value) if isinstance(value, dict) else value

def load_config(conf):
    with open(conf, "r") as f:
        config = json.load(f)
    return Dictobj(config)

def setup_opt(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.adam_betas is not None:
        adam_betas = tuple(float(_f) for _f in args.adam_betas.split(","))
        assert len(adam_betas) == 2
    else:
        adam_betas = (0.9, 0.999)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,betas=adam_betas,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.t_total)
    return model, optimizer, scheduler