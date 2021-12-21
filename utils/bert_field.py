import torch
import numpy
from typing import Dict, List, Optional

# class BertField:
#     def __init__(self, tokens, embs, padding_value, token_indexers) -> None:
#         self.tokens = tokens
#         self.embs = embs
#         self.padding_value = padding_value
#
#         if len(self.tokens) != self.embs.shape[0]:
#             raise ValueError("token {} and embedding size {} didn't match ".format(self.tokens, slef.embs.shape))
#         assert len(self.tokens) == self.embs.shape[0]
#
#     def sequence_length(self) -> int:
#         return len(self.tokens)
#
#     def get_padding_lengths(self) -> Dict[str, int]:
#         return {"num_tokens":self.sequence_length()}
#
#     def empty_field(self):
#         return BertField()
#
#
#     def __str__(self) -> str:
#         return f"BertField:{self.tokens} and {self.embs.shape}."
