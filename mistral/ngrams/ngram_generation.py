import copy
from typing import Any, Literal, List, Optional, Tuple

import torch

from ngrams.ngram_models import Bigram, Trigram

def ngram_generate(
  model: Any,
  model_type: Literal["bigram", "trigram"],
  prompt_tokens: List[List[int]],
  max_gen_len: int,
  eos_id: int
):
  generated_tokens = []
  generated_log_probs = []
  for sequence in prompt_tokens:
    sequence_probs = []
    new_sequence = copy.deepcopy(sequence)
    for i in range(max_gen_len):
      if model_type == "bigram":
        next_token, next_token_prob = model.generate(first_token=new_sequence[-1])
      else: 
        next_token, next_token_prob = model.generate(first_token=new_sequence[-2], second_token=new_sequence[-1])
      new_sequence += [next_token]
      sequence_probs.append(next_token_prob)
      if next_token == eos_id:
        break
    generated_tokens.append(new_sequence)
    generated_log_probs.append(torch.log(torch.tensor(sequence_probs)))
  return generated_tokens, generated_log_probs

