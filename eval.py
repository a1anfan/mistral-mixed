from typing import Any, List

import torch
from tqdm import tqdm

# run using notebook
# Can get mauve references easily through read file directly
# Mauve at end via generations stored 
# So can store tokens until end calculate mauve
# Calculate perplexity on beam side using cross entropy

def prepare_encodings(batch, tokenizer, length):
    tokens = tokenizer.encode(batch, True, False)
    new_encodings = []
    for i, encoded_text in enumerate(tokens):
        new_encodings.append(encoded_text[:length])
    return new_encodings

def evaluate_mixed_losses(data: List[List[str]],
                          model: Any,
                          tokenizer: Any,
                          prompt_len: int,
                          max_gen_len: int,
                          alpha: float,
                          temp: float,
                          n_drafts: int,
                          n_token_consider: int,
                          n_token_sample: int,
                          mixing_method: str,
                          smoothing: str,
                          debug: bool = False,
                          bsz=16,
                          i_weights = None,
                          i_length = None,
                          ngrams = None,
                          sample_beams = False,
                          diversity_boost=(None, None),
                          sample_tokens=False,
                          marker=True):
    """
    Evaluate perplexity for mixed embeddings.
    Args:
        data (List[List[String]]): Input data
        model (Any): Model
        tokenizer (Any): Llama tokenizer
        prompt_len (int): Number of tokens in starting prompt
        max_gen_len (int): Maximum numbers of tokens to generate
        alpha (float): Alpha value
        temp (float): Temperature
        n_drafts (int): Number of drafts
        mixing_method (str): Mixing method
        smoothing (str): Smoothing strategy
        debug (bool): Control whether to print debugging information (default False)
        bsz (int): Batch size (default = 16)
        
    Return:
        sequences (torch.Tensor): Generated sequences (n_prompts, n_drafts, prompt_len+max_gen_len)
        ppl (torch.Tensor): Perplexity (n_prompts, n_drafts)
    """
    if debug:
        print("### DEBUG MODE ON ###")
        data = data[0]
    
    it = range(0, len(data), bsz)
    if marker:
        it = tqdm(it)
    sequences = torch.zeros(len(data), n_drafts, prompt_len+max_gen_len, dtype=torch.long)
    ppl = torch.zeros(len(data), n_drafts)
    for b_start in it:
        b_end = b_start + bsz
        # preprocessing
        batch = data[b_start : b_end]
        truncated_tokens = prepare_encodings(batch, tokenizer, prompt_len)
        
        # inference
        (alive_seq, alive_ppl), (fin_seq, fin_ppl) = model.beam_generate(prompt_tokens=truncated_tokens, 
                                                                        max_gen_len=max_gen_len, 
                                                                        mixing_method=mixing_method,
                                                                        smoothing=smoothing,
                                                                        n_token_consider=n_token_consider,
                                                                        n_token_sample=n_token_sample,
                                                                        alpha=alpha, 
                                                                        temp=temp,
                                                                        n_drafts=n_drafts,
                                                                        debug=debug,
                                                                        i_weights=i_weights,
                                                                        i_length=i_length,
                                                                        ngrams=ngrams,
                                                                        sample_beams=sample_beams,
                                                                        diversity_boost=diversity_boost,
                                                                        sample_tokens=sample_tokens)
        # seq: n_prompts, n_drafts, prompt_len+max_gen_len
        # ppl: n_prompts, n_drafts
        combined_ppl = torch.cat([alive_ppl, fin_ppl], dim=1) # n_prompts, 2*n_drafts
        combined_seq = torch.cat([alive_seq, fin_seq], dim=1) # n_prompts, 2*n_drafts, prompt_len+max_gen_len
        top_ppl, top_idx = torch.topk(combined_ppl, n_drafts, dim=-1, largest=False)
        top_seq = torch.take_along_dim(combined_seq, top_idx.unsqueeze(dim=2), dim=1) # n_prompts, n_drafts, prompt_len+max_gen_len
        ppl[b_start : b_end, :] = top_ppl
        sequences[b_start : b_end, :, :] = top_seq
    return sequences, ppl

def evaluate_nucleus_losses(data,
                            model,
                            tokenizer,
                            prompt_len,
                            max_gen_len,
                            temp,
                            bsz=16,
                            marker=True):
    """
    Evaluate perplexity for nucleus sampling.
    Args:
        data (List[List[String]]): Input data
        model (Any): Model
        tokenizer (Any): Llama tokenizer
        prompt_len (int): Number of tokens in starting prompt
        max_gen_len (int): Maximum numbers of tokens to generate
        temp (float): Temperature
        bsz (int): Batch size (default = 16)
    Return:
        sequences (torch.Tensor): Generated sequences (n_prompts, prompt_len+max_gen_len)
        ppl (torch.Tensor): Perplexity (n_prompts)
    """
    it = range(0, len(data), bsz)
    if marker:
        it = tqdm(it)
    sequences = torch.zeros(len(data), prompt_len+max_gen_len, dtype=torch.long)
    ppl = torch.zeros(len(data), dtype=torch.float32)
    for b_start in it:
        b_end = b_start + bsz
        # preprocessing
        batch = data[b_start : b_end]
        truncated_tokens = prepare_encodings(batch, tokenizer, prompt_len)
        
        # inference
        # TODO: ensure torch tensor, ppl returned
        curr_seq, curr_ppl = model.generate(prompt_tokens=truncated_tokens,
                                  max_gen_len=max_gen_len,
                                  temperature=temp,
                                  top_p=0.9,
                                  logprobs=True)
        sequences[b_start : b_end, :] = curr_seq
        ppl[b_start : b_end] = curr_ppl
    return sequences, ppl
        
        