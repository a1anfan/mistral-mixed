from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
import os
import time
import sys
import json
from typing import List, Optional
from pathlib import Path

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from mistral.mixed_model import MixedTransformer
from mistral.tokenizer import Tokenizer
from mistral.beam import Beam
from mistral.mixed_model import ModelArgs


def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)

class MixedMistral:
    @staticmethod
    def build(
        folder: Path,
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device="cuda",
        dtype=torch.float16,
    ) -> "MixedMistral":
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        if num_pipeline_ranks > 1:
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
        with torch.device("meta"):
            model = MixedTransformer(
                model_args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
            )
        loaded = torch.load(str(folder / "consolidated.00.pth"), mmap=True)
        model.load_state_dict(loaded, assign=True)
        model = model.to(device=device, dtype=dtype)
        tokenizer = Tokenizer(str(folder / "tokenizer.model"))  # ASSUMES THAT TOKENIZER.MODEL IS IN THE FOLDER!
        return MixedMistral(model, tokenizer, device)

    def __init__(self, model: MixedTransformer, tokenizer: Tokenizer, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def beam_generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        mixing_method: str,
        smoothing: str,
        n_token_consider: int,
        n_token_sample: int,
        alpha: int, # weight on bigram probs
        temp: int,
        n_drafts: int = 1, # number of beams
        debug: bool = False,
        verbose: bool = False,
        i_weights = None,
        i_length = None,
        ngrams = None,
        sample_beams: bool = False,
        diversity_boost = (None, None),
        sample_tokens: bool = False
    ):
        """
        Run multi-sequence generation using mixed embeddings.
        Args:
            prompt_tokens (List[List[int]]): Initial tokenized prompts
            max_gen_len (int): Max generation length
            mixing_method (str): Mixing method
            smoothing (str): Smoothing method
            ngram_length (int): Length of ngrams for smoothing
            n_token_consider (int): Number of tokens to normalize for before running beam search
            n_token_sample (int): Number of tokens to consider from n_token_consider
            alpha (float): Weight for N-Gram probabilities
            temp (float): Temperature
            n_drafts (int): Number of drafts (Default 1)
            debug (bool): Whether to print outputs
            verbose (bool): Whether to store and return model hidden states (Default False)
            i_weights (list): List of weights corresponding to ngrams in i_length
            i_length (list): Ngram lengths to use in interpolation
            beam_selection_temp (float): Temperature for sampling beams (default None)
            diversity_boost (tuple(int, float)): Tuple of (a, b) denoting a additional tokens to mask for each draft and 
            sample_tokens (bool): Whether to sample next tokens passed into the beam search algorithm
        
        Returns:
            (alive_seq, alive_ppl), (fin_seq, fin_ppl)
        """
        # check batch size and prompt lengths
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert min_prompt_len == max_prompt_len, "Prompt lengths must be equal"
        prompt_len = min_prompt_len
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)
        pad_id = self.tokenizer.pad_id
        
        # initialize token tensor
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)
        
        # if no generation possible
        if min_prompt_len == total_len:
            raise RuntimeError("no generation possible")

        ### INTIALIZATION ###
        initial_tokens = tokens.unsqueeze(1).repeat(1, n_drafts, 1)
        beam_search = Beam(initial_tokens, 
                            tokenizer=self.tokenizer,
                            vocab_size=params.vocab_size,
                            mixing_method=mixing_method,
                            smoothing=smoothing,
                            alpha=alpha,
                            verbose=debug,
                            i_weights=i_weights,
                            i_length=i_length,
                            ngrams=ngrams,
                            sample_beams=sample_beams,
                            diversity_boost=diversity_boost,
                            sample_tokens=sample_tokens)
        unseen_first = torch.ones(bsz) # 1 if still parsing prompt
        token_weights = torch.zeros(bsz, self.model.vocab_size)
        if verbose:
            state_list = []
        prev_pos = 0
        ### INFERENCE ###
        for cur_pos in range(min_prompt_len, total_len):
            input_text_mask = tokens != pad_id
            
            # Model step
            if cur_pos == min_prompt_len:
                token_weights = None
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], 
                                        start_pos=prev_pos, 
                                        token_weights=token_weights, 
                                        verbose=verbose)
            if verbose:
                logits, states = logits
            
            # Softmax
            if temp > 0:
                probs = torch.softmax(logits[:, -1] / temp, dim=-1)
            else:
                raise RuntimeError("Temperature must be greater than 0 while mixing")
            if verbose:
                states["end_probs"] = probs
                state_list.append(states)

            # Flag prompts on first generation
            is_first = torch.mul(tokens[:, cur_pos] == pad_id, unseen_first)
            unseen_first[is_first.nonzero(as_tuple=True)[0]] = 0
            
            # Flag prompts not yet generating
            still_prompt = input_text_mask[:, cur_pos]
            
            # Beam pass
            token_weights = beam_search(probs, still_prompt, is_first, cur_pos, n_token_consider, n_token_sample, use_mix=True)
            
            # Do not mix for prompts not yet generating
            keep_idx = input_text_mask[:, cur_pos].ravel().nonzero()
            keep_token_weights = torch.zeros_like(token_weights)
            keep_token_weights[keep_idx, tokens[keep_idx, cur_pos]] = 1
            token_weights = torch.where(input_text_mask[:, cur_pos].unsqueeze(1).expand(-1, self.model.vocab_size), 
                                        keep_token_weights, token_weights)
            prev_pos = cur_pos
            
        ### RETURN ###
        results = beam_search.return_results(prompt_len)
        if verbose:
            return results, state_list
        else:
            return results


@torch.inference_mode()
def generate(prompts: List[str], model: MixedTransformer, tokenizer: Tokenizer, *, max_tokens: int,  temperature: float, chunk_size: int = None):
    model = model.eval()
    B, V = len(prompts), model.args.vocab_size

    # Tokenize
    encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    if model.args.sliding_window is not None and cache_window > model.args.sliding_window:
        cache_window = model.args.sliding_window
    cache = RotatingBufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()
    
    # Bookkeeping
    logprobs = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s:s+chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache
        )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(0, torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1)
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tokens = []
    assert last_token_prelogits is not None
    for i_token in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tokens.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * len(prompts), cache=cache)
        assert last_token_prelogits.shape == (B, V)

    generated_words = []
    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, 1)
        for i, x in enumerate(encoded_prompts):
            generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))

    return generated_words, logprobs


def interactive(model_path: str, max_tokens: int = 35, temperature: float = 0.7, instruct: bool = False):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = MixedTransformer.from_folder(Path(model_path), max_batch_size=3)

    while True:
        prompt = input("Prompt: ")
        if instruct:
            prompt = f"[INST] {prompt} [/INST]"
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(res[0])
        print("=====================")


def demo(
    model_path: str, max_tokens: int = 35, temperature: float = 0, num_pipeline_ranks=1
):
    if num_pipeline_ranks > 1:
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        should_print = torch.distributed.get_rank() == 0
    else:
        should_print = True
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = MixedTransformer.from_folder(
        Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks
    )

    res, _logprobs = generate(
        [
            "This is a test",
            "This is another great test",
            "This is a third test, mistral AI is very good at testing. ",
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if should_print:
        for x,l in zip(res, _logprobs):
            print(x)
            logging.debug('Logprobs: %s',l)
            print("=====================")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
    })
