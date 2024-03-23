# Implementation loosely based on https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/utils/beam_search.py#L554
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Literal

import torch
import torch.nn as nn
from mistral.tokenizer import Tokenizer

from mistral.utils import *
from ngrams.ngram_models import NGram

INF = 1. * 1e7

# Test by scaling # beams & verify work
class Beam(nn.Module): 
    def __init__(self, 
                 initial_tokens,
                 tokenizer,
                 vocab_size,
                 mixing_method: Literal["sample", "sample_new_weights_with_score", "sample_weights_with_current"],
                 smoothing: Optional[Literal["geom", "all"]],
                 alpha = None,
                 verbose = False,
                 i_weights = None,
                 i_length = None,
                 ngrams = None,
                 sample_beams = False,
                 diversity_boost = (None, None),
                 sample_tokens = False): # default no effect
        super().__init__()
        """
        Initialize a beam search class.
        
        Args:
            initial_tokens (torch.Tensor): Initial tokens
            n_prompts (int): Number of prompts
            tokenizer (Tokenizer): Mistral tokenizer
            vocab_size (int): Total vocab size
            mixing_method (str): Method to create mixing matrix
            smoothing (str): Smoothing method (either interpol or geom)
            ngram_length (int): N gram length to consider
            alpha (float): Alpha parameter
            debug (bool): Whether to print information
        """
        # primary parameters
        self.n_prompts, self.n_drafts, _ = initial_tokens.shape
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.eos_id = tokenizer.eos_id
        self.alive_seq = initial_tokens
        self.fin_seq = initial_tokens
        self.alive_log_probs = torch.zeros(self.n_prompts, self.n_drafts)
        self.fin_log_probs = torch.full((self.n_prompts, self.n_drafts), float("-inf"))
        self.alpha = alpha
        self.verbose = verbose
        self.sample_beams = sample_beams
        self.n_div_mask, self.div_boost = diversity_boost
        self.sample_tokens = sample_tokens
        
        # devices
        self.cpu = torch.device('cpu')
        self.gpu = torch.device('cuda')
        
        # settings
        self.mixing_method = mixing_method
        self.smoothing = smoothing
        
        # interpolation length and weights
        self.interpolation_weights = i_weights
        self.i_length = i_length
        
        # minfinigram
        self.bigram = ngrams[0] if len(ngrams) >= 1 else None
        self.trigram = ngrams[1] if len(ngrams) >= 2 else None
        self.fourgram = ngrams[2] if len(ngrams) >= 3 else None
        self.fivegram = ngrams[3] if len(ngrams) >= 4 else None
        self.sixgram = ngrams[4] if len(ngrams) >= 5 else None
        self.sevengram = ngrams[5] if len(ngrams) >= 6 else None

    def forward(self, probs, still_prompt, is_first, cur_pos, n_token_consider, n_token_sample, use_mix=False):
        """
        Apply beam decoding to update generations.
        
        Args:
            probs (torch.Tensor): Next token probability distribution
            still_prompt (torch.Tensor): Flags of prompts that should not generate yet (n_prompts, )
            is_first (torch.Tensor): Flags of prompts that are on their first generation (n_prompts, )
            cur_pos (int): Current generation position
            n_token_consider (int): Number of tokens to consider for next token sampling
            n_token_sample (int): Number of tokens to sample
            use_mix (bool): Whether mixed embeddings are being used or not (default False)
            
        Return:
            if standard beam search:
                attention_change_ids (torch.Tensor): New indices in kv cache (n_prompts, n_beams)
            if mixed:
                token_weights (torch.Tensor): Mixing weights (n_prompts, vocab_size)
        """
        
        # Adjust input probabilities
        if use_mix:
            probs = self.sample_top_k(probs, n_token_consider, n_token_sample)
            reshaped_probs = probs.reshape(self.n_prompts, 1, -1)
            reshaped_probs = reshaped_probs.repeat(1, self.n_drafts, 1)
        else:
            reshaped_probs = probs.reshape(self.n_prompts, self.n_drafts, -1)
        
        # Ngram smoothing 
        if self.smoothing is not None:
            if self.smoothing == "geom":
                ngram_probs = self.ngram_probs(self.alive_seq, cur_pos, probs=probs)              
                
                # make mask and normalize
                prob_mask = reshaped_probs != 0

                # apply
                ngram_probs *= prob_mask  
                
                # logprobs
                llm_log_probs = torch.log(reshaped_probs)
                ngram_log_probs = torch.log(ngram_probs)
                log_probs = (1 - self.alpha) * llm_log_probs + self.alpha * ngram_log_probs
                
                # replace
                is_all_inf = (log_probs != float("-inf")).sum(dim=-1, keepdims=True) == 0 
                log_probs = torch.where(is_all_inf, (1 - self.alpha) * llm_log_probs - 200, log_probs)
                
            elif self.smoothing == "all":
                ngram_probs = self.ngram_probs(self.alive_seq, cur_pos, probs=None)              
                log_probs = torch.log(ngram_probs)
        else:
            log_probs = torch.log(reshaped_probs)
        # Normalize options
        # Select top k
        curr_log_probs = self.alive_log_probs.unsqueeze(dim=2) + log_probs # [n_prompts, n_beams, vocab_size]
        # warning if nan
        if self.verbose:
            print(f"most probable tokens: {torch.sort(curr_log_probs, descending=True)}")
        if (torch.any(torch.isnan(curr_log_probs)).item()):
            raise RuntimeWarning("nan in sequence log probs")
        
        # boosting
        if self.div_boost is not None:
            # maybe could do a prob based heurestic, e.g. mask howver many of prev that got to 0.9 total or sum
            n_mask = torch.arange(0, self.n_drafts) * self.n_div_mask # (n_drafts,) number of seq to not change
            for p_idx in range(self.n_prompts):
                for d_idx in range(self.n_drafts):
                    if d_idx > 0:
                        _, b = torch.topk(curr_log_probs[p_idx][d_idx-1], n_mask[d_idx], dim=-1)
                        mult = torch.full((self.vocab_size,), self.div_boost)
                        mult[b] = 1
                        curr_log_probs[p_idx][d_idx] *= mult
        
        # Potential Sequences
        flat_curr_log_probs = curr_log_probs.reshape(-1, self.vocab_size*self.n_drafts) # Beams are consecutive on each row
        topk_log_probs, topk_idx = torch.topk(flat_curr_log_probs, 2 * self.n_drafts, dim=-1)
        topk_beam_id = topk_idx // self.vocab_size # [n_prompts, 2 * n_beams]
        topk_idx = topk_idx % self.vocab_size # [n_prompts, 2 * n_beams]
        
        # First generation uses top-k unique from one beam
        is_first_idx = is_first.nonzero(as_tuple=True)[0]
        if len(is_first_idx) != 0:
            first_time_log_probs = log_probs[is_first_idx][:, 0, :].squeeze(dim=1)
            first_time_log_probs, first_time_topk_idx = torch.topk(first_time_log_probs, 2 * self.n_drafts, dim=1)
            topk_idx[is_first_idx] = first_time_topk_idx
            topk_log_probs[is_first_idx] = self.alive_log_probs[is_first_idx, 0].unsqueeze(dim=1) + first_time_log_probs 
            # dont need to update topk_beam_id because all the same at first step
            
        # Construct new seqs
        topk_seq = torch.take_along_dim(self.alive_seq, topk_beam_id.unsqueeze(2), dim=1) # [n_prompts, 2 * n_beams, vocab_size]
        topk_seq[:, :, cur_pos] = topk_idx
        topk_finished = topk_idx == self.eos_id
    
        # Update for prompts that have begun generating
        new_alive_seq, new_alive_log_probs, attention_change_ids = self.grow_alive(topk_seq, topk_log_probs, topk_finished, topk_beam_id)
        new_fin_seq, new_fin_log_probs = self.grow_fin(topk_seq, topk_log_probs, topk_finished)
        
        still_prompt_probs = still_prompt.reshape(-1, 1)
        still_prompt_seqs = still_prompt.reshape(-1, 1, 1)
        still_prompt_idx = still_prompt.nonzero(as_tuple=True)[0]
        attention_change_ids[still_prompt_idx] = torch.arange(self.n_drafts)
        self.alive_seq = torch.where(still_prompt_seqs, self.alive_seq, new_alive_seq)
        self.alive_log_probs = torch.where(still_prompt_probs, self.alive_log_probs, new_alive_log_probs) 
        self.fin_seq = torch.where(still_prompt_seqs, self.fin_seq, new_fin_seq)
        self.fin_log_probs = torch.where(still_prompt_probs, self.fin_log_probs, new_fin_log_probs)

        if use_mix:
            topk_idx = self.alive_seq[:, :, cur_pos].reshape(self.n_prompts, -1)
            if self.mixing_method != "sample":
                token_weights = self.create_mixed_matrix_from_tokens(probs, topk_idx, self.mixing_method)
                return token_weights
            return probs
        else:
            return attention_change_ids
        
    def grow_alive(self, topk_seq, topk_log_probs, topk_finished, topk_beam_id):
        """
        Grow alive generations
        
        Args:
            topk_seq (torch.Tensor): Top k sequences (n_prompts, 2 * n_beams, vocab_size)
            topk_log_probs (torch.Tensor): Log probabilities (n_prompts, 2 * n_beams)
            topk_finished (torch.Tensor): Whether a sequence is finished (n_prompts, 2 * n_beams) 
            topk_beam_id (torch.Tensor): Original draft indices (n_prompts, 2 * n_beams)
            
        Returns:
            new_alive_seq, new_alive_log_probs, attention_change_ids
        """
        topk_log_probs = topk_log_probs + topk_finished * -INF 
        if not self.sample_beams:
            new_alive_log_probs, new_alive_idx = torch.topk(topk_log_probs, self.n_drafts, dim=1)
        else:
            probs = log_prob_to_prob(topk_log_probs)
            new_alive_idx = torch.multinomial(probs, num_samples=self.n_drafts)
            new_alive_log_probs = torch.gather(torch.log(probs), -1, new_alive_idx)
        new_alive_seq = torch.take_along_dim(topk_seq, new_alive_idx.unsqueeze(2), dim=1)
        attention_change_ids = torch.gather(topk_beam_id, 1, new_alive_idx)
        
        return new_alive_seq, new_alive_log_probs, attention_change_ids
        
    def grow_fin(self, topk_seq, topk_log_probs, topk_finished):
        """
        Grow alive generations
        
        Args:
            topk_seq (torch.Tensor): Top k sequences (n_prompts, 2 * n_beams, vocab_size)
            topk_log_probs (torch.Tensor): Log probabilities (n_prompts, 2 * n_beams)
            topk_finished (torch.Tensor): Whether a sequence is finished (n_prompts, 2 * n_beams) 
            
        Returns:
            new_fin_seq, new_fin_log_probs
        """
        
        topk_log_probs = topk_log_probs + ~topk_finished * -INF 
        new_fin_seq = torch.cat([self.fin_seq, topk_seq], dim=1)
        new_fin_log_probs = torch.cat([self.fin_log_probs, topk_log_probs], dim=1)
        # select seqs
        new_fin_log_probs, new_fin_idx = torch.topk(new_fin_log_probs, self.n_drafts, dim=1)
        new_fin_seq = torch.take_along_dim(new_fin_seq, new_fin_idx.unsqueeze(2), dim=1)
        
        return new_fin_seq, new_fin_log_probs

    def sample_top_k(self, probs, k, m):
        """
        Perform top-k sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            k (float): number of elements to consider top-k sampling.
            m (int): Number of indices to sample.

        Returns:
            torch.Tensor: New probability distribution based on renormalized indices. 
            
        """
        n_prompts, vocab_size = probs.shape
        # need to handle if m is too 
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        top_k_mask = torch.arange(probs.shape[-1])
        top_k_mask = top_k_mask.expand(probs.shape[0], -1) # duplicate along dim=0 to match probs array
        top_k_mask = top_k_mask >= k # set to 1 past k elements
        probs_sort[top_k_mask] = 0.0 # 0 wherever mask = 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # RESTORE
        if self.sample_tokens:
            next_token = torch.gather(probs_idx, -1, torch.multinomial(probs_sort, num_samples=m))
        else:
            next_token = torch.gather(probs_idx, -1, torch.topk(probs_sort, m, dim=-1)[1])       
        
        # set all other probs to 0
        new_probs_map = torch.zeros(probs.shape).bool()
        new_probs_map[torch.repeat_interleave(torch.arange(n_prompts), m), torch.flatten(next_token)] = True
        new_probs = torch.where(new_probs_map, probs, 0)
        
        # renormalize
        new_probs.div_(new_probs.sum(dim=-1, keepdim=True))
        
        return new_probs
    
    def create_mixed_matrix_from_tokens(self, probs, tokens, mixing_method: Literal["sample_new_weights_with_score", "sample_weights_with_current"]):
        """
        Create mixing matrix for each prompt based on provided tokens.
        
        Args:
            probs (torch.Tensor): Probability distribution tensor.
            tokens (torch.Tensor): Tokens to mix for each prompt
            mixing_method (str): How to construct mixing matrix

        Returns:
            torch.Tensor: Mixing matrix.
        """
        n_prompts, _ = probs.shape
        
        # indexing setup
        unique_tokens, _ = torch.unique(tokens, return_counts=True, dim=-1)
        prompt_idx = torch.arange(n_prompts)
        
        # create mixing matrix
        if mixing_method == "sample_new_weights_with_score":
            mixing_matrix = torch.zeros(self.n_prompts, self.vocab_size)
            weightings = log_prob_to_prob(self.alive_log_probs)
            for p_idx in range(self.n_prompts):
                for d_idx in range(self.n_drafts):
                    tok_idx = tokens[p_idx][d_idx]
                    mixing_matrix[p_idx][tok_idx] += weightings[p_idx][d_idx]
        else:
            lengths = torch.tensor([len(uniques) for uniques in unique_tokens])
            prompt_idx_map = torch.repeat_interleave(prompt_idx, lengths)
            probs_mask = torch.zeros(probs.shape).bool()
            probs_mask[prompt_idx_map, unique_tokens.flatten()] = True
            mixing_matrix = torch.where(probs_mask, probs, 0)

        if self.verbose:    
            print(f"tokens being mixed: {torch.nonzero(mixing_matrix)}")
        mixing_matrix.div_(mixing_matrix.sum(dim=-1, keepdims=True))
        if self.verbose:    
            print(f"mixture probabilities: {mixing_matrix[mixing_matrix.nonzero(as_tuple=True)]}")
        return mixing_matrix
    
    def ngram_probs(self, alive_seq, cur_pos, probs):
        next_token_probs = torch.zeros(self.n_prompts, self.n_drafts, 32000)
        if probs is not None:
            for p_idx in range(len(alive_seq)):
                nz = torch.nonzero(probs[p_idx, :], as_tuple=True)[0].tolist() # possible tokens
                for draft_idx in range(self.n_drafts):
                    i_mask = torch.sum(torch.tensor(self.i_length) <= cur_pos)
                    new_i_weights = self.interpolation_weights[:i_mask]
                    new_i_length = self.i_length[:i_mask]
                    for nt in nz:
                        for i, weight in zip(new_i_length, new_i_weights):
                            if cur_pos - i >= 0:
                                key = tuple(alive_seq[p_idx, draft_idx, cur_pos-i:cur_pos].tolist())
                                if i == 1:
                                    prob = self.bigram.prob(key, nt)
                                elif i == 2:
                                    prob = self.trigram.prob(key, nt)
                                elif i == 3:
                                    prob = self.fourgram.prob(key, nt)
                                elif i == 4:
                                    prob = self.fivegram.prob(key, nt)
                                elif i == 5:
                                    prob = self.sixgram.prob(key, nt)
                                elif i == 6:
                                    prob = self.sevengram.prob(key, nt)
                            if prob >= 0:
                                next_token_probs[p_idx, draft_idx, nt] += weight * prob
        else:
            # use ntd
            for p_idx in range(len(alive_seq)):
                for draft_idx in range(self.n_drafts):
                    i_mask = torch.sum(torch.tensor(self.i_length) <= cur_pos)
                    new_i_weights = self.interpolation_weights[:i_mask]
                    new_i_length = self.i_length[:i_mask]
                    for i, weight in zip(new_i_length, new_i_weights):
                        if cur_pos - i >= 0:
                            key = tuple(alive_seq[p_idx, draft_idx, cur_pos-i:cur_pos].tolist())
                            if i == 1:
                                ntd = self.bigram.ntd(key)
                            elif i == 2:
                                ntd = self.trigram.ntd(key)
                            elif i == 3:
                                ntd = self.fourgram.ntd(key)
                            elif i == 4:
                                ntd = self.fivegram.ntd(key)
                            elif i == 5:
                                ntd = self.sixgram.ntd(key)
                            elif i == 6:
                                ntd = self.sevengram.ntd(key)
                        if ntd is not None:
                            next_token_probs[p_idx, draft_idx, :] += weight * ntd
        return next_token_probs

    
    def return_results(self, prompt_len):
        """
        Return generations and perplexities
        
        Args:
            prompt_len (int): Length of prompt in tokens
        Returns:
            (self.alive_seq, alive_ppl), (self.fin_seq, fin_ppl)
            
        """
        # alive ppl
        alive_ppl = torch.exp(self.alive_log_probs / (-1 * (self.alive_seq.size(dim=-1)-prompt_len)))
                
        # fin ppl
        fin_seq_lengths = (self.fin_seq != self.tokenizer.pad_id).sum(dim=-1)
        fin_ppl = torch.exp(self.fin_log_probs / (-1 * (fin_seq_lengths - prompt_len)))
        fin_ppl += ((fin_ppl == 0) * float("inf"))
        
        return (self.alive_seq.to(torch.long), alive_ppl), (self.fin_seq.to(torch.long), fin_ppl)