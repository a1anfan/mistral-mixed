import math
from tqdm import tqdm

import torch
import nltk
from nltk.translate.bleu_score import SmoothingFunction

def calculate_perplexity(model, tokens, prompt_len, bsz=1, marker=False):
    it = range(0, len(tokens), bsz)
    if marker:
        it = tqdm(it)
    start = 0
    ppl = torch.zeros(len(tokens))
    for start in it:
        end = start + bsz
        data = tokens[start : end].tolist()
        for d_idx in range(len(data)):
            cur = data[d_idx]
            if -1 in cur:
                data[d_idx] = cur[:cur.index(-1)]
        ce_loss = model.generate(data, max_gen_len=0, temperature=-1, top_p=-1, grade=True)
        # only keep generation
        ce_loss = ce_loss[:, prompt_len-1:] # start at prompt_len-1 because we dropped first token
        # store in 
        ppl[start : end] = torch.exp(-1 * torch.mean(ce_loss, dim=-1))
    return ppl
    
def calculate_coherence():
    pass
    
def calculate_diversity(generations, k=4):
    """
    Calculate diversity of generations with k ngrams.
    Args:
        generations: List[List[List[int]]]
    """
    # generations is n_prompts, n_drafts, len_seq
    nltk.download('punkt')
    smooth = SmoothingFunction()
    bleus = []
    
    # generations needs List[List[str]], [n_prompts, n_drafts]
    for drafts in generations:
        tokenized_drafts = []
        # convert tokens to str versions
        for d in drafts:
            if -1 in d:
                d = d[:d.index(-1)]
            tokenized_drafts.append([str(n) for n in d])
        
        # make weights
        minlength = min([len(g) for g in tokenized_drafts])
        minlength = min(minlength, k)
        weights = tuple((1. / minlength for _ in range(minlength)))
        for i in range(len(drafts)):
            # need compare str to list of str
            src = tokenized_drafts[i]
            ref = tokenized_drafts[:i] + tokenized_drafts[i+1:]
            # print(src)
            # print(ref)
            tmp = nltk.translate.bleu_score.sentence_bleu(references=ref, 
                                                          hypothesis=src, 
                                                          weights=weights,
                                                          smoothing_function=smooth.method1)
            # print(tmp)
            bleus.append(tmp)
    bleus = torch.Tensor(bleus)
    return torch.mean(bleus)
