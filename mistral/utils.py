import torch

def create_mixed_matrix(probs, k):
    """
    Create token mixing matrix.
    
    Args:
        probs (torch.Tensor): Token probabilities
        k (int): Number of tokens to mix
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    top_k_mask = torch.arange(probs.shape[-1])
    top_k_mask = top_k_mask.expand(probs.shape[0], -1) # duplicate along dim=0 to match probs array
    top_k_mask = top_k_mask >= k # set to 1 past k elements
    probs_sort[top_k_mask] = 0.0 # 0 wherever mask = 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    restore_idx = torch.argsort(probs_idx, dim=-1)
    restored_probs = torch.gather(probs_sort, 1, restore_idx)
    return probs_sort, probs_idx, restored_probs

def log_prob_to_prob(log_probs, temp=1):
    """
    Convert log probabilities to probability distribution and normalize.
    Args:
        log_probs (torch.Tensor): Log probs (n_prompts, n_drafts, vocab_size)
    """
    # stability constant
    log_probs = log_probs + torch.max(log_probs, dim=-1, keepdim=True)[0]
    probs = torch.softmax(log_probs / temp, dim=-1)
    return probs