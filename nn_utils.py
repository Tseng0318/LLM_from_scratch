import torch

def cross_entropy(inputs, targets):
    # input = (batch, vocab_size)  target=(batch, )
    #     inputs = [[1.0, 3.0, 2.0, 0.0],      ← example 0's logits
    #               [0.0, 1.0, 0.0, 4.0]]      ← example 1's logits
    #               shape (2, 4)

    # targets = [1, 3]                      ← example 0's correct word is index 1. example 1's correct word is index 3
    #            shape (2,)                   
    x_max = inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = x_max.squeeze(-1) + torch.log(torch.exp(inputs - x_max).sum(dim=-1))
    target_logits = inputs[torch.arange(inputs.shape[0]), targets]   # advanced indexing
    #  inputs = [[ 1.0,  3.0,  2.0,  0.0],     ← row 0
    #            [ 0.0,  1.0,  0.0,  4.0],     ← row 1
    #            [ 5.0,  2.0,  1.0,  3.0]]     ← row 2

    # targets = [1, 3, 0] from row 0 grab index 1 
    loss = log_sum_exp - target_logits
    return loss.mean()