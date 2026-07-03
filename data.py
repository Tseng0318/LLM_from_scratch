import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    #  x = [x0, x1, x2, x3, x4, x5, ...] context_length=3, starting at i=2
    # input  = [x2, x3, x4]
    # target = [x3, x4, x5]    ← each target is the token that follows the input token
    n = len(dataset)
    # random start indices, one per batch element
    starts = np.random.randint(0, n - context_length, size=batch_size)

    # build input and target arrays
    inputs = np.stack([dataset[s : s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])

    # to tensors on the requested device
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs, targets

