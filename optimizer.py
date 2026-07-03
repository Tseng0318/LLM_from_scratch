import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # --- initialize state on the first step ---
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v, t = state["m"], state["v"], state["t"]

                # === YOU WRITE THIS PART (the 5 algorithm lines) ===
                # 1. update first moment:   m = β1·m + (1−β1)·g
                m = beta1*m + (1-beta1)*grad
                # 2. update second moment:  v = β2·v + (1−β2)·g²
                v = beta2*v + (1-beta2)*grad**2
                state["m"] = m
                state["v"] = v
                # 3. bias-corrected lr:     α_t = lr · √(1−β2^t) / (1−β1^t)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                # 4. main update:           p ← p − α_t · m / (√v + ε)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                # 5. weight decay:          p ← p − lr·λ·p
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1

        return loss


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return (it/warmup_iters)*max_learning_rate # (it / warmup_iters) * max_lr
    elif it <= cosine_cycle_iters:
        coeff = 0.5*(1+math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))   # 0.5 * (1 + cos((it - Tw)/(Tc - Tw) * pi))
        return min_learning_rate + coeff * (max_learning_rate-min_learning_rate)     # min_lr + coeff * (max_lr - min_lr)
    else:
        return min_learning_rate     # min_lr

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    # gather all grads that exist
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    # total L2 norm across all grads
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))         # sqrt of sum of squares of every grad element
    if total_norm > max_l2_norm:
        scale = max_l2_norm/(total_norm+eps)                                   # max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)                             # scale in place