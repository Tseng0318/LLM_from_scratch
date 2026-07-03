import numpy as np
import torch

from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.serialization import save_checkpoint


def train(
    data_path,
    vocab_size,
    context_length,
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta=10000.0,
    batch_size=32,
    max_steps=5000,
    max_lr=1e-3,
    min_lr=1e-4,
    warmup_steps=100,
    max_grad_norm=1.0,
    weight_decay=0.01,
    device="cpu",
    log_every=100,
    checkpoint_path="checkpoint.pt",
):
    # --- setup ---
    data = np.load(data_path, mmap_mode="r")           # memmap the token array

    model = TransformerLM(
        vocab_size, context_length, d_model, num_layers,
        num_heads, d_ff, rope_theta, device=device,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    # --- training loop ---
    for step in range(max_steps):
        # 1-2. set learning rate for this step
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 3. sample a batch
        inputs, targets = get_batch(data, batch_size, context_length, device)

        # 4-5. forward + loss
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        # 6-9. backward + clip + step
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_grad_norm)
        optimizer.step()

        # logging
        if step % log_every == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.6f}")

    save_checkpoint(model, optimizer, max_steps, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    # smoke test on random tokens
    import tempfile, os
    fake = np.random.randint(0, 1000, size=100_000).astype(np.uint16)
    path = os.path.join(tempfile.gettempdir(), "fake_tokens.npy")
    np.save(path, fake)

    train(
        data_path=path,
        vocab_size=1000,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        batch_size=8,
        max_steps=50,
        device="mps",
        log_every=10,
    )