# LLM From Scratch

This repository is a small educational implementation of core language-modeling
components from the Stanford CS336 basics track. It includes a byte-pair
encoding tokenizer, a decoder-only Transformer language model, custom AdamW
optimization utilities, data batching, and checkpoint save/load helpers.

The code is intentionally lightweight and readable, making it useful for
experiments, debugging, and understanding how the pieces of a GPT-style model
fit together.

## What's Included

- `train_bpe.py` - trains a byte-level BPE vocabulary and merge table.
- `tokenizer.py` - encodes and decodes text with learned BPE merges and optional
  special tokens.
- `model.py` - implements the Transformer LM stack, including RMSNorm, RoPE,
  multi-head causal self-attention, and SwiGLU feed-forward layers.
- `optimizer.py` - provides AdamW, cosine learning-rate scheduling, and gradient
  clipping.
- `data.py` - samples next-token prediction batches from token arrays.
- `serialization.py` - saves and loads model and optimizer checkpoints.
- `train.py` - contains a simple training loop plus a random-token smoke test.

## Requirements

The project uses Python with:

- `torch`
- `numpy`
- `einops`
- `regex`

Install those dependencies in your preferred environment before running the
examples below.

## Quick Start

From the parent directory of this package, run the built-in smoke test:

```bash
python -m cs336_basics.train
```

The smoke test creates random token data, trains a tiny Transformer for a few
steps, and writes a checkpoint file.

## Train a BPE Tokenizer

```python
from cs336_basics.train_bpe import train_bpe

vocab, merges = train_bpe(
    input_path="data/tiny-corpus.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
)
```

## Encode and Decode Text

```python
from cs336_basics.tokenizer import Tokenizer

tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=["<|endoftext|>"],
)

ids = tokenizer.encode("hello world<|endoftext|>")
text = tokenizer.decode(ids)
```

## Train a Model

`train.train` expects a `.npy` file containing integer token IDs:

```python
from cs336_basics.train import train

train(
    data_path="data/tokens.npy",
    vocab_size=10_000,
    context_length=256,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    batch_size=32,
    max_steps=5_000,
    device="cuda",
    checkpoint_path="checkpoint.pt",
)
```

Use `device="cpu"`, `device="cuda"`, or `device="mps"` depending on your
hardware.
