import os
import re
import regex
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _merge_word(word: tuple[bytes, ...], A: bytes, B: bytes) -> tuple[bytes, ...]:
    """Replace every adjacent (A, B) in `word` with a single A+B token."""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == A and word[i + 1] == B:
            new_word.append(A + B)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    vocab = {i: bytes([i]) for i in range(256)} 
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")  # special token
        next_id += 1

    # create the merged list here
    merges = []

    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        pieces = re.split(pattern, text)
    else:
        pieces = [text]

    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    # example pieces = ['low low low', 'widest newest']
    for piece in pieces:
        for m in regex.finditer(PAT, piece):
            word = m.group(0)
            key = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_freqs[key] += 1

    # ---------- Step 4: merge loop ----------
    num_merges = vocab_size - 256 - len(special_tokens)

    #     word_freqs = {
    #     (b'l', b'o', b'w'):                       1,
    #     (b' ', b'l', b'o', b'w'):                 1,
    #     (b'n', b'e', b'w', b'e', b's', b't'):     1,
    #     (b' ', b'n', b'e', b'w', b'e', b's', b't'): 1,
    # }

    # pair_counts = {
    #     (b'l',b'o'): 2, (b'o',b'w'): 2, (b' ',b'l'): 1,
    #     (b'n',b'e'): 2, (b'e',b'w'): 2, (b'w',b'e'): 2,
    #     (b'e',b's'): 2, (b's',b't'): 2, (b' ',b'n'): 1,
    # }

    # best_pair = (b'w', b'e')
    # A = b'w'
    # B = b'e'

# new_word_freqs doesn't exist yet
    for _ in range(num_merges):
        # 4.1: count adjacent pairs
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, count in word_freqs.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += count # tracking (b'l') and b'o'

        if not pair_counts:           # nothing left to merge (defensive)
            break

        # 4.2: pick the winning pair
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p)) # (2, (b'l', b'o'))
        A, B = best_pair
        #  pair_counts = {
        # (b'l', b'o'): 2,
        # (b'o', b'w'): 2,
        # (b' ', b'l'): 1,
        # (b'w', b'e'): 2,
        # (b'e', b's'): 2,
        # (b's', b't'): 2,}

        # 4.3: apply the merge — rebuild word_freqs
        new_word_freqs: Counter[tuple[bytes, ...]] = Counter()
        for word, count in word_freqs.items():
            new_word_freqs[_merge_word(word, A, B)] += count
            
        word_freqs = new_word_freqs

        # 4.4: record the merge
        merges.append((A, B))
        vocab[next_id] = A + B
        next_id += 1

    return vocab, merges

# # at the bottom of train_bpe.py, temporarily, for debugging
# if __name__ == "__main__":
#     # fake the function inputs
#     import tempfile, pathlib
#     p = pathlib.Path(tempfile.mktemp(suffix=".txt"))
#     p.write_text("low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest\n")
#     vocab, merges = train_bpe(p, vocab_size=262, special_tokens=["<|endoftext|>"])
#     print("merges:", merges)   # will be [] for now