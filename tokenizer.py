import re
import regex
from collections.abc import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from cs336_basics.train_bpe import _merge_word

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # TODO: store vocab, merges, special_tokens
        # TODO: build reverse map bytes -> id (for fast lookup in encode)
        self.vocab =vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {b: i for i, b in vocab.items()}

        for tok in self.special_tokens:
            b = tok.encode(encoding="utf-8")
            if b not in self.bytes_to_id:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = b
                self.bytes_to_id[b] = new_id

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        # skip this for now — tests don't use it.
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        # Phase 1: split on special tokens, keeping them
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
            pieces = re.split(pattern, text)
        else:
            pieces = [text]

        # Phase 2: process each piece  ← next step
        for piece in pieces:
            if piece in self.special_tokens:
                ids.append(self.bytes_to_id[piece.encode("utf-8")])
            else:
                # pre-tokenize + merge + lookup  ← we'll fill this in
                for m in regex.finditer(PAT, piece):
                    pretoken = m.group(0)
                    tokens = tuple(bytes([b]) for b in pretoken.encode("utf-8"))

                    for A, B in self.merges:
                        tokens = _merge_word(tokens, A, B)
                    for t in tokens:
                        ids.append(self.bytes_to_id[t])

        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")