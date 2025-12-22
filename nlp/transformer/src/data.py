from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

_TINY_CORPUS = """\
In the beginning there was attention.
Attention is all you need.
Transformers learn long-range dependencies efficiently.
This is a tiny corpus for a tiny character-level language model.
"""


@dataclass
class TextData:
    text: str
    stoi: Dict[str, int]
    itos: Dict[int, str]
    vocab_size: int
    data: torch.Tensor


def load_text(path: str) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return _TINY_CORPUS


def build_vocab(text: str) -> Tuple[Dict[str, int], dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(tokens: torch.Tensor, itos: Dict[int, str]) -> str:
    return "".join([itos[int(i)] for i in tokens])


def make_batch(
    stream: torch.Tensor, block_size: int, batch_size: int, device: torch.device
):
    """
    stream: (N,) tokens
    returns:
      x: (B, T)
      y: (B, T) next-token targets
    """
    N = stream.size(0)
    idx = torch.randint(0, N - block_size - 1, (batch_size,))
    x = torch.stack([stream[i : i + block_size] for i in idx])
    y = torch.stack([stream[i + 1 : i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)


def load_dataset(text_path: str) -> TextData:
    text = load_text(text_path)
    stoi, itos = build_vocab(text)
    stream = encode(text, stoi)
    return TextData(text=text, stoi=stoi, itos=itos, vocab_size=len(stoi), data=stream)
