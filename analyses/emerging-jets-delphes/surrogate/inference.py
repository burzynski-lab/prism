"""Preprocessing and scoring of padded particle arrays with the released
checkpoint (``surrogate.pt``: architecture, weights, normalization, PDG
vocabulary)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .features import model_space
from .model import ParticleTransformer


@dataclass
class Preprocessor:
    floats: list[str]
    cats: list[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    vocab: list[dict[int, int]] = field(default_factory=list)   # per cat: raw id -> index (0 pad, 1 unknown)

    def transform(self, padded: np.ndarray):
        valid = padded["valid"]
        x = np.stack([model_space(f, padded[f]) for f in self.floats], axis=-1)
        x = (x - self.mean) / self.std
        x = np.where(valid[..., None], x, 0.0).astype(np.float32)
        cats = np.zeros(padded.shape + (len(self.cats),), dtype=np.int64)
        for k, c in enumerate(self.cats):
            raw = padded[c]
            lut_keys = np.array(list(self.vocab[k].keys()), dtype=np.int64)
            lut_vals = np.array(list(self.vocab[k].values()), dtype=np.int64)
            pos = np.clip(np.searchsorted(lut_keys, raw), 0, len(lut_keys) - 1)
            known = lut_keys[pos] == raw
            cats[..., k] = np.where(valid, np.where(known, lut_vals[pos], 1), 0)
        return x, cats, valid

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(d["floats"], d["cats"], np.array(d["mean"], np.float32), np.array(d["std"], np.float32),
                   [{int(k): v for k, v in dd.items()} for dd in d["vocab"]])


def load_checkpoint(path: Path, device="cpu") -> tuple[ParticleTransformer, Preprocessor]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ParticleTransformer(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval().to(device), Preprocessor.from_dict(ck["preprocessor"])


@torch.no_grad()
def score_padded(model, pre: Preprocessor, padded: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    """Logits for a padded (jet, slot) array of any size, batched."""
    if len(padded) == 0:
        return np.zeros(0, np.float32)
    x, c, m = pre.transform(padded)
    x, c, m = torch.as_tensor(x), torch.as_tensor(c), torch.as_tensor(m)
    out = []
    for lo in range(0, len(x), batch_size):
        b = slice(lo, lo + batch_size)
        out.append(model(x[b].to(device), c[b].to(device), m[b].to(device)).float().cpu().numpy())
    return np.concatenate(out)
