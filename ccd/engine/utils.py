"""Training utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_device(pref: str = "cuda"):
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, path, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model.state_dict()}
    payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return ckpt
