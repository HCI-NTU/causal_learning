"""Transforms + counterfactual copy-paste.

All transforms operate on (PIL.Image | Tensor, target_dict) pairs and keep boxes
consistent. `build_transforms(level)` gives three matched strengths:

  level="none"  -> ToTensor only            (used by ERM eval and ERM train)
  level="weak"  -> hflip + light photometric (ERM train default)
  level="strong"-> hflip + photometric + scale jitter (the augmentation-matched
                   control; shared by `aug` and `causal` so the causal gain is
                   attributable to the head, not to augmentation)

Counterfactual copy-paste is implemented as a *dataset wrapper* (boxes only, per
your boxes-only constraint): with probability p it crops an instance from a
randomly chosen other image and pastes it at a random location, adding its box.
This breaks the object<->background correlation that the causal head then has to
become invariant to.
"""
from __future__ import annotations

import random
from typing import Callable, List

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class Compose:
    def __init__(self, ts: List[Callable]):
        self.ts = ts

    def __call__(self, img, target):
        for t in self.ts:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            w = img.shape[-1] if torch.is_tensor(img) else img.size[0]
            b = target["boxes"]
            if b.numel():
                b = b.clone()
                b[:, [0, 2]] = w - b[:, [2, 0]]
                target["boxes"] = b
        return img, target


class PhotometricDistort:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        self.b, self.c, self.s, self.h = brightness, contrast, saturation, hue

    def __call__(self, img, target):
        # operate on PIL; expects to run before ToTensor
        if random.random() < 0.5:
            img = F.adjust_brightness(img, 1 + random.uniform(-self.b, self.b))
        if random.random() < 0.5:
            img = F.adjust_contrast(img, 1 + random.uniform(-self.c, self.c))
        if random.random() < 0.5:
            img = F.adjust_saturation(img, 1 + random.uniform(-self.s, self.s))
        if random.random() < 0.5:
            img = F.adjust_hue(img, random.uniform(-self.h, self.h))
        return img, target


class RandomScaleJitter:
    """Resize the shorter side to a random target within [lo, hi]."""
    def __init__(self, sizes=(480, 512, 544, 576, 608, 640)):
        self.sizes = sizes

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        w, h = (img.size if not torch.is_tensor(img) else (img.shape[-1], img.shape[-2]))
        scale = size / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img = F.resize(img, [nh, nw])
        if target["boxes"].numel():
            target["boxes"] = target["boxes"] * scale
        return img, target


def build_transforms(level: str = "none", train: bool = True) -> Compose:
    ts: List[Callable] = []
    if train and level in ("weak", "strong"):
        ts.append(PhotometricDistort())
    if train and level == "strong":
        ts.append(RandomScaleJitter())
    ts.append(ToTensor())
    if train and level in ("weak", "strong"):
        ts.append(RandomHorizontalFlip(0.5))
    return Compose(ts)


# ----------------------------------------------------------------------------
# Counterfactual copy-paste (dataset wrapper, boxes only)
# ----------------------------------------------------------------------------
class CounterfactualCopyPaste(Dataset):
    """Wrap a detection dataset; paste a foreign instance crop with prob `p`.

    Works on tensor images (apply *after* ToTensor, i.e. wrap the dataset whose
    transforms already produce tensors). Boxes only — no masks.
    """

    def __init__(self, base: Dataset, p: float = 0.5, max_paste: int = 2,
                 min_box: int = 16):
        self.base = base
        self.p = p
        self.max_paste = max_paste
        self.min_box = min_box

    def __len__(self):
        return len(self.base)

    def _sample_instance(self):
        for _ in range(8):
            j = random.randrange(len(self.base))
            img, tgt = self.base[j]
            if tgt["boxes"].numel() == 0:
                continue
            k = random.randrange(tgt["boxes"].shape[0])
            x1, y1, x2, y2 = tgt["boxes"][k].round().int().tolist()
            if (x2 - x1) < self.min_box or (y2 - y1) < self.min_box:
                continue
            crop = img[:, y1:y2, x1:x2]
            if crop.numel() == 0:
                continue
            return crop, int(tgt["labels"][k])
        return None

    def __getitem__(self, idx):
        img, target = self.base[idx]
        if random.random() >= self.p or not torch.is_tensor(img):
            return img, target
        _, H, W = img.shape
        new_boxes, new_labels = [], []
        for _ in range(random.randint(1, self.max_paste)):
            s = self._sample_instance()
            if s is None:
                continue
            crop, label = s
            ch, cw = crop.shape[1], crop.shape[2]
            if ch >= H or cw >= W:
                continue
            px = random.randint(0, W - cw)
            py = random.randint(0, H - ch)
            img[:, py:py + ch, px:px + cw] = crop
            new_boxes.append([px, py, px + cw, py + ch])
            new_labels.append(label)
        if new_boxes:
            nb = torch.tensor(new_boxes, dtype=torch.float32)
            nl = torch.tensor(new_labels, dtype=torch.int64)
            target = dict(target)
            target["boxes"] = torch.cat([target["boxes"], nb], 0)
            target["labels"] = torch.cat([target["labels"], nl], 0)
            area = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
            target["area"] = torch.cat([target["area"], area], 0)
            target["iscrowd"] = torch.cat(
                [target["iscrowd"], torch.zeros(len(new_labels), dtype=torch.int64)], 0)
        return img, target
