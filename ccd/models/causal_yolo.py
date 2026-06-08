"""YOLOv11 partial causal port.

A dense YOLO head predicts over grid cells with no per-object feature, so the
per-object backdoor dictionary does not port. What *does* port is the
**causal/spurious disentanglement** of the head-input feature maps:

  * a channel-attention mask m in (0,1)^C splits each neck feature x into
    causal = x * m and spurious = x * (1-m)  (exact additive split),
  * a decorrelation loss pushes the channel-pooled causal/spurious apart,
  * an optional class-agnostic global context bank (over channel-pooled spurious)
    supplies a deconfounding context that is gated back in,
  * the module is channel-preserving, so the existing cv2/cv3 detection convs are
    untouched.

This is intentionally the *partial* mechanism (disentanglement + decorrelation +
global context), not the full per-object backdoor.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ccd.models.causal_common import PrototypeDictionary, decorrelation_loss


class ChannelCausalModule(nn.Module):
    """Channel-preserving causal/spurious disentanglement on a feature map."""

    def __init__(self, channels: int, use_bank: bool = True, slots: int = 32,
                 momentum: float = 0.2, lambda_decorr: float = 0.1):
        super().__init__()
        self.c = channels
        self.lambda_decorr = lambda_decorr
        r = max(channels // 8, 8)
        self.mask = nn.Sequential(
            nn.Linear(channels, r), nn.ReLU(inplace=True),
            nn.Linear(r, channels), nn.Sigmoid())
        self.use_bank = use_bank
        if use_bank:
            self.dict = PrototypeDictionary(num_fg=1, dim=channels, slots=slots,
                                            momentum=momentum)
            self.q = nn.Linear(channels, channels)
            self.k = nn.Linear(channels, channels)
            self.gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.last_aux: Tensor = torch.zeros(())

    def _context(self, pooled_causal: Tensor) -> Tensor:
        Z, prior = self.dict.prototypes()              # [M,C],[M]
        if Z.shape[0] == 0:
            return torch.zeros_like(pooled_causal)
        q = self.q(pooled_causal)                       # [B,C]
        k = self.k(Z)                                   # [M,C]
        att = (q @ k.t()) / (q.shape[-1] ** 0.5)
        att = att.softmax(dim=-1) * prior
        att = att / att.sum(-1, keepdim=True).clamp_min(1e-6)
        return att @ Z                                  # [B,C]

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        pooled = x.mean(dim=(2, 3))                      # [B,C] GAP
        m = self.mask(pooled).view(B, C, 1, 1)           # channel mask
        causal = x * m
        spurious = x - causal                            # = x*(1-m)

        if self.training:
            pc = pooled * m.view(B, C)
            ps = pooled - pc
            self.last_aux = self.lambda_decorr * decorrelation_loss(pc, ps)
            if self.use_bank:
                self.dict.update(ps.detach(),
                                 ps.new_zeros(B, dtype=torch.long))
        else:
            self.last_aux = x.new_zeros(())

        if self.use_bank:
            ctx = self._context(causal.mean(dim=(2, 3)))  # [B,C]
            g = self.gate(causal.mean(dim=(2, 3))).view(B, C, 1, 1)
            causal = causal + g * ctx.view(B, C, 1, 1)
        return causal                                    # channel-preserving


def find_detect(model):
    """Locate the YOLO Detect head inside an ultralytics YOLO / DetectionModel."""
    from ultralytics.nn.modules.head import Detect
    core = model.model if not isinstance(model, nn.Module) else model
    last = None
    for m in core.modules():
        if isinstance(m, Detect):
            last = m
    if last is None:
        raise RuntimeError("Detect head not found — patch find_detect.")
    return last


def _level_in_channels(detect) -> List[int]:
    """Infer per-level input channel counts from the cv2 (box) conv stack."""
    chans = []
    for i in range(detect.nl):
        first = detect.cv2[i][0]
        conv = getattr(first, "conv", first)            # ultralytics Conv wraps nn.Conv2d
        chans.append(conv.in_channels)
    return chans


def install_causal_yolo(model, use_bank: bool = True, slots: int = 32,
                        momentum: float = 0.2, lambda_decorr: float = 0.1):
    """Convert the Detect head to CausalDetect and attach per-level causal modules."""
    from ultralytics.nn.modules.head import Detect

    detect = find_detect(model)
    in_ch = _level_in_channels(detect)
    detect.causal = nn.ModuleList([
        ChannelCausalModule(c, use_bank=use_bank, slots=slots,
                            momentum=momentum, lambda_decorr=lambda_decorr)
        for c in in_ch])
    detect._aux: Tensor = torch.zeros(())

    # build the subclass dynamically so we inherit the exact Detect at runtime
    class CausalDetect(type(detect)):
        def forward(self, x):
            self._aux = x[0].new_zeros(()) if isinstance(x, (list, tuple)) else 0
            for i in range(self.nl):
                x[i] = self.causal[i](x[i])
                self._aux = self._aux + self.causal[i].last_aux
            return super().forward(x)

    detect.__class__ = CausalDetect
    return detect
