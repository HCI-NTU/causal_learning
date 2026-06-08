"""RT-DETR query-level causal port (full method).

RT-DETR's decoder produces a per-object query embedding (analogous to a Faster
R-CNN RoI feature), then a per-layer classification head `dec_score_head[i]`
(Linear hd->nc) scores it. We replace each score head with a `CausalQueryScoreHead`
that:
  1. disentangles the query into causal/spurious (residual split),
  2. updates a class-agnostic spurious-prototype dictionary (the confounder),
  3. backdoor-adjusts the causal feature against that dictionary,
  4. classifies the deconfounded feature.
A decorrelation loss between causal and spurious is exposed via `.last_aux` and
summed by the trainer (see ccd/engine/ultralytics_causal.py).

This swaps modules only — it does not edit the decoder's forward — so it is robust
to most ultralytics versions. The one version-sensitive assumption is the
attribute name `dec_score_head` on `RTDETRDecoder`; patch `find_decoder` if it
differs in your install.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ccd.models.causal_common import (BackdoorCombine, PrototypeDictionary,
                                      ResidualDisentangler, decorrelation_loss)


class CausalQueryScoreHead(nn.Module):
    """Drop-in replacement for a decoder score head (Linear hd -> nc)."""

    def __init__(self, orig_linear: nn.Linear, dictionary: PrototypeDictionary,
                 mode: str = "concat", lambda_decorr: float = 0.1):
        super().__init__()
        hd = orig_linear.in_features
        nc = orig_linear.out_features
        self.dict = dictionary
        self.disentangle = ResidualDisentangler(hd)
        self.combine = BackdoorCombine(causal_dim=hd, dict_dim=hd,
                                       out_dim=nc, mode=mode)
        self.lambda_decorr = lambda_decorr
        self.last_aux: Tensor = torch.zeros(())

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, nq, hd] (or [N, hd]); operate on the last dim throughout.
        causal, spurious = self.disentangle(x)
        if self.training:
            flat_s = spurious.detach().reshape(-1, spurious.shape[-1])
            self.dict.update(flat_s, flat_s.new_zeros(flat_s.shape[0],
                                                      dtype=torch.long))
            self.last_aux = self.lambda_decorr * decorrelation_loss(
                causal.reshape(-1, causal.shape[-1]),
                spurious.reshape(-1, spurious.shape[-1]))
        else:
            self.last_aux = x.new_zeros(())
        return self.combine(causal, self.dict)


def find_decoder(model):
    """Locate the RTDETRDecoder inside an ultralytics RTDETR / RTDETRDetectionModel."""
    from ultralytics.nn.modules.head import RTDETRDecoder
    core = model.model if not isinstance(model, nn.Module) else model
    for m in core.modules():
        if isinstance(m, RTDETRDecoder):
            return m
    raise RuntimeError("RTDETRDecoder not found — patch find_decoder for your "
                       "ultralytics version.")


def install_causal_rtdetr(model, slots: int = 64, momentum: float = 0.2,
                          mode: str = "concat",
                          lambda_decorr: float = 0.1) -> List[CausalQueryScoreHead]:
    """Replace every decoder score head with a causal one (shared dictionary).

    Returns the list of installed causal heads (their `.last_aux` are summed by
    the trainer).
    """
    decoder = find_decoder(model)
    if not hasattr(decoder, "dec_score_head"):
        raise RuntimeError("decoder has no `dec_score_head`; patch for your version.")
    hd = decoder.dec_score_head[0].in_features
    dictionary = PrototypeDictionary(num_fg=1, dim=hd, slots=slots,
                                     momentum=momentum)
    heads: List[CausalQueryScoreHead] = []
    for i, lin in enumerate(decoder.dec_score_head):
        head = CausalQueryScoreHead(lin, dictionary, mode=mode,
                                    lambda_decorr=lambda_decorr)
        decoder.dec_score_head[i] = head
        heads.append(head)
    # keep references so the dictionary/heads move with .to() and are saved
    decoder.add_module("_causal_dict", dictionary)
    decoder._causal_heads = heads
    return heads
