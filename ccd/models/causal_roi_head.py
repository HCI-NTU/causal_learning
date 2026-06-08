"""Causal box head: backdoor adjustment over a confounder dictionary.

The intervention sits between the RoI feature extractor (`box_head`, output dim
D) and the classifier. We approximate

    P(Y | do(X=x)) = sum_c P(Y | x, c) P(c)

via the NWGM/attention approximation used by VC R-CNN / IFSL: a single attended
context vector c_hat = sum over dictionary prototypes z of softmax(x.z) * z * P(c)
is fused with x and classified once. The dictionary is a per-class EMA memory
bank of real RoI prototypes (Zhang TPAMI'24 style: ~K slots/class, momentum
gamma), updated online from positive RoIs during training.

`CausalRoIHeads` reproduces torchvision's RoIHeads.forward (boxes only — no
mask/keypoint branch) and inserts the dictionary update. Targets torchvision
>=0.15; if your version changed the RoIHeads API, this is the one place to patch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss


class ConfounderDictionary(nn.Module):
    """Per-class EMA memory bank of RoI prototypes."""

    def __init__(self, num_fg: int, dim: int, slots: int = 50,
                 momentum: float = 0.3):
        super().__init__()
        self.num_fg = num_fg
        self.dim = dim
        self.slots = slots
        self.momentum = momentum
        # bank[c] holds `slots` prototypes for foreground class c (0..num_fg-1)
        self.register_buffer("bank", torch.zeros(num_fg, slots, dim))
        self.register_buffer("filled", torch.zeros(num_fg, slots, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros(num_fg, dtype=torch.long))
        self.register_buffer("class_count", torch.ones(num_fg))  # for P(c)

    @torch.no_grad()
    def update(self, feats: Tensor, labels: Tensor):
        """feats[N,D], labels[N] in 0..num_fg (0 == background, skipped)."""
        for f, l in zip(feats, labels):
            c = int(l) - 1
            if c < 0:                      # background
                continue
            self.class_count[c] += 1
            p = int(self.ptr[c])
            if self.filled[c, p]:
                self.bank[c, p] = (1 - self.momentum) * self.bank[c, p] + \
                    self.momentum * f
            else:
                self.bank[c, p] = f
                self.filled[c, p] = True
            self.ptr[c] = (p + 1) % self.slots

    def prototypes(self) -> Tuple[Tensor, Tensor]:
        """Return (Z[M,D], prior[M]) over all filled prototypes; P(c) split evenly
        across that class's filled slots."""
        zs, ws = [], []
        prior_c = self.class_count / self.class_count.sum()
        for c in range(self.num_fg):
            mask = self.filled[c]
            n = int(mask.sum())
            if n == 0:
                continue
            zs.append(self.bank[c, mask])
            ws.append(prior_c[c].repeat(n) / n)
        if not zs:
            return (self.bank.new_zeros(0, self.dim),
                    self.bank.new_zeros(0))
        return torch.cat(zs, 0), torch.cat(ws, 0)


class CausalPredictor(nn.Module):
    """FastRCNNPredictor with a backdoor-adjusted feature.

    cls/bbox heads consume the fused [x ; c_hat] vector. When the dictionary is
    empty (start of training) it degrades gracefully to a vanilla predictor.
    """

    def __init__(self, in_dim: int, num_classes: int,
                 dictionary: ConfounderDictionary, fuse: str = "concat"):
        super().__init__()
        self.dict = dictionary
        self.fuse = fuse
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Linear(in_dim, in_dim)
        feat_dim = in_dim * 2 if fuse == "concat" else in_dim
        if fuse == "gate":
            self.gate = nn.Linear(in_dim * 2, in_dim)
        self.cls_score = nn.Linear(feat_dim, num_classes)
        self.bbox_pred = nn.Linear(feat_dim, num_classes * 4)
        self.scale = in_dim ** 0.5

    def _context(self, x: Tensor) -> Tensor:
        Z, prior = self.dict.prototypes()         # [M,D], [M]
        if Z.shape[0] == 0:
            return torch.zeros_like(x)
        q = self.q(x)                              # [N,D]
        k = self.k(Z)                              # [M,D]
        att = (q @ k.t()) / self.scale             # [N,M]
        att = Fn.softmax(att, dim=1) * prior.unsqueeze(0)   # weight by P(c)
        att = att / att.sum(1, keepdim=True).clamp_min(1e-6)
        return att @ Z                             # [N,D]

    def forward(self, x: Tensor):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        c_hat = self._context(x)
        if self.fuse == "concat":
            fused = torch.cat([x, c_hat], dim=1)
        elif self.fuse == "gate":
            g = torch.sigmoid(self.gate(torch.cat([x, c_hat], dim=1)))
            fused = x + g * c_hat
        else:  # residual
            fused = x + c_hat
        return self.cls_score(fused), self.bbox_pred(fused)


class CausalRoIHeads(RoIHeads):
    """RoIHeads that updates the confounder dictionary and uses CausalPredictor.

    Build a vanilla torchvision FasterRCNN, then convert its roi_heads with
    `convert_roi_heads(...)` in build.py.
    """

    def set_causal(self, dictionary: ConfounderDictionary):
        self.cdict = dictionary

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, _, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)         # [N, D]

        if self.training and hasattr(self, "cdict"):
            self.cdict.update(box_features.detach(), torch.cat(labels, 0))

        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, Tensor]] = []
        losses: Dict[str, Tensor] = {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier,
                      "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, lbls = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": lbls[i],
                               "scores": scores[i]})
        return result, losses
