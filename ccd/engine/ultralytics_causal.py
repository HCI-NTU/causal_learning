"""Glue between the causal ports and ultralytics training.

`build_causal_model(arch, ...)` loads a YOLOv11 or RT-DETR model, installs the
appropriate causal port, and patches the model's `loss` so the decorrelation /
backdoor auxiliary terms are added to the detection loss.

VERSION CAVEAT (read this): ultralytics computes the loss inside the model
(`DetectionModel.loss` / `RTDETRDetectionModel.loss`). We wrap that method so the
auxiliary loss from the same forward pass is added before backward. This is the
most version-sensitive seam in the repo. If your ultralytics build reconstructs
the model inside the trainer (so the patch is lost), use the custom-trainer path
sketched at the bottom of this file instead. Verify on-device.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ccd.models.causal_rtdetr import (CausalQueryScoreHead,
                                      install_causal_rtdetr)
from ccd.models.causal_yolo import install_causal_yolo

RTDETR_ARCHES = {"rtdetr-l", "rtdetr-x"}
YOLO_ARCHES = {"yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"}


def _core(model) -> nn.Module:
    return model.model if not isinstance(model, nn.Module) else model


def collect_aux(model) -> torch.Tensor:
    """Sum the current auxiliary loss from all installed causal modules."""
    core = _core(model)
    total = None
    for m in core.modules():
        aux = None
        if isinstance(m, CausalQueryScoreHead):
            aux = m.last_aux
        elif hasattr(m, "_aux") and hasattr(m, "causal"):   # CausalDetect
            aux = m._aux
        if aux is None:
            continue
        aux = aux if torch.is_tensor(aux) else torch.as_tensor(float(aux))
        total = aux if total is None else total + aux
    if total is None:
        # no causal module produced a graph-connected scalar this step
        p = next(core.parameters())
        return p.new_zeros(())
    return total


def attach_causal_loss(model, lambda_aux: float = 1.0):
    """Monkeypatch core.loss to add the collected auxiliary loss."""
    core = _core(model)
    orig_loss = core.loss  # bound method

    def causal_loss(batch, preds=None):
        total, items = orig_loss(batch, preds)
        aux = collect_aux(core)
        return total + lambda_aux * aux, items

    core.loss = causal_loss
    return model


def build_causal_model(arch: str, weights: str = None,
                       lambda_aux: float = 1.0, lambda_decorr: float = 0.1,
                       slots: int = 64, momentum: float = 0.2,
                       use_bank: bool = True, mode: str = "concat"):
    """Load an ultralytics model and install the causal port + aux loss."""
    from ultralytics import RTDETR, YOLO

    if arch in RTDETR_ARCHES:
        model = RTDETR(weights or f"{arch}.pt")
        install_causal_rtdetr(model, slots=slots, momentum=momentum, mode=mode,
                              lambda_decorr=lambda_decorr)
    elif arch in YOLO_ARCHES:
        model = YOLO(weights or f"{arch}.pt")
        install_causal_yolo(model, use_bank=use_bank, slots=slots,
                            momentum=momentum, lambda_decorr=lambda_decorr)
    else:
        raise ValueError(f"unknown arch '{arch}'")

    attach_causal_loss(model, lambda_aux=lambda_aux)
    return model


# ---------------------------------------------------------------------------
# Fallback if the loss monkeypatch does not survive your ultralytics trainer:
# subclass the trainer and re-install in get_model. Sketch (verify on-device):
#
#   from ultralytics.models.yolo.detect import DetectionTrainer
#   from ccd.models.causal_yolo import install_causal_yolo
#   class CausalDetectionTrainer(DetectionTrainer):
#       def get_model(self, cfg=None, weights=None, verbose=True):
#           model = super().get_model(cfg, weights, verbose)
#           install_causal_yolo(model)            # model is the nn.Module here
#           orig = model.loss
#           def loss(batch, preds=None):
#               t, it = orig(batch, preds); return t + collect_aux(model), it
#           model.loss = loss
#           return model
#   # then: CausalDetectionTrainer(overrides=dict(model='yolo11m.pt',
#   #         data='data_yolo/mocs/data.yaml', epochs=100)).train()
# ---------------------------------------------------------------------------
