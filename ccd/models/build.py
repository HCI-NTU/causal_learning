"""Detector construction for the Faster R-CNN family.

build_detector(model="erm"|"aug"|"causal") returns a torchvision FasterRCNN.
  * erm / aug -> identical vanilla detector (they differ only in the *data*
                 augmentation level, set in the config, so any gap is attributable
                 to augmentation, not architecture).
  * causal    -> same detector with roi_heads converted to CausalRoIHeads and the
                 box predictor replaced by CausalPredictor over an EMA dictionary.

YOLOv11 / RT-DETR are built through the ultralytics package, handled separately
in scripts/train_ultralytics.py (they are baseline-only alternative detectors).
"""
from __future__ import annotations

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ccd.data.class_mapping import NUM_CLASSES
from ccd.models.causal_roi_head import (CausalPredictor, CausalRoIHeads,
                                        ConfounderDictionary)


def build_detector(model: str = "erm", pretrained_backbone: bool = True,
                   dict_slots: int = 50, dict_momentum: float = 0.3,
                   fuse: str = "concat", min_size: int = 800,
                   max_size: int = 1333):
    num_classes = NUM_CLASSES + 1  # +1 background

    weights_backbone = "DEFAULT" if pretrained_backbone else None
    net = fasterrcnn_resnet50_fpn(weights=None,
                                  weights_backbone=weights_backbone,
                                  num_classes=num_classes,
                                  min_size=min_size, max_size=max_size)

    if model in ("erm", "aug"):
        return net

    if model == "causal":
        in_dim = net.roi_heads.box_predictor.cls_score.in_features
        cdict = ConfounderDictionary(num_fg=NUM_CLASSES, dim=in_dim,
                                     slots=dict_slots, momentum=dict_momentum)
        net.roi_heads.box_predictor = CausalPredictor(
            in_dim, num_classes, cdict, fuse=fuse)
        # convert the RoIHeads instance to the causal subclass in place
        net.roi_heads.__class__ = CausalRoIHeads
        net.roi_heads.set_causal(cdict)
        return net

    raise ValueError(f"unknown model '{model}' (use erm | aug | causal)")
