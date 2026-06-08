"""Datasets and the split-resolution registry.

Directory layout this assumes (exactly as specified):

  data/MOCS/images/{train,val}
  data/MOCS/instances_{train,val}.json
  data/MOCS/instances_{train,val}_{earthmoving,foundation,superstructure}.json

  data/CIS/images/{train,val,test}
  data/CIS/instances_{train,val,test}.json

  data/SODA/images/{train,test}
  data/SODA/annotations/*.xml            (PASCAL VOC; converted on the fly)

  data/ACID/images/{train,test}
  data/ACID/instances_{all,train,test}.json

  data/ExtCon/images
  data/ExtCon/extcon_gt.json

Split-resolution rule (your spec):
  * No separate `test` split  -> use `val` as the test set.
  * A separate `test` split    -> merge `val` into `train` as `trainval`; `test`
                                  is the held-out test set.

For the two headline experiments we only ever *train* on MOCS, so the trainval
merge matters only if you later train a single-source model on CIS/ACID/SODA.
It is encoded generally so those experiments are one config away.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


# ----------------------------------------------------------------------------
# COCO-format detection dataset
# ----------------------------------------------------------------------------
class CocoDetection(Dataset):
    """Minimal COCO detection dataset returning torchvision-style targets.

    target = dict(boxes[N,4] xyxy float, labels[N] int64, image_id, area,
                  iscrowd, orig_size)
    """

    def __init__(self, img_dir: str, ann_file: str,
                 transforms: Optional[Callable] = None,
                 remap: Optional[Dict[int, int]] = None,
                 keep_unmapped: bool = False):
        from pycocotools.coco import COCO  # local import; heavy dep
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms
        self.remap = remap            # ext_cat_id -> target_id (for eval remap)
        self.keep_unmapped = keep_unmapped

    def __len__(self) -> int:
        return len(self.ids)

    def _load_target(self, img_id: int):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            cat = a["category_id"]
            if self.remap is not None:
                if cat not in self.remap:
                    if not self.keep_unmapped:
                        continue
                    continue
                cat = self.remap[cat]
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(cat)
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))
        return boxes, labels, areas, iscrowd

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info["file_name"])
        img = Image.open(path).convert("RGB")
        boxes, labels, areas, iscrowd = self._load_target(img_id)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "orig_size": torch.tensor([info["height"], info["width"]]),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
@dataclass
class SplitSpec:
    name: str
    img_dir: str
    ann_file: str
    label_space: str = "mocs"          # "mocs" or "external"
    role: str = "test"                 # "train" | "test" | "trainval"
    extra: dict = field(default_factory=dict)


def _p(*parts) -> str:
    return os.path.join(*parts)


def build_registry(root: str = "data") -> Dict[str, SplitSpec]:
    """Resolve every split used by the experiments, applying the merge rule."""
    R: Dict[str, SplitSpec] = {}

    # ----- MOCS (no test split -> val is the in-domain test) -----
    mocs = _p(root, "MOCS")
    R["mocs_train"] = SplitSpec("mocs_train", _p(mocs, "images/train"),
                                _p(mocs, "instances_train.json"), "mocs", "train")
    R["mocs_test"] = SplitSpec("mocs_test", _p(mocs, "images/val"),
                               _p(mocs, "instances_val.json"), "mocs", "test")
    for stage in ("earthmoving", "foundation", "superstructure"):
        R[f"mocs_train_{stage}"] = SplitSpec(
            f"mocs_train_{stage}", _p(mocs, "images/train"),
            _p(mocs, f"instances_train_{stage}.json"), "mocs", "train")
        # stage val splits are pure test (cross-stage targets)
        R[f"mocs_test_{stage}"] = SplitSpec(
            f"mocs_test_{stage}", _p(mocs, "images/val"),
            _p(mocs, f"instances_val_{stage}.json"), "mocs", "test")

    # ----- CIS (has test -> val merged to train as trainval; test = target) -----
    cis = _p(root, "CIS")
    R["cis_trainval"] = SplitSpec(  # consumed as ConcatDataset, see make_dataset
        "cis_trainval", _p(cis, "images"),  # img_dir is a base; see extra
        "", "external", "trainval",
        extra={"merge": [(_p(cis, "images/train"), _p(cis, "instances_train.json")),
                          (_p(cis, "images/val"), _p(cis, "instances_val.json"))]})
    R["cis_test"] = SplitSpec("cis_test", _p(cis, "images/test"),
                              _p(cis, "instances_test.json"), "external", "test")

    # ----- ACID (has test -> test = target). No val listed; train stays train. -----
    acid = _p(root, "ACID")
    R["acid_train"] = SplitSpec("acid_train", _p(acid, "images/train"),
                                _p(acid, "instances_train.json"), "external", "train")
    R["acid_test"] = SplitSpec("acid_test", _p(acid, "images/test"),
                               _p(acid, "instances_test.json"), "external", "test")

    # ----- SODA (VOC; has test -> test = target). Converted to COCO json. -----
    soda = _p(root, "SODA")
    R["soda_train"] = SplitSpec("soda_train", _p(soda, "images/train"),
                                _p(soda, "instances_train.json"), "external", "train",
                                extra={"voc_xml_dir": _p(soda, "annotations")})
    R["soda_test"] = SplitSpec("soda_test", _p(soda, "images/test"),
                               _p(soda, "instances_test.json"), "external", "test",
                               extra={"voc_xml_dir": _p(soda, "annotations")})

    # ----- ExtCon (single set -> all test) -----
    extcon = _p(root, "ExtCon")
    R["extcon_test"] = SplitSpec("extcon_test", _p(extcon, "images"),
                                 _p(extcon, "extcon_gt.json"), "external", "test")

    return R


def make_dataset(spec: SplitSpec, transforms=None, remap=None,
                 keep_unmapped: bool = False) -> Dataset:
    """Instantiate a torch Dataset from a SplitSpec (handles trainval merge)."""
    if spec.role == "trainval" and "merge" in spec.extra:
        parts = [CocoDetection(img_dir, ann, transforms, remap, keep_unmapped)
                 for img_dir, ann in spec.extra["merge"]]
        return ConcatDataset(parts)
    return CocoDetection(spec.img_dir, spec.ann_file, transforms, remap, keep_unmapped)
