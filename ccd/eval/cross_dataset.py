"""Cross-dataset evaluation helpers.

The model predicts in the MOCS label space. To score it on an external target we:
  1. load the external GT,
  2. remap its category_id -> MOCS id (drop unmapped annotations),
  3. relabel categories to the shared MOCS subset,
  4. keep only predictions whose label is in that shared subset,
  5. run COCOeval restricted to the shared ids.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

from ccd.data.class_mapping import (MOCS_ID_TO_NAME, build_external_to_mocs,
                                    shared_mocs_ids)


def remapped_gt_in_mocs_space(ann_file: str) -> Tuple["COCO", List[int]]:
    """Return (COCO gt remapped to MOCS ids, sorted shared MOCS ids)."""
    from pycocotools.coco import COCO

    base = COCO(ann_file)
    cats = list(base.cats.values())
    ext2mocs = build_external_to_mocs(cats)
    shared = shared_mocs_ids(cats)

    new = {
        "images": list(base.dataset["images"]),
        "annotations": [],
        "categories": [{"id": cid, "name": MOCS_ID_TO_NAME[cid]} for cid in shared],
    }
    aid = 0
    for a in base.dataset["annotations"]:
        if a["category_id"] not in ext2mocs:
            continue
        aid += 1
        na = copy.deepcopy(a)
        na["category_id"] = ext2mocs[a["category_id"]]
        na["id"] = aid
        new["annotations"].append(na)

    gt = COCO()
    gt.dataset = new
    gt.createIndex()
    return gt, shared


def filter_predictions_to_shared(detections: List[dict],
                                 shared_ids: List[int]) -> List[dict]:
    s = set(shared_ids)
    return [d for d in detections if d["category_id"] in s]
