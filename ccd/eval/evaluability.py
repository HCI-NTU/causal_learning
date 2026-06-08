"""Evaluability mask.

Per-class AP is only trustworthy when a target split has enough instances of that
class. We tag each (target, class) as:
    ok        >= 30 instances
    borderline 10..29
    drop      < 10
Reports should print AP only for `ok` (optionally `borderline`, greyed) classes.
"""
from __future__ import annotations

import collections
from typing import Dict

OK, BORDERLINE, DROP = "ok", "borderline", "drop"


def instance_counts(ann_file: str) -> Dict[int, int]:
    from pycocotools.coco import COCO
    c = COCO(ann_file)
    cnt = collections.Counter(a["category_id"] for a in c.dataset["annotations"])
    return dict(cnt)


def trust_level(n: int) -> str:
    if n >= 30:
        return OK
    if n >= 10:
        return BORDERLINE
    return DROP


def evaluability_for(ann_file: str) -> Dict[int, str]:
    return {cid: trust_level(n) for cid, n in instance_counts(ann_file).items()}
