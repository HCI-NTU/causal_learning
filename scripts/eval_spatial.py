"""Experiment (i) — spatial (cross-dataset) evaluation.

Train on MOCS; test on MOCS-val (in-domain) and ExtCon / CIS / ACID test sets
(SODA optional, worker-only). External GT and predictions are remapped to the
MOCS label space and scored on shared classes only.

    python -m scripts.eval_spatial --ckpt runs/spatial_causal/final.pth \
        --root data --model causal --out runs/spatial_causal/spatial.json
"""
from __future__ import annotations

import argparse
import json

import torch

from ccd.data.class_mapping import MOCS_ID_TO_NAME
from ccd.data.datasets import CocoDetection, build_registry
from ccd.data.transforms import build_transforms
from ccd.engine.utils import get_device, load_checkpoint
from ccd.eval.coco_eval import predict_coco, run_cocoeval
from ccd.eval.cross_dataset import (filter_predictions_to_shared,
                                    remapped_gt_in_mocs_space)
from ccd.eval.evaluability import instance_counts, trust_level
from ccd.models.build import build_detector

# targets: (registry_key, is_external). SODA included but easy to drop.
TARGETS = [
    ("mocs_test", False),
    ("extcon_test", True),
    ("cis_test", True),
    ("acid_test", True),
    # ("soda_test", True),   # uncomment to include the worker-only target
]


def eval_one(model, spec, device, is_external, score_thresh=0.0, limit=0):
    tfm = build_transforms(level="none", train=False)
    base = CocoDetection(spec.img_dir, spec.ann_file, transforms=tfm)
    ds = base
    if limit and limit < len(base):
        from torch.utils.data import Subset
        ds = Subset(base, list(range(limit)))
    preds = predict_coco(model, ds, device, score_thresh=score_thresh)

    if is_external:
        gt, shared = remapped_gt_in_mocs_space(spec.ann_file)
        preds = filter_predictions_to_shared(preds, shared)
        cat_ids = shared
        counts = evaluability_for_remapped(spec.ann_file)
    else:
        gt = base.coco
        cat_ids = sorted(gt.getCatIds())
        counts = instance_counts(spec.ann_file)

    res = run_cocoeval(gt, preds, cat_ids=cat_ids)
    res["evaluability"] = {int(k): trust_level(v) for k, v in counts.items()}
    return res


def evaluability_for_remapped(ann_file):
    """Instance counts in MOCS space for an external target (for trust tags)."""
    import collections
    from ccd.data.class_mapping import build_external_to_mocs
    from pycocotools.coco import COCO
    c = COCO(ann_file)
    ext2mocs = build_external_to_mocs(list(c.cats.values()))
    cnt = collections.Counter()
    for a in c.dataset["annotations"]:
        if a["category_id"] in ext2mocs:
            cnt[ext2mocs[a["category_id"]]] += 1
    return dict(cnt)


def print_report(name, res):
    print(f"\n##### {name}  mAP={res['mAP']:.3f}  mAP50={res['mAP50']:.3f}")
    print(f"  {'class':<18}{'AP':>8}  trust")
    for cid, ap in sorted(res["per_class_AP"].items()):
        trust = res["evaluability"].get(cid, "drop")
        print(f"  {MOCS_ID_TO_NAME.get(cid, cid):<18}{ap:>8.3f}  {trust}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--model", default="causal", choices=["erm", "aug", "causal"])
    ap.add_argument("--out", default="spatial_results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="eval on first N images only (sanity)")
    args = ap.parse_args()

    device = get_device(args.device)
    model = build_detector(model=args.model).to(device)
    load_checkpoint(model, args.ckpt, map_location=device)

    reg = build_registry(args.root)
    all_res = {}
    for key, is_ext in TARGETS:
        if key not in reg:
            continue
        res = eval_one(model, reg[key], device, is_ext, limit=args.limit)
        all_res[key] = res
        print_report(key, res)

    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n[saved] {args.out}")
