"""Experiment (ii) — temporal (cross-stage) evaluation.

Train on MOCS earthmoving; test on MOCS earthmoving (+0, in-domain), foundation
(+1) and superstructure (+2) val sets. All in the MOCS label space. Reports
per-class AP at each shift distance and the degradation curve (absolute and
relative drop vs the in-domain stage).

    python -m scripts.eval_temporal --ckpt runs/temporal_causal/final.pth \
        --root data --model causal --out runs/temporal_causal/temporal.json
"""
from __future__ import annotations

import argparse
import json

from ccd.data.class_mapping import MOCS_ID_TO_NAME
from ccd.data.datasets import CocoDetection, build_registry
from ccd.data.transforms import build_transforms
from ccd.engine.utils import get_device, load_checkpoint
from ccd.eval.coco_eval import predict_coco, run_cocoeval
from ccd.eval.evaluability import instance_counts, trust_level
from ccd.models.build import build_detector

STAGES = [("mocs_test_earthmoving", 0),
          ("mocs_test_foundation", 1),
          ("mocs_test_superstructure", 2)]


def eval_stage(model, spec, device, limit=0):
    tfm = build_transforms(level="none", train=False)
    base = CocoDetection(spec.img_dir, spec.ann_file, transforms=tfm)
    ds = base
    if limit and limit < len(base):
        from torch.utils.data import Subset
        ds = Subset(base, list(range(limit)))
    preds = predict_coco(model, ds, device)
    cat_ids = sorted(base.coco.getCatIds())
    res = run_cocoeval(base.coco, preds, cat_ids=cat_ids)
    counts = instance_counts(spec.ann_file)
    res["evaluability"] = {int(k): trust_level(v) for k, v in counts.items()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--model", default="causal", choices=["erm", "aug", "causal"])
    ap.add_argument("--out", default="temporal_results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="eval on first N images only (sanity)")
    args = ap.parse_args()

    device = get_device(args.device)
    model = build_detector(model=args.model).to(device)
    load_checkpoint(model, args.ckpt, map_location=device)
    reg = build_registry(args.root)

    results = {}
    for key, dist in STAGES:
        res = eval_stage(model, reg[key], device, limit=args.limit)
        results[key] = {"distance": dist, **res}
        print(f"\n##### {key} (+{dist})  mAP={res['mAP']:.3f}  mAP50={res['mAP50']:.3f}")

    base = results["mocs_test_earthmoving"]["mAP"]
    print("\n=== degradation curve (mAP) ===")
    print(f"  {'stage':<26}{'mAP':>8}{'abs drop':>10}{'rel drop':>10}")
    for key, _ in STAGES:
        m = results[key]["mAP"]
        absd = base - m
        reld = absd / base if base > 0 else float("nan")
        print(f"  {key:<26}{m:>8.3f}{absd:>10.3f}{reld:>9.1%}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out}")
