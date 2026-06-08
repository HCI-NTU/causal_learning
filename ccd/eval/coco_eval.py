"""COCO evaluation: predict over a dataset and score with pycocotools."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ccd.engine.utils import collate_fn


@torch.no_grad()
def predict_coco(model, dataset, device, batch_size: int = 4,
                 num_workers: int = 4, score_thresh: float = 0.0) -> List[dict]:
    """Return detections in COCO results format [{image_id,category_id,bbox,score}]."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)
    results: List[dict] = []
    for images, targets in loader:
        images = [im.to(device) for im in images]
        outputs = model(images)
        for tgt, out in zip(targets, outputs):
            image_id = int(tgt["image_id"].item())
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                if s < score_thresh:
                    continue
                x1, y1, x2, y2 = b
                results.append({
                    "image_id": image_id,
                    "category_id": int(l),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                })
    return results


def run_cocoeval(coco_gt, detections: List[dict],
                 cat_ids: Optional[List[int]] = None) -> Dict:
    """COCOeval over given category ids. Returns summary + per-class AP@[.5:.95]."""
    from pycocotools.cocoeval import COCOeval

    if len(detections) == 0:
        return {"mAP": 0.0, "mAP50": 0.0, "per_class_AP": {}}

    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if cat_ids is not None:
        ev.params.catIds = cat_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()

    out = {"mAP": float(ev.stats[0]), "mAP50": float(ev.stats[1]),
           "per_class_AP": {}}
    # per-class AP@[.5:.95]: precision dims [T,R,K,A,M]
    prec = ev.eval["precision"]
    cat_list = ev.params.catIds
    for k, cid in enumerate(cat_list):
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        out["per_class_AP"][int(cid)] = float(p.mean()) if p.size else float("nan")
    return out
