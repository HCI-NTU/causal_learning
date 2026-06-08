# Experiment protocol

## Conditions
- `erm`    — vanilla Faster R-CNN R50-FPN, no augmentation.
- `aug`    — same detector + strong augmentation (photometric + scale jitter +
             counterfactual copy-paste). The "is it just augmentation?" control.
- `causal` — same as `aug` + the EMA confounder-dictionary backdoor head.

Identical optimizer, schedule, backbone, input size, and batch size across all
three. Tune once on the source-internal selection split (`selection_frac`); never
on a target.

## Metrics
- COCO mAP@[.5:.95] and mAP50, plus **per-class AP** (lead with per-class).
- Evaluability mask: ok ≥30 inst, borderline 10–29, drop <10 per target.
- Temporal headline: degradation curve (abs & rel mAP drop vs +0).
- Spatial headline: in-domain (MOCS-val) vs each target, per shared-class AP.
- Report ≥3 seeds, mean ± std.

## Architecture generality (optional)
Run `erm`/`aug` on YOLOv11-m and RT-DETR-l via ultralytics to show the shift
degrades every modern detector. The causal head stays on Faster R-CNN; an
RT-DETR query-level port is a separate follow-up.
