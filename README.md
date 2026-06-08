# CCD — Causal construction-site object detection

A repository for two distribution-shift object-detection experiments on
construction imagery, with a causal RoI head over a vanilla Faster R-CNN and
clean augmentation-matched baselines. YOLOv11 and RT-DETR are wired as optional
architecture-generality baselines.

## The two experiments

**(i) Spatial / cross-dataset.** Train on MOCS; test on MOCS-val (in-domain) and
ExtCon / CIS / ACID test sets. External GT + predictions are remapped to the MOCS
label space and scored on shared classes only. *(SODA is supported but off by
default — it appears in the data tree but not in experiment (i); enable it in
`scripts/eval_spatial.py:TARGETS` if you want the worker-only target.)*

**(ii) Temporal / cross-stage.** Train on MOCS earthmoving; test on MOCS
earthmoving (+0, in-domain), foundation (+1), superstructure (+2) val sets.
Reports the degradation-vs-distance curve.

## Conditions (run all three, identically)

| name     | detector              | augmentation        | causal head |
|----------|-----------------------|---------------------|-------------|
| `erm`    | Faster R-CNN R50-FPN  | none                | no          |
| `aug`    | Faster R-CNN R50-FPN  | strong + copy-paste | no          |
| `causal` | Faster R-CNN R50-FPN  | strong + copy-paste | yes (EMA dict backdoor) |

`causal` vs `aug` isolates the causal mechanism over augmentation — the key
comparison. Optimizer/schedule/backbone are identical across all three.

## Split-resolution rule (encoded in `ccd/data/datasets.py`)
- No separate `test` split → `val` is the test set (MOCS).
- A separate `test` split → `val` merged into `train` as `trainval`; `test` is the
  held-out target (CIS, ACID, SODA). We only *train* on MOCS here, so the merge
  matters only for future single-source training on those sets.

## Setup
```bash
pip install -r requirements.txt
python -m scripts.prepare_data --root data --convert-soda --validate --inspect
```
`--inspect` prints how each external dataset's categories resolve to MOCS — **verify
this before trusting cross-dataset numbers** and edit `ccd/data/class_mapping.py:ALIAS_TO_MOCS`.

## Run — Faster R-CNN (primary)
```bash
# Experiment (i)
for m in erm aug causal; do
  python -m scripts.train --config configs/spatial_$m.yaml
  python -m scripts.eval_spatial --ckpt runs/spatial_$m/final.pth --model $m \
      --out runs/spatial_$m/spatial.json
done
# Experiment (ii)
for m in erm aug causal; do
  python -m scripts.train --config configs/temporal_$m.yaml
  python -m scripts.eval_temporal --ckpt runs/temporal_$m/final.pth --model $m \
      --out runs/temporal_$m/temporal.json
done
```
Use ≥3 seeds (`seed=1 out_dir=...`) and report mean ± std.

## Run — YOLOv11 / RT-DETR (optional architectures, baseline only)
```bash
# build YOLO-format MOCS, then a remapped target (e.g. CIS)
python -m scripts.coco_to_yolo --img data/MOCS/images/train --ann data/MOCS/instances_train.json --out data_yolo/mocs --split train
python -m scripts.coco_to_yolo --img data/MOCS/images/val   --ann data/MOCS/instances_val.json   --out data_yolo/mocs --split val --write-yaml
python -m scripts.coco_to_yolo --img data/CIS/images/test   --ann data/CIS/instances_test.json   --out data_yolo/cis  --split val --remap-external --write-yaml

python -m scripts.train_ultralytics --arch yolo11m --data data_yolo/mocs/data.yaml --aug erm --project runs/yolo11m_erm
python -m scripts.train_ultralytics --arch rtdetr-l --data data_yolo/mocs/data.yaml --aug strong --project runs/rtdetr_aug
python -m scripts.train_ultralytics --arch yolo11m --weights runs/yolo11m_erm/yolo11m/weights/best.pt --eval --data data_yolo/cis/data.yaml
```

### Causal ports on RT-DETR (full) and YOLOv11 (partial)
The causal mechanism is also ported onto the other detectors as a generality
experiment (`ccd/models/causal_{common,rtdetr,yolo}.py`, glue in
`ccd/engine/ultralytics_causal.py`):
- **RT-DETR — full method.** Each decoder query is split into causal/spurious;
  a class-agnostic spurious-prototype dictionary is backdoor-adjusted before the
  per-layer classification head. The query is the per-object locus, so the whole
  mechanism transfers.
- **YOLOv11 — partial.** The dense head has no per-object locus, so only the
  channel-wise causal/spurious disentanglement (+ decorrelation, + optional
  global context bank) ports, applied to the neck features feeding the head.

```bash
python -m scripts.train_ultralytics_causal --arch rtdetr-l --data data_yolo/mocs/data.yaml --project runs/rtdetr_causal
python -m scripts.train_ultralytics_causal --arch yolo11m --data data_yolo/mocs/data.yaml --project runs/yolo11m_causal
python -m scripts.train_ultralytics_causal --arch yolo11m --no-bank --data data_yolo/mocs/data.yaml --project runs/yolo11m_causal_nobank
```
Always compare causal-vs-strong-aug **within the same architecture**; never read
a cross-architecture absolute-mAP gap as the method effect.

**The version-sensitive seam:** the causal aux loss is added by patching the
model's `loss` method. If your ultralytics build rebuilds the model inside its
trainer and the patch is lost (symptom: the decorrelation term never moves),
switch to the `CausalDetectionTrainer` sketch at the bottom of
`ccd/engine/ultralytics_causal.py`. The mechanism modules themselves are
syntax-checked and their math is verified against a NumPy mirror, but the
ultralytics swap/loss integration must be confirmed on your installed version.

## Honest caveats
- Not trained end-to-end in this repo's authoring environment (no GPU there);
  expect integration debugging on your box. `CausalRoIHeads.forward` reproduces
  torchvision's RoIHeads (boxes only) and targets torchvision ≥0.15 — if your
  version changed that API, patch it there.
- The causal head is an NWGM/attention *approximation* of `P(Y|do(X))`, not exact
  do-calculus.
- Always report **per-class AP** with the evaluability mask, never lead with
  pooled mAP (Worker dominates every split).
- Cross-architecture: the method effect is the within-architecture delta, never
  YOLO-vs-Faster-R-CNN absolute mAP.

See `docs/data.md` and `docs/experiments.md` for details.
