"""Train the causal ports of RT-DETR (full) and YOLOv11 (partial) via ultralytics.

    # RT-DETR — full query-level causal method
    python -m scripts.train_ultralytics_causal --arch rtdetr-l \
        --data data_yolo/mocs/data.yaml --epochs 100 --project runs/rtdetr_causal

    # YOLOv11 — partial channel-wise causal disentanglement
    python -m scripts.train_ultralytics_causal --arch yolo11m \
        --data data_yolo/mocs/data.yaml --epochs 100 --project runs/yolo11m_causal

Pair each with the matching ERM/strong-aug baseline from train_ultralytics.py;
the method effect is the within-architecture delta (causal vs strong-aug on the
SAME detector), never a cross-architecture absolute-mAP comparison.

NOTE: the causal aux loss is injected by patching the model's loss method. If
your ultralytics version rebuilds the model inside the trainer and the patch is
lost (watch for the aux term not moving / decorrelation not decreasing), switch
to the CausalDetectionTrainer sketch in ccd/engine/ultralytics_causal.py.
"""
from __future__ import annotations

import argparse

from ccd.engine.ultralytics_causal import build_causal_model

ERM_AUG_OFF = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
                   hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
                   translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                   flipud=0.0, fliplr=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    help="rtdetr-l|rtdetr-x (full) or yolo11{n,s,m,l,x} (partial)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--aug", default="strong", choices=["erm", "strong"],
                    help="match the baseline you compare against")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/ultra_causal")
    # causal hyperparameters
    ap.add_argument("--lambda-aux", type=float, default=1.0)
    ap.add_argument("--lambda-decorr", type=float, default=0.1)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.2)
    ap.add_argument("--no-bank", action="store_true",
                    help="YOLO only: disable the global context bank (pure disentangle)")
    ap.add_argument("--fuse", default="concat", choices=["concat", "gate"])
    args = ap.parse_args()

    model = build_causal_model(
        args.arch, weights=args.weights, lambda_aux=args.lambda_aux,
        lambda_decorr=args.lambda_decorr, slots=args.slots,
        momentum=args.momentum, use_bank=not args.no_bank, mode=args.fuse)

    extra = ERM_AUG_OFF if args.aug == "erm" else {}
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=args.project, name=args.arch + "_causal",
                **extra)


if __name__ == "__main__":
    main()
