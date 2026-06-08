"""YOLOv11 / RT-DETR baselines via ultralytics (architecture-generality role).

These are ERM / strong-aug baselines on alternative detectors — they show the
shift degrades every modern detector, not just Faster R-CNN. The causal head is
NOT ported here (dense YOLO has no per-object locus; an RT-DETR query-level port
is a separate effort).

Augmentation control:
  --aug erm    : disable ultralytics' built-in aug (mosaic/mixup/copy_paste off)
                 so it is comparable to the Faster R-CNN ERM baseline.
  --aug strong : keep ultralytics defaults (mosaic + copy_paste etc.).

    # train
    python -m scripts.train_ultralytics --arch yolo11m --data data_yolo/mocs/data.yaml \
        --aug erm --epochs 100 --project runs/yolo11m_erm
    python -m scripts.train_ultralytics --arch rtdetr-l --data data_yolo/mocs/data.yaml \
        --aug strong --epochs 100 --project runs/rtdetr_aug

    # eval a trained model on a target data.yaml (val split = the target)
    python -m scripts.train_ultralytics --arch yolo11m --weights runs/yolo11m_erm/weights/best.pt \
        --eval --data data_yolo/cis/data.yaml
"""
from __future__ import annotations

import argparse

# ultralytics maps arch name -> loader class
YOLO_ARCHES = {"yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"}
RTDETR_ARCHES = {"rtdetr-l", "rtdetr-x"}

ERM_AUG_OFF = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
                   hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
                   translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                   flipud=0.0, fliplr=0.0)


def load_model(arch, weights=None):
    from ultralytics import RTDETR, YOLO
    ckpt = weights or (f"{arch}.pt")
    if arch in RTDETR_ARCHES:
        return RTDETR(ckpt)
    if arch in YOLO_ARCHES:
        return YOLO(ckpt)
    raise ValueError(f"unknown arch '{arch}'. "
                     f"YOLO: {sorted(YOLO_ARCHES)} RT-DETR: {sorted(RTDETR_ARCHES)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--data", required=True, help="ultralytics data.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--aug", default="erm", choices=["erm", "strong"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/ultra")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    model = load_model(args.arch, args.weights)

    if args.eval:
        metrics = model.val(data=args.data, imgsz=args.imgsz, split="val")
        print("mAP50-95:", float(metrics.box.map))
        print("mAP50   :", float(metrics.box.map50))
        try:
            print("per-class AP50-95:", list(map(float, metrics.box.maps)))
        except Exception:
            pass
        return

    extra = ERM_AUG_OFF if args.aug == "erm" else {}
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=args.project, name=args.arch,
                **extra)


if __name__ == "__main__":
    main()
