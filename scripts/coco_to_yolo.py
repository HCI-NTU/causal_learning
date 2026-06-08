"""Convert COCO-format splits to YOLO format for ultralytics (YOLOv11 / RT-DETR).

Writes:
  <out>/images/<split>/*.jpg         (symlinks to originals)
  <out>/labels/<split>/*.txt         (class cx cy w h, normalized; class 0-based)

For external targets pass --remap-external so labels land in the MOCS 13-class
space (unmapped categories are skipped), making cross-dataset val comparable to
the Faster R-CNN pipeline.

    # MOCS train + val (native 13-class space)
    python -m scripts.coco_to_yolo --img data/MOCS/images/train \
        --ann data/MOCS/instances_train.json --out data_yolo/mocs --split train
    python -m scripts.coco_to_yolo --img data/MOCS/images/val \
        --ann data/MOCS/instances_val.json --out data_yolo/mocs --split val

    # external target remapped to MOCS space
    python -m scripts.coco_to_yolo --img data/CIS/images/test \
        --ann data/CIS/instances_test.json --out data_yolo/cis --split val \
        --remap-external
"""
from __future__ import annotations

import argparse
import os

from ccd.data.class_mapping import MOCS_CLASSES, build_external_to_mocs


def convert(img_dir, ann_file, out, split, remap_external=False):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)

    remap = None
    if remap_external:
        remap = build_external_to_mocs(list(coco.cats.values()))  # ext_id -> mocs_id(1-based)

    img_out = os.path.join(out, "images", split)
    lbl_out = os.path.join(out, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    n_img = n_box = 0
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        fn = info["file_name"]
        W, H = info["width"], info["height"]
        src = os.path.join(img_dir, fn)
        dst = os.path.join(img_out, os.path.basename(fn))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except FileExistsError:
                pass
        lines = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            cat = a["category_id"]
            if remap is not None:
                if cat not in remap:
                    continue
                cls0 = remap[cat] - 1            # MOCS 1-based -> 0-based
            else:
                cls0 = cat - 1                    # native MOCS already 1-based
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"{cls0} {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
            n_box += 1
        stem = os.path.splitext(os.path.basename(fn))[0]
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n_img += 1
    print(f"[coco_to_yolo] {split}: {n_img} imgs, {n_box} boxes -> {out}")


def write_data_yaml(out, train_split="train", val_split="val"):
    path = os.path.join(out, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        f.write(f"train: images/{train_split}\n")
        f.write(f"val: images/{val_split}\n")
        f.write(f"nc: {len(MOCS_CLASSES)}\n")
        f.write("names:\n")
        for i, n in enumerate(MOCS_CLASSES):
            f.write(f"  {i}: {n}\n")
    print(f"[coco_to_yolo] wrote {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--remap-external", action="store_true")
    ap.add_argument("--write-yaml", action="store_true")
    args = ap.parse_args()
    convert(args.img, args.ann, args.out, args.split, args.remap_external)
    if args.write_yaml:
        write_data_yaml(args.out)
