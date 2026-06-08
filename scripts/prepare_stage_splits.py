"""Generate stage-specific COCO JSON splits from the main MOCS annotations.

The stage label is stored somewhere in each image's metadata dict inside
instances_train.json / instances_val.json, but the exact field name varies by
MOCS release. Run with --inspect first to see what's there:

    python -m scripts.prepare_stage_splits --root data --inspect

Then run the actual split (assuming the field is called 'stage'):

    python -m scripts.prepare_stage_splits --root data --field stage

If the values are not the canonical strings (earthmoving / foundation /
superstructure), pass a mapping with --map:

    python -m scripts.prepare_stage_splits --root data --field stage \
        --map "0=earthmoving,1=foundation,2=superstructure"

Writes:
    data/MOCS/instances_train_earthmoving.json
    data/MOCS/instances_train_foundation.json
    data/MOCS/instances_train_superstructure.json
    data/MOCS/instances_val_earthmoving.json
    data/MOCS/instances_val_foundation.json
    data/MOCS/instances_val_superstructure.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Dict, List

STAGES = ("earthmoving", "foundation", "superstructure")


def inspect(ann_file: str):
    with open(ann_file) as f:
        data = json.load(f)
    images = data["images"]
    standard = {"id", "file_name", "width", "height", "license",
                 "coco_url", "flickr_url", "date_captured"}
    extra_keys = collections.Counter()
    for img in images:
        for k in img:
            if k not in standard:
                extra_keys[k] += 1
    print(f"\n{ann_file}  ({len(images)} images)")
    if not extra_keys:
        print("  no extra keys found beyond standard COCO image fields")
        return
    print("  extra image-level keys (name: count):")
    for k, n in extra_keys.most_common():
        # show a sample of unique values (up to 8)
        vals = list({str(img.get(k, "")) for img in images})[:8]
        print(f"    {k!r:30s}  n={n}  sample values: {vals}")


def split(ann_file: str, out_dir: str, split_name: str,
          field: str, value_map: Dict[str, str]):
    with open(ann_file) as f:
        data = json.load(f)

    images: List[dict] = data["images"]
    annotations: List[dict] = data["annotations"]
    categories: List[dict] = data["categories"]

    # build ann lookup
    ann_by_img: Dict[int, List[dict]] = collections.defaultdict(list)
    for a in annotations:
        ann_by_img[a["image_id"]].append(a)

    stage_images: Dict[str, List[dict]] = {s: [] for s in STAGES}
    n_unmapped = 0
    for img in images:
        raw = str(img.get(field, ""))
        stage = value_map.get(raw, raw.lower())
        if stage not in STAGES:
            n_unmapped += 1
            continue
        stage_images[stage].append(img)

    if n_unmapped:
        print(f"  [warn] {n_unmapped} images had unrecognised stage values "
              f"and were skipped. Use --map to remap them.")

    for stage, imgs in stage_images.items():
        img_ids = {img["id"] for img in imgs}
        anns = [a for a in annotations if a["image_id"] in img_ids]
        out = {"images": imgs, "annotations": anns, "categories": categories}
        path = os.path.join(out_dir, f"instances_{split_name}_{stage}.json")
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"  wrote {path}  ({len(imgs)} imgs, {len(anns)} anns)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--inspect", action="store_true",
                    help="print extra image-level fields and sample values, then exit")
    ap.add_argument("--field", default="stage",
                    help="image-dict key holding the stage label (default: 'stage')")
    ap.add_argument("--map", default="",
                    help="comma-separated value remappings, e.g. "
                         "'0=earthmoving,1=foundation,2=superstructure'")
    args = ap.parse_args()

    mocs = os.path.join(args.root, "MOCS")
    train_ann = os.path.join(mocs, "instances_train.json")
    val_ann = os.path.join(mocs, "instances_val.json")

    if args.inspect:
        inspect(train_ann)
        inspect(val_ann)
        return

    # parse --map
    value_map: Dict[str, str] = {}
    if args.map:
        for pair in args.map.split(","):
            k, _, v = pair.partition("=")
            value_map[k.strip()] = v.strip()

    print(f"[stage-split] field='{args.field}'  map={value_map or '(identity)'}")
    for ann, sp in [(train_ann, "train"), (val_ann, "val")]:
        split(ann, mocs, sp, args.field, value_map)
    print("[stage-split] done — re-run prepare_data --validate to confirm paths")


if __name__ == "__main__":
    main()
