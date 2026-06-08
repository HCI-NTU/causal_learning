"""Prepare data and sanity-check the class mappings.

    python -m scripts.prepare_data --root data --convert-soda
    python -m scripts.prepare_data --root data --inspect

--convert-soda : build data/SODA/instances_{train,test}.json from VOC xml.
--inspect      : print each external dataset's categories and how they resolve to
                 the MOCS label space, so you can verify ALIAS_TO_MOCS before any
                 cross-dataset evaluation.
"""
from __future__ import annotations

import argparse
import os

from ccd.data.class_mapping import (MOCS_ID_TO_NAME, build_external_to_mocs,
                                    resolve_to_mocs, shared_mocs_ids)
from ccd.data.datasets import build_registry


def convert_soda(root):
    from ccd.data.voc_to_coco import convert
    for split in ("train", "test"):
        if os.path.isdir(os.path.join(root, "SODA", "images", split)):
            convert(root, split)


def inspect(root):
    from pycocotools.coco import COCO
    targets = {
        "ExtCon": os.path.join(root, "ExtCon", "extcon_gt.json"),
        "CIS-test": os.path.join(root, "CIS", "instances_test.json"),
        "ACID-test": os.path.join(root, "ACID", "instances_test.json"),
        "SODA-test": os.path.join(root, "SODA", "instances_test.json"),
    }
    for name, path in targets.items():
        if not os.path.exists(path):
            print(f"\n=== {name}: NOT FOUND ({path}) ===")
            continue
        coco = COCO(path)
        cats = list(coco.cats.values())
        ext2mocs = build_external_to_mocs(cats)
        shared = shared_mocs_ids(cats)
        print(f"\n=== {name}: {len(cats)} categories, "
              f"{len(shared)} shared with MOCS ===")
        for c in cats:
            canon = resolve_to_mocs(c["name"])
            tag = canon if canon else "—  (DROPPED, not in MOCS)"
            print(f"  {c['id']:>3} {c['name']:<22} -> {tag}")
        print("  shared MOCS classes scored:",
              [MOCS_ID_TO_NAME[i] for i in shared])


def validate(root):
    reg = build_registry(root)
    print("\n=== path check ===")
    for key, spec in reg.items():
        if spec.role == "trainval":
            for d, a in spec.extra.get("merge", []):
                print(f"  {key:<20} img:{os.path.isdir(d)}  ann:{os.path.isfile(a)}  ({a})")
        else:
            print(f"  {key:<20} img:{os.path.isdir(spec.img_dir)}  "
                  f"ann:{os.path.isfile(spec.ann_file)}  ({spec.ann_file})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--convert-soda", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.convert_soda:
        convert_soda(args.root)
    if args.validate:
        validate(args.root)
    if args.inspect:
        inspect(args.root)
    if not any([args.convert_soda, args.inspect, args.validate]):
        validate(args.root)
        inspect(args.root)
