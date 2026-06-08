"""Convert SODA's PASCAL VOC XML annotations to COCO detection json.

SODA ships annotations as `data/SODA/annotations/*.xml` (one per image). This
writes `instances_<split>.json` next to the other COCO files so the rest of the
pipeline treats SODA identically to the other datasets.

A split is defined by which images live in `images/<split>/`. Each xml is matched
to its image by `<filename>` (falling back to the xml stem).

Usage:
    python -m ccd.data.voc_to_coco --root data --split train
    python -m ccd.data.voc_to_coco --root data --split test
"""
from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List


def _index_images(img_dir: str) -> Dict[str, str]:
    out = {}
    if not os.path.isdir(img_dir):
        return out
    for f in os.listdir(img_dir):
        out[os.path.splitext(f)[0]] = f
    return out


def convert(root: str, split: str) -> str:
    soda = os.path.join(root, "SODA")
    xml_dir = os.path.join(soda, "annotations")
    img_dir = os.path.join(soda, "images", split)
    out_path = os.path.join(soda, f"instances_{split}.json")
    img_index = _index_images(img_dir)

    images: List[dict] = []
    annotations: List[dict] = []
    cat_name_to_id: Dict[str, int] = {}
    img_id = 0
    ann_id = 0

    for xml in sorted(os.listdir(xml_dir)):
        if not xml.endswith(".xml"):
            continue
        stem = os.path.splitext(xml)[0]
        tree = ET.parse(os.path.join(xml_dir, xml))
        r = tree.getroot()
        fname_el = r.find("filename")
        fname = fname_el.text if fname_el is not None else None
        key = os.path.splitext(fname)[0] if fname else stem
        if key not in img_index:
            continue  # this xml's image is not in this split
        file_name = img_index[key]

        size = r.find("size")
        width = int(size.findtext("width")) if size is not None else 0
        height = int(size.findtext("height")) if size is not None else 0

        img_id += 1
        images.append({"id": img_id, "file_name": file_name,
                       "width": width, "height": height})

        for obj in r.findall("object"):
            name = obj.findtext("name").strip()
            if name not in cat_name_to_id:
                cat_name_to_id[name] = len(cat_name_to_id) + 1
            b = obj.find("bndbox")
            xmin = float(b.findtext("xmin"))
            ymin = float(b.findtext("ymin"))
            xmax = float(b.findtext("xmax"))
            ymax = float(b.findtext("ymax"))
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0:
                continue
            ann_id += 1
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_name_to_id[name],
                "bbox": [xmin, ymin, w, h], "area": w * h, "iscrowd": 0,
            })

    categories = [{"id": cid, "name": n} for n, cid in
                  sorted(cat_name_to_id.items(), key=lambda kv: kv[1])]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[voc_to_coco] {split}: {len(images)} imgs, {len(annotations)} anns, "
          f"{len(categories)} cats -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", required=True, choices=["train", "test", "val"])
    args = ap.parse_args()
    convert(args.root, args.split)
