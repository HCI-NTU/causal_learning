"""Train a Faster R-CNN-family model from a YAML config.

    python -m scripts.train --config configs/spatial_causal.yaml
    python -m scripts.train --config configs/temporal_erm.yaml seed=1 out_dir=runs/t_erm_s1
"""
from __future__ import annotations

import argparse
import ast

import yaml

from ccd.engine.trainer import train


def parse_overrides(pairs):
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        low = v.strip().lower()
        if low in ("true", "false"):
            out[k] = (low == "true")
        elif low in ("none", "null"):
            out[k] = None
        else:
            try:
                out[k] = ast.literal_eval(v)
            except Exception:
                out[k] = v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.update(parse_overrides(args.overrides))
    print("[config]", cfg)
    train(cfg)
