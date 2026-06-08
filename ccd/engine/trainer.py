"""Training loop for the Faster R-CNN family (erm / aug / causal)."""
from __future__ import annotations

import math
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from ccd.data.datasets import build_registry, make_dataset
from ccd.data.transforms import CounterfactualCopyPaste, build_transforms
from ccd.engine.utils import (collate_fn, get_device, save_checkpoint, set_seed)
from ccd.models.build import build_detector


def _build_train_dataset(cfg):
    reg = build_registry(cfg["data_root"])
    spec = reg[cfg["train_split"]]
    level = cfg.get("aug_level", "none")
    tfm = build_transforms(level=level, train=True)
    ds = make_dataset(spec, transforms=tfm)
    if cfg.get("copy_paste", False):
        ds = CounterfactualCopyPaste(ds, p=cfg.get("cp_prob", 0.5),
                                     max_paste=cfg.get("cp_max", 2))
    return ds


def _maybe_selection_split(ds, cfg):
    """Carve a source-internal selection split for model selection (no target
    leakage). Returns (train_subset, sel_subset) or (ds, None)."""
    frac = cfg.get("selection_frac", 0.0)
    if frac <= 0:
        return ds, None
    n = len(ds)
    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    perm = torch.randperm(n, generator=g).tolist()
    n_sel = int(n * frac)
    sel_idx, tr_idx = perm[:n_sel], perm[n_sel:]
    return Subset(ds, tr_idx), Subset(ds, sel_idx)


def build_optimizer(model, cfg, base_lr=None):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = base_lr if base_lr is not None else cfg["lr"]
    return torch.optim.SGD(params, lr=lr, momentum=0.9,
                           weight_decay=cfg.get("weight_decay", 1e-4))


def train(cfg: Dict):
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "cuda"))

    ds = _build_train_dataset(cfg)
    limit = cfg.get("limit_train", 0)
    if limit and limit < len(ds):
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(limit)))
        print(f"[sanity] limiting train set to {limit} images")
    ds, _sel = _maybe_selection_split(ds, cfg)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 2), shuffle=True,
                        num_workers=cfg.get("num_workers", 4),
                        collate_fn=collate_fn, drop_last=True)

    model = build_detector(model=cfg["model"],
                           dict_slots=cfg.get("dict_slots", 50),
                           dict_momentum=cfg.get("dict_momentum", 0.3),
                           fuse=cfg.get("fuse", "concat"),
                           min_size=cfg.get("min_size", 800),
                           max_size=cfg.get("max_size", 1333)).to(device)

    # Linear LR scaling: config `lr` is the reference for `ref_batch_size`
    # (default 16, the canonical COCO setting); scale to the actual batch size.
    bs = cfg.get("batch_size", 2)
    ref_bs = cfg.get("ref_batch_size", 16)
    base_lr = cfg["lr"] * bs / ref_bs if cfg.get("lr_auto_scale", True) else cfg["lr"]
    print(f"[lr] base_lr={base_lr:.5f}  (config lr={cfg['lr']} for batch {ref_bs}, "
          f"scaled to batch {bs}; auto_scale={cfg.get('lr_auto_scale', True)})")

    opt = build_optimizer(model, cfg, base_lr)
    epochs = cfg.get("epochs", 12)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=cfg.get("lr_steps", [8, 11]), gamma=0.1)
    warmup = cfg.get("warmup_iters", 500)

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    it = 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if it < warmup:
                lr_scale = (it + 1) / warmup
                for g in opt.param_groups:
                    g["lr"] = base_lr * lr_scale

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not math.isfinite(loss.item()):
                print(f"[warn] non-finite loss at it={it}, skipping")
                opt.zero_grad(); it += 1; continue
            opt.zero_grad(); loss.backward(); opt.step()
            if it % cfg.get("log_every", 50) == 0:
                msg = " ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items())
                print(f"ep{epoch} it{it} loss={loss.item():.3f} {msg}")
            it += 1
        sched.step()
        print(f"[epoch {epoch}] done in {time.time()-t0:.0f}s")
        save_checkpoint(model, os.path.join(out_dir, "last.pth"),
                        epoch=epoch, cfg=cfg)
    save_checkpoint(model, os.path.join(out_dir, "final.pth"), cfg=cfg)
    print(f"[train] saved -> {os.path.join(out_dir, 'final.pth')}")
    return os.path.join(out_dir, "final.pth")
