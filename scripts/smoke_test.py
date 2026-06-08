"""Data-free smoke test for the Faster R-CNN family.

Builds the erm / aug / causal detectors and runs a training forward+backward and
an eval forward on synthetic tensors — no dataset, no pretrained download, CPU is
fine. This is the fastest way to catch wiring bugs in the causal head before any
real run.

    python -m scripts.smoke_test
"""
from __future__ import annotations

import torch

from ccd.data.class_mapping import NUM_CLASSES
from ccd.models.build import build_detector


def fake_batch(n=2, h=256, w=256):
    images = [torch.rand(3, h, w) for _ in range(n)]
    boxes = torch.tensor([[10, 10, 60, 60], [30, 40, 120, 160], [70, 80, 200, 220]],
                         dtype=torch.float32)
    targets = [{"boxes": boxes.clone(),
                "labels": torch.randint(1, NUM_CLASSES + 1, (boxes.shape[0],))}
               for _ in range(n)]
    return images, targets


def run(model_name, device, steps=3):
    net = build_detector(model=model_name, pretrained_backbone=False).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    last = None
    for _ in range(steps):                       # a few steps so the causal dict fills
        images, targets = fake_batch()
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = net(images, targets)
        loss = sum(losses.values())
        opt.zero_grad(); loss.backward(); opt.step()
        last = (loss, losses)

    net.eval()
    with torch.no_grad():
        images, _ = fake_batch(n=1)
        out = net([images[0].to(device)])

    loss, losses = last
    assert torch.isfinite(loss), f"{model_name}: non-finite loss"
    print(f"[{model_name:6s}] train loss={loss.item():.3f} "
          f"keys={list(losses)} | eval boxes={out[0]['boxes'].shape[0]}  OK")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    for m in ("erm", "aug", "causal"):
        run(m, device)
    print("\nSMOKE TEST PASSED")
