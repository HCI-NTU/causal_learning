"""Shared primitives for the causal heads (Faster R-CNN / RT-DETR / YOLO).

Two ideas are reused everywhere:
  * a feature is split into a causal part and a spurious part, with the spurious
    part defined as the residual (guarantees an exact additive decomposition);
  * a decorrelation penalty pushes the two parts to be statistically independent
    (zero cross-covariance), which is what makes the split meaningful.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def decorrelation_loss(a: Tensor, b: Tensor) -> Tensor:
    """Squared Frobenius norm of the cross-covariance between a[N,D] and b[N,D].

    Zero iff every causal channel is linearly uncorrelated with every spurious
    channel. Cheap, batch-wise, and 0 when N<2 (no statistics).
    """
    if a.dim() != 2:
        a = a.flatten(0, -2)
        b = b.flatten(0, -2)
    n = a.shape[0]
    if n < 2:
        return a.new_zeros(())
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    cov = (a.t() @ b) / (n - 1)              # [D, D]
    return (cov ** 2).sum() / a.shape[1]


class PrototypeDictionary(nn.Module):
    """Per-class EMA memory bank of prototypes (works for RoI feats or queries)."""

    def __init__(self, num_fg: int, dim: int, slots: int = 50,
                 momentum: float = 0.3):
        super().__init__()
        self.num_fg, self.dim, self.slots, self.momentum = \
            num_fg, dim, slots, momentum
        self.register_buffer("bank", torch.zeros(num_fg, slots, dim))
        self.register_buffer("filled", torch.zeros(num_fg, slots, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros(num_fg, dtype=torch.long))
        self.register_buffer("class_count", torch.ones(num_fg))

    @torch.no_grad()
    def update(self, feats: Tensor, labels: Tensor):
        """feats[N,D] spurious/context features; labels[N] in 0..num_fg-1."""
        for f, l in zip(feats, labels):
            c = int(l)
            if c < 0 or c >= self.num_fg:
                continue
            self.class_count[c] += 1
            p = int(self.ptr[c])
            if self.filled[c, p]:
                self.bank[c, p] = (1 - self.momentum) * self.bank[c, p] + \
                    self.momentum * f
            else:
                self.bank[c, p] = f
                self.filled[c, p] = True
            self.ptr[c] = (p + 1) % self.slots

    def prototypes(self):
        """Return (Z[M,D], prior[M]) over filled slots; P(c) split across its slots."""
        zs, ws = [], []
        prior_c = self.class_count / self.class_count.sum()
        for c in range(self.num_fg):
            mask = self.filled[c]
            n = int(mask.sum())
            if n == 0:
                continue
            zs.append(self.bank[c, mask])
            ws.append(prior_c[c].repeat(n) / n)
        if not zs:
            return self.bank.new_zeros(0, self.dim), self.bank.new_zeros(0)
        return torch.cat(zs, 0), torch.cat(ws, 0)


# ---------------------------------------------------------------------------
# Added for the RT-DETR / YOLO ports (built on the residual-split philosophy
# above). No torch.nn.functional import needed — tensor methods are used.
# ---------------------------------------------------------------------------
class ResidualDisentangler(nn.Module):
    """causal = ReLU(W x); spurious = x - causal  (exact additive decomposition).

    Works on the last dim, so it accepts [N, D] (RoI/query rows) or [B, nq, D].
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        causal = torch.relu(self.proj(x))
        spurious = x - causal
        return causal, spurious


def backdoor_context(causal: Tensor, dictionary, q_proj: nn.Module,
                     k_proj: nn.Module) -> Tensor:
    """Attend `causal` (..., Dc) over dictionary prototypes -> context (..., Dz).

    NWGM approximation of the confounder marginalisation: prior-weighted softmax
    attention, renormalised, weighted sum of prototypes. Accepts 2D or 3D causal.
    """
    Z, prior = dictionary.prototypes()                  # [M, Dz], [M]
    if Z.shape[0] == 0:
        return causal.new_zeros(*causal.shape[:-1], Z.shape[-1] if Z.dim() else 0)
    q = q_proj(causal)                                  # (..., Dz)
    k = k_proj(Z)                                        # [M, Dz]
    att = (q @ k.t()) / (q.shape[-1] ** 0.5)            # (..., M)
    att = att.softmax(dim=-1) * prior
    att = att / att.sum(-1, keepdim=True).clamp_min(1e-6)
    return att @ Z                                       # (..., Dz)


class BackdoorCombine(nn.Module):
    """Deconfound a causal feature with a spurious-prototype dictionary.

    Degrades gracefully to (a projection of) the causal feature while the
    dictionary is still empty at the start of training.
    """

    def __init__(self, causal_dim: int, dict_dim: int, out_dim: int,
                 mode: str = "gate"):
        super().__init__()
        self.mode = mode
        self.q = nn.Linear(causal_dim, dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim)
        if mode == "gate":
            self.ctx_to_causal = nn.Linear(dict_dim, causal_dim)
            self.gate = nn.Linear(causal_dim + dict_dim, causal_dim)
            self.out = (nn.Linear(causal_dim, out_dim)
                        if causal_dim != out_dim else nn.Identity())
        else:  # concat
            self.out = nn.Linear(causal_dim + dict_dim, out_dim)

    def forward(self, causal: Tensor, dictionary) -> Tensor:
        ctx = backdoor_context(causal, dictionary, self.q, self.k)
        if self.mode == "gate":
            g = torch.sigmoid(self.gate(torch.cat([causal, ctx], dim=-1)))
            fused = causal + g * self.ctx_to_causal(ctx)
            return self.out(fused)
        return self.out(torch.cat([causal, ctx], dim=-1))
