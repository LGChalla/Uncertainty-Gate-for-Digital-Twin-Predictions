"""
Cell rules (protocol steps 7-11). Every rule acts on the whole batch at once.

All state changes use the waiting-time rule from Table 4.1:
    P(event in dt) = 1 - exp(-rate * dt)
so results do not depend on the step size the way the old linear rule did.

Neighbourhoods:
  - "local" = a disc of radius R squares (M2 driver, M2 suppression of CD8)
  - "adjacent" = the 8 surrounding squares (division, CD8 contact)
"""

import math

import torch
import torch.nn.functional as F

from .grid import EMPTY, TUMOR, M0, M2, FIB, CAF, CD8

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def p_event(rate, dt):
    """Waiting-time probability. rate may be a tensor."""
    return 1.0 - torch.exp(-torch.as_tensor(rate, dtype=torch.float64) * dt)


def disc_kernel(radius_sq: int, device="cpu"):
    r = radius_sq
    yy, xx = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
    k = ((yy ** 2 + xx ** 2) <= r * r).double()
    return k.to(device)[None, None]


def local_sum(mask, kernel):
    """Count of True squares in each square's neighbourhood (zero outside the grid)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(mask.double().unsqueeze(1), kernel, padding=pad).squeeze(1)


def local_fraction(mask, kernel):
    """Fraction of in-grid neighbourhood squares that are True (edges handled fairly)."""
    ones = torch.ones_like(mask, dtype=torch.float64)
    return local_sum(mask, kernel) / local_sum(ones.bool(), kernel)


def adjacent_count(mask):
    k = torch.ones(1, 1, 3, 3, dtype=torch.float64, device=mask.device)
    k[0, 0, 1, 1] = 0.0
    return F.conv2d(mask.double().unsqueeze(1), k, padding=1).squeeze(1)


def shift(x, dy, dx):
    """Move every value by (dy, dx); squares shifted in from outside are False/0."""
    out = torch.zeros_like(x)
    ny, nx = x.shape[-2:]
    ys, yd = (slice(0, ny - dy), slice(dy, ny)) if dy >= 0 else (slice(-dy, ny), slice(0, ny + dy))
    xs, xd = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
    out[..., yd, xd] = x[..., ys, xs]
    return out


def rand_like(t, generator):
    return torch.rand(t.shape, generator=generator, device=t.device, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Step 7: M0 -> M2, driven by the local tumour fraction
# ---------------------------------------------------------------------------
def polarize_macrophages(kind, f_tumour, k_m2, dt, generator):
    p = p_event(k_m2 * f_tumour, dt)
    flip = (kind == M0) & (rand_like(f_tumour, generator) < p)
    return torch.where(flip, torch.full_like(kind, M2), kind), int(flip.sum())


# ---------------------------------------------------------------------------
# Step 9: fibroblast -> CAF, driven by local TGF-beta (absorbing state)
# ---------------------------------------------------------------------------
def activate_fibroblasts(kind, tgfb, k_caf, dt, generator):
    p = p_event(k_caf * tgfb, dt)
    flip = (kind == FIB) & (rand_like(tgfb, generator) < p)
    return torch.where(flip, torch.full_like(kind, CAF), kind), int(flip.sum())


# ---------------------------------------------------------------------------
# Step 10: tumour death (background + necrosis when starved of oxygen)
# ---------------------------------------------------------------------------
def tumour_death(kind, o2, d0, r_nec, o_nec, dt, generator):
    rate = d0 + r_nec * (o2 < o_nec).double()
    die = (kind == TUMOR) & (rand_like(o2, generator) < p_event(rate, dt))
    return torch.where(die, torch.full_like(kind, EMPTY), kind), int(die.sum())


# ---------------------------------------------------------------------------
# Step 19: cisplatin kill. exposure = this hour's effective drug exposure (uM s),
# already scaled by the oxygen and IL-6 factors (drug.py).
# ---------------------------------------------------------------------------
def drug_kill(kind, exposure, k_kill, generator):
    p = 1.0 - torch.exp(-k_kill * exposure)
    die = (kind == TUMOR) & (rand_like(exposure, generator) < p)
    return torch.where(die, torch.full_like(kind, EMPTY), kind), int(die.sum())


# ---------------------------------------------------------------------------
# Step 11: CD8 contact killing, suppressed by nearby M2
# ---------------------------------------------------------------------------
def cd8_kill(kind, m2_local, k_T, K_sup, dt, generator):
    n_cd8 = adjacent_count(kind == CD8)
    rate = k_T * n_cd8 / (1.0 + m2_local / K_sup)
    die = (kind == TUMOR) & (rand_like(rate, generator) < p_event(rate, dt))
    return torch.where(die, torch.full_like(kind, EMPTY), kind), int(die.sum())


# ---------------------------------------------------------------------------
# Step 10: tumour division into an empty adjacent square, oxygen-limited
# ---------------------------------------------------------------------------
def tumour_divide(kind, o2, r_div, K_o, dt, generator):
    """
    Each tumour cell that divides picks one of its 8 neighbours at random and
    succeeds only if that square is empty. Directions are processed one at a
    time in a random order, so two cells can never fill the same square and no
    direction is favoured.
    """
    growth = r_div * o2 / (o2 + K_o)
    wants = (kind == TUMOR) & (rand_like(o2, generator) < p_event(growth, dt))
    direction = torch.randint(0, 8, kind.shape, generator=generator, device=kind.device)
    order = torch.randperm(8, generator=generator, device=generator.device).tolist()
    born = 0
    for d in order:
        dy, dx = MOORE[d]
        arrive = shift(wants & (direction == d), dy, dx) & (kind == EMPTY)
        kind = torch.where(arrive, torch.full_like(kind, TUMOR), kind)
        born += int(arrive.sum())
    return kind, born
