"""
The tissue grid (protocol steps 5-6).

One integer grid per tissue, shape (B, ny, nx): every 20 um square holds at
most one thing, identified by its KIND code. Vessels are squares too: they hold
no cell and act as fixed sources of oxygen (and later drug).
"""

import torch

# Kind codes. Order matters only for readability; never compare with < or >.
EMPTY, TUMOR, M0, M2, FIB, CAF, CD8, VESSEL = 0, 1, 2, 3, 4, 5, 6, 7
NAMES = {EMPTY: "empty", TUMOR: "tumour", M0: "M0", M2: "M2", FIB: "fibroblast",
         CAF: "CAF", CD8: "CD8", VESSEL: "vessel"}
# Cell kinds a composition can set, in this order (columns of `fractions`).
PLACEABLE = (TUMOR, M0, M2, FIB, CAF, CD8)


def area_mm2(nx: int, ny: int, dx_cm: float) -> float:
    return (nx * dx_cm * 10.0) * (ny * dx_cm * 10.0)


def make_vessels(B, ny, nx, dx_cm, density_per_mm2, generator, device="cpu"):
    """
    Vessels at uniformly random squares. density_per_mm2: float or (B,) tensor,
    so each tissue (patient) can have its own vessel density.
    Returns a bool mask (B, ny, nx).
    """
    dens = torch.as_tensor(density_per_mm2, dtype=torch.float64, device=device).expand(B)
    n_v = torch.round(dens * area_mm2(nx, ny, dx_cm)).long().clamp(min=0, max=ny * nx)
    keys = torch.rand(B, ny * nx, generator=generator, device=device)
    order = keys.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(ny * nx, device=device).expand(B, -1))
    return (ranks < n_v[:, None]).view(B, ny, nx)


def place_cells(vessels, fractions, generator):
    """
    Fill non-vessel squares at random so that each tissue gets exactly
    round(fraction * free squares) cells of each kind in PLACEABLE; the rest stay empty.
    fractions: (B, len(PLACEABLE)) or (len(PLACEABLE),), each row summing to <= 1.
    Returns kind grid (B, ny, nx) int8.
    """
    B, ny, nx = vessels.shape
    dev = vessels.device
    fr = torch.as_tensor(fractions, dtype=torch.float64, device=dev)
    fr = fr.expand(B, len(PLACEABLE)) if fr.dim() == 1 else fr
    if bool((fr.sum(1) > 1 + 1e-9).any()):
        raise ValueError("composition fractions sum to more than 1")
    flat_v = vessels.view(B, -1)
    n_free = (~flat_v).sum(1).double()
    counts = torch.round(fr * n_free[:, None]).long()                 # (B, K)
    bounds = counts.cumsum(1)                                          # rank thresholds

    keys = torch.rand(B, ny * nx, generator=generator, device=dev)
    keys = keys.masked_fill(flat_v, 2.0)                               # vessels rank last
    order = keys.argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(ny * nx, device=dev).expand(B, -1))

    kind = torch.full((B, ny * nx), EMPTY, dtype=torch.int8, device=dev)
    lower = torch.zeros(B, 1, dtype=torch.long, device=dev)
    for j, k in enumerate(PLACEABLE):
        upper = bounds[:, j:j + 1]
        kind[(ranks >= lower) & (ranks < upper)] = k
        lower = upper
    kind[flat_v] = VESSEL
    return kind.view(B, ny, nx)


def counts(kind) -> dict:
    """Number of squares of each kind, per tissue."""
    return {NAMES[k]: (kind == k).sum((1, 2)) for k in NAMES}
