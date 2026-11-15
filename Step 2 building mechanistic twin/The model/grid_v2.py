"""
The tissue grid, version 2. tme_twin/grid.py is unchanged and still describes
every run before Nov 5, 2026.

What changed: two new cell kinds, CD4 T cells and B cells, so that the twin
models the same compartments EPIC reports. EPIC's TRef reference returns eight
compartments; v1 modelled four of the cell ones and dropped B, CD4 and NK cells,
which left the twin's fractions summing to 1 while the patients' summed to
0.70-0.86 (scripts/phase5_basis_check.py, Nov 5). NK cells are not added: EPIC
puts them at a median of 0.0000 in all four cohorts.

Existing kind codes keep their values, so a v1 grid is a valid v2 grid.
"""

import torch

from .grid import (EMPTY, TUMOR, M0, M2, FIB, CAF, CD8, VESSEL,       # noqa: F401
                   area_mm2, make_vessels)

CD4, BCELL = 8, 9

NAMES = {EMPTY: "empty", TUMOR: "tumour", M0: "M0", M2: "M2", FIB: "fibroblast",
         CAF: "CAF", CD8: "CD8", VESSEL: "vessel", CD4: "CD4", BCELL: "B"}

# Columns of a starting composition, in this order.
PLACEABLE = (TUMOR, M0, M2, FIB, CAF, CD8, CD4, BCELL)

# Cells that move (step 7b). Tumour cells, fibroblasts and CAFs stay put.
MOTILE_LYMPHOID = (CD8, CD4, BCELL)
MOTILE_MYELOID = (M0, M2)


def place_cells(vessels, fractions, generator):
    """
    Same rule as grid.place_cells, over the v2 kinds: every tissue gets exactly
    round(fraction * free squares) cells of each kind, at random squares.
    fractions: (B, 8) or (8,), in PLACEABLE order, each row summing to <= 1.
    """
    B, ny, nx = vessels.shape
    dev = vessels.device
    fr = torch.as_tensor(fractions, dtype=torch.float64, device=dev)
    fr = fr.expand(B, len(PLACEABLE)) if fr.dim() == 1 else fr
    if fr.shape[1] != len(PLACEABLE):
        raise ValueError(f"need {len(PLACEABLE)} fractions, got {fr.shape[1]}")
    if bool((fr.sum(1) > 1 + 1e-9).any()):
        raise ValueError("composition fractions sum to more than 1")
    flat_v = vessels.view(B, -1)
    n_free = (~flat_v).sum(1).double()
    counts = torch.round(fr * n_free[:, None]).long()
    bounds = counts.cumsum(1)

    keys = torch.rand(B, ny * nx, generator=generator, device=dev)
    keys = keys.masked_fill(flat_v, 2.0)
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
