"""
Cell rules, version 2. tme_twin/cells.py is unchanged; every v1 rule is reused
from it as is (tumour division, tumour death, drug kill, M0 -> M2, fibroblast -> CAF).

What v1 lacked
--------------
The design had every population as an independent agent with its own source
and sink. v1 implemented only the conversions, so no stromal or immune cell was
ever recruited, born or killed. Each population was its starting count diluted
by tumour growth: no steady state, and fractions that were initial conditions
rather than rate balances (we have recorded this in the corrections log, items 6, 7 and 15- easier to track project wise. Also the foundational set up for the future direction of this project for 3D representations
).

What v2 adds, per population
----------------------------
  macrophages (M0, M2)  recruited at vessels as M0; die; move
  CD8, CD4, B cells     recruited at vessels; divide; die; move
  fibroblasts, CAFs     resident, not recruited; divide into empty space while
                        local lineage density is below a set point; die; CAFs
                        divide faster than resting fibroblasts
  reversion             M2 -> M0 and CAF -> fibroblast, rate 0 in the main
                        analysis, switched on for one sensitivity check
  CD4                   a single net effect on CD8 killing, sign unknown:
                        helper cells raise it, Tregs lower it, and EPIC's CD4
                        compartment mixes both

Every rule uses the waiting-time probability P = 1 - exp(-rate * dt), as in v1.
Rates and their sources: docs/population_rates_v1.md.

Space: recruitment, division and movement all need an empty neighbouring
square, and two cells can never land in the same square. Directions are taken
one at a time in random order, the same scheme cells.tumour_divide uses.
"""

import torch

from .cells import (p_event, rand_like, shift, local_fraction, local_sum,   # noqa: F401
                    adjacent_count, MOORE)
from .grid_v2 import EMPTY, TUMOR, M0, M2, FIB, CAF, CD8, CD4


def _place_into_neighbours(kind, sources, values, generator):
    """
    Each True square in `sources` sends `values` (that square's value, or one
    kind code everywhere) to one random neighbour, if that neighbour is empty.
    Returns (new kind grid, arrivals mask, departures mask).
    """
    direction = torch.randint(0, 8, kind.shape, generator=generator, device=kind.device)
    order = torch.randperm(8, generator=generator, device=generator.device).tolist()
    arrived = torch.zeros_like(sources)
    departed = torch.zeros_like(sources)
    pending = sources.clone()
    for d in order:
        dy, dx = MOORE[d]
        go = pending & (direction == d)
        arrive = shift(go, dy, dx) & (kind == EMPTY)
        moved_val = shift(torch.where(go, values, torch.zeros_like(kind)), dy, dx)
        kind = torch.where(arrive, moved_val, kind)
        back = shift(arrive, -dy, -dx)
        arrived |= arrive
        departed |= back
        pending &= ~back
    return kind, arrived, departed


# ---------------------------------------------------------------------------
# Recruitment at vessels
# ---------------------------------------------------------------------------
def recruit(kind, vessels, new_kind, rate_per_vessel, dt, generator):
    """Each vessel recruits one cell of new_kind with P = 1 - exp(-rate dt),
    into a random neighbouring square if it is empty. rate: 1/s per vessel."""
    p = p_event(rate_per_vessel, dt)
    fire = vessels & (rand_like(kind.double(), generator) < p)
    vals = torch.full_like(kind, new_kind)
    kind, arrived, _ = _place_into_neighbours(kind, fire, vals, generator)
    return kind, int(arrived.sum())


# ---------------------------------------------------------------------------
# Division and death, any population
# ---------------------------------------------------------------------------
def divide(kind, mask, rate, dt, generator):
    """Cells in `mask` divide with P = 1 - exp(-rate dt); the daughter has the
    parent's kind and goes to an empty neighbour, or the division fails."""
    wants = mask & (rand_like(kind.double(), generator) < p_event(rate, dt))
    kind, arrived, _ = _place_into_neighbours(kind, wants, kind, generator)
    return kind, int(arrived.sum())


def die(kind, kinds, rate, dt, generator):
    """Cells of any kind in `kinds` die with P = 1 - exp(-rate dt)."""
    alive = torch.zeros_like(kind, dtype=torch.bool)
    for k in kinds:
        alive |= kind == k
    gone = alive & (rand_like(kind.double(), generator) < p_event(rate, dt))
    return torch.where(gone, torch.full_like(kind, EMPTY), kind), int(gone.sum())


def fibroblast_divide(kind, kernel, K_fib, rate_fib, caf_mult, dt, generator):
    """
    Resident fibroblasts and CAFs divide only where the local lineage fraction
    (fibroblasts + CAFs over the neighbourhood disc) is below the set point
    K_fib: contact inhibition with a patient-specific density. CAFs divide
    caf_mult times faster than resting fibroblasts.
    """
    lineage = (kind == FIB) | (kind == CAF)
    room = local_fraction(lineage, kernel) < K_fib
    rate = torch.where(kind == CAF, rate_fib * caf_mult, rate_fib * torch.ones_like(kind.double()))
    wants = lineage & room & (rand_like(kind.double(), generator) < p_event(rate, dt))
    kind, arrived, _ = _place_into_neighbours(kind, wants, kind, generator)
    return kind, int(arrived.sum())


# ---------------------------------------------------------------------------
# Reversion (rate 0 in the main analysis)
# ---------------------------------------------------------------------------
def revert(kind, from_kind, to_kind, rate, dt, generator):
    p = p_event(rate, dt)
    flip = (kind == from_kind) & (rand_like(kind.double(), generator) < p)
    return torch.where(flip, torch.full_like(kind, to_kind), kind), int(flip.sum())


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------
def move(kind, kinds, steps, generator):
    """
    Random walk: `steps` times per hour, every cell of the given kinds tries to
    step into a random neighbouring square, and succeeds if it is empty.
    Without this, recruited cells pile up beside the vessel that recruited them
    and block further recruitment, and CD8 contact killing is set by where
    cells happened to be placed rather than by migration.
    """
    total = 0
    for _ in range(int(steps)):
        movers = torch.zeros_like(kind, dtype=torch.bool)
        for k in kinds:
            movers |= kind == k
        kind, arrived, departed = _place_into_neighbours(kind, movers, kind, generator)
        kind = torch.where(departed & ~arrived, torch.full_like(kind, EMPTY), kind)
        # A square both vacated and refilled in the same pass holds the newcomer.
        total += int(arrived.sum())
    return kind, total


# ---------------------------------------------------------------------------
# CD8 killing with M2 suppression and a net CD4 effect
# ---------------------------------------------------------------------------
def cd8_kill(kind, m2_local, cd4_frac_local, k_T, K_sup, gamma_cd4, dt, generator):
    """
    v1 rule, k_T * n_adjacent_CD8 / (1 + M2_local / K_sup), times
    exp(gamma_cd4 * local CD4 fraction). gamma_cd4 > 0 means CD4 help
    dominates, < 0 means Treg suppression dominates, 0 recovers v1 exactly.
    """
    n_cd8 = adjacent_count(kind == CD8)
    rate = k_T * n_cd8 * torch.exp(gamma_cd4 * cd4_frac_local) / (1.0 + m2_local / K_sup)
    die_ = (kind == TUMOR) & (rand_like(rate, generator) < p_event(rate, dt))
    return torch.where(die_, torch.full_like(kind, EMPTY), kind), int(die_.sum())
