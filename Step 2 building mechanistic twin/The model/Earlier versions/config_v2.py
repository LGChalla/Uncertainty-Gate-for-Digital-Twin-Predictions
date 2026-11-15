"""
Model constants, version 2. config.py is unchanged and still describes every
run reported up to Nov 5, 2026.

What v2 adds: a source and a sink for every stromal and immune population,
which the design called for and cells.py never implemented (see
docs/population_rates_v1.md and corrections log item 15). Two new populations,
CD4 T cells and B cells, so the twin's compartments match EPIC's.

Every rate here is a working value at the centre of its prior in priors_v3.py.
status follows config.py: ok | verify | find | calibrate.

Units: rates in 1/s, as everywhere else in the model. The /day value is given in
the source string because that is how the papers report them.
"""

from .config import TME, P

DAY = 86400.0

NEW = {
    # ---------------- recruitment (patient parameters) ----------------
    # Each vessel recruits into one of its 8 neighbouring squares, if empty.
    # Recruited cells then move (below), so the squares around a vessel clear
    # and recruitment is not capped by crowding.
    "r_mac":  P(0.03 / DAY, "cells/vessel/s", "derived: steady state R = N*(death - division); "
                "docs/population_rates_v1.md", "calibrate"),
    "r_cd8":  P(0.03 / DAY, "cells/vessel/s", "derived, as r_mac", "calibrate"),
    "r_cd4":  P(0.05 / DAY, "cells/vessel/s", "derived, as r_mac", "calibrate"),
    "r_b":    P(0.005 / DAY, "cells/vessel/s", "derived, as r_mac", "calibrate"),

    # ---------------- macrophages ----------------
    "d_mac":  P(0.10 / DAY, "1/s", "dwell 3-30 d prior, geometric middle ~10 d. Loyher 2018 JEM "
                "215:2536 (monocyte-derived TAMs 58-70% by day 10-15); Lee 2025 Cancer Immunol Res "
                "(mdTAM pool continually replenished)", "find"),
    "p_mac":  P(0.0, "1/s", "local division set to 0: monocyte-derived majority (Loyher 2018); "
                "resident IM proliferation reported without a rate", "find"),
    # Reversion: off in the main analysis (LC, Nov 5). Turned on for one
    # sensitivity run; discussed as a limitation. From one bulk snapshot,
    # reversion and death both remove M2s and cannot be told apart.
    "k_m2_rev": P(0.0, "1/s", "M2 -> M0 reversion; 0 in main analysis, sensitivity run only", "ok"),

    # ---------------- fibroblasts ----------------
    # A resident lineage: no vascular recruitment. Division into an empty
    # neighbouring square at rate p * (1 - local lineage fraction / K_fib),
    # i.e. logistic contact inhibition (tissue_v2._fibroblast_divide_logistic).
    "K_fib":  P(0.10, "local fraction", "density set point; replaces f0_fibroblast", "calibrate"),
    # Division is set as a multiple of death, u_fib = division / death > 1.
    # With division only where there is room and only into an empty square,
    # births always fall short of deaths when division = death, and the lineage
    # declines toward extinction (first v2 steady-state test, Nov 5: -15% over
    # 60 days and still falling). u_fib > 1 lets contact inhibition, not
    # attrition, set the level. No rate source; prior 2-20.
    "u_fib":  P(5.0, "-", "fibroblast division / death; must exceed 1; prior 2-20", "find"),
    "m_caf":  P(2.0, "-", "CAF division multiplier over resting; TGF-beta-driven; prior 1-5x", "find"),
    "d_fib":  P(0.005 / DAY, "1/s", "no source found; prior 0.001-0.02/d", "find"),
    # CAF reversion is ON, unlike M2 reversion. Fibroblasts are resident: no
    # fresh resting fibroblasts arrive from blood, so with one-way activation
    # the only steady state is every fibroblast a CAF (caf_share = 1), and the
    # observed CAF share becomes a clock again, as in v1. Macrophages escape
    # this because M2s die and are replaced by new M0s. Found in the first v2
    # smoke test, Nov 5. No rate source; wide prior 0.01-1/day.
    "k_caf_rev": P(0.1 / DAY, "1/s", "CAF -> fibroblast reversion; prior 0.01-1/d; required for a "
                   "steady CAF share; no rate source", "find"),

    # ---------------- CD8 ----------------
    "d_cd8":  P(0.113 / DAY, "1/s", "Macallan 2004 JEM 200:255: CD4 effector-memory disappearance "
                "11.3%/d in blood (t1/2 6 d); no CD8-in-tumour measurement found", "verify"),
    # Division as a share of death keeps the population recruitment-sustained:
    # p = s * d with s < 1, so it cannot grow without bound.
    "s_cd8":  P(0.30, "-", "division / death; Thommen 2018 Nat Med 24:994 S/G2/M 0.5-9.1% "
                "-> 0.01-0.36/d (conversion assumes S+G2+M = 6-12 h)", "verify"),

    # ---------------- CD4 ----------------
    # One net effect on CD8 killing (LC, Nov 5: "give CD4 a role"). EPIC's CD4
    # compartment mixes helper cells, which raise killing, and Tregs, which
    # lower it, and bulk data cannot split them. So the effect is a single
    # shared parameter whose prior spans both signs:
    #   CD8 kill rate x exp(gamma_cd4 * local CD4 fraction)
    # gamma_cd4 = 0 recovers the v1 rule exactly.
    "gamma_cd4": P(0.0, "-", "net CD4 effect on CD8 killing; prior -5 to +5 (factor 0.22-4.5 "
                   "at a local CD4 fraction of 0.3); no rate source", "find"),
    "d_cd4":  P(0.068 / DAY, "1/s", "Macallan 2004: TCM 4.1%/d, TEM 11.3%/d; geometric middle", "verify"),
    "s_cd4":  P(0.40, "-", "division / death; Macallan 2004 TCM 1.5/4.1, TEM 4.7/11.3", "verify"),

    # ---------------- B cells (no effector rule) ----------------
    "d_b":    P(0.039 / DAY, "1/s", "Macallan 2005 Blood 105:3633: memory B 3.9%/d (t1/2 ~18 d)", "verify"),
    "s_b":    P(0.49, "-", "division / death; Macallan 2005: 1.9%/d / 3.9%/d", "verify"),

    # ---------------- starting state, fibroblast lineage ----------------
    # The mean-field set point misses spatial clustering (daughters land beside
    # parents), so the lineage starts ~8% above its true equilibrium and then
    # relaxes with a time constant of ~50 days, longer than the burn-in. Its
    # equilibrium depends only on rate ratios (division/death, activation/
    # reversion), not on absolute speed, so the lineage alone is run with every
    # one of its rates multiplied by `fib_preeq_speedup` before the burn-in.
    # A numerical device that leaves the equilibrium unchanged.
    "fib_preeq_days":    P(0, "days", "pre-equilibration length at the sped-up rates; 0 = off. "
                           "Not needed once fibroblasts move (move_fibroblast), kept for checks", "ok"),
    "fib_preeq_speedup": P(20, "-", "rate multiplier during pre-equilibration", "ok"),

    # ---------------- movement (random walk, cells_v2.move) ----------------
    # One step = one neighbouring square; mean squared step over the 8
    # neighbours is 600 um^2, so in 2D the motility coefficient is
    # M = 600 * steps / (4 * 60) = 2.5 * steps um^2/min (before crowding).
    # Salmon 2012 JCI 122:899, fresh human lung tumour slices: stromal hot
    # spots have M > 10 um^2/min (~20% of stroma); tumour islets ~3x lower.
    # 2 steps/h gives M ~ 5, inside that range; crowding lowers it further in
    # dense tumour, as observed. Integer steps, fixed across tissues.
    "move_lymphoid": P(2, "steps/h", "CD8, CD4, B; Salmon 2012 (M ~ 5 um^2/min)", "verify"),
    "move_myeloid":  P(1, "steps/h", "M0, M2; slower than T cells; no number found", "find"),
    # Fibroblasts and CAFs are migratory. Without movement the lineage is a
    # 2D contact process: daughters crowd their parents, local density outruns
    # the mean, and the population declines slowly even at division/death = 5
    # (v2 test, Nov 5: -10% over 60 days after 100 effective days of
    # pre-equilibration). Slow movement keeps the lineage mixed, so its
    # closed-form steady state holds.
    "move_fibroblast": P(1, "steps/h", "fibroblasts, CAFs; ~20 um/h; no number found", "find"),
}

TME_V2 = {**TME, **NEW}
