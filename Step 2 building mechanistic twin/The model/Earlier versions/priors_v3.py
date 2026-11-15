"""
Priors, version 3, for the v2 model (tissue_v2). priors.py and priors_v2.py
are unchanged and still describe the runs made with them.

What changed from v2
--------------------
The starting fractions f0_macrophage, f0_fibroblast and f0_cd8 are gone. In v2
the stroma had no source or sink, so a population's size was its starting
count diluted by tumour growth, and f0_* had to be tuned against EPIC. In the
v2 model every population is a rate balance:

  macrophages, CD8, CD4, B   recruitment at vessels (patient), death and
                             division (cohort)
  fibroblasts                density set point K_fib (patient), death,
                             division/death ratio, CAF multiplier, CAF
                             reversion (cohort)

Recruitment is the patient parameter because what varies between tumours is
chemokine signalling and vascular access, not how long a T cell lives. From
one snapshot only recruitment / (death - division) is identifiable, so death
and division are shared, sampled and marginalised.

Two structural constraints keep every population at a finite, positive
steady state: recruited populations divide at s * death with s < 1, and
fibroblasts divide at u * death with u > 1 (docs/population_rates_v1.md,
corrections log item 15).

Sources for every range: tme_twin/config_v2.py and docs/population_rates_v1.md.
"""

import math

DAY = 86400.0

REGIMES = {
    "residual": {"f0_tumour": (0.005, 0.05),
                 "settle": ("M0", "M2", "fibroblast", "CAF", "CD8", "CD4", "B"),
                 "note": "sparse post-resection disease; tumour still expanding at treatment start"},
    "primary":  {"f0_tumour": (0.20, 0.45),
                 "settle": ("M0", "M2", "fibroblast", "CAF", "CD8", "CD4", "B", "tumour"),
                 "note": "confluent primary patch"},
}
DEFAULT_REGIME = "residual"

# name -> (low, high, scale, class, what it controls); "log" = log-uniform
PRIORS = {
    # --- tissue structure
    "vessel_density": (10.0, 60.0, "lin", "patient", "vessels per mm^2; oxygen, drug, recruitment"),
    "f0_tumour":      (0.005, 0.05, "log", "patient", "starting tumour fraction (regime sets the range)"),

    # --- recruitment at vessels (patient): derived range, rates file
    "r_mac": (1e-4 / DAY, 1.0 / DAY, "log", "patient", "macrophage (as M0) recruitment per vessel"),
    "r_cd8": (1e-4 / DAY, 1.0 / DAY, "log", "patient", "CD8 recruitment per vessel"),
    "r_cd4": (1e-4 / DAY, 1.0 / DAY, "log", "patient", "CD4 recruitment per vessel"),
    "r_b":   (1e-4 / DAY, 1.0 / DAY, "log", "patient", "B-cell recruitment per vessel"),
    "K_fib": (0.01, 0.50, "log", "patient", "fibroblast density set point (contact inhibition)"),

    # --- cell-state and tumour rates (unchanged from v2)
    "k_m2":   (0.02 / 3600, 0.50 / 3600, "log", "patient", "M0->M2 rate per unit local tumour fraction"),
    "k_caf":  (0.02 / 3600, 0.30 / 3600, "log", "patient", "fibroblast->CAF rate per nM TGF-beta"),
    "r_div":  (math.log(2) / (72 * 3600), math.log(2) / (24 * 3600), "log", "patient", "tumour division rate"),
    "r_il6":  (0.08 / 3600, 0.80 / 3600, "log", "patient", "IL-6 per M2"),
    "Vmax_o2": (3.5, 14.0, "log", "patient", "oxygen use per occupied square"),

    # --- population turnover (cohort): measured or bracketed
    "d_mac":     (1 / 30 / DAY, 1 / 3 / DAY, "log", "cohort", "macrophage death/egress (dwell 3-30 d); find"),
    "d_fib":     (0.001 / DAY, 0.02 / DAY, "log", "cohort", "fibroblast death; find"),
    "u_fib":     (2.0, 20.0, "log", "cohort", "fibroblast division / death (> 1); find"),
    "m_caf":     (1.0, 5.0, "log", "cohort", "CAF division multiplier; find"),
    "k_caf_rev": (0.01 / DAY, 1.0 / DAY, "log", "cohort", "CAF -> fibroblast reversion; find"),
    "d_cd8":     (0.05 / DAY, 0.30 / DAY, "log", "cohort", "CD8 death; Macallan 2004 anchor 0.113/d"),
    "s_cd8":     (0.0, 0.9, "lin", "cohort", "CD8 division / death; Thommen 2018"),
    "d_cd4":     (0.03 / DAY, 0.20 / DAY, "log", "cohort", "CD4 death; Macallan 2004 0.041-0.113/d, widened"),
    "s_cd4":     (0.2, 0.6, "lin", "cohort", "CD4 division / death; Macallan 2004 ~0.4"),
    "d_b":       (0.02 / DAY, 0.10 / DAY, "log", "cohort", "B death; Macallan 2005 0.039/d"),
    "s_b":       (0.1, 0.8, "lin", "cohort", "B division / death; Macallan 2005 ~0.49"),
    "gamma_cd4": (-5.0, 5.0, "lin", "cohort", "net CD4 effect on CD8 killing; spans help and Treg"),

    # --- CD8 killing, signalling, oxygen, tumour death (unchanged from v2)
    "k_T":    (0.01 / 3600, 0.20 / 3600, "log", "cohort", "CD8 kill rate per adjacent CD8"),
    "K_sup":  (1.0, 10.0, "log", "cohort", "M2 count for half-suppression of CD8 killing"),
    "r_tgfb": (0.01 / 3600, 0.16 / 3600, "log", "cohort", "TGF-beta per M2"),
    "o_vessel": (20.0, 60.0, "lin", "cohort", "pO2 held at vessels"),
    "d0":     (0.0002 / 3600, 0.005 / 3600, "log", "cohort", "background tumour death"),
    "r_nec":  (0.01 / 3600, 0.20 / 3600, "log", "cohort", "necrosis rate below the oxygen threshold"),

    # --- drug and transport: latent by construction (untreated tissue at H)
    "D0_drug":       (3.5e-8, 8.8e-7, "log", "latent", "cisplatin tissue diffusivity"),
    "mu_drug":       (1e-5, 1e-3, "log", "latent", "cisplatin uptake per occupied square"),
    "beta_caf":      (0.0, 5.0, "lin", "latent", "how much CAF density slows the drug"),
    "k_kill":        (5e-6, 5e-5, "log", "latent", "cisplatin kill rate"),
    "K_o_kill":      (5.0, 60.0, "log", "latent", "oxygen for half-maximal kill"),
    "K_il6_protect": (0.01, 10.0, "log", "latent", "IL-6 for half protection; prior spans no effect"),
}

SCREEN = tuple(k for k, v in PRIORS.items() if v[3] != "latent")


def regime_priors(regime=DEFAULT_REGIME):
    out = dict(PRIORS)
    lo, hi = REGIMES[regime]["f0_tumour"]
    out["f0_tumour"] = (lo, hi, "log" if regime == "residual" else "lin", "patient",
                        "starting tumour fraction")
    return out


def settle_populations(regime=DEFAULT_REGIME):
    return REGIMES[regime]["settle"]


def bounds(names=None, regime=DEFAULT_REGIME):
    pri = regime_priors(regime)
    names = list(names or SCREEN)
    b = [[math.log(pri[n][0]), math.log(pri[n][1])] if pri[n][2] == "log" else [pri[n][0], pri[n][1]]
         for n in names]
    return {"num_vars": len(names), "names": names, "bounds": b}


def to_native(row, names, regime=DEFAULT_REGIME):
    pri = regime_priors(regime)
    return {n: (math.exp(x) if pri[n][2] == "log" else float(x)) for x, n in zip(row, names)}


def split(values):
    """(f0_tumour, overrides) for TissueV2."""
    return values.get("f0_tumour", 0.02), {k: v for k, v in values.items() if k != "f0_tumour"}


def by_class(cls):
    return tuple(k for k, v in PRIORS.items() if v[3] == cls)
