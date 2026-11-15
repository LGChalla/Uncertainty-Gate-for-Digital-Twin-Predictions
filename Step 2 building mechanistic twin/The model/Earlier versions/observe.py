"""
The observation operator H (Phase 5, step 26).

H maps a burned-in twin to the same row a real patient's expression data
produce, so the two can be compared during calibration.

Two parts, kept separate on purpose:

  summarise(T)       -> raw twin quantities in physical units. No fitting.
  apply_plan(raw, …) -> the vector comparable to patient data, under a named
                        observation plan.

The split matters because pathway scores are not concentrations. GSVA returns
an enrichment score standardised within a dataset, so the map from a simulated
concentration to a reported score carries its own scale and offset. Those are
nuisance parameters of the ASSAY, not of the patient: fitted once per dataset
and shared across its patients. Keeping them out of summarise() leaves the
physics in physical units and the modelling choice visible.

Observation plans
-----------------
What the data can see is a choice, not a fact, so it is configurable and the
identifiability analysis is run under each plan.

EPIC reports CAFs, endothelial cells, CD8 T cells, macrophages and a residual
"other" (tumour) fraction. It does not split macrophages into M0 and M2, nor
resting fibroblasts from activated CAFs. Under the base plan those two twin
quantities are unobserved and constrained only indirectly, through the IL-6 and
TGF-beta scores that M2 cells drive.

Two plans add a direct handle on m2_share:

  epic+cibersort  CIBERSORTx with the LM22 signature matrix reports M0, M1 and
                  M2 macrophages separately, so m2_share is compared as a
                  fraction. LM22's three macrophage states are highly
                  correlated and are among its least reliable distinctions, so
                  this feature carries a deliberately wide noise term rather
                  than being treated as precise.
  epic+gsva_m2    An M2 polarisation signature scored by GSVA, compared through
                  the same monotone link as the other pathway scores. No single
                  canonical M2 gene set exists, so the chosen set is recorded
                  with the plan and this route is reported as a sensitivity
                  analysis rather than the primary.

caf_share is included as a target under every plan, but it is the weakest one
and is reported as such. EPIC's CAF reference profile was built from
cancer-associated fibroblasts, so its CAF fraction should in principle map to
the activated compartment alone. In practice, deconvolving a mixed fibroblast
population against a CAF-only reference recovers some of the resting
compartment too, by an unknown amount. The observed quantity is therefore a
blend,

    EPIC CAF fraction  ~  w * f_CAF + (1 - w) * f_fibroblast,

with w unknown. Rather than assert a value, we compare caf_share directly and
give it the widest noise term of any target, so the identifiability analysis
measures what that ambiguity costs instead of hiding it. The expectation is
poor contraction, and Section~\ref{sec:disc_caf} of the paper sets out why the
CAF arm is weak at three levels at once: the transport mechanism it encodes is
negligible for a small molecule, the quantity itself is only ambiguously
measurable, and its direction check reversed for reasons of tissue density
rather than biology.
"""

import torch

from . import grid as G

# Rows H returns, in a fixed order.
FEATURES = (
    "f_tumour",         # tumour / all cells
    "f_macrophage",     # (M0 + M2) / all cells
    "f_fibroblast",     # (fibroblast + CAF) / all cells
    "f_cd8",            # CD8 / all cells
    "f_endothelial",    # vessels / (all cells + vessels)
    "hypoxia",          # fraction of cells below the hypoxia cutoff
    "tgfb",             # mean TGF-beta over the patch, nM
    "il6",              # mean IL-6 over the patch, nM
    "m2_share",         # M2 / (M0 + M2)          -- not observable from EPIC
    "caf_share",        # CAF / (fibroblast + CAF) -- not observable from EPIC
)

# Quantities that reach the data through a fitted monotone link rather than
# directly. Fractions are compared as fractions; pathway scores are not.
LINKED_BASE = ("hypoxia", "tgfb", "il6")

# How each plan sees the twin. "direct" = compared as a fraction,
# "linked" = through offset + scale * log(.), absent = not a calibration target.
# noise is the standard deviation added when training the posterior estimator,
# on the scale the feature is compared on.
PLANS = {
    "epic": {
        "direct": {"f_tumour": 0.08, "f_macrophage": 0.06, "f_fibroblast": 0.06,
                   "f_cd8": 0.04, "f_endothelial": 0.04,
                   # EPIC's CAF reference cannot cleanly separate resting
                   # fibroblasts from activated CAFs; widest term of any target.
                   "caf_share": 0.25},
        "linked": {"hypoxia": 0.25, "tgfb": 0.25, "il6": 0.25},
        "note": "EPIC fractions + GSVA pathway scores. m2_share unobserved.",
    },
    "epic+cibersort": {
        "direct": {"f_tumour": 0.08, "f_macrophage": 0.06, "f_fibroblast": 0.06,
                   "f_cd8": 0.04, "f_endothelial": 0.04,
                   # LM22 separates M0/M1/M2 poorly; wide on purpose.
                   "m2_share": 0.20, "caf_share": 0.25},
        "linked": {"hypoxia": 0.25, "tgfb": 0.25, "il6": 0.25},
        "note": "Adds CIBERSORTx LM22 macrophage polarisation as a direct target.",
    },
    "epic+gsva_m2": {
        "direct": {"f_tumour": 0.08, "f_macrophage": 0.06, "f_fibroblast": 0.06,
                   "f_cd8": 0.04, "f_endothelial": 0.04, "caf_share": 0.25},
        "linked": {"hypoxia": 0.25, "tgfb": 0.25, "il6": 0.25, "m2_share": 0.30},
        "note": "Adds a GSVA-scored M2 polarisation signature through the link. "
                "Record the gene set used; sensitivity analysis only.",
    },
}
DEFAULT_PLAN = "epic+cibersort"


def plan_features(plan=DEFAULT_PLAN):
    """Feature names a plan uses as calibration targets, in FEATURES order."""
    p = PLANS[plan]
    keep = set(p["direct"]) | set(p["linked"])
    return tuple(f for f in FEATURES if f in keep)


def plan_noise(plan=DEFAULT_PLAN):
    """Measurement noise per target feature, matching plan_features order."""
    p = PLANS[plan]
    merged = {**p["direct"], **p["linked"]}
    return torch.tensor([merged[f] for f in plan_features(plan)], dtype=torch.float64)


def summarise(T):
    """Raw twin quantities, in physical units. Returns (B, len(FEATURES))."""
    c = G.counts(T.kind)
    dev = T.kind.device
    f = lambda k: c[k].to(dev).double()
    cells = (f("tumour") + f("M0") + f("M2") + f("fibroblast") + f("CAF")
             + f("CD8")).clamp(min=1)
    macs = (f("M0") + f("M2")).clamp(min=1)
    fibs = (f("fibroblast") + f("CAF")).clamp(min=1)
    vessels = f("vessel")

    rows = {
        "f_tumour":      f("tumour") / cells,
        "f_macrophage":  (f("M0") + f("M2")) / cells,
        "f_fibroblast":  (f("fibroblast") + f("CAF")) / cells,
        "f_cd8":         f("CD8") / cells,
        "f_endothelial": vessels / (cells + vessels),
        "hypoxia":       T.hypoxic_fraction().to(dev).double(),
        "tgfb":          T.tgfb.mean((1, 2)),
        "il6":           T.il6.mean((1, 2)),
        "m2_share":      f("M2") / macs,
        "caf_share":     f("CAF") / fibs,
    }
    return torch.stack([rows[k] for k in FEATURES], 1)


def apply_plan(raw, plan=DEFAULT_PLAN, scale=None, offset=None):
    """
    Reduce raw twin quantities to a plan's targets, applying the monotone link
    to the linked ones.

    scale and offset are one value per linked feature of this plan, in
    plan_features order, fitted per dataset and shared across its patients.
    Omitted, the link is the identity on log(x).
    """
    p = PLANS[plan]
    names = plan_features(plan)
    cols = [FEATURES.index(n) for n in names]
    out = raw[:, cols].clone()
    linked_pos = [i for i, n in enumerate(names) if n in p["linked"]]
    if not linked_pos:
        return out, names
    k = len(linked_pos)
    sc = torch.ones(k, dtype=raw.dtype, device=raw.device) if scale is None else \
        torch.as_tensor(scale, dtype=raw.dtype, device=raw.device).reshape(-1, k)
    of = torch.zeros(k, dtype=raw.dtype, device=raw.device) if offset is None else \
        torch.as_tensor(offset, dtype=raw.dtype, device=raw.device).reshape(-1, k)
    out[:, linked_pos] = of + sc * torch.log(out[:, linked_pos].clamp(min=1e-12))
    return out, names


def observe(T, plan=DEFAULT_PLAN, scale=None, offset=None):
    """summarise then apply_plan. Returns (rows, feature names)."""
    return apply_plan(summarise(T), plan, scale, offset)


def add_noise(rows, plan=DEFAULT_PLAN, generator=None):
    """Add the plan's measurement noise, for building the calibration training set."""
    sd = plan_noise(plan).to(rows.device)
    eps = torch.randn(rows.shape, generator=generator, device=rows.device, dtype=rows.dtype)
    return rows + eps * sd


def as_dict(rows, names=FEATURES):
    """One dict per tissue, for the run log."""
    return [{n: float(r[i]) for i, n in enumerate(names)} for r in rows]
