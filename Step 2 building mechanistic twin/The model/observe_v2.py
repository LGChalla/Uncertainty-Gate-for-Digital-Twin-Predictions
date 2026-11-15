"""
Observation operator H, version 2. observe.py is unchanged.

Three corrections from the Nov 5 audit (corrections log, items 12-14), each
checked by scripts/phase5_basis_check.py:

1. One basis for every fraction. EPIC's compartments are fractions of the
   same whole, vessels' endothelium included. v1 divided cell fractions by
   cells only and the endothelial fraction by cells + vessels, so the twin's
   four cell fractions summed to exactly 1 while the patients' summed to
   0.70-0.86. Here every fraction is over all cells plus vessels.

2. CD4 T cells and B cells are compartments, as in EPIC (TRef). NK cells are
   not modelled (EPIC median 0.0000 in three cohorts, 0.0002 in the fourth);
   on the data side the seven kept EPIC compartments are divided by 1 - NK so
   both sides sum to 1 over the same compartments (data_pipeline/features_v2.py).

3. caf_share goes through the fitted monotone link, like m2_share. Both are
   GSVA enrichment scores, not fractions. v1 rank-normalised the data and
   compared caf_share directly against the twin's CAF/(fibroblast + CAF).

Raw twin quantities stay in physical units here; the link, its floor and the
noise model are applied at training time, as in v1.
"""

import torch

from . import grid_v2 as G

FEATURES = (
    "f_tumour",         # tumour / (all cells + vessels)
    "f_macrophage",     # (M0 + M2) / same
    "f_fibroblast",     # (fibroblast + CAF) / same
    "f_cd8",            # CD8 / same
    "f_cd4",            # CD4 / same
    "f_b",              # B / same
    "f_endothelial",    # vessels / same
    "hypoxia",          # fraction of cells below the hypoxia cutoff
    "tgfb",             # mean TGF-beta over the patch, nM
    "il6",              # mean IL-6 over the patch, nM
    "m2_share",         # M2 / (M0 + M2)
    "caf_share",        # CAF / (fibroblast + CAF)
)
FRACTIONS = FEATURES[:7]          # sum to 1 in every tissue

PLANS = {
    "epic+gsva_m2": {
        # direct targets: noise is a coefficient of variation, applied at
        # training time (phase5_train_npe_v2 --noise-mode relative)
        "direct": {f: 0.15 for f in FRACTIONS},
        # linked targets: noise on the latent log scale, carried through the
        # fitted link's scale factor at training time
        "linked": {"hypoxia": 0.25, "tgfb": 0.25, "il6": 0.25, "m2_share": 0.30, "caf_share": 0.30},
        "note": "Primary plan. EPIC fractions over one shared basis, plus GSVA scores "
                "for hypoxia, TGF-beta, IL-6, M2 polarisation and CAF activation through "
                "the fitted monotone link.",
    },
}
DEFAULT_PLAN = "epic+gsva_m2"


def summarise(T):
    """Raw twin quantities, in physical units. Returns (B, len(FEATURES))."""
    c = G.counts(T.kind)
    dev = T.kind.device
    f = lambda k: c[k].to(dev).double()
    vessels = f("vessel")
    cells = (f("tumour") + f("M0") + f("M2") + f("fibroblast") + f("CAF")
             + f("CD8") + f("CD4") + f("B"))
    total = (cells + vessels).clamp(min=1)
    macs = (f("M0") + f("M2")).clamp(min=1)
    fibs = (f("fibroblast") + f("CAF")).clamp(min=1)
    rows = {
        "f_tumour":      f("tumour") / total,
        "f_macrophage":  (f("M0") + f("M2")) / total,
        "f_fibroblast":  (f("fibroblast") + f("CAF")) / total,
        "f_cd8":         f("CD8") / total,
        "f_cd4":         f("CD4") / total,
        "f_b":           f("B") / total,
        "f_endothelial": vessels / total,
        "hypoxia":       T.hypoxic_fraction().to(dev).double(),
        "tgfb":          T.tgfb.mean((1, 2)),
        "il6":           T.il6.mean((1, 2)),
        "m2_share":      f("M2") / macs,
        "caf_share":     f("CAF") / fibs,
    }
    return torch.stack([rows[k] for k in FEATURES], 1)


def share_floors(T):
    """Smallest non-zero value each share could take in each tissue (one cell),
    for the log link at training time. Returns {feature: (B,) tensor}."""
    c = G.counts(T.kind)
    dev = T.kind.device
    f = lambda k: c[k].to(dev).double()
    return {"m2_share": 1.0 / (f("M0") + f("M2")).clamp(min=1),
            "caf_share": 1.0 / (f("fibroblast") + f("CAF")).clamp(min=1),
            "hypoxia": 1.0 / (f("tumour") + f("M0") + f("M2") + f("fibroblast") + f("CAF")
                              + f("CD8") + f("CD4") + f("B")).clamp(min=1)}
