"""
Patient features for the v2 observation operator, built from the EPIC and
ssGSEA output that deconvolve.R already wrote. deconvolve.py and
qc_features.py are unchanged; no R step is re-run.

    python data_pipeline/features_v2.py
    python data_pipeline/features_v2.py --datasets GSE14814

Reads  data/processed/<DS>/deconv_raw.csv     EPIC (all eight TRef columns) + ssGSEA
Writes data/processed/<DS>/features_v2_qc.csv one row per kept sample, observe_v2.FEATURES
       data/processed/<DS>/qc_report_v2.json   what was dropped and why, NK mass

What differs from features_qc.csv (v1):
  - CD4_Tcells and Bcells are kept, as f_cd4 and f_b.
  - The seven kept EPIC compartments are divided by 1 - NKcells, so they sum
    to 1, matching the twin, which has no NK agent. NK is reported, not
    modelled: its median is 0.0000 in three cohorts and 0.0002 in GSE42127.
  - m2_share and caf_share are the raw ssGSEA scores. No rank normalisation:
    both go through the fitted monotone link at training time, which absorbs
    the scale. v1 rank-normalised them, and compared caf_share directly.

QC is the same as qc_features.py: TCGA primary tumours only (barcode type
01), and fits with any fraction >= 0.999 or tumour <= 0.001 are dropped.
Nothing here reads treatment or survival.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from tme_twin.observe_v2 import FEATURES, FRACTIONS                  # noqa: E402

DATASETS = ["GSE14814", "GSE68465", "GSE42127", "TCGA-LUAD"]
EPIC_MAP = {
    "otherCells":  "f_tumour",
    "Macrophages": "f_macrophage",
    "CAFs":        "f_fibroblast",
    "CD8_Tcells":  "f_cd8",
    "CD4_Tcells":  "f_cd4",
    "Bcells":      "f_b",
    "Endothelial": "f_endothelial",
}
GSVA_MAP = {
    "HALLMARK_HYPOXIA":                 "hypoxia",
    "HALLMARK_TGF_BETA_SIGNALING":      "tgfb",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "il6",
    "M2_POLARISATION":                  "m2_share",
    "CAF_ACTIVATION":                   "caf_share",
}
KEEP_TCGA_TYPES = {"01"}


def run_one(ds):
    src = os.path.join(ROOT, "data", "processed", ds, "deconv_raw.csv")
    if not os.path.exists(src):
        print(f"{ds}: no deconv_raw.csv, skipping")
        return None
    d = pd.read_csv(src).set_index("sample")
    need = list(EPIC_MAP) + list(GSVA_MAP) + ["NKcells"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise SystemExit(f"{ds}: deconv_raw.csv missing {missing}")
    n0 = len(d)
    rep = {"dataset": ds, "n_input": n0, "dropped": {}}

    if ds.startswith("TCGA"):
        code = pd.Series(d.index.str[13:15], index=d.index)
        bad = ~code.isin(KEEP_TCGA_TYPES)
        rep["sample_type_counts"] = code.value_counts().to_dict()
        rep["dropped"]["not_primary_tumour"] = int(bad.sum())
        d = d[~bad]

    nk = d["NKcells"].astype(float)
    kept = d[list(EPIC_MAP)].astype(float)
    denom = (1.0 - nk).clip(lower=1e-6)
    out = pd.DataFrame(index=d.index)
    for s, t in EPIC_MAP.items():
        out[t] = kept[s] / denom
    for s, t in GSVA_MAP.items():
        out[t] = d[s].astype(float)
    out = out[list(FEATURES)]

    fr = out[list(FRACTIONS)]
    degenerate = (fr.max(axis=1) >= 0.999) | (fr["f_tumour"] <= 0.001)
    rep["dropped"]["degenerate_fit"] = int(degenerate.sum())
    if degenerate.any():
        rep["degenerate_samples"] = list(out.index[degenerate])
    out = out[~degenerate]
    nk = nk[~degenerate]

    s7 = out[list(FRACTIONS)].sum(axis=1)
    rep["fraction_sum_after_nk_rescale"] = {"min": float(s7.min()), "max": float(s7.max())}
    rep["nk_fraction"] = {"median": float(nk.median()), "p99": float(nk.quantile(0.99)),
                          "max": float(nk.max())}
    rep["n_kept"] = len(out)
    out.index.name = "sample"
    out.to_csv(os.path.join(ROOT, "data", "processed", ds, "features_v2_qc.csv"))
    with open(os.path.join(ROOT, "data", "processed", ds, "qc_report_v2.json"), "w") as f:
        json.dump(rep, f, indent=2)
    drops = ", ".join(f"{k} {v}" for k, v in rep["dropped"].items() if v)
    print(f"{ds:>10}: {n0} -> {len(out)}" + (f"   dropped: {drops}" if drops else "")
          + f"   NK median {rep['nk_fraction']['median']:.4f} max {rep['nk_fraction']['max']:.4f}"
          + f"   fractions sum {s7.min():.6f}-{s7.max():.6f}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    a = ap.parse_args()
    reps = [r for r in (run_one(ds) for ds in a.datasets) if r]
    print(f"\ntotal kept: {sum(r['n_kept'] for r in reps)} of {sum(r['n_input'] for r in reps)}")


if __name__ == "__main__":
    main()
