"""
Quality control on the deconvolved features, between deconvolve.py and
calibration.

    python data_pipeline/qc_features.py
    python data_pipeline/qc_features.py --datasets TCGA-LUAD

Reads  data/processed/<DS>/features.csv
Writes data/processed/<DS>/features_qc.csv   the rows that survive
       data/processed/<DS>/qc_report.json    what was dropped and why

Two things are removed.

Sample type. TCGA barcodes carry a sample-type code in characters 14-15:
01 primary solid tumour, 02 recurrent, 11 solid tissue normal. The LUAD
download contains 59 normals and 2 recurrences alongside 528 primaries, and
the normals are a different tissue -- EPIC returns degenerate fits on them
(f_fibroblast reaching 1.000, f_macrophage 0.509) because the TRef reference
has no profile for healthy lung. Only primary tumours are kept, matching what
the twin models and what the GEO cohorts already contain.

Impossible fractions. A sample whose EPIC fractions are all but zero for
tumour, or where any single fraction reaches 1.0, is a failed fit rather than
a measurement. These are reported separately from the sample-type drops so the
two reasons stay distinguishable in the flow table.

Nothing here looks at treatment or survival.
"""

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASETS = ["GSE14814", "GSE68465", "GSE42127", "TCGA-LUAD"]

KEEP_TCGA_TYPES = {"01"}        # primary solid tumour only
FRACTIONS = ["f_tumour", "f_macrophage", "f_fibroblast", "f_cd8", "f_endothelial"]


def tcga_sample_type(index: pd.Index) -> pd.Series:
    return pd.Series(index.str[13:15], index=index)


def run_one(ds: str):
    path = os.path.join(ROOT, "data", "processed", ds, "features.csv")
    if not os.path.exists(path):
        print(f"{ds}: no features.csv, skipping")
        return None
    d = pd.read_csv(path, index_col=0)
    n0 = len(d)
    rep = {"dataset": ds, "n_input": n0, "dropped": {}}

    if ds.startswith("TCGA"):
        code = tcga_sample_type(d.index)
        counts = code.value_counts().to_dict()
        bad = ~code.isin(KEEP_TCGA_TYPES)
        rep["sample_type_counts"] = counts
        rep["dropped"]["not_primary_tumour"] = int(bad.sum())
        d = d[~bad]

    fr = d[FRACTIONS]
    degenerate = (fr.max(axis=1) >= 0.999) | (fr["f_tumour"] <= 0.001)
    rep["dropped"]["degenerate_fit"] = int(degenerate.sum())
    if degenerate.any():
        rep["degenerate_samples"] = list(d.index[degenerate])
    d = d[~degenerate]

    rep["n_kept"] = len(d)
    out = os.path.join(ROOT, "data", "processed", ds, "features_qc.csv")
    d.to_csv(out)
    with open(os.path.join(ROOT, "data", "processed", ds, "qc_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    drops = ", ".join(f"{k} {v}" for k, v in rep["dropped"].items() if v)
    print(f"{ds:>10}: {n0} -> {len(d)}" + (f"   dropped: {drops}" if drops else "   nothing dropped"))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    a = ap.parse_args()
    reps = [r for r in (run_one(ds) for ds in a.datasets) if r]
    print(f"\ntotal kept: {sum(r['n_kept'] for r in reps)} of {sum(r['n_input'] for r in reps)}")


if __name__ == "__main__":
    main()
