"""
Turn each cohort's bulk expression into the features the twin reports
(tme_twin/observe.py FEATURES), by calling data_pipeline/deconvolve.R.

    python data_pipeline/deconvolve.py --renv ~/renv
    python data_pipeline/deconvolve.py --renv ~/renv --datasets GSE14814

Per dataset writes:
    data/processed/<DS>/features.csv     one row per sample, FEATURES columns
    data/processed/<DS>/deconv_raw.csv   EPIC fractions and ssGSEA scores as R produced them

Scale. EPIC needs linear, TPM-like values and will return plausible but wrong
fractions from log data without complaining. The four cohorts do not agree:
GSE68465 is linear while GSE14814 is log2 on the same Affymetrix platform, so
platform alone cannot be trusted. SCALE below records what each one is, and
check_scale() verifies the guess against the data before anything is converted.

m2_share and caf_share. EPIC's reference has one fibroblast-lineage
compartment and no M1/M2 split, so neither share is recoverable from it. Both
come instead from ssGSEA on short marker sets, rank-normalised to [0, 1]
within a cohort. That makes them relative to the cohort rather than absolute
fractions -- a real limitation, and the reason observe.py gives caf_share the
widest noise term of any target (0.25) and treats these as the weakest arm.
The Phase 5 sensitivity screen bears that out: caf_share was the least
informative observed feature.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.observe import FEATURES                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# dataset -> (platform for deconvolve.R, how the stored values are scaled)
SCALE = {
    "GSE14814":  ("affy_hgu133a", "log2"),
    "GSE68465":  ("affy_hgu133a", "linear"),
    "GSE42127":  ("illumina_v3",  "log2"),
    "TCGA-LUAD": ("ensembl",      "log2p1"),
}

# EPIC TRef column -> twin feature
EPIC_MAP = {
    "otherCells":  "f_tumour",
    "Macrophages": "f_macrophage",
    "CAFs":        "f_fibroblast",
    "CD8_Tcells":  "f_cd8",
    "Endothelial": "f_endothelial",
}
GSVA_MAP = {
    "HALLMARK_HYPOXIA":                 "hypoxia",
    "HALLMARK_TGF_BETA_SIGNALING":      "tgfb",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "il6",
    "M2_POLARISATION":                  "m2_share",
    "CAF_ACTIVATION":                   "caf_share",
}
SHARES = ("m2_share", "caf_share")      # rank-normalised to [0, 1] per cohort


def check_scale(x: pd.DataFrame, declared: str, ds: str):
    """Verify the declared scale against the data before converting."""
    hi = float(np.nanmax(x.values))
    lo = float(np.nanmin(x.values))
    looks_log = hi < 100
    if declared == "linear" and looks_log:
        raise SystemExit(f"{ds}: declared linear but max is {hi:.1f} -- looks log-scaled")
    if declared in ("log2", "log2p1") and not looks_log:
        raise SystemExit(f"{ds}: declared {declared} but max is {hi:.1f} -- looks linear")
    if declared == "log2p1" and lo < -1e-9:
        raise SystemExit(f"{ds}: declared log2(TPM+1) but min is {lo:.3f} -- expected >= 0")
    return hi, lo


def to_linear(x: pd.DataFrame, declared: str) -> pd.DataFrame:
    if declared == "linear":
        return x
    if declared == "log2":
        return np.power(2.0, x)
    if declared == "log2p1":
        return np.power(2.0, x) - 1.0
    raise ValueError(declared)


def rank_to_unit(s: pd.Series) -> pd.Series:
    """Rank-normalise to (0, 1) within the cohort. Ties share the mean rank."""
    r = s.rank(method="average")
    return (r - 0.5) / len(r)


def run_one(ds: str, renv: str, force: bool = False):
    platform, declared = SCALE[ds]
    raw = os.path.join(ROOT, "data", "raw", ds, "expression.parquet")
    outdir = os.path.join(ROOT, "data", "processed", ds)
    os.makedirs(outdir, exist_ok=True)
    feat_path = os.path.join(outdir, "features.csv")
    if os.path.exists(feat_path) and not force:
        print(f"{ds}: features.csv exists, skipping (use --force to redo)")
        return

    print(f"\n=== {ds} ({platform}, stored as {declared}) ===", flush=True)
    x = pd.read_parquet(raw)
    hi, lo = check_scale(x, declared, ds)
    print(f"  {x.shape[0]} features x {x.shape[1]} samples; stored range {lo:.2f} to {hi:.2f}")
    x = to_linear(x, declared)
    print(f"  converted to linear; range {float(np.nanmin(x.values)):.2f} "
          f"to {float(np.nanmax(x.values)):.1f}")

    with tempfile.TemporaryDirectory() as tmp:
        csv_in = os.path.join(tmp, "expr.csv")
        csv_out = os.path.join(outdir, "deconv_raw.csv")
        x.to_csv(csv_in)
        cmd = [os.path.expanduser(os.path.join(renv, "bin", "Rscript")),
               os.path.join(HERE, "deconvolve.R"), csv_in, platform, csv_out]
        print(f"  calling R: {' '.join(cmd[-3:])}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write("".join(f"  R| {l}\n" for l in r.stderr.strip().splitlines()))
        if r.returncode != 0:
            raise SystemExit(f"{ds}: R failed with code {r.returncode}")

    d = pd.read_csv(csv_out).set_index("sample")
    missing = [c for c in list(EPIC_MAP) + list(GSVA_MAP) if c not in d.columns]
    if missing:
        raise SystemExit(f"{ds}: R output missing columns {missing}; got {list(d.columns)}")

    out = pd.DataFrame(index=d.index)
    for src, dst in {**EPIC_MAP, **GSVA_MAP}.items():
        out[dst] = d[src]
    for c in SHARES:
        out[c] = rank_to_unit(out[c])
    out = out[list(FEATURES)]
    out.index.name = "sample"
    out.to_csv(feat_path)
    print(f"  wrote {feat_path}")
    with pd.option_context("display.width", 140):
        print(out.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renv", default="~/renv", help="path to the micromamba R environment")
    ap.add_argument("--datasets", nargs="*", default=list(SCALE))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    renv = os.path.expanduser(a.renv)
    rscript = os.path.join(renv, "bin", "Rscript")
    if not os.path.exists(rscript):
        raise SystemExit(f"Rscript not found at {rscript} -- check --renv")
    for ds in a.datasets:
        if ds not in SCALE:
            raise SystemExit(f"unknown dataset {ds}; known: {list(SCALE)}")
        run_one(ds, renv, a.force)


if __name__ == "__main__":
    main()
