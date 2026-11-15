"""
Phase 4 direction checks on the v2 model (tissue_v2: population equations,
CD4 and B cells). phase4_directions.py is unchanged and still describes the
v1 checks reported in the paper.

    CUDA_VISIBLE_DEVICES=5 python scripts/phase4_directions_v2.py --device cuda --param vessel_density
    python scripts/phase4_directions_v2.py --size 40 --B 2 --levels 2 --burn-in-days 2 --param r_mac   # CPU smoke test

Change one parameter at a time across its range, holding every other at its
config_v2 working value; build the tissue at its analytic steady state; burn in
for a fixed 20 days (as the Phase 5 simulations do); run paired chemo /
no-chemo to day 9; read B_i.

What changed from the v1 checks
-------------------------------
- Macrophage burden is varied through recruitment (r_mac), not a starting
  fraction, so it no longer trades space against tumour by construction.
- The verdict comes from a weighted slope of B_i on the level (log level for
  log-scaled parameters) with its standard error, not from the sign of an
  unweighted slope. Three outcomes: as expected (|z| >= 2, right sign),
  reversed (|z| >= 2, wrong sign), no detectable effect (|z| < 2). Runs too
  small to mean anything (B < 16, or grid < 100) are labelled underpowered
  whatever they show, so a smoke test cannot read as a result.
- Parameters with no expected sign are run as exploratory and reported with
  their slope and z only.
- Burn-in convergence is reported per tissue.

Writes runs/phase4_v2/<param>/report.json and <param>.png.
"""

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.config_v2 import TME_V2                               # noqa: E402
from tme_twin.config import config_hash                              # noqa: E402
from tme_twin.tissue_v2 import TissueV2                              # noqa: E402
from tme_twin.response_v2 import burn_in, paired_response, occupancy, PRIMARY_HOUR  # noqa: E402
from tme_twin.runlog import RunLog, set_deterministic                # noqa: E402

torch.set_default_dtype(torch.float64)
DAY = 86400.0
F0_TUMOUR = {"residual": 0.02, "primary": 0.35}

# name -> (what it is, expected sign of dB_i/dlevel or 0 if exploratory, levels, log-scaled?)
PARAMS = {
    "vessel_density": ("vessels per mm^2 (oxygen, drug, recruitment)", +1, [10, 20, 30, 45, 60], False),
    "Vmax_o2": ("oxygen use per cell (more = more hypoxia)", -1, [3.5, 5.0, 7.0, 10.0, 14.0], True),
    "r_mac": ("macrophage recruitment per vessel (more M2: IL-6 protection, CD8 suppression)", -1,
              [0.003 / DAY, 0.01 / DAY, 0.03 / DAY, 0.1 / DAY, 0.3 / DAY], True),
    "k_caf": ("fibroblast->CAF rate per nM TGF-beta; v1 bound predicts no detectable effect", -1,
              [0.02 / 3600, 0.05 / 3600, 0.10 / 3600, 0.18 / 3600, 0.30 / 3600], True),
    "r_cd8": ("CD8 recruitment per vessel; exploratory: killing acts in both arms", 0,
              [0.003 / DAY, 0.01 / DAY, 0.03 / DAY, 0.1 / DAY, 0.3 / DAY], True),
    "gamma_cd4": ("net CD4 effect on CD8 killing (<0 Treg, >0 help); exploratory", 0,
                  [-5.0, -2.5, 0.0, 2.5, 5.0], False),
    "K_fib": ("fibroblast set point; exploratory: also takes space from tumour", 0,
              [0.02, 0.05, 0.10, 0.20, 0.40], True),
}


def weighted_slope(x, m, se):
    """Slope of m on x weighted by 1/se^2, with its standard error."""
    w = [1.0 / max(s, 1e-12) ** 2 for s in se]
    W = sum(w)
    xb = sum(wi * xi for wi, xi in zip(w, x)) / W
    yb = sum(wi * yi for wi, yi in zip(w, m)) / W
    sxx = sum(wi * (xi - xb) ** 2 for wi, xi in zip(w, x))
    if sxx <= 0:
        return 0.0, float("inf")
    b = sum(wi * (xi - xb) * (yi - yb) for wi, xi, yi in zip(w, x, m)) / sxx
    return b, math.sqrt(1.0 / sxx)


def figure(name, rows, out, logx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = [r["level"] for r in rows]
    m = [r["B_i_day9"]["mean"] for r in rows]
    s = [r["B_i_day9"]["sem"] for r in rows]
    ax[0].errorbar(x, m, yerr=s, fmt="o-", capsize=3)
    if logx:
        ax[0].set_xscale("log")
    ax[0].set(xlabel=PARAMS[name][0][:60], ylabel="B_i at day 9 (mean +/- SEM)", title=f"B_i vs {name}")
    ax[1].plot(x, [r["at_treatment_start"]["occupancy"] for r in rows], "o-", label="occupancy")
    ax[1].plot(x, [r["at_treatment_start"]["f_cd8"] for r in rows], "s-", label="CD8 / cells")
    ax[1].plot(x, [r["at_treatment_start"]["m2_per_cell"] for r in rows], "^-", label="M2 / cells")
    if logx:
        ax[1].set_xscale("log")
    ax[1].set(xlabel=name, title="Tissue at treatment start")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--param", required=True, choices=list(PARAMS))
    ap.add_argument("--B", type=int, default=32, help="tissues (random layouts) per level")
    ap.add_argument("--size", type=int, default=100, help="grid side; Phase 5 uses 100")
    ap.add_argument("--levels", type=int, default=5, help="use the first N levels (smoke tests)")
    ap.add_argument("--burn-in-days", type=int, default=20)
    ap.add_argument("--regime", default="residual", choices=["primary", "residual"])
    ap.add_argument("--out", default="runs/phase4_v2")
    a = ap.parse_args()
    out = os.path.join(a.out, a.param)
    os.makedirs(out, exist_ok=True)
    set_deterministic()
    desc, sign, levels, logx = PARAMS[a.param]
    levels = levels[:a.levels]
    underpowered = a.B < 16 or a.size < 100 or len(levels) < 3
    t0 = time.time()
    rows = []
    for i, level in enumerate(levels):
        ov = {"nx": a.size, "ny": a.size, a.param: level}
        T = TissueV2(a.B, F0_TUMOUR[a.regime], seed=4000 + i, device=a.device, overrides=ov)
        tb = time.time()
        bi = burn_in(T, fixed_days=a.burn_in_days)
        c = T.counts()
        cells = sum(c[k].double() for k in ("tumour", "M0", "M2", "fibroblast", "CAF", "CD8", "CD4", "B"))
        start = {"tumour": float(c["tumour"].double().mean()),
                 "occupancy": float(occupancy(T).mean()),
                 "hypoxic_fraction": float(T.hypoxic_fraction().mean()),
                 "f_cd8": float((c["CD8"].double() / cells.clamp(min=1)).mean()),
                 "m2_per_cell": float((c["M2"].double() / cells.clamp(min=1)).mean()),
                 "caf_share": float((c["CAF"].double() / (c["fibroblast"] + c["CAF"]).double().clamp(min=1)).mean()),
                 "crowded_init": int(T.crowded_init.sum())}
        resp = paired_response(T)
        b = torch.tensor(resp[PRIMARY_HOUR]["B_i"])
        sem = float(b.std() / math.sqrt(len(b))) if len(b) > 1 else float("inf")
        row = {"level": level, "burn_in": {k: bi[k] for k in ("days", "worst_daily_change", "binding",
                                                               "occupancy", "converged")},
               "converged_tissues": int(sum(bi["converged_per_tissue"])),
               "B_i_day9": {"mean": float(b.mean()), "sem": sem, "per_tissue": b.tolist()},
               "B_i_day1": {"mean": float(torch.tensor(resp[24]["B_i"]).mean())},
               "at_treatment_start": start, "seconds": round(time.time() - tb, 1)}
        rows.append(row)
        print(f"{a.param}={level:.4g}: B_i day9 {row['B_i_day9']['mean']:.4f} +/- {sem:.4f} (SEM), "
              f"converged {row['converged_tissues']}/{a.B}, start {start}, {row['seconds']} s", flush=True)
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump({"param": a.param, "partial": True, "rows": rows}, f, indent=2)

    x = [math.log(r["level"]) if logx else float(r["level"]) for r in rows]
    m = [r["B_i_day9"]["mean"] for r in rows]
    se = [r["B_i_day9"]["sem"] for r in rows]
    slope, slope_se = weighted_slope(x, m, se) if len(rows) > 1 else (0.0, float("inf"))
    z = slope / slope_se if slope_se > 0 and math.isfinite(slope_se) else 0.0
    if underpowered:
        verdict = "underpowered (smoke test): no conclusion"
    elif sign == 0:
        verdict = f"exploratory: slope z = {z:+.2f}"
    elif abs(z) < 2:
        verdict = "no detectable effect"
    elif z * sign > 0:
        verdict = "as expected"
    else:
        verdict = "reversed"
    span = max(m) - min(m) if m else 0.0
    report = {"param": a.param, "description": desc, "expected_sign": sign, "levels": levels,
              "x_scale": "log" if logx else "linear",
              "slope": slope, "slope_se": slope_se, "z": z, "verdict": verdict,
              "B_i_range_across_levels": span, "underpowered": underpowered,
              "model": "tissue_v2", "config": "config_v2", "regime": a.regime,
              "device": a.device, "B": a.B, "grid": a.size, "burn_in_days": a.burn_in_days,
              "total_seconds": round(time.time() - t0, 1), "rows": rows}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    figure(a.param, rows, os.path.join(out, f"{a.param}.png"), logx)
    RunLog("runs/runlog.jsonl").append(
        "phase4_direction_v2", config_hash=config_hash(TME_V2), seed="4000+",
        inputs={"param": a.param, "levels": levels, "B": a.B, "grid": a.size},
        outputs={"verdict": verdict, "z": z, "B_i_day9": m})
    print(f"\n{a.param}: expected sign {sign:+d}; slope {slope:+.4g} +/- {slope_se:.2g} "
          f"(z = {z:+.2f}); VERDICT: {verdict}; {report['total_seconds']} s; wrote {out}/report.json")


if __name__ == "__main__":
    main()
