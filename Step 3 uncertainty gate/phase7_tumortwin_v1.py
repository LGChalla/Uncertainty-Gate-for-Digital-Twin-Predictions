"""
Phase 7 comparator: a TumorTwin-style tumour-only twin, given each patient's
inputs. B_TT per patient draw, on the same scale and read hour as B_i.

    python scripts/phase7_tumortwin_v1.py --npe runs/phase5_v3_fib120_npe_logithyp_endolinked \
        --out runs/phase7_tumortwin_v1

What TumorTwin is. TumorTwin (Kapteyn et al. 2025, arXiv:2505.00670; Oden
Institute, github.com/OncologyModelingGroup/TumorTwin) is a current framework
for patient-specific oncology twins. Its default model is a reaction-diffusion
equation for tumour cell density N:

    dN/dt = div(D grad N) + k N (1 - N/theta) - sum_i sum_j alpha_i C_i exp(-beta_i (t - tau_ij)) N

that is, invasion, logistic growth and a chemotherapy kill term, calibrated to
repeated MRI scans. It has no microenvironment: no vessels, oxygen, immune or
stromal cells.

How it is applied here (LC decision, Nov 6: its equations, our inputs). Our
cohorts have one RNA sample per patient and no serial imaging, so TumorTwin's
own calibration cannot run. Instead its equation is given the same per-patient
inputs our twin has:
  - k (growth): the patient's posterior tumour division rate r_div, minus the
    background death d0 (cohort parameter, sampled as in phase6).
  - N at the start: the patient's posterior f0_tumour, then 20 days of growth
    without drug, as our twin's burn-in.
  - theta = 1 (the patch's free space, as in our twin).
  - chemotherapy: the same cisplatin plasma curve (JBR.10 schedule, one cycle,
    tme_twin/drug.plasma_uM) and the same kill constant k_kill (sampled as in
    phase6), so alpha C(t) = k_kill * plasma(t).
  - diffusion: with a uniform start the D term is zero everywhere, so the
    equation is solved in its well-mixed form. Stated in the Methods.
So the two twins share growth, drug and kill; what differs is everything the
microenvironment adds (drug delivery from vessels, hypoxia, IL-6, CD8
killing, M2 suppression, CAF barrier). B_TT is computed for exactly the same
posterior draws and the same sampled cohort parameters as our B_i.

Writes <out>/btt_<cohort>.npz (B_TT per patient x draw, hour 192) and report.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.config_v3 import TME_V3                                 # noqa: E402
from tme_twin.config import values                                    # noqa: E402
from tme_twin.drug import plasma_uM                                   # noqa: E402
from phase6_patient_chemo_v1 import cohort_params                     # noqa: E402

DATASETS = ["GSE14814", "GSE68465", "GSE42127", "TCGA-LUAD"]
PRIMARY_HOUR = 192


def logistic_kill(N0, k, kill_rate_per_h, hours, dt_h):
    """dN/dt = k N (1 - N) - kill(t) N, forward in small steps (vectorised over draws).
    k: (n,) per hour. kill_rate_per_h: (n_steps,) or (n, n_steps) per hour, or None."""
    N = N0.copy()
    steps = int(round(hours / dt_h))
    for s in range(steps):
        g = k * N * (1.0 - N)
        if kill_rate_per_h is not None:
            kr = kill_rate_per_h[:, s] if kill_rate_per_h.ndim == 2 else kill_rate_per_h[s]
            # exact over the step for the kill term, explicit for growth
            N = (N + dt_h * g) * np.exp(-kr * dt_h)
        else:
            N = N + dt_h * g
        N = np.clip(N, 1e-12, 1.0)
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npe", required=True)
    ap.add_argument("--cohorts", nargs="*", default=DATASETS)
    ap.add_argument("--burn-in-days", type=int, default=20)
    ap.add_argument("--dt-min", type=float, default=2.0, help="step in minutes")
    ap.add_argument("--seed", type=int, default=9100, help="same as phase6 --seed, so cohort draws match")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep_npe = json.load(open(os.path.join(a.npe, "report.json")))
    regime = rep_npe["sim_meta"]["regime"]
    v = values(TME_V3)
    dt_h = a.dt_min / 60.0
    t = np.arange(int(round(PRIMARY_HOUR / dt_h))) * dt_h + 0.5 * dt_h        # midpoints
    plasma = plasma_uM(torch.tensor(t), v, cycles=1).numpy()                   # uM
    rep = {"model": "TumorTwin reaction term, well-mixed (logistic growth + chemo kill)",
           "source": "Kapteyn et al. 2025 arXiv:2505.00670", "primary_hour": PRIMARY_HOUR,
           "burn_in_days": a.burn_in_days, "dt_min": a.dt_min, "cohorts": {}}

    for ds in a.cohorts:
        f = os.path.join(a.npe, f"draws_{ds}.npz")
        if not os.path.exists(f):
            print(f"{ds}: no draws file, skipping"); continue
        d = np.load(f, allow_pickle=True)
        draws, samples, targets = d["draws"], d["samples"], list(d["targets"])
        npat, nd, _ = draws.shape
        p_idx, d_idx = np.repeat(np.arange(npat), nd), np.tile(np.arange(nd), npat)
        seeds = a.seed + p_idx * 1000 + d_idx
        cn, tc = cohort_params(len(seeds), seeds, regime)
        flat = draws.reshape(-1, len(targets))
        good = np.isfinite(flat).all(1)
        r_div = flat[:, targets.index("r_div")] * 3600.0
        d0 = tc[:, cn.index("d0")] * 3600.0
        k_kill = tc[:, cn.index("k_kill")]                                     # 1/(uM s)
        k = np.where(good, r_div - d0, 0.0)
        N0 = np.where(good, flat[:, targets.index("f0_tumour")], 0.01)
        N_start = logistic_kill(N0, k, None, 24 * a.burn_in_days, dt_h)
        kill = k_kill[:, None] * plasma[None, :] * 3600.0                      # per hour
        N_c = logistic_kill(N_start, k, kill, PRIMARY_HOUR, dt_h)
        N_n = logistic_kill(N_start, k, None, PRIMARY_HOUR, dt_h)
        btt = np.where(good, np.log10(N_n / N_c), np.nan).reshape(npat, nd)
        np.savez_compressed(os.path.join(a.out, f"btt_{ds}.npz"), btt=btt, samples=samples, hour=PRIMARY_HOUR)
        rep["cohorts"][ds] = {"n_patients": int(npat), "draws": int(nd),
                              "btt_median": float(np.nanmedian(btt)),
                              "between_patient_sd": float(np.nanstd(np.nanmean(btt, 1))),
                              "within_patient_sd": float(np.nanmean(np.nanstd(btt, 1)))}
        print(f"{ds}: B_TT median {np.nanmedian(btt):.3f}; between-patient SD "
              f"{rep['cohorts'][ds]['between_patient_sd']:.3f}, within {rep['cohorts'][ds]['within_patient_sd']:.3f}")
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"wrote {a.out}/report.json")


if __name__ == "__main__":
    main()
