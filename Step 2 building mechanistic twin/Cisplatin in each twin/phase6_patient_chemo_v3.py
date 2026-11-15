"""
Phase 6, version 3: as phase6_patient_chemo_v2.py (unchanged), plus
--shared-world. With it, draw d of EVERY patient uses the same values of the 24
cohort-shared and latent parameters (drug potency, hypoxia protection, cell
turnover...), sampled once per draw index d. Each draw is then one consistent
"world", and patients are compared like-for-like inside it.

Why (Nov 8, sensitivity ranking): k_kill and K_o_kill, which are shared by all
patients and known only as ranges, explain ~46% of B_i variance. Sampling them
separately for each patient buries patient differences under drug-potency
noise. With shared worlds that noise moves every patient together within a
draw, so the ranking of patients is what the draws vary over. The Cox test
fits once per world and pools by Rubin's rules (pre-registration v0.5).
Chosen from the outcome-free sensitivity analysis, before the freeze.

    CUDA_VISIBLE_DEVICES=5 python scripts/phase6_patient_chemo_v3.py --device cuda --shared-world \
        --npe <dir> --cohort GSE14814 --draws-per-patient 10 --hours 192 --every 24 \
        --out runs/phase6_chemo_v3_GSE14814_1cycle_10draws_sharedworld

--- v2 docstring ---
Phase 6, version 2: as phase6_patient_chemo_v1.py (unchanged), plus --cycles,
so the twin can receive the full neoadjuvant course (3 cycles = 63 days) and
its residual tumour can be compared with real pathological response:

    viable fraction = tumour cells in the chemo arm at the end / at chemo start
    twin "MPR"  : viable fraction <= 0.10   (major pathological response)
    twin "pCR"  : no tumour cells left       (pathological complete response)

Real reference, chemotherapy-alone arm of CheckMate 816 (3 cycles of platinum
doublet before surgery): pCR 2.2%, MPR 12.7%. The comparison is computed
afterwards from trajectories.npz by phase6_absolute_response_v1.py.

    CUDA_VISIBLE_DEVICES=3 python scripts/phase6_patient_chemo_v2.py --device cuda --cycles 3 \
        --hours 1512 --every 168 --npe <dir> --cohort GSE14814 --draws-per-patient 5 \
        --out runs/phase6_chemo_v2_GSE14814_3cycles_5draws

--- v1 docstring ---
Phase 6: give each patient's fitted twin chemotherapy, and record what happens.

    CUDA_VISIBLE_DEVICES=7 python scripts/phase6_patient_chemo_v1.py --device cuda \
        --npe runs/phase5_v3_fib120_npe_logithyp_endolinked --cohort GSE14814 \
        --draws-per-patient 5 --grid-patients <id1> <id2> \
        --out runs/phase6_chemo_v1_GSE14814_5draws
    python scripts/phase6_patient_chemo_v1.py --npe <dir> --cohort GSE14814 --size 30 \
        --burn-in-days 1 --hours 48 --draws-per-patient 1 --max-patients 2 --out /tmp/p6_smoke   # CPU smoke test

What it does, per patient
-------------------------
1. Takes the first --draws-per-patient of the patient's posterior draws over
   the 12 patient parameters (phase5_train_npe_v4, draws_<cohort>.npz).
2. The 24 cohort-shared parameters are not known per patient; for each draw
   they are sampled from their prior (the same prior the campaign used), so
   their uncertainty is carried into the patient's result. Seeded per
   (patient, draw), so a rerun gives the same tissue.
3. Builds the tissue at its steady state (tissue_v3), burns in for 20 days
   without drug, exactly as the campaign did.
4. Copies it into a chemo arm and a no-chemo arm that share the same random
   numbers (as response.paired_response), runs both for --hours, and records
   every --every hours, in both arms:
     tumour, M0, M2, fibroblast, CAF, CD8, CD4, B counts; mean oxygen;
     hypoxic share of cells; mean IL-6 and TGF-beta; CD8 cells touching a
     tumour cell ("CD8 reach"); M2 share; CAF share.
   B_i = log10(tumour no chemo / tumour chemo) at every recorded hour.
5. For the patients named in --grid-patients, saves the full tissue grid
   (cell kinds) and the oxygen field in both arms at --grid-hours, for the
   figures.

Output: <out>/trajectories.npz, one row per (patient, draw), and grids.npz.
Checkpoints every --checkpoint-every batches and resumes when rerun with the
same arguments, like phase5_simulate_v2/v3.

Nothing here reads treatment or survival.
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.tissue_v3 import TissueV3                               # noqa: E402
from tme_twin.config_v3 import TME_V3                                 # noqa: E402
from tme_twin.config import config_hash                               # noqa: E402
from tme_twin.response_v2 import burn_in                              # noqa: E402
from tme_twin import grid_v2 as G                                     # noqa: E402
from tme_twin import cells as C                                       # noqa: E402
from tme_twin import priors_v4 as P                                   # noqa: E402

torch.set_default_dtype(torch.float64)
KINDS = ("tumour", "M0", "M2", "fibroblast", "CAF", "CD8", "CD4", "B")
FIELDS = ("o2_mean", "hypoxic_share", "il6_mean", "tgfb_mean", "cd8_reach", "m2_share", "caf_share")
REC = KINDS + FIELDS


def snapshot(T):
    """(B, len(REC)) record of one tissue state."""
    c = G.counts(T.kind)
    dev = T.kind.device
    f = lambda k: c[k].to(dev).double()
    tum = T.kind == G.TUMOR
    cd8 = T.kind == G.CD8
    reach = (cd8 & (C.adjacent_count(tum) > 0)).sum((1, 2)).double()
    vals = [f(k) for k in KINDS] + [
        T.o2.mean((1, 2)), T.hypoxic_fraction().to(dev).double(),
        T.il6.mean((1, 2)), T.tgfb.mean((1, 2)), reach,
        f("M2") / (f("M0") + f("M2")).clamp(min=1),
        f("CAF") / (f("fibroblast") + f("CAF")).clamp(min=1)]
    return torch.stack(vals, 1).cpu().numpy()


def cohort_params(n, seeds, regime):
    """Cohort-class parameters from their prior, one row per tissue, seeded per tissue."""
    pri = P.regime_priors(regime)
    names = [k for k, v in P.PRIORS.items() if v[3] != "patient"]
    out = np.empty((n, len(names)))
    for i, s in enumerate(seeds):
        r = np.random.default_rng(int(s))
        for j, nm in enumerate(names):
            lo, hi, scale, _, _ = pri[nm]
            out[i, j] = (math.exp(r.uniform(math.log(lo), math.log(hi))) if scale == "log"
                         else r.uniform(lo, hi))
    return names, out


def run_block(theta_pat, targets, cohort_names, theta_coh, a, seed, keep_grid):
    B = len(theta_pat)
    ov = {n: torch.tensor(theta_pat[:, j]) for j, n in enumerate(targets) if n != "f0_tumour"}
    ov.update({n: torch.tensor(theta_coh[:, j]) for j, n in enumerate(cohort_names)})
    ov.update({"nx": a.size, "ny": a.size})
    f0 = torch.tensor(theta_pat[:, targets.index("f0_tumour")])
    T = TissueV3(B, f0, seed=seed, device=a.device, overrides=ov)
    bi = burn_in(T, fixed_days=a.burn_in_days)
    chemo, ctrl = copy.deepcopy(T), copy.deepcopy(T)
    chemo.start_chemo(a.cycles)
    ctrl.start_chemo(0)
    hours = list(range(0, a.hours + 1, a.every))
    rc, rn = [snapshot(chemo)], [snapshot(ctrl)]
    gk = {"chemo": [], "no_chemo": []}
    go = {"chemo": [], "no_chemo": []}
    def grab():
        for arm, X in (("chemo", chemo), ("no_chemo", ctrl)):
            gk[arm].append(X.kind[keep_grid].cpu().numpy().astype(np.int8))
            go[arm].append(X.o2[keep_grid].cpu().numpy().astype(np.float32))
    if keep_grid.any() and 0 in a.grid_hours:
        grab()
    for h in range(1, a.hours + 1):
        chemo.step_hour()
        ctrl.step_hour()
        if h % a.every == 0:
            rc.append(snapshot(chemo)); rn.append(snapshot(ctrl))
        if keep_grid.any() and h in a.grid_hours:
            grab()
    rc, rn = np.stack(rc, 1), np.stack(rn, 1)                     # (B, n_hours, len(REC))
    ti = REC.index("tumour")
    b = np.log10(rn[:, :, ti] / np.clip(rc[:, :, ti], 1, None))
    grids = None
    if keep_grid.any():
        grids = {arm: (np.stack(gk[arm], 1), np.stack(go[arm], 1)) for arm in gk}
    conv = np.array(bi["converged_per_tissue"], dtype=float)
    return hours, rc, rn, b, conv, grids


def save(path, parts, meta):
    tmp = path + ".tmp.npz"
    arr = {k: np.concatenate(v) if v else np.empty((0,)) for k, v in parts.items()}
    np.savez_compressed(tmp, **arr, meta=json.dumps(meta))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npe", required=True, help="phase5_train_npe_v4 output folder")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--draws-per-patient", type=int, default=5)
    ap.add_argument("--max-patients", type=int, default=0, help="0 = all; for tests")
    ap.add_argument("--patient-start", type=int, default=0, help="shard: first patient index")
    ap.add_argument("--patient-end", type=int, default=-1, help="shard: one past last patient index")
    ap.add_argument("--grid-patients", nargs="*", default=[])
    ap.add_argument("--grid-hours", nargs="*", type=int, default=[0, 48, 96, 192])
    ap.add_argument("--hours", type=int, default=192)
    ap.add_argument("--shared-world", action="store_true", help="same cohort/latent parameters for every patient within a draw")
    ap.add_argument("--cycles", type=int, default=1, help="cisplatin cycles (each: days 1 and 8 of 21... as config)")
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--burn-in-days", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rep = json.load(open(os.path.join(a.npe, "report.json")))
    if rep.get("priors_module") != "priors_v4":
        raise SystemExit(f"{a.npe} was not trained on priors_v4 simulations")
    regime = rep["sim_meta"]["regime"]
    z = np.load(os.path.join(a.npe, f"draws_{a.cohort}.npz"), allow_pickle=True)
    draws, samples, targets = z["draws"], list(z["samples"]), list(z["targets"])
    end = len(samples) if a.patient_end < 0 else min(a.patient_end, len(samples))
    pats = list(range(a.patient_start, end))
    if a.max_patients:
        pats = pats[:a.max_patients]
    nd = min(a.draws_per_patient, draws.shape[1])
    rows = [(p, d) for p in pats for d in range(nd) if np.isfinite(draws[p, d]).all()]
    skipped = len(pats) * nd - len(rows)

    meta = {"script": "phase6_patient_chemo_v3", "cycles": a.cycles, "shared_world": a.shared_world, "model": "tissue_v3", "priors_module": "priors_v4",
            "npe": a.npe, "cohort": a.cohort, "regime": regime, "size": a.size,
            "burn_in_days": a.burn_in_days, "hours": a.hours, "every": a.every,
            "draws_per_patient": nd, "patient_range": [a.patient_start, end], "seed": a.seed,
            "record": list(REC), "targets": targets, "grid_patients": list(a.grid_patients),
            "grid_hours": list(a.grid_hours), "config_hash": config_hash(TME_V3)}

    part_path = os.path.join(a.out, "partial.npz")
    keys = ("patient", "draw", "theta", "theta_cohort", "chemo", "no_chemo", "B_i", "burnin_converged")
    parts = {k: [] for k in keys}
    gparts = {k: [] for k in ("g_patient", "g_draw", "g_kind_chemo", "g_kind_no_chemo",
                              "g_o2_chemo", "g_o2_no_chemo")}
    done = set()
    if os.path.exists(part_path):
        zz = np.load(part_path, allow_pickle=True)
        old = json.loads(str(zz["meta"]))
        for k in ("cohort", "size", "burn_in_days", "hours", "every", "draws_per_patient", "cycles", "shared_world",
                  "patient_range", "seed", "config_hash", "npe"):
            if old[k] != meta[k]:
                raise SystemExit(f"partial.npz was made with {k}={old[k]}, now {meta[k]}: use a new --out")
        for k in keys:
            parts[k].append(zz[k])
        done = set(zip(zz["patient"].tolist(), zz["draw"].tolist()))
        gp = os.path.join(a.out, "grids_partial.npz")
        if os.path.exists(gp):
            gz = np.load(gp, allow_pickle=True)
            for k in gparts:
                gparts[k].append(gz[k])
        print(f"resuming: {len(done)} tissues already done", flush=True)

    todo = [r for r in rows if r not in done]
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"{a.cohort}: patients {a.patient_start}..{end} ({len(pats)}), {nd} draws each -> "
          f"{len(rows)} tissues ({skipped} draws missing), {len(todo)} to do in {len(batches)} batches; "
          f"grid {a.size}, burn-in {a.burn_in_days} d, {a.hours} h after chemo starts", flush=True)

    t0, n_done = time.time(), 0
    for bi_, blk in enumerate(batches):
        p_idx = np.array([p for p, _ in blk]); d_idx = np.array([d for _, d in blk])
        th = draws[p_idx, d_idx]
        seeds = a.seed + p_idx * 1000 + d_idx                      # tissue randomness: per patient and draw
        world = (a.seed + 777 + d_idx) if a.shared_world else seeds  # shared/latent parameters
        cn, tc = cohort_params(len(blk), world, regime)
        keep = torch.tensor([samples[p] in a.grid_patients for p in p_idx], device=a.device)
        hours, rc, rn, b, conv, grids = run_block(th, targets, cn, tc, a, int(seeds[0]), keep)
        for k, v in zip(keys, (p_idx, d_idx, th, tc, rc, rn, b, conv)):
            parts[k].append(v)
        if grids is not None:
            sel = keep.cpu().numpy()
            gparts["g_patient"].append(p_idx[sel]); gparts["g_draw"].append(d_idx[sel])
            gparts["g_kind_chemo"].append(grids["chemo"][0]); gparts["g_kind_no_chemo"].append(grids["no_chemo"][0])
            gparts["g_o2_chemo"].append(grids["chemo"][1]); gparts["g_o2_no_chemo"].append(grids["no_chemo"][1])
        n_done += len(blk)
        el = time.time() - t0
        print(f"  {len(done) + n_done}/{len(rows)}  {el / n_done:.2f} s/tissue  "
              f"eta {(len(todo) - n_done) * el / n_done / 60:.0f} min  "
              f"B_i at {hours[-1]} h: median {np.median(b[:, -1]):.3f}", flush=True)
        if (bi_ + 1) % a.checkpoint_every == 0:
            meta["hours_recorded"] = hours
            save(part_path, parts, meta)
            if gparts["g_patient"]:
                save(os.path.join(a.out, "grids_partial.npz"), gparts, meta)

    meta["hours_recorded"] = list(range(0, a.hours + 1, a.every))
    meta["samples"] = samples
    meta["seconds"] = round(time.time() - t0, 1)
    save(os.path.join(a.out, "trajectories.npz"), parts, meta)
    if gparts["g_patient"]:
        save(os.path.join(a.out, "grids.npz"), gparts, meta)
    for f in ("partial.npz", "grids_partial.npz"):
        if os.path.exists(os.path.join(a.out, f)):
            os.remove(os.path.join(a.out, f))
    print(f"\nwrote {a.out}/trajectories.npz ({len(rows)} tissues)"
          + (f" and grids.npz" if gparts["g_patient"] else ""))


if __name__ == "__main__":
    main()
