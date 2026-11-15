"""
Phase 5: fit a twin to each patient, version 5: identical to v4 (unchanged)
except that it runs on a GPU. v4 built the prior on the CPU, and sbi refuses
to train on cuda with a CPU prior (Nov 7 overnight run stopped there). Here the
prior box and every tensor given to sbi are put on --device; draws come back
to the CPU for the bookkeeping. Nothing in the fitting itself changes.

    CUDA_VISIBLE_DEVICES=3 python scripts/phase5_train_npe_v5.py --device cuda \
        --sims runs/phase5_v3_fib120_sims --out runs/phase5_v3_fib120_npe_logithyp_endolinked

--- v4 docstring ---
Phase 5: fit a twin to each patient, version 4. Trains on the v3 campaign
(tissue_v3, priors_v4). phase5_train_npe.py, _v2 and _v3 are unchanged.

    CUDA_VISIBLE_DEVICES=1 python scripts/phase5_train_npe_v4.py --device cuda \
        --sims runs/phase5_v3_fib120_sims --out runs/phase5_v3_fib120_npe_logithyp_endolinked

What changed from v3: the observation choices made from the outcome-free
coverage check on Nov 6 (pre-registration v0.5, Section 2), applied exactly as
scripts/phase5_coverage_check_v5.py applies them, so training sees the same
comparison the coverage check passed:
  - f_endothelial goes through the within-cohort link (latent noise 0.30),
    not compared directly; --endothelial direct is the sensitivity analysis.
  - hypoxia uses logit inside the link; every other linked feature uses log.
    --logit-features changes the set (sensitivity only).
  - the priors module is read from the simulation file's meta, so the prior
    box always matches the prior the simulations were drawn from.
  - the per-cohort link offset and scale are saved, so the same mapping can
    be reused for the chemo runs and the figures.

--- v3 docstring ---

What is the same as v2: the monotone link offset + scale * log(.) is fitted per
cohort; linked noise is a relative error carried through the link's scale;
fraction noise is a coefficient of variation.

What changed, and why
---------------------
1. The v2 model and features (observe_v2): 12 features, every fraction on one
   basis that includes vessels, CD4 and B cells present, caf_share and m2_share
   both on the link. Patient features come from features_v2_qc.csv.

2. Share floors per draw. A tissue with no hypoxic cells, no M2 or no CAF gives
   a share of exactly 0, which the log link turns into a spike. The floor is
   the smallest non-zero value that tissue could show: one cell of its own
   population. phase5_simulate_v2 stores it for every draw; v2 used a single
   1/n_cells for hypoxia only.

3. Sampling can no longer hang. v2 used rejection sampling, which waited
   forever once a patient's posterior sat outside the prior box (acceptance
   0.000%, Nov 5). Here each patient's draws are taken without rejection, the
   share inside the box is recorded as that patient's acceptance, and only the
   inside draws are kept. A patient with fewer than --min-inside draws inside
   is flagged, not waited on. Acceptance per patient is a pre-registered
   check (preregistration v0.3, Section 7).

4. Patient specificity is measured, not assumed. Per target parameter the
   report gives prior-to-posterior contraction AND the spread of posterior
   medians across patients relative to the mean within-patient posterior SD.
   Contraction alone looked fine in v1 while every patient got the same answer
   (corrections log, item 8).

5. Posterior draws are saved: --keep-draws per patient (default 50, the number
   the pre-registration fixes for Exp 4's Rubin's-rules pooling), in model units.

Writes <out>/report.json, and per cohort posterior_<DS>.csv (median, 5%, 95%
per parameter, acceptance, flag) and draws_<DS>.npz.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.observe_v2 import FEATURES, PLANS, FRACTIONS, DEFAULT_PLAN   # noqa: E402
import importlib                                                           # noqa: E402

torch.set_default_dtype(torch.float32)
DATASETS = ["GSE14814", "GSE68465", "GSE42127", "TCGA-LUAD"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def apply_floors(x, feats, floors, floor_names):
    out = x.copy()
    for k, fn in enumerate(floor_names):
        j = feats.index(fn)
        out[:, j] = np.maximum(out[:, j], floors[:, k])
    return out


BOUNDED = ("hypoxia", "m2_share", "caf_share")


def transform(sim_linked, names, logit_set):
    """log, or logit for shares in logit_set (as phase5_coverage_check_v5)."""
    v = np.clip(sim_linked, 1e-12, None)
    out = np.log(v)
    for j, fn in enumerate(names):
        if fn in logit_set:
            p = np.clip(v[:, j], 1e-6, 1 - 1e-6)
            out[:, j] = np.log(p) - np.log1p(-p)
    return out


def fit_link(ls, obs_linked):
    """Match transformed twin values to one cohort's scores by mean and spread."""
    scale = obs_linked.std(0) / np.clip(ls.std(0), 1e-12, None)
    offset = obs_linked.mean(0) - scale * ls.mean(0)
    return offset, scale


def draw(post, x, n, lo, hi, device="cpu"):
    """n draws without rejection; returns (inside draws, share inside the box)."""
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad(), warnings.catch_warnings():
        # sbi warns on every call that draws may fall outside the prior; that
        # is intended here, since the share inside is measured below.
        warnings.simplefilter("ignore", UserWarning)
        try:
            s = post.sample((n,), x=xt, show_progress_bars=False, reject_outside_prior=False)
        except TypeError:                                # older sbi: no such flag
            s = post.sample((n,), x=xt, show_progress_bars=False, max_sampling_time=30)
    s = s.detach().cpu().numpy()
    inside = np.all((s >= lo) & (s <= hi), axis=1)
    return s[inside], float(inside.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="runs/phase5_v3_fib120_sims")
    ap.add_argument("--endothelial", choices=["linked", "direct"], default="linked")
    ap.add_argument("--logit-features", nargs="*", default=["hypoxia"], choices=list(BOUNDED))
    ap.add_argument("--plan", default=DEFAULT_PLAN, choices=list(PLANS))
    ap.add_argument("--out", default="runs/phase5_v3_fib120_npe_logithyp_endolinked")
    ap.add_argument("--calibrate", nargs="*", default=DATASETS)
    ap.add_argument("--n-posterior", type=int, default=1000, help="draws per patient before the box check")
    ap.add_argument("--min-inside", type=int, default=200, help="flag a patient with fewer draws inside the box")
    ap.add_argument("--keep-draws", type=int, default=50, help="draws saved per patient for Exp 4")
    ap.add_argument("--fraction-cv", type=float, default=None,
                    help="override the plan's coefficient of variation for the fractions")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    try:
        from sbi.inference import NPE
        from sbi.utils import BoxUniform
    except ImportError:
        raise SystemExit("sbi is not installed: pip install sbi")

    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    if meta.get("observe") != "observe_v2":
        raise SystemExit(f"{a.sims} was not made with observe_v2 (meta says {meta.get('observe')}); "
                         "use phase5_train_npe_v2.py for v1 simulations")
    P = importlib.import_module(f"tme_twin.{meta['priors_module']}")
    names, feats = list(z["names"]), list(z["features"])
    theta_all = z["theta"]
    x_all = apply_floors(z["x"], feats, z["floors"], meta["floors"])
    targets = meta["target_names"]
    ti = [names.index(t) for t in targets]

    plan = {"direct": dict(PLANS[a.plan]["direct"]), "linked": dict(PLANS[a.plan]["linked"])}
    if a.endothelial == "linked":
        plan["linked"]["f_endothelial"] = plan["direct"].pop("f_endothelial") * 2   # as coverage v5
    logit_set = tuple(a.logit_features)
    direct = [f for f in FEATURES if f in plan["direct"]]
    linked = [f for f in FEATURES if f in plan["linked"]]
    used = direct + linked
    cv = {f: (a.fraction_cv if a.fraction_cv is not None else plan["direct"][f]) for f in direct}
    print(f"{len(theta_all)} simulations, {len(targets)} targets, plan {a.plan}, grid {meta['size']}")
    print(f"  direct (CV): {cv}")
    print(f"  linked: {linked}; logit on {list(logit_set) or 'none'}; endothelial {a.endothelial}")
    sim_direct = x_all[:, [feats.index(f) for f in direct]]
    sim_linked = x_all[:, [feats.index(f) for f in linked]]

    pri = P.regime_priors(meta["regime"])
    tscale = [pri[t][2] for t in targets]
    th = theta_all[:, ti].copy()
    lo, hi = np.empty(len(targets)), np.empty(len(targets))
    for j, t in enumerate(targets):
        l, h, sc, _, _ = pri[t]
        if sc == "log":
            th[:, j] = np.log(th[:, j]); lo[j], hi[j] = np.log(l), np.log(h)
        else:
            lo[j], hi[j] = l, h
    prior_sd = (hi - lo) / np.sqrt(12.0)                 # SD of a uniform on the box
    prior = BoxUniform(low=torch.as_tensor(lo, dtype=torch.float32), high=torch.as_tensor(hi, dtype=torch.float32),
                       device=a.device)

    report = {"version": 5, "plan": a.plan, "endothelial": a.endothelial,
              "logit_features": list(logit_set), "priors_module": meta["priors_module"], "targets": targets, "target_scale": tscale,
              "direct_cv": cv, "linked": linked, "n_sims": int(len(theta_all)),
              "sim_meta": meta, "cohorts": {}}
    t0 = time.time()
    for ds in a.calibrate:
        f = os.path.join(ROOT, "data", "processed", ds, "features_v2_qc.csv")
        if not os.path.exists(f):
            print(f"{ds}: no features_v2_qc.csv (run data_pipeline/features_v2.py), skipping")
            continue
        obs = pd.read_csv(f, index_col=0)
        obs_direct, obs_linked = obs[direct].to_numpy(), obs[linked].to_numpy()
        tl = transform(sim_linked, linked, logit_set)
        off, sc = fit_link(tl, obs_linked)
        print(f"\n=== {ds}: {len(obs)} patients ===")
        for j, fn in enumerate(linked):
            print(f"  link {fn:>10}: offset {off[j]:+.4f}  scale {sc[j]:+.4f}")

        xs = np.concatenate([sim_direct, off + sc * tl], 1)
        sd = np.empty_like(xs)
        for j, fn in enumerate(used):
            if fn in linked:
                sd[:, j] = plan["linked"][fn] * abs(sc[linked.index(fn)])
            else:
                sd[:, j] = np.maximum(cv[fn] * np.abs(xs[:, j]), 1e-4)
        xs = xs + rng.normal(0.0, 1.0, xs.shape) * sd
        xo = np.concatenate([obs_direct, obs_linked], 1)
        snr = {fn: float(np.std(xo[:, j]) / np.median(sd[:, j])) for j, fn in enumerate(used)}
        print("  signal-to-noise per feature (observed spread / noise):")
        for fn in used:
            print(f"    {fn:>14} {snr[fn]:>7.2f}" + ("  <-- weak" if snr[fn] < 1 else ""))
        ok = np.isfinite(xs).all(1)
        print(f"  training on {int(ok.sum())} of {len(xs)} simulations", flush=True)

        inf = NPE(prior=prior, device=a.device)
        inf.append_simulations(torch.as_tensor(th[ok], dtype=torch.float32, device=a.device),
                               torch.as_tensor(xs[ok], dtype=torch.float32, device=a.device))
        density = inf.train(max_num_epochs=a.max_epochs, show_train_summary=False)
        post = inf.build_posterior(density)
        print(f"  trained ({inf.summary['epochs_trained'][-1] if 'epochs_trained' in inf.summary else '?'} epochs)",
              flush=True)

        rows, keep, med, psd = [], [], [], []
        for i, pid in enumerate(obs.index):
            s, acc = draw(post, xo[i], a.n_posterior, lo, hi, a.device)
            n_in = len(s)
            flag = n_in < a.min_inside
            if n_in:
                m, q05, q95, sdev = np.median(s, 0), np.quantile(s, 0.05, 0), np.quantile(s, 0.95, 0), s.std(0)
                k = s[rng.choice(n_in, size=a.keep_draws, replace=n_in < a.keep_draws)]
            else:
                m = q05 = q95 = sdev = np.full(len(targets), np.nan)
                k = np.full((a.keep_draws, len(targets)), np.nan)
            med.append(m); psd.append(sdev); keep.append(k)
            to_model = lambda v, j: float(np.exp(v)) if tscale[j] == "log" else float(v)
            rows.append({"sample": pid, "acceptance": acc, "n_inside": n_in, "flagged": flag,
                         **{f"{t}_median": to_model(m[j], j) for j, t in enumerate(targets)},
                         **{f"{t}_q05": to_model(q05[j], j) for j, t in enumerate(targets)},
                         **{f"{t}_q95": to_model(q95[j], j) for j, t in enumerate(targets)}})
            if (i + 1) % 50 == 0:
                print(f"  calibrated {i + 1}/{len(obs)}", flush=True)

        med, psd, keep = np.array(med), np.array(psd), np.array(keep)
        acc_all = np.array([r["acceptance"] for r in rows])
        spec = {}
        print("  patient specificity (on the prior's own scale):")
        print(f"    {'parameter':>16} {'contraction':>12} {'between/within':>15}")
        for j, t in enumerate(targets):
            contraction = float(1 - np.nanmean(psd[:, j]) / prior_sd[j])
            ratio = float(np.nanstd(med[:, j]) / np.nanmean(psd[:, j]))
            spec[t] = {"contraction": contraction, "between_over_within": ratio}
            print(f"    {t:>16} {contraction:>12.2f} {ratio:>15.2f}" + ("" if ratio >= 1 else "  <-- patients alike"))
        print(f"  acceptance: median {np.median(acc_all):.3f}, min {acc_all.min():.3f}; "
              f"{int(sum(r['flagged'] for r in rows))} of {len(rows)} patients flagged")

        out = pd.DataFrame(rows).set_index("sample")
        out.to_csv(os.path.join(a.out, f"posterior_{ds}.csv"))
        draws_model = keep.copy()
        for j in range(len(targets)):
            if tscale[j] == "log":
                draws_model[:, :, j] = np.exp(draws_model[:, :, j])
        np.savez_compressed(os.path.join(a.out, f"draws_{ds}.npz"), draws=draws_model,
                            samples=np.array(obs.index), targets=np.array(targets))
        report["cohorts"][ds] = {"n_patients": int(len(obs)), "linked": linked, "link_offset": off.tolist(),
                                 "link_scale": sc.tolist(), "snr": snr, "specificity": spec,
                                 "acceptance_median": float(np.median(acc_all)),
                                 "acceptance_min": float(acc_all.min()),
                                 "n_flagged": int(sum(r["flagged"] for r in rows))}
        print(f"  wrote {a.out}/posterior_{ds}.csv and draws_{ds}.npz")

    report["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\ntotal {report['total_seconds'] / 60:.0f} min; wrote {a.out}/report.json")


if __name__ == "__main__":
    main()
