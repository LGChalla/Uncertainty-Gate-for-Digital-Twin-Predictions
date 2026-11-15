"""
Positive control for the twin's guardrail (pre-registration v0.12). Does the
check "between-patient SD of B_i >= mean within-patient SD" PASS when patients
truly differ in a way the pipeline can see, and FAIL when they do not?

    CUDA_VISIBLE_DEVICES=5 python scripts/phase6_guardrail_positive_control_v1.py --device cuda \
        --sims runs/phase5_v3_fib120_sims --cohort GSE14814 \
        --real-draws runs/phase5_v3_fib120_npe_logithyp_endolinked/draws_GSE14814.npz \
        --cohorts 20 --out runs/phase6_guardrail_positive_control_v1_GSE14814_20cohorts

Why a surrogate. Running new chemotherapy simulations for every synthetic patient,
posterior draw and world (4 scenarios x 20 cohorts x 133 patients x 10 draws) would
take days of GPU. Instead B_i is computed by a surrogate f(theta) fitted to the
4,504 campaign tissues with a simulated B_i (all 36 parameters -> B_i at 192 h,
gradient boosting), plus run-to-run chance drawn from its held-out residuals.
The surrogate is checked first: applied to the REAL JBR.10 posterior draws with
shared worlds, it must reproduce the direct-simulation check (0.036 vs 0.144) in
order of magnitude. This is a surrogate-based demonstration of how the check
behaves, not new simulation evidence.

Synthetic patients are held-out campaign tissues whose true parameters are known.
Their features get the same link and observation noise as in fitting
(as phase5_sbc_v1); a neural posterior estimator is trained on the remaining
tissues. Drug parameters (the six latent ones) do not affect untreated features,
so each synthetic cohort gets ONE true drug "world", shared by its patients, as
for real patients. A patient's true response is f(its true parameters).

Scenarios (each over --cohorts synthetic cohorts of --n-patients):
  S1 measured     patient parameters known to +-10%; shared parameters known.
                  Real differences, well measured. The check should PASS.
  S2 RNA, pinned  patient parameters from the RNA posterior; shared parameters known.
  S3 RNA, as run  patient parameters from the RNA posterior; shared parameters drawn
                  from the prior per world, the same for all patients (as the study).
  S4 null         every patient has the SAME true parameters; features differ only by
                  observation noise; posterior from RNA; shared known. No real
                  differences. The check should FAIL.
Per cohort: the check, in two forms, and how
well the estimated patient medians rank the true responses (Spearman rho; not
defined for S4).

  primary    between >= within, as pre-registered (v0.4 onward).
  secondary  reliability R = 1 - (pi/2) * mean(within^2) / draws / between^2, the share of
             the between-patient spread of medians that is not explained by the noise of a
             median of `draws` runs (pi/2: variance of a median vs a mean); pass if R >= 0.5,
             i.e. at least half of the apparent patient differences are real. An intraclass-
             correlation-type rule; candidate replacement if the primary form is too strict.

Pre-specified criteria (v0.12): the check discriminates if S1 passes in >= 90% of
cohorts AND S4 fails in >= 95%; it is informative about ranking if, over S1-S3
cohorts, "pass" agrees with "rho >= 0.5" in >= 80% of cohorts. Each criterion is
reported for the primary and the secondary form.

Writes <out>/positive_control.json and cohorts.csv.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase5_train_npe_v5 import apply_floors, transform, fit_link      # noqa: E402
from tme_twin.observe_v2 import FEATURES, PLANS, DEFAULT_PLAN             # noqa: E402
import importlib                                                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOUR = 192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--cohort", default="GSE14814")
    ap.add_argument("--real-draws", required=True, help="draws_<cohort>.npz from phase5_train_npe_v5")
    ap.add_argument("--direct-check", nargs=2, type=float, default=[0.036, 0.144],
                    help="between, within from direct simulation, for the surrogate check")
    ap.add_argument("--cohorts", type=int, default=20)
    ap.add_argument("--n-patients", type=int, default=133)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--pool", type=int, default=3000, help="tissues held out as synthetic patients")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    P = importlib.import_module(f"tme_twin.{meta['priors_module']}")
    names, feats = list(z["names"]), list(z["features"])
    pri = P.regime_priors(meta["regime"])
    cls = {n: P.PRIORS[n][3] for n in names}
    pat = [n for n in names if cls[n] == "patient"]
    shared = [n for n in names if cls[n] != "patient"]
    latent = [n for n in names if cls[n] == "latent"]
    pi = [names.index(n) for n in pat]
    is_log = np.array([pri[n][2] == "log" for n in names])

    def to_scaled(T):           # all 36, on the prior's own scale (log where log-uniform)
        T = np.array(T, float)
        T[:, is_log] = np.log(T[:, is_log])
        return T

    # ---- surrogate for B_i -------------------------------------------------
    col = list(meta["read_hours"]).index(HOUR)
    okb = np.isfinite(z["bi"][:, col])
    Xs, yb = to_scaled(z["theta"][okb]), z["bi"][okb, col]
    res, r2s = np.empty(len(yb)), []
    for tr, te in KFold(5, shuffle=True, random_state=a.seed).split(Xs):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=a.seed)
        m.fit(Xs[tr], yb[tr]); p = m.predict(Xs[te]); res[te] = yb[te] - p
        r2s.append(1 - np.sum((yb[te] - p) ** 2) / np.sum((yb[te] - yb[te].mean()) ** 2))
    sigma = float(res.std())
    f = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=a.seed).fit(Xs, yb)
    print(f"surrogate: {len(yb)} tissues, held-out R^2 {np.mean(r2s):.3f}; run-to-run chance SD {sigma:.3f}", flush=True)

    def prior_draw(n_draw, which):
        out = np.empty((n_draw, len(which)))
        for j, n in enumerate(which):
            lo, hi, sc = pri[n][:3]
            out[:, j] = np.exp(rng.uniform(np.log(lo), np.log(hi), n_draw)) if sc == "log" else rng.uniform(lo, hi, n_draw)
        return out

    def bi_matrix(pat_draws, shared_per_world):
        """pat_draws: patients x draws x 12 (native); shared_per_world: draws x 24 (same for all patients)
        or patients x draws x 24. Returns patients x draws B_i with run-to-run chance."""
        npat, nd, _ = pat_draws.shape
        T = np.empty((npat, nd, len(names)))
        T[:, :, pi] = pat_draws
        sw = shared_per_world if shared_per_world.ndim == 3 else np.broadcast_to(shared_per_world, (npat, nd, len(shared)))
        T[:, :, [names.index(n) for n in shared]] = sw
        b = f.predict(to_scaled(T.reshape(-1, len(names)))).reshape(npat, nd)
        return b + rng.normal(0, sigma, b.shape)

    def check(B):
        med = np.median(B, 1)
        between, within = float(np.std(med)), float(np.mean(np.std(B, 1)))
        return med, between, within

    def reliability(between, within_sq_mean, n):
        return 1 - (np.pi / 2) * within_sq_mean / n / max(between ** 2, 1e-12)

    # ---- surrogate check on the real patients -------------------------------
    rd = np.load(a.real_draws, allow_pickle=True)
    rdraw = rd["draws"][:, :a.draws, :]
    tg = list(rd["targets"])
    rdraw = rdraw[:, :, [tg.index(n) for n in pat]]
    okp = np.isfinite(rdraw).all((1, 2))
    worlds = prior_draw(a.draws, shared)
    Br = bi_matrix(rdraw[okp], worlds[None, :, :].repeat(okp.sum(), 0))
    _, rb, rw = check(Br)
    rR = reliability(rb, float(np.mean(np.var(Br, 1))), a.draws)
    dR = reliability(a.direct_check[0], a.direct_check[1] ** 2, a.draws)
    print(f"surrogate on the real {a.cohort} posteriors: between {rb:.3f}, within {rw:.3f}, R {rR:.2f} "
          f"(direct simulation: {a.direct_check[0]:.3f}, {a.direct_check[1]:.3f}, R {dR:.2f})", flush=True)

    # ---- synthetic patients: features and estimator ---------------------------
    x_all = apply_floors(z["x"], feats, z["floors"], meta["floors"])
    plan = {"direct": dict(PLANS[DEFAULT_PLAN]["direct"]), "linked": dict(PLANS[DEFAULT_PLAN]["linked"])}
    plan["linked"]["f_endothelial"] = plan["direct"].pop("f_endothelial") * 2
    direct = [q for q in FEATURES if q in plan["direct"]]
    linked = [q for q in FEATURES if q in plan["linked"]]
    used = direct + linked
    obs = pd.read_csv(os.path.join(ROOT, "data", "processed", a.cohort, "features_v2_qc.csv"), index_col=0)
    tl = transform(x_all[:, [feats.index(q) for q in linked]], linked, ("hypoxia",))
    off, sc = fit_link(tl, obs[linked].to_numpy())
    clean = np.concatenate([x_all[:, [feats.index(q) for q in direct]], off + sc * tl], 1)

    def noisy(xc):
        sd = np.empty_like(xc)
        for j, q in enumerate(used):
            sd[:, j] = (plan["linked"][q] * abs(sc[linked.index(q)]) if q in linked
                        else np.maximum(plan["direct"][q] * np.abs(xc[:, j]), 1e-4))
        return xc + rng.normal(0, 1, xc.shape) * sd

    xs = noisy(clean)
    th_s = to_scaled(z["theta"])[:, pi]
    lo = np.array([np.log(pri[n][0]) if pri[n][2] == "log" else pri[n][0] for n in pat])
    hi = np.array([np.log(pri[n][1]) if pri[n][2] == "log" else pri[n][1] for n in pat])
    good = np.where(np.isfinite(xs).all(1) & np.isfinite(clean).all(1))[0]
    perm = rng.permutation(good)
    pool, train = perm[:a.pool], perm[a.pool:]
    dev = a.device
    prior = BoxUniform(low=torch.as_tensor(lo, dtype=torch.float32), high=torch.as_tensor(hi, dtype=torch.float32), device=dev)
    inf = NPE(prior=prior, device=dev)
    inf.append_simulations(torch.as_tensor(th_s[train], dtype=torch.float32, device=dev),
                           torch.as_tensor(xs[train], dtype=torch.float32, device=dev))
    post = inf.build_posterior(inf.train(max_num_epochs=a.max_epochs, show_train_summary=False))
    print(f"estimator trained on {len(train)} tissues; {len(pool)} held out as synthetic patients", flush=True)
    plog = np.array([pri[n][2] == "log" for n in pat])

    def posterior_draws(xrow):
        with torch.no_grad():
            s = post.sample((a.draws * 8,), x=torch.as_tensor(xrow, dtype=torch.float32, device=dev),
                            reject_outside_prior=False, show_progress_bars=False).cpu().numpy()
        s = s[((s >= lo) & (s <= hi)).all(1)]
        if len(s) == 0:
            s = np.clip(post.sample((a.draws,), x=torch.as_tensor(xrow, dtype=torch.float32, device=dev),
                                    reject_outside_prior=False, show_progress_bars=False).cpu().numpy(), lo, hi)
        s = s[rng.choice(len(s), a.draws, replace=len(s) < a.draws)]
        s[:, plog] = np.exp(s[:, plog])
        return s

    lat_i = [shared.index(n) for n in latent]
    rows = []
    for c in range(a.cohorts):
        ids = rng.choice(pool, a.n_patients, replace=False)
        true_shared = z["theta"][ids][:, [names.index(n) for n in shared]].copy()
        true_shared[:, lat_i] = prior_draw(1, latent)[0]          # one drug world for the cohort
        true_pat = z["theta"][ids][:, pi]
        T = np.empty((a.n_patients, len(names)))
        T[:, pi] = true_pat; T[:, [names.index(n) for n in shared]] = true_shared
        b_true = f.predict(to_scaled(T))
        known_sh = np.repeat(true_shared[:, None, :], a.draws, 1)
        # S1 measured
        jit = np.exp(rng.normal(0, 0.10, (a.n_patients, a.draws, len(pat))))
        d1 = np.where(plog, true_pat[:, None, :] * jit, true_pat[:, None, :] * (1 + rng.normal(0, 0.10, jit.shape)))
        d1 = np.clip(d1, [pri[n][0] for n in pat], [pri[n][1] for n in pat])
        # S2/S3 from the RNA posterior of each patient's own noisy features
        dr = np.array([posterior_draws(xs[k]) for k in ids])
        world = prior_draw(a.draws, shared)
        # S4 null: one tissue for all; features differ only by noise
        k0 = ids[0]
        x0 = noisy(np.repeat(clean[k0][None, :], a.n_patients, 0))
        d4 = np.array([posterior_draws(x0[i]) for i in range(a.n_patients)])
        null_sh = np.repeat(np.repeat(true_shared[:1][:, None, :], a.draws, 1), a.n_patients, 0)
        for sname, B, truth in [("S1 measured", bi_matrix(d1, known_sh), b_true),
                                ("S2 RNA, pinned", bi_matrix(dr, known_sh), b_true),
                                ("S3 RNA, as run", bi_matrix(dr, np.repeat(world[None], a.n_patients, 0)), b_true),
                                ("S4 null", bi_matrix(d4, null_sh), None)]:
            med, bw, wi = check(B)
            R = reliability(bw, float(np.mean(np.var(B, 1))), a.draws)
            rho = float(stats.spearmanr(med, truth).statistic) if truth is not None else np.nan
            rows.append({"cohort": c, "scenario": sname, "between": bw, "within": wi, "ratio": bw / wi,
                         "pass": bw >= wi, "reliability": R, "pass_secondary": R >= 0.5, "rho_true": rho,
                         "true_sd": float(np.std(truth)) if truth is not None else 0.0})
        print(f"  cohort {c + 1}/{a.cohorts} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "cohorts.csv"), index=False)
    print(f"\n{'scenario':>16} {'between':>8} {'within':>8} {'ratio':>6} {'pass':>6} {'R':>6} {'pass(2nd)':>10} "
          f"{'rho with truth':>16} {'true SD':>8}")
    summ = {}
    for sname, g in df.groupby("scenario", sort=False):
        q = lambda v: f"{np.nanmedian(v):.2f} [{np.nanquantile(v, .05):.2f}, {np.nanquantile(v, .95):.2f}]" if np.isfinite(v).any() else "-"
        print(f"{sname:>16} {g.between.mean():>8.3f} {g.within.mean():>8.3f} {g.ratio.mean():>6.2f} "
              f"{g['pass'].mean():>6.0%} {g.reliability.median():>6.2f} {g.pass_secondary.mean():>10.0%} {q(g.rho_true.to_numpy()):>16} {g.true_sd.mean():>8.3f}")
        summ[sname] = {"between": g.between.mean(), "within": g.within.mean(), "pass_rate": g["pass"].mean(),
                       "pass_rate_secondary": g.pass_secondary.mean(), "reliability_median": float(g.reliability.median()), "rho_median": float(np.nanmedian(g.rho_true)),
                       "true_sd": g.true_sd.mean()}
    crit = {}
    for form, key in [("primary", "pass"), ("secondary", "pass_secondary")]:
        s1 = df[df.scenario == "S1 measured"][key].mean()
        s4 = 1 - df[df.scenario == "S4 null"][key].mean()
        sub = df[df.scenario != "S4 null"]
        agree = float(((sub[key]) == (sub.rho_true >= 0.5)).mean())
        crit[form] = {"S1_pass": float(s1), "S4_fail": float(s4), "agreement_with_rho_0.5": agree,
                      "discriminates": bool(s1 >= 0.90 and s4 >= 0.95), "informative": bool(agree >= 0.80)}
        print(f"\n{form} check: S1 passes {s1:.0%} (need >= 90%), S4 fails {s4:.0%} (need >= 95%) -> "
              f"{'discriminates' if crit[form]['discriminates'] else 'does NOT discriminate'}; "
              f"agrees with rho >= 0.5 in {agree:.0%} of cohorts (need >= 80%) -> "
              f"{'informative' if crit[form]['informative'] else 'NOT informative'}")
    with open(os.path.join(a.out, "positive_control.json"), "w") as fh:
        json.dump({"surrogate": {"r2_heldout": float(np.mean(r2s)), "sigma_chance": sigma, "n": int(len(yb))},
                   "surrogate_on_real": {"between": rb, "within": rw, "R": float(rR), "direct": a.direct_check, "direct_R": float(dR)},
                   "scenarios": summ, "criteria": crit,
                   "settings": {k: v for k, v in vars(a).items()}}, fh, indent=2)
    print(f"\nwrote {a.out}/positive_control.json and cohorts.csv")


if __name__ == "__main__":
    main()
