"""
Simulation-based calibration (SBC) of the fitting step (reviewer item 3).
Is each patient's posterior honest: not over- or under-confident?

    CUDA_VISIBLE_DEVICES=1 python scripts/phase5_sbc_v1.py --device cuda \
        --sims runs/phase5_v3_fib120_sims --cohort GSE14814 --n-test 300 \
        --out runs/phase5_sbc_v1_GSE14814_300

Same data preparation as phase5_train_npe_v5 (floors, link fitted to the
cohort, logit on hypoxia, endothelial linked, the same observation noise).
--n-test simulations are held out; the estimator is trained on the rest.
For each held-out tissue, --n-post posterior draws are taken and the rank of
the true value among them is recorded, per parameter (Talts et al. 2018).
A calibrated posterior gives ranks uniform on 0..n_post.

Reported per parameter:
  - chi-square test of rank uniformity (10 bins) and its p value
  - coverage of central 50% and 90% intervals (should be ~50% and ~90%)
  - direction of any miss: ranks piled at both ends = over-confident
    (intervals too narrow); piled in the middle = under-confident; piled at one
    end = biased.
Draws outside the prior box are kept here (no rejection), so the check is of
the estimator itself; the share outside is reported.

Writes <out>/sbc.json and ranks.npz.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase5_train_npe_v5 import apply_floors, transform, fit_link, BOUNDED   # noqa: E402
from tme_twin.observe_v2 import FEATURES, PLANS, DEFAULT_PLAN                # noqa: E402
import importlib                                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--cohort", default="GSE14814", help="whose link and noise scale to use")
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-post", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    P = importlib.import_module(f"tme_twin.{meta['priors_module']}")
    names, feats = list(z["names"]), list(z["features"])
    x_all = apply_floors(z["x"], feats, z["floors"], meta["floors"])
    targets = meta["target_names"]
    ti = [names.index(t) for t in targets]
    plan = {"direct": dict(PLANS[DEFAULT_PLAN]["direct"]), "linked": dict(PLANS[DEFAULT_PLAN]["linked"])}
    plan["linked"]["f_endothelial"] = plan["direct"].pop("f_endothelial") * 2
    direct = [f for f in FEATURES if f in plan["direct"]]
    linked = [f for f in FEATURES if f in plan["linked"]]
    used = direct + linked
    pri = P.regime_priors(meta["regime"])
    th = z["theta"][:, ti].copy()
    lo, hi = np.empty(len(targets)), np.empty(len(targets))
    for j, t in enumerate(targets):
        l, h, sc, _, _ = pri[t]
        if sc == "log":
            th[:, j] = np.log(th[:, j]); lo[j], hi[j] = np.log(l), np.log(h)
        else:
            lo[j], hi[j] = l, h

    obs = pd.read_csv(os.path.join(ROOT, "data", "processed", a.cohort, "features_v2_qc.csv"), index_col=0)
    tl = transform(x_all[:, [feats.index(f) for f in linked]], linked, ("hypoxia",))
    off, sc = fit_link(tl, obs[linked].to_numpy())
    xs = np.concatenate([x_all[:, [feats.index(f) for f in direct]], off + sc * tl], 1)
    sd = np.empty_like(xs)
    for j, fn in enumerate(used):
        sd[:, j] = (plan["linked"][fn] * abs(sc[linked.index(fn)]) if fn in linked
                    else np.maximum(plan["direct"][fn] * np.abs(xs[:, j]), 1e-4))
    xs = xs + rng.normal(0, 1, xs.shape) * sd
    ok = np.where(np.isfinite(xs).all(1))[0]
    perm = rng.permutation(ok)
    test, train = perm[:a.n_test], perm[a.n_test:]
    print(f"{len(train)} training, {len(test)} held-out simulations; cohort link {a.cohort}", flush=True)

    dev = a.device
    prior = BoxUniform(low=torch.as_tensor(lo, dtype=torch.float32), high=torch.as_tensor(hi, dtype=torch.float32), device=dev)
    inf = NPE(prior=prior, device=dev)
    inf.append_simulations(torch.as_tensor(th[train], dtype=torch.float32, device=dev),
                           torch.as_tensor(xs[train], dtype=torch.float32, device=dev))
    density = inf.train(max_num_epochs=a.max_epochs, show_train_summary=False)
    post = inf.build_posterior(density)

    ranks = np.empty((len(test), len(targets)), int)
    in50 = np.zeros((len(test), len(targets)), bool)
    in90 = np.zeros((len(test), len(targets)), bool)
    outside = []
    for i, k in enumerate(test):
        with torch.no_grad():
            s = post.sample((a.n_post,), x=torch.as_tensor(xs[k], dtype=torch.float32, device=dev),
                            reject_outside_prior=False, show_progress_bars=False).detach().cpu().numpy()
        outside.append(float(np.mean(((s < lo) | (s > hi)).any(1))))
        ranks[i] = (s < th[k]).sum(0)
        q = np.quantile(s, [0.05, 0.25, 0.75, 0.95], axis=0)
        in50[i] = (th[k] >= q[1]) & (th[k] <= q[2])
        in90[i] = (th[k] >= q[0]) & (th[k] <= q[3])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)}", flush=True)

    rep = {"cohort": a.cohort, "n_train": int(len(train)), "n_test": int(len(test)), "n_post": a.n_post,
           "share_draws_outside_box": float(np.mean(outside)), "parameters": {}}
    print(f"\n{'parameter':>16} {'chi2 p':>8} {'cover 50%':>10} {'cover 90%':>10}  verdict")
    nb = 10
    for j, t in enumerate(targets):
        h, _ = np.histogram(ranks[:, j], bins=nb, range=(0, a.n_post + 1))
        chi = stats.chisquare(h)
        c50, c90 = float(in50[:, j].mean()), float(in90[:, j].mean())
        ends = (h[0] + h[-1]) / (2 * len(test) / nb)
        mid = h[nb // 2 - 1:nb // 2 + 1].sum() / (2 * len(test) / nb)
        skew = (h[:nb // 2].sum() - h[nb // 2:].sum()) / len(test)
        if chi.pvalue >= 0.01 and 0.80 <= c90 <= 0.97:
            verdict = "calibrated"
        elif c90 < 0.80 or ends > 1.6:
            verdict = "over-confident (intervals too narrow)"
        elif c90 > 0.97 or mid > 1.6:
            verdict = "under-confident (intervals too wide)"
        else:
            verdict = "biased " + ("low" if skew > 0 else "high")
        rep["parameters"][t] = {"chi2_p": float(chi.pvalue), "coverage50": c50, "coverage90": c90,
                                "rank_hist": h.tolist(), "verdict": verdict}
        print(f"{t:>16} {chi.pvalue:>8.3f} {c50:>10.2f} {c90:>10.2f}  {verdict}")
    print(f"\nshare of posterior draws outside the prior box: {rep['share_draws_outside_box']:.3f}")
    with open(os.path.join(a.out, "sbc.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    np.savez_compressed(os.path.join(a.out, "ranks.npz"), ranks=ranks, targets=np.array(targets))
    print(f"wrote {a.out}/sbc.json")


if __name__ == "__main__":
    main()
