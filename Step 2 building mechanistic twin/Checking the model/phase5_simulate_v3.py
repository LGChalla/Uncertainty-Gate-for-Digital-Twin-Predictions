"""
Phase 5 simulations on the v3 model. phase5_simulate.py and
phase5_simulate_v2.py are unchanged; the v2 shards (runs/phase5_v2_sims_*)
are kept as the record of the campaign that showed the fibroblast miss.

Only change from v2: tissue_v3 (fibroblast contact inhibition over a 120 um
disc, config_v3) and priors_v4 (K_fib 0.005-0.2). Everything else, including
the seed scheme, B_i subset, checkpointing and output format, is as v2.

    CUDA_VISIBLE_DEVICES=1 python scripts/phase5_simulate_v3.py --device cuda \
        --n 20000 --start 0 --end 5000 --out runs/phase5_v3_fib120_sims_a
    python scripts/phase5_simulate_v3.py --combine runs/phase5_v3_fib120_sims_a ... \
        --out runs/phase5_v3_fib120_sims
    python scripts/phase5_simulate_v3.py --n 8 --batch 4 --size 30 --burn-in-days 1 --bi-every 2 \
        --out /tmp/p5v3_smoke                                          # CPU smoke test

--- v2 docstring ---

    CUDA_VISIBLE_DEVICES=1 python scripts/phase5_simulate_v2.py --device cuda \
        --n 20000 --start 0 --end 5000 --out runs/phase5_v2_sims_a
    python scripts/phase5_simulate_v2.py --combine runs/phase5_v2_sims_a runs/phase5_v2_sims_b \
        runs/phase5_v2_sims_c runs/phase5_v2_sims_d --out runs/phase5_v2_sims
    python scripts/phase5_simulate_v2.py --n 8 --batch 4 --size 30 --burn-in-days 1 --bi-every 2 \
        --out /tmp/p5v2_smoke                                          # CPU smoke test

Each draw samples every priors_v3 parameter, builds a tissue at its analytic
steady state, burns in for a fixed 20 days, and records the raw features
observe_v2 reports (12 of them, physical units; link and noise are applied at
training time). Features are read on untreated tissue.

What is new
-----------
B_i on a subset. Every --bi-every-th draw (by global index, so the subset is
spread evenly over shards and over the prior) also gets a paired chemo /
no-chemo run, and its B_i is stored at every read hour. Those draws train the
B_i emulator (pre-registration: held-out R^2 >= 0.95). Recording B_i for every
draw would add the two 8-day arms to the 20-day burn-in for all 20,000 draws,
roughly +80% cost. Draws without B_i store NaN. B_i draws and feature-only
draws are batched separately so a batch never mixes the two costs.

Checkpointing. v1 wrote only at the end, so a shard killed after 22 hours
lost everything. Here the shard writes <out>/partial.npz every
--checkpoint-every batches (atomically: write then rename) and, when started
again with the same arguments, skips the draws already done.

Per draw it also records: occupancy, hypoxic fraction and tumour count at
treatment start; whether the starting composition had to be scaled for
crowding; burn-in worst daily change and whether it met tolerance; and the
share floors the training script needs for the log link.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.config_v3 import TME_V3                               # noqa: E402
from tme_twin.config import config_hash                              # noqa: E402
from tme_twin.tissue_v3 import TissueV3                              # noqa: E402
from tme_twin.response_v2 import burn_in, occupancy, paired_response, READ_HOURS  # noqa: E402
from tme_twin.observe_v2 import summarise, share_floors, FEATURES    # noqa: E402
from tme_twin import priors_v4 as P                                  # noqa: E402
from tme_twin.runlog import RunLog, set_deterministic                # noqa: E402

torch.set_default_dtype(torch.float64)
TARGET_CLASS = "patient"
STATE = ("occupancy", "hypoxic_fraction", "tumour", "crowded_init", "burnin_worst_daily_change",
         "burnin_converged")
FLOORS = ("m2_share", "caf_share", "hypoxia")
KEYS = ("idx", "theta", "x", "state", "floors", "bi")


def sample_prior(n, names, regime, rng):
    pri = P.regime_priors(regime)
    out = np.empty((n, len(names)))
    for j, nm in enumerate(names):
        lo, hi, scale, _, _ = pri[nm]
        out[:, j] = (np.exp(rng.uniform(math.log(lo), math.log(hi), n)) if scale == "log"
                     else rng.uniform(lo, hi, n))
    return out


def save(path, parts, meta):
    """Atomic write: a killed process leaves the previous file intact."""
    tmp = path + ".tmp.npz"
    arrays = {k: (np.concatenate(parts[k]) if parts[k] else np.empty((0,))) for k in KEYS}
    np.savez_compressed(tmp, **arrays, names=np.array(meta["names"]),
                        features=np.array(meta["features"]), meta=json.dumps(meta))
    os.replace(tmp, path)


def simulate_block(blk, names, a, seed, with_bi):
    f0, ov = [], {}
    f0 = torch.tensor(blk[:, names.index("f0_tumour")], dtype=torch.float64)
    ov = {n: torch.tensor(blk[:, j], dtype=torch.float64) for j, n in enumerate(names) if n != "f0_tumour"}
    ov.update({"nx": a.size, "ny": a.size})
    T = TissueV3(len(blk), f0, seed=seed, device=a.device, overrides=ov)
    bi = burn_in(T, fixed_days=a.burn_in_days)
    x = summarise(T).cpu().numpy()
    fl = share_floors(T)
    floors = np.stack([fl[k].cpu().numpy() for k in FLOORS], 1)
    st = np.stack([occupancy(T).numpy(), T.hypoxic_fraction().cpu().numpy(),
                   T.counts()["tumour"].numpy().astype(float),
                   T.crowded_init.cpu().numpy().astype(float),
                   np.array(bi["worst_daily_change_per_tissue"]),
                   np.array(bi["converged_per_tissue"], dtype=float)], 1)
    b = np.full((len(blk), len(READ_HOURS)), np.nan)
    if with_bi:
        resp = paired_response(T)
        for j, h in enumerate(READ_HOURS):
            b[:, j] = resp[h]["B_i"]
    return x, st, floors, b


def combine(a, names, feat):
    parts = {k: [] for k in KEYS}
    meta = None
    for d in a.combine:
        z = np.load(os.path.join(d, "sims.npz"), allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if meta is None:
            meta = m
        for k in ("regime", "size", "burn_in_days", "n", "bi_every", "seed", "priors_module"):
            if meta[k] != m[k]:
                raise SystemExit(f"{d}: {k} is {m[k]}, expected {meta[k]}")
        for k in KEYS:
            parts[k].append(z[k])
        print(f"  {d}: {len(z['idx'])} draws, {int(np.isfinite(z['bi'][:, -1]).sum())} with B_i")
    arr = {k: np.concatenate(parts[k]) for k in KEYS}
    order = np.argsort(arr["idx"])
    arr = {k: v[order] for k, v in arr.items()}
    if len(np.unique(arr["idx"])) != len(arr["idx"]):
        raise SystemExit("duplicate draw indices across shards: a shard was listed twice")
    ok = np.isfinite(arr["x"]).all(1)
    print(f"\ncombined {len(ok)} draws; {int((~ok).sum())} dropped as non-finite; "
          f"{int(np.isfinite(arr['bi'][ok][:, -1]).sum())} with B_i")
    meta.update({"combined_from": list(a.combine), "n_total": int(ok.sum())})
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "sims.npz"), **{k: v[ok] for k, v in arr.items()},
                        names=np.array(names), features=np.array(feat), meta=json.dumps(meta))
    print(f"wrote {a.out}/sims.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=20000, help="total draws across all shards")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--burn-in-days", type=int, default=20)
    ap.add_argument("--bi-every", type=int, default=4, help="B_i for every k-th draw (4 -> 5,000 of 20,000); 0 = none")
    ap.add_argument("--checkpoint-every", type=int, default=4, help="batches between checkpoints")
    ap.add_argument("--regime", default=P.DEFAULT_REGIME, choices=["primary", "residual"])
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--combine", nargs="*", default=None)
    ap.add_argument("--out", default="runs/phase5_v3_fib120_sims")
    a = ap.parse_args()
    names, feat = list(P.PRIORS), list(FEATURES)
    if a.combine:
        return combine(a, names, feat)
    os.makedirs(a.out, exist_ok=True)

    end = a.n if a.end < 0 else a.end
    set_deterministic()
    rng = np.random.default_rng(a.seed)
    allv = sample_prior(a.n, names, a.regime, rng)        # draw all, then take this shard
    idx = np.arange(a.start, end)
    meta = {"model": "tissue_v3", "config": "config_v3", "priors_module": "priors_v4",
            "observe": "observe_v2", "regime": a.regime, "size": a.size,
            "burn_in_days": a.burn_in_days, "n": a.n, "start": a.start, "end": end,
            "seed": a.seed, "device": a.device, "bi_every": a.bi_every,
            "read_hours": list(READ_HOURS), "state": list(STATE), "floors": list(FLOORS),
            "target_class": TARGET_CLASS,
            "target_names": [n for n in names if P.PRIORS[n][3] == TARGET_CLASS],
            "config_hash": config_hash(TME_V3), "names": names, "features": feat}

    # Resume
    part_path = os.path.join(a.out, "partial.npz")
    parts = {k: [] for k in KEYS}
    done = set()
    if os.path.exists(part_path):
        z = np.load(part_path, allow_pickle=True)
        old = json.loads(str(z["meta"]))
        for k in ("n", "start", "end", "seed", "size", "burn_in_days", "bi_every", "regime", "config_hash"):
            if old[k] != meta[k]:
                raise SystemExit(f"partial.npz was made with {k}={old[k]}, now {meta[k]}: "
                                 f"use a new --out or delete it deliberately")
        for k in KEYS:
            parts[k].append(z[k])
        done = set(z["idx"].tolist())
        print(f"resuming: {len(done)} draws already done", flush=True)

    with_bi = (idx % a.bi_every == 0) if a.bi_every > 0 else np.zeros(len(idx), bool)
    todo = [(i, w) for i, w in zip(idx, with_bi) if i not in done]
    groups = [[i for i, w in todo if w], [i for i, w in todo if not w]]
    batches = [(g[k:k + a.batch], gi == 0) for gi, g in enumerate(groups) for k in range(0, len(g), a.batch)]
    n_target = sum(1 for p in P.PRIORS.values() if p[3] == TARGET_CLASS)
    print(f"{len(names)} parameters, {a.n} draws total, this shard {a.start}..{end} ({len(idx)}), "
          f"grid {a.size}, burn-in {a.burn_in_days} d; {int(with_bi.sum())} with B_i; "
          f"{len(todo)} to do in {len(batches)} batches", flush=True)
    print(f"inferring the {n_target} {TARGET_CLASS}-class parameters; the rest are sampled and marginalised\n",
          flush=True)

    t0, n_done = time.time(), 0
    for b_i, (ids, wb) in enumerate(batches):
        ids = np.array(ids)
        blk = allv[ids]
        x, st, fl, b = simulate_block(blk, names, a, seed=a.seed + int(ids[0]), with_bi=wb)
        for k, v in zip(KEYS, (ids, blk, x, st, fl, b)):
            parts[k].append(v)
        n_done += len(ids)
        el = time.time() - t0
        rem = len(todo) - n_done
        print(f"  {len(done) + n_done}/{len(idx)}  {'B_i ' if wb else 'feat'}  {el / n_done:.3f} s/draw  "
              f"eta {rem * el / n_done / 60:.0f} min  occ {st[:, 0].mean():.3f}  "
              f"converged {int(st[:, 5].sum())}/{len(ids)}", flush=True)
        if (b_i + 1) % a.checkpoint_every == 0:
            save(part_path, parts, meta)

    meta["seconds"] = round(time.time() - t0, 1)
    save(os.path.join(a.out, "sims.npz"), parts, meta)
    if os.path.exists(part_path):
        os.remove(part_path)
    RunLog("runs/runlog.jsonl").append(
        "phase5_simulate_v3", config_hash=meta["config_hash"], seed=str(a.seed),
        inputs={"n": a.n, "shard": [a.start, end], "grid": a.size, "bi_every": a.bi_every},
        outputs={"draws": len(idx), "seconds": meta["seconds"]})
    print(f"\nwrote {a.out}/sims.npz  ({len(idx)} draws, {int(with_bi.sum())} with B_i) "
          f"in {meta['seconds'] / 60:.0f} min")


if __name__ == "__main__":
    main()
