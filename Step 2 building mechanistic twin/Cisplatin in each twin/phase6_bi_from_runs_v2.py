"""
Version 2 (v1 unchanged). Recomputes B_i from the two arms with BOTH counts
floored at one cell: B_i = log10(max(N_no chemo, 1) / max(N_chemo, 1)).
v1 used the B_i stored by the runs, which floors only the chemo arm, so a
tissue whose tumour died out without chemo gave log10(0) = -inf (23 of 133
JBR.10 patients had at least one such draw, Nov 8). With both floored, a
tumour that disappears in both arms has B_i = 0: chemo had nothing to remove.
Chosen before any outcome was seen. Also reports how often it happens.

--- v1 docstring ---
Collect B_i from direct chemo runs (phase6_patient_chemo_v2/v3, one or more
shards) into the file the outcome analysis reads, in place of the emulator
(which failed its pre-registered R^2 check on Nov 8).

    python scripts/phase6_bi_from_runs_v1.py --runs runs/phase6_chemo_v3_GSE14814_1cycle_10draws_sharedworld_p0 ... \
        --cohort GSE14814 --out runs/phase6_bi_direct_v1

Writes <out>/bi_<cohort>.npz: bi (patients x draws, B_i at --hour), samples,
hour, shared_world. Column d is draw d for every patient; with --shared-world
runs, column d is one consistent "world" of the shared parameters, which is
what phase8_outcomes_v1 pools over by Rubin's rules. Patients or draws that
were not run are NaN.
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--hour", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P, D, B, meta = [], [], [], None
    ext_nc, ext_both = [], []
    for d in a.runs:
        f = os.path.join(d, "trajectories.npz")
        if not os.path.exists(f):
            raise SystemExit(f"{d}: no trajectories.npz (shard not finished)")
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if meta is None:
            meta = m
        for k in ("cohort", "npe", "shared_world", "hours", "every"):
            if m.get(k) != meta.get(k):
                raise SystemExit(f"{d}: {k}={m.get(k)} differs from the first run ({meta.get(k)})")
        hrs = m["hours_recorded"]
        if a.hour not in hrs:
            raise SystemExit(f"{d}: hour {a.hour} not recorded")
        j = hrs.index(a.hour)
        ti = m["record"].index("tumour")
        nc, ch = z["no_chemo"][:, j, ti], z["chemo"][:, j, ti]
        ext_nc.append(nc == 0); ext_both.append((nc == 0) & (ch == 0))
        P.append(z["patient"]); D.append(z["draw"]); B.append(np.log10(np.maximum(nc, 1) / np.maximum(ch, 1)))
        print(f"  {d}: {len(z['patient'])} tissues")
    P, D, B = np.concatenate(P), np.concatenate(D), np.concatenate(B)
    ext_nc, ext_both = np.concatenate(ext_nc), np.concatenate(ext_both)
    print(f"  tumour gone without chemo in {int(ext_nc.sum())} of {len(B)} tissues "
          f"({int(ext_both.sum())} gone in both arms -> B_i = 0)")
    if len(set(zip(P.tolist(), D.tolist()))) != len(P):
        raise SystemExit("duplicate (patient, draw): a shard was listed twice")
    samples = meta["samples"]
    bi = np.full((len(samples), int(D.max()) + 1), np.nan)
    bi[P, D] = B
    np.savez_compressed(os.path.join(a.out, f"bi_{a.cohort}.npz"), bi=bi, samples=np.array(samples),
                        hour=a.hour, shared_world=bool(meta.get("shared_world", False)), source="direct simulation, both arms floored at 1 cell")
    done = np.isfinite(bi).all(1)
    pm = np.nanmean(bi, 1)
    print(f"\n{a.cohort}: {int(done.sum())}/{len(samples)} patients with all draws; B_i at {a.hour} h median "
          f"{np.nanmedian(bi):.3f}; between-patient SD {np.nanstd(pm):.3f}; "
          f"within-patient SD {np.nanmean(np.nanstd(bi, 1)):.3f}")
    if meta.get("shared_world"):
        r = np.nanmean([np.corrcoef(bi[done, i], bi[done, k])[0, 1]
                        for i in range(bi.shape[1]) for k in range(i + 1, bi.shape[1])])
        print(f"  shared worlds: mean correlation of patient rankings between worlds {r:.2f} "
              f"(how stable the ordering of patients is across the shared-parameter uncertainty)")
    print(f"wrote {a.out}/bi_{a.cohort}.npz")


if __name__ == "__main__":
    main()
