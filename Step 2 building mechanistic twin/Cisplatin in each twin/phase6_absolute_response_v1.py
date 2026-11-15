"""
Does a single twin respond to chemotherapy the way real tumours do?
Reads one or more phase6_patient_chemo_v2 outputs (shards) and reports the
twin's absolute response, not differences between patients.

    python scripts/phase6_absolute_response_v1.py \
        --runs runs/phase6_chemo_v2_GSE14814_3cycles_5draws_p0 ... --out runs/phase6_absolute_response_v1

Per tissue (one posterior draw of one patient):
  viable fraction after k cycles = tumour cells in the chemo arm at the end of
      cycle k / tumour cells when chemo started
  twin MPR  = viable fraction <= 0.10 at the end of the course
  twin pCR  = no tumour cells left at the end of the course
  chemo vs no chemo at the end = tumour cells in the chemo arm / in the untreated
      arm of the same twin (how much the course removed, net of regrowth)
  B_i at 192 h (day 9) = log10(no chemo / chemo), the paper's primary read
Per patient: averaged over that patient's draws (the probability of MPR is the
share of the patient's draws reaching it).

Real references (chemotherapy-alone arm, CheckMate 816, 3 cycles of platinum
doublet before surgery; Forde et al. 2022, NEJM): pCR 2.2%, MPR 12.7%. The
twin's schedule is JBR.10 cisplatin (days 1 and 8, every 28 days) without the
partner drug, so the comparison is an order-of-magnitude check, stated so in
the paper; it is not a fit and nothing is tuned to it.

Writes <out>/absolute_response.json and per_patient.csv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

REF = {"source": "CheckMate 816 chemotherapy-alone arm (Forde et al. 2022)", "pCR": 0.022, "MPR": 0.127}


def load(dirs):
    parts, meta = [], None
    for d in dirs:
        f = os.path.join(d, "trajectories.npz")
        if not os.path.exists(f):
            f = os.path.join(d, "partial.npz")
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        meta = meta or m
        if m.get("cycles") != meta.get("cycles") or m["hours"] != meta["hours"] or m["every"] != meta["every"]:
            raise SystemExit(f"{d}: different cycles/hours/every from the first run")
        parts.append({k: z[k] for k in ("patient", "draw", "chemo", "no_chemo", "B_i")})
        print(f"  {f}: {len(z['patient'])} tissues")
    arr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    key = arr["patient"] * 1000 + arr["draw"]
    if len(np.unique(key)) != len(key):
        raise SystemExit("duplicate (patient, draw) across runs")
    return arr, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arr, meta = load(a.runs)
    rec = meta["record"]
    ti = rec.index("tumour")
    hours = np.arange(0, meta["hours"] + 1, meta["every"])
    cyc_h = 28 * 24
    n_cyc = int(meta.get("cycles", 1))
    N0 = np.clip(arr["chemo"][:, 0, ti], 1, None)
    viable = arr["chemo"][:, :, ti] / N0[:, None]
    ends = {}
    for k in range(1, n_cyc + 1):
        h = min(k * cyc_h, hours[-1])
        ends[k] = viable[:, int(np.argmin(np.abs(hours - h)))]
    final = viable[:, -1]
    vs_nochemo = arr["chemo"][:, -1, ti] / np.clip(arr["no_chemo"][:, -1, ti], 1, None)   # surviving share vs untreated twin
    mpr, pcr = final <= 0.10, arr["chemo"][:, -1, ti] == 0
    i192 = int(np.argmin(np.abs(hours - 192)))
    nc = arr["no_chemo"][:, i192, ti]; ch = np.clip(arr["chemo"][:, i192, ti], 1, None)
    bi192 = np.log10(np.clip(nc, 1, None) / ch)

    df = pd.DataFrame({"patient": arr["patient"], "draw": arr["draw"], "viable_final": final,
                       "chemo_vs_no_chemo_final": vs_nochemo, "mpr": mpr.astype(float), "pcr": pcr.astype(float), "B_i_192h": bi192,
                       **{f"viable_after_cycle_{k}": v for k, v in ends.items()}})
    if meta.get("samples"):
        df["sample"] = [meta["samples"][p] for p in df["patient"]]
    per = df.groupby("patient").agg(viable_final_median=("viable_final", "median"),
                                    p_mpr=("mpr", "mean"), p_pcr=("pcr", "mean"),
                                    B_i_median=("B_i_192h", "median"), draws=("draw", "count"))
    per.to_csv(os.path.join(a.out, "per_patient.csv"))

    rep = {"runs": a.runs, "cycles": n_cyc, "hours": int(meta["hours"]), "n_tissues": int(len(df)),
           "n_patients": int(len(per)), "reference": REF,
           "twin_MPR_rate": float(per["p_mpr"].mean()), "twin_pCR_rate": float(per["p_pcr"].mean()),
           "viable_final_median": float(np.median(final)),
           "viable_final_iqr": [float(np.percentile(final, 25)), float(np.percentile(final, 75))],
           "chemo_vs_no_chemo_final_median": float(np.median(vs_nochemo)),
           "viable_after_cycle_median": {k: float(np.median(v)) for k, v in ends.items()},
           "B_i_192h_median": float(np.median(bi192)),
           "B_i_192h_between_patient_sd": float(per["B_i_median"].std()),
           "B_i_192h_within_patient_sd": float(df.groupby("patient")["B_i_192h"].std().mean())}
    with open(os.path.join(a.out, "absolute_response.json"), "w") as fh:
        json.dump(rep, fh, indent=2)

    print(f"\n{rep['n_patients']} patients, {rep['n_tissues']} tissues, {n_cyc} cycle(s), {meta['hours']} h")
    for k, v in rep["viable_after_cycle_median"].items():
        print(f"  median viable tumour after cycle {k}: {100 * v:.1f}% of the starting count")
    print(f"  at the end: median {100 * rep['viable_final_median']:.1f}% "
          f"(middle half {100 * rep['viable_final_iqr'][0]:.1f}–{100 * rep['viable_final_iqr'][1]:.1f}%)")
    print(f"  tumour cells left vs the same twin untreated: median {100 * rep['chemo_vs_no_chemo_final_median']:.1f}%")
    print(f"  twin MPR (<=10% viable): {100 * rep['twin_MPR_rate']:.1f}%   real: {100 * REF['MPR']:.1f}%")
    print(f"  twin pCR (no tumour left): {100 * rep['twin_pCR_rate']:.1f}%   real: {100 * REF['pCR']:.1f}%")
    print(f"  B_i at day 9: median {rep['B_i_192h_median']:.3f}; between-patient SD "
          f"{rep['B_i_192h_between_patient_sd']:.3f}, within {rep['B_i_192h_within_patient_sd']:.3f}")
    print(f"wrote {a.out}/absolute_response.json and per_patient.csv")


if __name__ == "__main__":
    main()
