"""
Harmonise clinical fields and build a flow table per dataset.

    python data_pipeline/flow.py --raw data/raw --out data/processed

Outputs per dataset:
    data/processed/<DS>/clinical_harmonised.csv   id, adeno, treated, os_months, os_event, stage, age, sex
    data/processed/<DS>/flow.csv                  counts at every filter, by treatment arm
    data/processed/flow_summary.md                all flow tables in one place

The script never looks at the relationship between treatment and survival.
It counts patients; Survival is analysed only in Step 5.” 
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)

# Patterns used to guess a field when field_maps.json has no key yet.
GUESS = {
    "histology": r"hist|cell_type|disease_state|diagnosis",
    "treatment": r"chemo|adjuvant|post_surgical|treatment|^act",
    "os_time":   r"(^os|overall).*(time|month|year|day)|months_to_last|survival.*(month|time|year)|follow",
    "os_event":  r"(^os|overall).*(status|event|censor)|vital|dead",
    "stage":     r"stage",
    "age":       r"^age|age_",
    "sex":       r"sex|gender",
}


def guess_key(cols, concept, taken):
    pat = re.compile(GUESS[concept], re.I)
    for c in cols:
        if c not in taken and pat.search(c):
            return c
    return None


def to_months(x, unit):
    x = pd.to_numeric(x, errors="coerce")
    return {"years": x * 12, "days": x / 30.4375}.get(unit, x)


def unit_from_key(key):
    k = (key or "").lower()
    if "year" in k:
        return "years"
    if "day" in k:
        return "days"
    return "months"


def harmonise_geo(ds, samples, fmap):
    m = fmap.get(ds, {})
    cols = list(samples.columns)
    used, chosen = set(), {}
    for concept in ("histology", "treatment", "os_time", "os_event", "stage", "age", "sex"):
        key = (m.get(concept) or {}).get("key") or guess_key(cols, concept, used)
        chosen[concept] = key
        if key:
            used.add(key)
    lc = lambda s: s.astype(str).str.strip().str.lower()

    out = pd.DataFrame(index=samples.index)
    if chosen["histology"]:
        out["adeno"] = lc(samples[chosen["histology"]]).str.contains(m["histology"]["adeno_values"], regex=True)
    else:
        out["adeno"] = np.nan
    if chosen["treatment"]:
        t = lc(samples[chosen["treatment"]])
        out["treated"] = np.where(t.str.match(m["treatment"]["treated_values"]), 1,
                                  np.where(t.str.match(m["treatment"]["untreated_values"]), 0, np.nan))
    else:
        out["treated"] = np.nan
    unit = (m.get("os_time") or {}).get("unit") or unit_from_key(chosen["os_time"])
    out["os_months"] = to_months(samples[chosen["os_time"]], unit) if chosen["os_time"] else np.nan
    if chosen["os_event"]:
        out["os_event"] = lc(samples[chosen["os_event"]]).str.match(m["os_event"]["event_values"]).astype(float)
    else:
        out["os_event"] = np.nan
    for c in ("stage", "age", "sex"):
        out[c] = samples[chosen[c]] if chosen[c] else np.nan
    chosen["os_time_unit"] = unit
    return out, chosen


def harmonise_tcga(raw, fmap):
    m = fmap["TCGA-LUAD"]
    clin = pd.read_csv(os.path.join(raw, "clinical.csv")).drop_duplicates("case")
    treat = pd.read_csv(os.path.join(raw, "treatments.csv"))
    surv = pd.read_csv(os.path.join(raw, "survival.csv"))
    expr_cols = pd.read_parquet(os.path.join(raw, "expression.parquet")).columns
    primary = [c for c in expr_cols if c[13:15] == "01"]          # primary tumour samples
    has_expr = {c[:12] for c in primary}

    out = clin.set_index("case")
    out["has_expression"] = out.index.isin(has_expr)
    out["adeno"] = True                     # every TCGA-LUAD case is an adenocarcinoma by project
    cls = classify_tcga_treatment(treat, m)
    for col in cls.columns:
        out[col] = cls[col].reindex(out.index)
    out["treated"] = out["adjuvant_platinum"]          # adjuvant platinum: the definition used in the paper
    s = surv.rename(columns={"_PATIENT": "case"}).drop_duplicates("case").set_index("case")
    out["os_months"] = to_months(s.reindex(out.index).get("OS.time"), "days")
    out["os_event"] = pd.to_numeric(s.reindex(out.index).get("OS"), errors="coerce")
    out["age"] = pd.to_numeric(out["age_at_diagnosis_days"], errors="coerce") / 365.25
    out = out.rename(columns={"gender": "sex"})
    chosen = {"treated_definition": "platinum agent with intent=Adjuvant (or no intent and start day "
                                    f"{m['window_days'][0]}-{m['window_days'][1]}); untreated = explicit "
                                    "'no' chemo record and no platinum; otherwise unknown; "
                                    "neoadjuvant excluded"}
    return out, chosen


def classify_tcga_treatment(treat: pd.DataFrame, m: dict) -> pd.DataFrame:
    """
    Three-way adjuvant platinum status per TCGA case: 1 = treated, 0 = untreated, NaN = unknown.

    treated   : a platinum record whose intent is Adjuvant, or whose intent is missing
                but whose start day falls in the window. Platinum can be filed under
                'Chemotherapy' or 'Pharmaceutical Therapy', so both are accepted.
    untreated : no adjuvant platinum, and an explicit chemo/pharmaceutical record
                that says treatment_or_therapy = 'no'.
    unknown   : anything else, including no usable record at all.
    Any Neoadjuvant record makes the case ineligible (set to unknown, flagged).
    """
    t = treat.copy()
    for c in ("treatment_type", "therapeutic_agents", "treatment_or_therapy", "treatment_intent_type"):
        t[c] = t[c].fillna("").astype(str).str.strip().str.lower()
    is_drug_type = t["treatment_type"].str.contains("chemotherapy|pharmaceutical", regex=True)
    is_platinum = t["therapeutic_agents"].str.contains(m["platinum_pattern"])
    start = pd.to_numeric(t["days_to_treatment_start"], errors="coerce")
    lo, hi = m["window_days"]
    intent = t["treatment_intent_type"]
    no_intent = intent.isin(["nan", "", "not reported", "unknown"])
    adj_plat = is_platinum & ((intent == "adjuvant") | (no_intent & start.between(lo, hi)))
    explicit_no = is_drug_type & (t["treatment_or_therapy"] == "no")
    neo = intent == "neoadjuvant"

    g = t.assign(adj_plat=adj_plat, any_plat=is_platinum, explicit_no=explicit_no, neo=neo,
                 adj_any_chemo=is_drug_type & (intent == "adjuvant") & (t["treatment_or_therapy"] == "yes"))
    per = g.groupby("case")[["adj_plat", "any_plat", "explicit_no", "neo", "adj_any_chemo"]].any()
    status = pd.Series(float("nan"), index=per.index)
    status[per["explicit_no"] & ~per["adj_plat"]] = 0.0
    status[per["adj_plat"]] = 1.0
    status[per["neo"]] = float("nan")
    return pd.DataFrame({
        "adjuvant_platinum": status,
        "any_platinum_record": per["any_plat"].astype(int),
        "adjuvant_any_chemo": per["adj_any_chemo"].astype(int),
        "neoadjuvant_flag": per["neo"],
    })


def flow_table(df, need_expression=True):
    steps = []

    def add(label, mask):
        sub = df[mask]
        steps.append({"step": label, "n": len(sub),
                      "treated": int((sub["treated"] == 1).sum()),
                      "untreated": int((sub["treated"] == 0).sum()),
                      "treatment_unknown": int(sub["treated"].isna().sum())})
        return mask

    m = add("all records", pd.Series(True, index=df.index))
    if need_expression and "has_expression" in df:
        m = add("with tumour expression", m & df["has_expression"])
    m = add("adenocarcinoma", m & (df["adeno"] == True))          # noqa: E712
    m = add("treatment known", m & df["treated"].notna())
    m = add("survival known", m & df["os_months"].notna() & df["os_event"].notna())
    return pd.DataFrame(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    with open(os.path.join(HERE, "field_maps.json")) as f:
        fmap = json.load(f)

    md = ["# Flow tables (protocol step 24)\n",
          "Counts only. Treatment and survival are never analysed together here.\n"]
    for ds in ("GSE14814", "GSE68465", "GSE42127", "TCGA-LUAD"):
        raw = os.path.join(a.raw, ds)
        if not os.path.isdir(raw):
            print(f"{ds}: not downloaded, skipped")
            continue
        if ds == "TCGA-LUAD":
            df, chosen = harmonise_tcga(raw, fmap)
            extra = df
        else:
            df, chosen = harmonise_geo(ds, pd.read_csv(os.path.join(raw, "samples.csv"), index_col=0), fmap)
            extra = None
        od = os.path.join(a.out, ds)
        os.makedirs(od, exist_ok=True)
        df.to_csv(os.path.join(od, "clinical_harmonised.csv"))
        flow = flow_table(df)
        flow.to_csv(os.path.join(od, "flow.csv"), index=False)

        status = "confirmed" if fmap.get(ds, {}).get("confirmed") else "UNCONFIRMED: check field map"
        md.append(f"\n## {ds} ({status})\n")
        md.append("Fields used: " + ", ".join(f"{k} = `{v}`" for k, v in chosen.items()) + "\n")
        md.append(flow.to_markdown(index=False) + "\n")
        if extra is not None:
            e = extra[extra["has_expression"]]
            alt = (f"Counts under other definitions (cases with tumour expression): "
                   f"any platinum record, any intent = {int(e['any_platinum_record'].fillna(0).sum())}; "
                   f"adjuvant chemo of any drug = {int(e['adjuvant_any_chemo'].fillna(0).sum())}; "
                   f"neoadjuvant (excluded) = {int(e['neoadjuvant_flag'].fillna(False).sum())}")
            md.append("\n" + alt + "\n")
            print("  " + alt)
        print(f"{ds} [{status}]")
        print("  fields:", chosen)
        print(flow.to_string(index=False))

    with open(os.path.join(a.out, "flow_summary.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {a.out}/flow_summary.md")


if __name__ == "__main__":
    main()
