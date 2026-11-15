"""
TCGA-LUAD expression and clinical data.

    python data_pipeline/tcga.py --out data/raw/TCGA-LUAD

Expression: the GDC-harmonised STAR TPM matrix from the UCSC Xena GDC hub
(one file instead of ~600 per-sample downloads).
Clinical and treatment records: the GDC API cases endpoint, which carries
diagnoses.treatments (type, agents, start day). Survival: Xena GDC hub table.

Writes expression.parquet, clinical.csv, treatments.csv, survival.csv and a
treatment_fields.txt summary to check by eye.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

XENA = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
FILES = {
    "expression": f"{XENA}/TCGA-LUAD.star_tpm.tsv.gz",
    "survival": f"{XENA}/TCGA-LUAD.survival.tsv.gz",
}
GDC_CASES = "https://api.gdc.cancer.gov/cases"
FIELDS = [
    "submitter_id",
    "demographic.gender", "demographic.vital_status", "demographic.days_to_death",
    "diagnoses.primary_diagnosis", "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_m",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.treatments.treatment_type", "diagnoses.treatments.therapeutic_agents",
    "diagnoses.treatments.treatment_or_therapy", "diagnoses.treatments.days_to_treatment_start",
    "diagnoses.treatments.days_to_treatment_end", "diagnoses.treatments.treatment_intent_type",
]


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  already have {dest}")
        return dest
    print(f"  downloading {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def gdc_cases():
    filt = {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-LUAD"]}}
    params = {"filters": json.dumps(filt), "fields": ",".join(FIELDS),
              "format": "JSON", "size": "2000"}
    url = GDC_CASES + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=300) as r:
        hits = json.load(r)["data"]["hits"]
    clin, treat = [], []
    for h in hits:
        demo = h.get("demographic") or {}
        for dg in h.get("diagnoses") or [{}]:
            clin.append({
                "case": h["submitter_id"],
                "gender": demo.get("gender"), "vital_status": demo.get("vital_status"),
                "days_to_death": demo.get("days_to_death"),
                "primary_diagnosis": dg.get("primary_diagnosis"),
                "age_at_diagnosis_days": dg.get("age_at_diagnosis"),
                "stage": dg.get("ajcc_pathologic_stage"),
                "pT": dg.get("ajcc_pathologic_t"), "pN": dg.get("ajcc_pathologic_n"),
                "pM": dg.get("ajcc_pathologic_m"),
                "days_to_last_follow_up": dg.get("days_to_last_follow_up"),
            })
            for t in dg.get("treatments") or []:
                treat.append({"case": h["submitter_id"], **{k: t.get(k) for k in (
                    "treatment_type", "therapeutic_agents", "treatment_or_therapy",
                    "days_to_treatment_start", "days_to_treatment_end",
                    "treatment_intent_type")}})
    return pd.DataFrame(clin), pd.DataFrame(treat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/TCGA-LUAD")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    e = fetch(FILES["expression"], os.path.join(a.out, "star_tpm.tsv.gz"))
    expr = pd.read_csv(e, sep="\t", index_col=0)
    expr.to_parquet(os.path.join(a.out, "expression.parquet"))
    print(f"  expression: {expr.shape[0]} genes x {expr.shape[1]} samples")

    s = fetch(FILES["survival"], os.path.join(a.out, "survival.tsv.gz"))
    pd.read_csv(s, sep="\t").to_csv(os.path.join(a.out, "survival.csv"), index=False)

    clin, treat = gdc_cases()
    clin.to_csv(os.path.join(a.out, "clinical.csv"), index=False)
    treat.to_csv(os.path.join(a.out, "treatments.csv"), index=False)
    with open(os.path.join(a.out, "treatment_fields.txt"), "w") as f:
        for col in ("treatment_type", "treatment_or_therapy", "treatment_intent_type",
                    "therapeutic_agents"):
            f.write(f"== {col}\n{treat[col].value_counts(dropna=False).head(40).to_string()}\n\n")
        f.write(f"== days_to_treatment_start missing: "
                f"{treat['days_to_treatment_start'].isna().mean():.0%}\n")
    print(f"  clinical: {clin['case'].nunique()} cases; treatment records: {len(treat)}")


if __name__ == "__main__":
    main()
