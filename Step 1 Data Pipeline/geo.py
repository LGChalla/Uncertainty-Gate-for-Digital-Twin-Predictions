"""
Download and parse GEO series-matrix files (step 1 currenlty).

    python data_pipeline/geo.py --gse GSE14814 GSE68465 GSE42127 --out data/raw

For each series this writes:
    data/raw/<GSE>/expression.parquet      probes x samples (as deposited)
    data/raw/<GSE>/samples.csv             one row per sample, every
                                           characteristic split into a column
    data/raw/<GSE>/characteristics.txt     every clinical field and its values,
                                           to check the field map by eye

No outcome data are analysed here. This script only downloads and reshapes.
"""

import argparse
import gzip
import io
import os
import re
import urllib.request
from collections import Counter

import pandas as pd


def series_matrix_url(gse: str) -> str:
    stub = gse[:-3] + "nnn"
    return (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/matrix/"
            f"{gse}_series_matrix.txt.gz")


def fetch(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  already have {dest}")
        return dest
    print(f"  downloading {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def parse_series_matrix(path: str):
    """Returns (expression DataFrame, samples DataFrame). Handles multi-platform files poorly
    on purpose: if a series has several platforms GEO splits them into _1, _2 files."""
    meta_lines, table_lines, in_table = [], [], False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            (table_lines if in_table else meta_lines).append(line.rstrip("\n"))

    expr = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", index_col=0)
    expr.columns = [c.strip('"') for c in expr.columns]

    rows = {}
    char_count = Counter()
    for line in meta_lines:
        if not line.startswith("!Sample_"):
            continue
        parts = [p.strip().strip('"') for p in line.split("\t")]
        key, vals = parts[0][len("!Sample_"):], parts[1:]
        if key == "characteristics_ch1":
            # each line holds one characteristic; the key is before the first ':'
            n = char_count[key]
            char_count[key] += 1
            name = None
            parsed = []
            for v in vals:
                if ":" in v:
                    k, val = v.split(":", 1)
                    name = name or k.strip()
                    parsed.append(val.strip())
                else:
                    parsed.append(v)
            col = _clean(name) if name else f"characteristic_{n}"
            while col in rows:
                col += "_dup"
            rows[col] = parsed
        elif key in ("geo_accession", "title", "source_name_ch1", "platform_id"):
            rows[key] = vals
    samples = pd.DataFrame(rows)
    if "geo_accession" in samples:
        samples = samples.set_index("geo_accession")
    return expr, samples


def _clean(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()


def describe(samples: pd.DataFrame, path: str):
    with open(path, "w") as f:
        for col in samples.columns:
            vc = samples[col].value_counts(dropna=False)
            f.write(f"== {col}  ({vc.size} distinct values)\n")
            for val, n in vc.head(15).items():
                f.write(f"   {n:5d}  {val}\n")
            if vc.size > 15:
                f.write("   ...\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gse", nargs="+", default=["GSE14814", "GSE68465", "GSE42127"])
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    for gse in a.gse:
        print(gse)
        d = os.path.join(a.out, gse)
        os.makedirs(d, exist_ok=True)
        gz = fetch(series_matrix_url(gse), os.path.join(d, f"{gse}_series_matrix.txt.gz"))
        expr, samples = parse_series_matrix(gz)
        expr.to_parquet(os.path.join(d, "expression.parquet"))
        samples.to_csv(os.path.join(d, "samples.csv"))
        describe(samples, os.path.join(d, "characteristics.txt"))
        print(f"  {expr.shape[0]} probes x {expr.shape[1]} samples; "
              f"{samples.shape[1]} clinical fields -> {d}/characteristics.txt")


if __name__ == "__main__":
    main()
