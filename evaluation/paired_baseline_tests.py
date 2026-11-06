"""
paired_baseline_tests.py

Paired comparison of Clinical Diagnosis exact match between the learned
router and the static and random baselines (McNemar exact test and paired
bootstrap interval), plus router confidence and per-route accuracy.

Reproduces the static-baseline comparison reported in the paper:
  Learned vs Static: +0.008, 95% CI [-0.014, +0.031], p = 0.55
  Learned vs Random: +0.097, 95% CI [+0.074, +0.121], p < 0.001

Usage:
  python evaluation/paired_baseline_tests.py \
      --learned results/qwen/downstream_sanity_final/generations_real_encoder.csv \
      --static  results/qwen/downstream_sanity_static/generations_static_base.csv \
      --random  results/qwen/downstream_sanity_random/<generations file>.csv
"""
import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def load_cd(path):
    d = pd.read_csv(path)
    d = d[d.task == "Clinical Diagnosis"].copy()
    d["em"] = (d.extracted_prediction.map(norm) == d.extracted_reference.map(norm)).astype(int)
    return d.set_index("example_id")


def paired(a, b, name, n_boot=5000, seed=0):
    j = a[["em"]].join(b[["em"]], lsuffix="_a", rsuffix="_b", how="inner")
    b10 = int(((j.em_a == 1) & (j.em_b == 0)).sum())
    b01 = int(((j.em_a == 0) & (j.em_b == 1)).sum())
    p = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    d = (j.em_a - j.em_b).values
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{name}: n={len(j)}  EM {j.em_a.mean():.3f} vs {j.em_b.mean():.3f}  "
          f"diff={d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"McNemar p={p:.4f}  (discordant {b10} vs {b01})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learned", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--random", default=None)
    args = ap.parse_args()

    L, S = load_cd(args.learned), load_cd(args.static)
    paired(L, S, "Learned vs Static")
    if args.random:
        R = load_cd(args.random)
        paired(L, R, "Learned vs Random")
        paired(S, R, "Static vs Random")

    full = pd.read_csv(args.learned)
    if "max_prob" in full:
        print("\nRouter confidence (probability of the selected function):")
        print(full.groupby("task").max_prob.describe()[["mean", "50%", "max"]].round(3))

    if "resolved_function_label" in L:
        print("\nDiagnosis exact match by learned route (same examples under static):")
        j = L[["resolved_function_label", "em"]].join(S[["em"]], rsuffix="_static")
        print(j.groupby("resolved_function_label")
               .agg(n=("em", "size"), learned=("em", "mean"), static=("em_static", "mean"))
               .round(3))


if __name__ == "__main__":
    main()
