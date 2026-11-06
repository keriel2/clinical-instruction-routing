"""
Paired statistical tests: Real Encoder (ICIF-Router) vs Random Baseline
- Clinical Diagnosis: McNemar (exact match) + Wilcoxon (token-F1)
- Medical Question Answering: Wilcoxon (ROUGE-L, token-F1)
- Clinical Reasoning: Wilcoxon (full & conclusion ROUGE-L, token-F1)

Usage:
    pip install statsmodels rouge-score
    python paired_tests.py
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from rouge_score import rouge_scorer as rs

np.random.seed(42)
scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)

REAL_PATH = "/home/share/henry/ICIP_ROUTER/QWEN/downstream_sanity_final/generations_real_encoder.csv"
RAND_PATH = "/home/share/henry/ICIP_ROUTER/QWEN/downstream_sanity_random/downstream_sanity_generations.csv"

# ── metric helpers ─────────────────────────────────────────────────────────────

def tf1(p, r):
    def tok(x): return re.sub(r"[^a-z0-9\s]", " ", str(x).lower()).split()
    p, r = tok(p), tok(r)
    if not p and not r: return 1.0
    if not p or not r: return 0.0
    pc = {t: p.count(t) for t in set(p)}
    rc = {t: r.count(t) for t in set(r)}
    ov = sum(min(pc.get(t, 0), rc.get(t, 0)) for t in pc)
    if ov == 0: return 0.0
    pr = ov / len(p); re_ = ov / len(r)
    return 2 * pr * re_ / (pr + re_ + 1e-12)
 
def rl(p, r):
    p, r = str(p).strip(), str(r).strip()
    if not p or not r: return 0.0
    return float(scorer.score(r, p)["rougeL"].fmeasure)
 
# ── test helpers ───────────────────────────────────────────────────────────────
 
def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"
 
def wilcoxon_test(a, b, label):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    # drop pairs where both are nan
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if (a == b).all():
        print(f"  {label}: no differences — test not applicable")
        return
    stat, p = wilcoxon(a, b, alternative="greater")  # H1: real > random
    print(f"  {label}: W={stat:.1f}, p={p:.4e} {sig(p)}")
    print(f"    real={a.mean():.4f}, random={b.mean():.4f}, Δ={a.mean()-b.mean():+.4f}, n={len(a)}")
 
def mcnemar_test(a, b, label):
    a, b = np.array(a, dtype=int), np.array(b, dtype=int)
    n11 = ((a==1) & (b==1)).sum()
    n10 = ((a==1) & (b==0)).sum()  # real correct, random wrong
    n01 = ((a==0) & (b==1)).sum()  # real wrong, random correct
    n00 = ((a==0) & (b==0)).sum()
    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)
    p = result.pvalue
    print(f"  {label}: χ²={result.statistic:.2f}, p={p:.4e} {sig(p)}")
    print(f"    real correct={a.sum()}, random correct={b.sum()}, "
          f"real-only={n10}, random-only={n01}, n={len(a)}")
 
# ── load & align ───────────────────────────────────────────────────────────────
 
print("Loading files...")
real = pd.read_csv(REAL_PATH)
rand = pd.read_csv(RAND_PATH)
 
print(f"Real encoder:     {len(real)} rows, tasks: {real['task'].unique()}")
print(f"Random baseline:  {len(rand)} rows, tasks: {rand['task'].unique()}")
 
# ── Clinical Diagnosis ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Clinical Diagnosis")
print("="*60)
cd_r = real[real["task"]=="Clinical Diagnosis"].sort_values("example_id").reset_index(drop=True)
cd_n = rand[rand["task"]=="Clinical Diagnosis"].sort_values("example_id").reset_index(drop=True)
assert (cd_r["example_id"].values == cd_n["example_id"].values).all(), "CD IDs not aligned"
 
real_em = (cd_r["extracted_prediction"].str.lower().str.strip() ==
           cd_r["extracted_reference"].str.lower().str.strip()).astype(int).tolist()
rand_em = (cd_n["extracted_prediction"].str.lower().str.strip() ==
           cd_n["extracted_reference"].str.lower().str.strip()).astype(int).tolist()
mcnemar_test(real_em, rand_em, "Exact match (McNemar)")
 
real_tf1 = [tf1(p, r) for p, r in zip(cd_r["extracted_prediction"], cd_r["extracted_reference"])]
rand_tf1 = [tf1(p, r) for p, r in zip(cd_n["extracted_prediction"], cd_n["extracted_reference"])]
wilcoxon_test(real_tf1, rand_tf1, "Token-F1   (Wilcoxon, one-sided)")
 
# ── Medical Question Answering ─────────────────────────────────────────────────
print("\n" + "="*60)
print("Medical Question Answering")
print("="*60)
mqa_r = real[real["task"]=="Medical Question Answering"].sort_values("example_id").reset_index(drop=True)
mqa_n = rand[rand["task"]=="Medical Question Answering"].sort_values("example_id").reset_index(drop=True)
assert (mqa_r["example_id"].values == mqa_n["example_id"].values).all(), "MQA IDs not aligned"
 
real_rl  = [rl(p, r) for p, r in zip(mqa_r["prediction"], mqa_r["reference"])]
rand_rl  = [rl(p, r) for p, r in zip(mqa_n["prediction"], mqa_n["reference"])]
wilcoxon_test(real_rl, rand_rl, "ROUGE-L  (Wilcoxon, one-sided)")
 
real_tf1 = [tf1(p, r) for p, r in zip(mqa_r["prediction"], mqa_r["reference"])]
rand_tf1 = [tf1(p, r) for p, r in zip(mqa_n["prediction"], mqa_n["reference"])]
wilcoxon_test(real_tf1, rand_tf1, "Token-F1 (Wilcoxon, one-sided)")
 
# ── Clinical Reasoning ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Clinical Reasoning")
print("="*60)
cr_r = real[real["task"]=="Clinical Reasoning"].sort_values("example_id").reset_index(drop=True)
cr_n = rand[rand["task"]=="Clinical Reasoning"].sort_values("example_id").reset_index(drop=True)
assert (cr_r["example_id"].values == cr_n["example_id"].values).all(), "CR IDs not aligned"
 
real_rl  = [rl(p, r) for p, r in zip(cr_r["prediction"], cr_r["reference"])]
rand_rl  = [rl(p, r) for p, r in zip(cr_n["prediction"], cr_n["reference"])]
wilcoxon_test(real_rl,  rand_rl,  "ROUGE-L  (Wilcoxon, one-sided)")
 
real_tf1 = [tf1(p, r) for p, r in zip(cr_r["prediction"], cr_r["reference"])]
rand_tf1 = [tf1(p, r) for p, r in zip(cr_n["prediction"], cr_n["reference"])]
wilcoxon_test(real_tf1, rand_tf1, "Token-F1 (Wilcoxon, one-sided)")
 
print("\nDone. Significance: * p<0.05  ** p<0.01  *** p<0.001  ns=not significant")
 