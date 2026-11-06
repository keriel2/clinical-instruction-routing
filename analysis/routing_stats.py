"""
routing_stats.py — v2
======================
Fixes from ChatGPT review:
  1. scipy.jensenshannon returns JS DISTANCE (sqrt of divergence) — renamed correctly
  2. Training-time stats reported separately (non-independent samples)
  3. chi2_two_sample drops all-zero columns to avoid invalid expected frequencies
  4. p-values not rounded in JSON, only in display
  5. Caption and column labels corrected

Run:
    python routing_stats.py \
        --real_dir      /home/share/henry/ICIP_ROUTER/QWEN/real \
        --shuffled_dir  /home/share/henry/ICIP_ROUTER/QWEN/shuffled_control \
        --taskprior_dir /home/share/henry/ICIP_ROUTER/QWEN/task_prior_control \
        --output_dir    /home/share/henry/ICIP_ROUTER/QWEN/routing_stats
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

TASKS = [
    "Clinical Diagnosis",
    "Clinical Reasoning",
    "Medical Question Answering",
]
K = 6


def get_counts(df, task, mode=None):
    """Get raw integer counts per function (length K)."""
    sub = df[df["task"] == task]
    if mode is not None and "mode" in df.columns:
        sub = sub[sub["mode"] == mode]
    counts = sub["template_idx"].value_counts().reindex(range(K), fill_value=0)
    return counts.values.astype(float)


def bootstrap_js_distance(dist1, dist2, counts1, counts2, n=10000, ci=0.95):
    """
    Bootstrap CI for Jensen-Shannon DISTANCE (scipy convention).
    JS distance = sqrt(JS divergence), bounded [0, 1].
    Resamples routing decisions via multinomial sampling.
    """
    n1, n2 = int(counts1.sum()), int(counts2.sum())
    # Normalise distributions — add epsilon then renormalise to ensure sum=1
    p1 = dist1 + 1e-10; p1 = p1 / p1.sum()
    p2 = dist2 + 1e-10; p2 = p2 / p2.sum()
    jsd_samples = []
    for _ in range(n):
        s1 = np.random.multinomial(n1, p1) / n1
        s2 = np.random.multinomial(n2, p2) / n2
        s1 = np.clip(s1, 1e-10, None); s1 /= s1.sum()
        s2 = np.clip(s2, 1e-10, None); s2 /= s2.sum()
        jsd_samples.append(float(jensenshannon(s1, s2, base=2)))
    arr = np.array(jsd_samples)
    observed = float(jensenshannon(
        dist1 + 1e-10, dist2 + 1e-10, base=2
    ))
    return {
        "js_distance": round(observed, 4),
        "lower":       round(float(np.percentile(arr, (1-ci)/2*100)), 4),
        "upper":       round(float(np.percentile(arr, (1+ci)/2*100)), 4),
        "note":        "scipy jensenshannon returns JS distance = sqrt(JSD)",
    }


def chi2_vs_uniform(counts):
    """Chi-square goodness-of-fit: observed counts vs uniform expected."""
    total    = counts.sum()
    expected = np.full(K, total / K)
    chi2, p  = stats.chisquare(counts, f_exp=expected)
    return {"chi2": round(float(chi2), 3), "p": float(p), "df": K - 1}


def chi2_two_sample(counts1, counts2):
    """
    Chi-square test of homogeneity.
    Drops all-zero columns (functions unused by both conditions)
    to avoid invalid expected frequencies.
    """
    table = np.array([counts1, counts2], dtype=float)
    keep  = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        return {"chi2": 0.0, "p": 1.0, "df": 0,
                "note": "fewer than 2 non-zero columns"}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"chi2": round(float(chi2), 3), "p": float(p), "df": int(dof)}


def format_p(p):
    if p < 0.001:
        return "$p{<}0.001$"
    return f"$p={p:.3f}$"


def run_phase(real_df, shuf_df, tp_df, mode, n_boot):
    """Compute all statistics for one phase (training or eval)."""
    results = {}
    for task in TASKS:
        r  = get_counts(real_df, task, mode)
        s  = get_counts(shuf_df, task, mode)
        t  = get_counts(tp_df,   task, mode)
        u  = np.full(K, r.sum() / K)

        rd = r / r.sum(); sd = s / s.sum()
        td = t / t.sum(); ud = u / u.sum()

        results[task] = {
            "counts": {
                "real": r.tolist(), "shuffled": s.tolist(), "task_prior": t.tolist()
            },
            "chi2_vs_uniform": {
                "real":       chi2_vs_uniform(r),
                "shuffled":   chi2_vs_uniform(s),
                "task_prior": chi2_vs_uniform(t),
            },
            "chi2_two_sample": {
                "real_vs_shuffled":      chi2_two_sample(r, s),
                "real_vs_taskprior":     chi2_two_sample(r, t),
                "shuffled_vs_taskprior": chi2_two_sample(s, t),
            },
            "js_distance": {
                "real_vs_shuffled":  bootstrap_js_distance(rd, sd, r, s, n=n_boot),
                "real_vs_taskprior": bootstrap_js_distance(rd, td, r, t, n=n_boot),
                "real_vs_uniform":   bootstrap_js_distance(rd, ud, r, u, n=n_boot),
            },
        }
    return results


def print_phase(results, phase_label, lines):
    lines.append(f"\n{'='*70}")
    lines.append(f"PHASE: {phase_label.upper()}")
    lines.append(f"{'='*70}")
    for task in TASKS:
        d = results[task]
        lines.append(f"\n--- {task} ---")

        lines.append("  Chi-square vs uniform (df=5):")
        for cond in ["real", "shuffled", "task_prior"]:
            c = d["chi2_vs_uniform"][cond]
            lines.append(f"    {cond:<16} χ²={c['chi2']:>9.3f}  p={c['p']:.2e}")

        lines.append("  Chi-square two-sample:")
        for pair, label in [
            ("real_vs_shuffled",      "Real vs Shuffled  "),
            ("real_vs_taskprior",     "Real vs Task-prior"),
            ("shuffled_vs_taskprior", "Shuf vs Task-prior"),
        ]:
            c = d["chi2_two_sample"][pair]
            lines.append(f"    {label}  χ²={c['chi2']:>9.3f}  p={c['p']:.2e}  df={c['df']}")

        lines.append("  JS distance (95% bootstrap CI):")
        for pair, label in [
            ("real_vs_shuffled",  "Real vs Shuffled  "),
            ("real_vs_taskprior", "Real vs Task-prior"),
            ("real_vs_uniform",   "Real vs Uniform   "),
        ]:
            j = d["js_distance"][pair]
            lines.append(
                f"    {label}  JS={j['js_distance']:.4f}  "
                f"[{j['lower']:.4f}, {j['upper']:.4f}]"
            )


def generate_latex_table(eval_results, train_results):
    """
    Main-paper table: eval-time statistics.
    Training-time in appendix note.
    """
    TASK_SHORT = {
        "Clinical Diagnosis":         "CD",
        "Clinical Reasoning":         "CR",
        "Medical Question Answering": "MQA",
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Routing-distribution statistics for ICIP-Router. "
        r"Jensen-Shannon distance (base-2, bounded $[0,1]$) with 95\% "
        r"bootstrap confidence intervals (10{,}000 iterations) quantifies "
        r"divergence between held-out greedy routing profiles. "
        r"Chi-square tests of homogeneity assess whether pairs of conditions "
        r"produce significantly different routing distributions. "
        r"Effect sizes (JS distance) are emphasized over $p$-values because "
        r"large sample sizes ($n \approx 988$--$990$) make small differences "
        r"statistically significant. Results are for held-out greedy routing; "
        r"training-time statistics are provided in the supplementary material.}",
        r"\label{tab:routing_stats}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Task} & "
        r"\textbf{JS dist.\ Real vs Shuf.} & "
        r"\textbf{JS dist.\ Real vs T-prior} & "
        r"$\boldsymbol{\chi^2}$ \textbf{Real vs Shuf.} & "
        r"$\boldsymbol{\chi^2}$ \textbf{Real vs T-prior} \\",
        r"\midrule",
    ]

    for task in TASKS:
        d    = eval_results[task]
        jrs  = d["js_distance"]["real_vs_shuffled"]
        jrt  = d["js_distance"]["real_vs_taskprior"]
        c2rs = d["chi2_two_sample"]["real_vs_shuffled"]
        c2rt = d["chi2_two_sample"]["real_vs_taskprior"]

        lines.append(
            f"{TASK_SHORT[task]} & "
            f"{jrs['js_distance']:.4f} [{jrs['lower']:.4f},{jrs['upper']:.4f}] & "
            f"{jrt['js_distance']:.4f} [{jrt['lower']:.4f},{jrt['upper']:.4f}] & "
            f"{c2rs['chi2']:.1f} ({format_p(c2rs['p'])}) & "
            f"{c2rt['chi2']:.1f} ({format_p(c2rt['p'])}) \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir",       required=True)
    parser.add_argument("--shuffled_dir",   required=True)
    parser.add_argument("--taskprior_dir",  required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--n_boot",         type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(42)

    print("Loading CSVs...")
    real_sel  = pd.read_csv(os.path.join(args.real_dir,      "icip_selection_history.csv"))
    shuf_sel  = pd.read_csv(os.path.join(args.shuffled_dir,  "icip_selection_history.csv"))
    tp_sel    = pd.read_csv(os.path.join(args.taskprior_dir, "icip_selection_history.csv"))
    real_eval = pd.read_csv(os.path.join(args.real_dir,      "icip_eval_routing.csv"))
    shuf_eval = pd.read_csv(os.path.join(args.shuffled_dir,  "icip_eval_routing.csv"))
    tp_eval   = pd.read_csv(os.path.join(args.taskprior_dir, "icip_eval_routing.csv"))

    print("Computing eval-time statistics...")
    eval_results  = run_phase(real_eval, shuf_eval, tp_eval,   mode=None,     n_boot=args.n_boot)
    print("Computing training-time statistics...")
    train_results = run_phase(real_sel,  shuf_sel,  tp_sel,    mode="policy", n_boot=args.n_boot)

    lines = []
    print_phase(eval_results,  "Held-out greedy routing (eval-time)", lines)
    print_phase(train_results, "Training-time policy phase", lines)
    report = "\n".join(lines)
    print(report)

    with open(os.path.join(args.output_dir, "routing_statistics.txt"), "w") as f:
        f.write(report)

    all_results = {"eval_time": eval_results, "training_time": train_results}
    with open(os.path.join(args.output_dir, "routing_statistics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    latex = generate_latex_table(eval_results, train_results)
    with open(os.path.join(args.output_dir, "routing_stats_table.tex"), "w") as f:
        f.write(latex)

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()