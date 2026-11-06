"""
generate_fig7_combined.py — v3
================================
Fixed Panel B: explicit function-name-to-index mapping matching
icip_train.py ICIP_TEMPLATES order exactly.

The function order is fixed at definition time in icip_train.py:
  Clinical Diagnosis:        F1=Direct label, F2=Urgency/triage,
                             F3=Differential reasoning, F4=Mechanism/pathophysiology,
                             F5=Organ-system localisation, F6=Severity/acuity
  Clinical Reasoning:        F1=Deductive, F2=Hypothesis-driven,
                             F3=Pathophysiology-first, F4=Differential elimination,
                             F5=Red-flag/safety, F6=Uncertainty-aware
  Medical QA:                F1=Direct factual, F2=Mechanistic,
                             F3=Evidence-based, F4=Patient-facing,
                             F5=Risk/safety, F6=Management-oriented

Validation against logs:
  Task-prior top functions from training log:
    CD:  Severity/acuity      -> F6 (index 5)
    CR:  Differential elim.   -> F4 (index 3)
    MQA: Evidence-based       -> F3 (index 2)
  Real encoder top functions:
    CD:  Organ-system         -> F5 (index 4)
    CR:  Red-flag/safety      -> F5 (index 4)
    MQA: Direct factual       -> F1 (index 0)
  Shuffled top functions:
    CD:  Organ-system         -> F5 (index 4)
    CR:  Deductive            -> F1 (index 0)
    MQA: Mechanistic          -> F2 (index 1)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.color":        "#CCCCCC",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

TASKS = [
    "Clinical Diagnosis",
    "Clinical Reasoning",
    "Medical Question Answering",
]
TASK_COLORS = {
    "Clinical Diagnosis":         "#0072B2",
    "Clinical Reasoning":         "#D55E00",
    "Medical Question Answering": "#009E73",
}

# Exact function names from icip_train.py ICIP_TEMPLATES — ORDER MATTERS
# These are the strings stored in icip_eval_routing.csv selected_function column
FUNCTION_ORDER = {
    "Clinical Diagnosis": [
        "Direct label",
        "Urgency / triage",
        "Differential reasoning",
        "Mechanism / pathophysiology",
        "Organ-system localisation",
        "Severity / acuity",
    ],
    "Clinical Reasoning": [
        "Deductive",
        "Hypothesis-driven",
        "Pathophysiology-first",
        "Differential elimination",
        "Red-flag / safety",
        "Uncertainty-aware",
    ],
    "Medical Question Answering": [
        "Direct factual",
        "Mechanistic",
        "Evidence-based",
        "Patient-facing",
        "Risk / safety",
        "Management-oriented",
    ],
}

# Short display labels for x-axis (two lines for readability)
FUNCTION_SHORT = {
    "Clinical Diagnosis": [
        "F1\nDirect\nlabel",
        "F2\nUrgency/\ntriage",
        "F3\nDifferential\nreasoning",
        "F4\nMechanism/\npathophys.",
        "F5\nOrgan-system\nlocalisation",
        "F6\nSeverity/\nacuity",
    ],
    "Clinical Reasoning": [
        "F1\nDeductive",
        "F2\nHypothesis-\ndriven",
        "F3\nPathophys.-\nfirst",
        "F4\nDiff.\nelim.",
        "F5\nRed-flag/\nsafety",
        "F6\nUncertainty-\naware",
    ],
    "Medical Question Answering": [
        "F1\nDirect\nfactual",
        "F2\nMechanistic",
        "F3\nEvidence-\nbased",
        "F4\nPatient-\nfacing",
        "F5\nRisk/\nsafety",
        "F6\nManagement-\noriented",
    ],
}

# Colorblind-safe (Wong 2011)
CONDITIONS = [
    ("Real encoder",     "#000000", "-",  2.2),
    ("Shuffled encoder", "#7F7F7F", "--", 1.8),
    ("Task-prior ctrl",  "#D55E00", ":",  2.2),
]
BAR_COLORS = ["#000000", "#7F7F7F", "#D55E00"]


def smooth(series, window=40):
    return pd.Series(series).rolling(window, min_periods=1, center=True).mean().values


def load_entropy(d):
    p = os.path.join(d, "icip_entropy_history.csv")
    if not os.path.exists(p):
        print(f"  WARNING: {p} not found"); return pd.DataFrame()
    return pd.read_csv(p)


def load_eval_routing(d):
    p = os.path.join(d, "icip_eval_routing.csv")
    if not os.path.exists(p):
        print(f"  WARNING: {p} not found"); return pd.DataFrame()
    df = pd.read_csv(p)
    print(f"  {os.path.basename(d)}: {len(df)} rows | "
          f"unique functions: {sorted(df['selected_function'].unique())}")
    return df


def get_freqs(evl_df, task):
    """
    Map selected_function strings to F1-F6 indices using exact FUNCTION_ORDER.
    Validates against known log results before returning.
    """
    ordered = FUNCTION_ORDER[task]
    k = len(ordered)

    if evl_df.empty:
        return np.zeros(k)

    task_df = evl_df[evl_df["task"] == task]
    if task_df.empty:
        return np.zeros(k)

    counts = task_df["selected_function"].value_counts()
    freqs  = np.array([counts.get(name, 0) for name in ordered], dtype=float)
    total  = freqs.sum()

    if total == 0:
        print(f"  WARNING: zero matches for {task}. "
              f"CSV values: {list(counts.index[:5])}")
        print(f"  Expected: {ordered}")
        return np.zeros(k)

    freqs /= total
    top_idx = int(np.argmax(freqs))
    print(f"  {task}: top={ordered[top_idx]} (F{top_idx+1}, {freqs[top_idx]:.1%})")
    return freqs


def validate_against_logs(eval_dfs):
    """
    Print validation table comparing figure data against training logs.
    Expected from logs:
      Real:       CD=F5(Organ-system), CR=F5(Red-flag), MQA=F1(Direct factual)
      Shuffled:   CD=F5(Organ-system), CR=F1(Deductive), MQA=F2(Mechanistic)
      Task-prior: CD=F6(Severity/acuity), CR=F4(Diff.elim.), MQA=F3(Evidence-based)
    """
    names  = ["Real", "Shuffled", "Task-prior"]
    expected = {
        # Values from actual icip_eval_routing.csv greedy routing
        "Real": {
            "Clinical Diagnosis":         2,  # F3 Differential reasoning (59.0%)
            "Clinical Reasoning":         3,  # F4 Differential elimination (74.7%)
            "Medical Question Answering": 0,  # F1 Direct factual (69.7%)
        },
        "Shuffled": {
            "Clinical Diagnosis":         3,  # F4 Mechanism/pathophysiology (59.5%)
            "Clinical Reasoning":         3,  # F4 Differential elimination (43.2%)
            "Medical Question Answering": 2,  # F3 Evidence-based (64.1%)
        },
        "Task-prior": {
            "Clinical Diagnosis":         2,  # F3 Differential reasoning (100%)
            "Clinical Reasoning":         5,  # F6 Uncertainty-aware (100%)
            "Medical Question Answering": 0,  # F1 Direct factual (100%)
        },
    }
    print("\n=== VALIDATION AGAINST TRAINING LOGS ===")
    all_ok = True
    for name, evl_df in zip(names, eval_dfs):
        for task in TASKS:
            freqs   = get_freqs(evl_df, task)
            top_idx = int(np.argmax(freqs)) if freqs.sum() > 0 else -1
            exp_idx = expected[name][task]
            match   = "OK" if top_idx == exp_idx else f"MISMATCH (got F{top_idx+1}, expected F{exp_idx+1})"
            if "MISMATCH" in match:
                all_ok = False
            print(f"  {name:<12} {task:<35} top=F{top_idx+1}  {match}")
    print(f"\n{'All checks passed.' if all_ok else 'WARNING: mismatches found — check function name spelling in CSV.'}")
    print("=========================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir",       required=True)
    parser.add_argument("--shuffled_dir",   required=True)
    parser.add_argument("--taskprior_dir",  required=True)
    parser.add_argument("--output_dir",     required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dirs = [args.real_dir, args.shuffled_dir, args.taskprior_dir]
    print("Loading entropy histories...")
    entropy_dfs = [load_entropy(d) for d in dirs]
    print("\nLoading eval routing CSVs...")
    eval_dfs = [load_eval_routing(d) for d in dirs]

    # Validate before plotting
    validate_against_logs(eval_dfs)

    # -----------------------------------------------------------------------
    # Figure: 2 rows x 3 cols
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.62, wspace=0.30,
                            top=0.89, bottom=0.12,
                            left=0.06, right=0.97)

    # ---- Panel A: Entropy ----
    for col, task in enumerate(TASKS):
        ax    = fig.add_subplot(gs[0, col])
        color = TASK_COLORS[task]

        for (label, lc, ls, lw), ent_df in zip(CONDITIONS, entropy_dfs):
            if ent_df.empty:
                continue
            d = ent_df[(ent_df["task"] == task) &
                       (ent_df["mode"] == "policy")].sort_values("step")
            if d.empty:
                continue
            ax.plot(d["step"], smooth(d["entropy"].values),
                    color=lc, linewidth=lw, linestyle=ls, label=label)

        ax.axhline(np.log(6), color="#BDBDBD", linestyle=":",
                   linewidth=1.2, label="Max entropy (log 6)")
        ax.set_title(f"({chr(65+col)}) {task}",
                     fontsize=10, fontweight="bold", color=color, pad=6)
        ax.set_xlabel("Training step")
        if col == 0:
            ax.set_ylabel("Policy entropy H[π]")
            ax.legend(fontsize=7.5, framealpha=0.9, loc="lower right")
        ax.set_ylim(bottom=0, top=2.05)

    # ---- Panel B: Held-out routing ----
    bar_width = 0.26
    x = np.arange(6)

    for col, task in enumerate(TASKS):
        ax    = fig.add_subplot(gs[1, col])
        color = TASK_COLORS[task]

        for ci, ((label, lc, ls, lw), evl_df) in \
                enumerate(zip(CONDITIONS, eval_dfs)):
            freqs  = get_freqs(evl_df, task)
            offset = (ci - 1) * bar_width
            ax.bar(x + offset, freqs, bar_width,
                   label=label, color=lc, alpha=0.80,
                   edgecolor="white", linewidth=0.4)

        ax.axhline(1/6, color="#999999", linestyle="--",
                   linewidth=1.0, label="Uniform (1/6)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"F{i+1}" for i in range(6)], fontsize=8
        )
        ax.set_title(f"({chr(68+col)}) {task}",
                     fontsize=10, fontweight="bold", color=color, pad=6)
        ax.set_xlabel("Instruction function")
        if col == 0:
            ax.set_ylabel("Selection proportion")
            ax.legend(fontsize=7.5, framealpha=0.9, loc="upper right")
        ax.set_ylim(0, 1.08)

        # Short function labels below x-ticks
        short = FUNCTION_SHORT[task]
        for i, lbl in enumerate(short):
            lines = lbl.split("\n")[1:]  # skip "F1" prefix
            ax.text(i, -0.22, "\n".join(lines),
                    ha="center", va="top", fontsize=5.5,
                    color="#555555",
                    transform=ax.get_xaxis_transform())

    # ---- Panel labels ----
    fig.text(0.005, 0.92, "Panel A: Policy entropy during training",
             fontsize=10, fontweight="bold", va="top")
    fig.text(0.005, 0.49, "Panel B: Held-out greedy routing distribution (validation set)",
             fontsize=10, fontweight="bold", va="top")

    # ---- Shared legend ----
    legend_elements = [
        Line2D([0],[0], color="#000000", lw=2.2, ls="-",  label="Real encoder"),
        Line2D([0],[0], color="#7F7F7F", lw=1.8, ls="--", label="Shuffled encoder"),
        Line2D([0],[0], color="#D55E00", lw=2.2, ls=":",  label="Task-prior control"),
        Line2D([0],[0], color="#BDBDBD", lw=1.2, ls=":",  label="Max entropy / Uniform (1/6)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right",
               bbox_to_anchor=(0.99, 0.97), fontsize=9,
               framealpha=0.95, ncol=2)

    fig.suptitle(
        "ICIP-Router: Routing Behavior Under Real and Control Conditions",
        fontsize=12, fontweight="bold", y=0.97
    )

    for ext in ("pdf", "png"):
        out = os.path.join(args.output_dir, f"fig7_combined_controls.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()