import argparse
import json
import os
import csv as _csv
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style — publication quality, serif font, white background
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         18,
    "axes.titlesize":    20,
    "axes.labelsize":    18,
    "xtick.labelsize":   16,
    "ytick.labelsize":   16,
    "legend.fontsize":   15,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.color":        "#BBBBBB",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASK_COLORS = {
    "Clinical Diagnosis":         "#2166AC",
    "Clinical Reasoning":         "#D6604D",
    "Medical Question Answering": "#1A7D3A",
}
TASK_LIGHT = {
    "Clinical Diagnosis":         "#D1E5F0",
    "Clinical Reasoning":         "#FDDBC7",
    "Medical Question Answering": "#D4EFD9",
}
TASK_SHORT = {
    "Clinical Diagnosis":         "CD",
    "Clinical Reasoning":         "CR",
    "Medical Question Answering": "MQA",
}
TASKS = list(TASK_COLORS.keys())

FUNCTION_LABELS = {
    "Clinical Diagnosis": [
        "Direct label", "Urgency / triage", "Differential reasoning",
        "Mechanism / pathophysiology", "Organ-system localisation", "Severity / acuity",
    ],
    "Clinical Reasoning": [
        "Deductive", "Hypothesis-driven", "Pathophysiology-first",
        "Differential elimination", "Red-flag / safety", "Uncertainty-aware",
    ],
    "Medical Question Answering": [
        "Direct factual", "Mechanistic", "Evidence-based",
        "Patient-facing", "Risk / safety", "Management-oriented",
    ],
}

LINESTYLES = ["-", "--", "-."]
MARKERS    = ["o", "s", "^"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig, figures_dir, name):
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir, f"{name}.{ext}"),
                    facecolor="white", bbox_inches="tight")
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


def smooth(series, window=30):
    s = pd.Series(series)
    return s.rolling(window=window, min_periods=1, center=True).mean().values


def load_csv(path):
    if not os.path.exists(path):
        print(f"  WARNING: {os.path.basename(path)} not found — skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {os.path.basename(path)}: {len(df)} rows")
    return df


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

def load_all(output_dir):
    print(f"\nLoading data from: {output_dir}")
    sel   = load_csv(os.path.join(output_dir, "icip_selection_history.csv"))
    ent   = load_csv(os.path.join(output_dir, "icip_entropy_history.csv"))
    rew   = load_csv(os.path.join(output_dir, "icip_reward_history.csv"))
    evl   = load_csv(os.path.join(output_dir, "icip_eval_routing.csv"))
    stats = load_json(os.path.join(output_dir, "icip_final_stats.json"))
    analysis = load_json(os.path.join(output_dir, "icip_function_analysis.json"))
    return sel, ent, rew, evl, stats, analysis


# ===========================================================================
# Fig 1 — Function selection distribution (training-time, policy phase)
# ===========================================================================

def fig_function_distribution(sel, figures_dir):
    print("\nGenerating Fig 1: Function selection distribution...")
    policy = sel[sel["mode"] == "policy"] if not sel.empty else pd.DataFrame()
    if policy.empty:
        print("  Skipped — no policy-phase data"); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, task in zip(axes, TASKS):
        color  = TASK_COLORS[task]
        colorl = TASK_LIGHT[task]
        labels = FUNCTION_LABELS[task]
        K      = len(labels)

        data   = policy[policy["task"] == task]
        counts = data["template_idx"].value_counts().reindex(range(K), fill_value=0)
        freqs  = counts.values.astype(float)
        if freqs.sum() > 0:
            freqs /= freqs.sum()

        top3    = set(np.argsort(freqs)[-3:])
        bcolors = [color if i in top3 else colorl for i in range(K)]
        ax.bar(range(K), freqs, color=bcolors, edgecolor=color,
               linewidth=0.9, zorder=3, width=0.8)
        ax.axhline(1/K, color="#888888", linestyle="--",
                   linewidth=1.4, zorder=4)

        for i in top3:
            ax.text(i, freqs[i] + 0.005, f"{freqs[i]:.1%}",
                    ha="center", va="bottom", fontsize=15,
                    fontweight="bold", color=color)

        ax.set_xticks(range(K))
        short = [f"F{i+1}\n{l.split('/')[0].strip()[:10]}" for i, l in enumerate(labels)]
        ax.set_xticklabels(short, fontsize=14, rotation=45, ha="right")
        ax.set_ylabel("Selection proportion" if task == TASKS[0] else "")
        ax.set_ylim(0, max(freqs.max() * 1.35, (1/K) * 2.2))
        ax.set_title(task, fontsize=15, fontweight="bold", color=color, pad=10)

        handles = [
            mpatches.Patch(color=color,  label="Top-3 functions"),
            mpatches.Patch(color=colorl, edgecolor=color, label="Other functions"),
            plt.Line2D([0],[0], color="#888888", linestyle="--",
                       label=f"Uniform (1/{K})"),
        ]
        ax.legend(handles=handles, fontsize=14, framealpha=0.9,
                  loc="upper right")

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig1_function_distribution")


# ===========================================================================
# Fig 2 — Policy entropy over training
# ===========================================================================

def fig_entropy_over_training(ent, figures_dir, warmup_steps=200):
    print("\nGenerating Fig 2: Entropy over training...")
    policy = ent[ent["mode"] == "policy"] if not ent.empty else pd.DataFrame()
    if policy.empty:
        print("  Skipped — no policy-phase data"); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, task, ls in zip(axes, TASKS, LINESTYLES):
        color = TASK_COLORS[task]
        data  = policy[policy["task"] == task].sort_values("step")
        if data.empty:
            ax.set_title(task); continue

        steps = data["step"].values
        ents  = data["entropy"].values
        sm    = smooth(ents)

        ax.fill_between(steps, sm - data["entropy"].rolling(30, min_periods=1).std().values,
                        sm + data["entropy"].rolling(30, min_periods=1).std().values,
                        alpha=0.12, color=color)
        ax.plot(steps, sm, color=color, linewidth=2.2, linestyle=ls,
                label="Entropy (smoothed)")
        ax.axhline(np.log(6), color="#AAAAAA", linestyle=":",
                   linewidth=1.2, label="Max entropy (uniform)")
        ax.axvline(warmup_steps, color="#555555", linestyle=":",
                   linewidth=1.2)
        ax.text(warmup_steps + 5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                "Policy\nactivates", fontsize=14, color="#555555", va="top")

        ax.set_title(task, fontsize=15, fontweight="bold", color=color, pad=10)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Entropy H[π]" if task == TASKS[0] else "")
        ax.legend(fontsize=14, framealpha=0.9)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig2_entropy_over_training")


# ===========================================================================
# Fig 3 — Reward and LM-loss trajectory
# ===========================================================================

def fig_reward_trajectory(rew, figures_dir):
    print("\nGenerating Fig 3: Reward / LM-loss trajectory...")
    policy = rew[(rew["mode"] == "policy") &
                 rew["lm_loss"].notna() &
                 rew["reward"].notna()] if not rew.empty else pd.DataFrame()
    if policy.empty:
        print("  Skipped — no reward data"); return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, task in enumerate(TASKS):
        color = TASK_COLORS[task]
        data  = policy[policy["task"] == task].sort_values("step")
        if data.empty:
            continue

        ax = axes[0, col]
        ax.plot(data["step"], smooth(data["lm_loss"].values),
                color=color, linewidth=2.0)
        ax.set_title(f"{task}\nLM Loss", fontsize=17, color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel("LM Loss" if col == 0 else "")

        ax = axes[1, col]
        ax.plot(data["step"], smooth(data["reward"].values),
                color=color, linewidth=2.0, label="Reward")
        if "baseline" in data.columns and data["baseline"].notna().any():
            ax.plot(data["step"], smooth(data["baseline"].values),
                    color="#888888", linewidth=1.4, linestyle="--",
                    label="Baseline (EMA)")
        ax.set_title(f"{task}\nReward", fontsize=17, color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward (−LM loss)" if col == 0 else "")
        ax.legend(fontsize=14)

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig3_reward_trajectory")


# ===========================================================================
# Fig 4 — Policy confidence (max_prob) over training
# ===========================================================================

def fig_policy_confidence(ent, figures_dir):
    print("\nGenerating Fig 4: Policy confidence...")
    policy = ent[ent["mode"] == "policy"] if not ent.empty else pd.DataFrame()
    if policy.empty or "max_prob" not in policy.columns:
        print("  Skipped — no max_prob data"); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, task, ls in zip(axes, TASKS, LINESTYLES):
        color = TASK_COLORS[task]
        data  = policy[policy["task"] == task].sort_values("step")
        if data.empty:
            continue

        ax.plot(data["step"], smooth(data["max_prob"].values),
                color=color, linewidth=2.2, linestyle=ls)
        ax.axhline(1/6, color="#888888", linestyle="--",
                   linewidth=1.2, label="Uniform (1/6)")
        ax.set_ylim(0, 1)
        ax.set_title(task, fontsize=15, fontweight="bold", color=color, pad=10)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Max probability" if task == TASKS[0] else "")
        ax.legend(fontsize=14)

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig4_policy_confidence")


# ===========================================================================
# Fig 5 — Probability heatmap (function specialisation over training deciles)
# ===========================================================================

def fig_probability_heatmap(sel, figures_dir):
    print("\nGenerating Fig 5: Probability heatmap...")
    policy = sel[sel["mode"] == "policy"].dropna(subset=["prob_vector"]) \
        if not sel.empty else pd.DataFrame()
    if policy.empty:
        print("  Skipped — no prob_vector data"); return

    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, task in zip(axes, TASKS):
        color  = TASK_COLORS[task]
        labels = FUNCTION_LABELS[task]
        K      = len(labels)
        data   = policy[policy["task"] == task]
        if data.empty:
            continue

        try:
            prob_matrix = np.array([json.loads(v) for v in data["prob_vector"]])
        except Exception as e:
            print(f"  Could not parse prob_vector for {task}: {e}"); continue

        n_bins   = 10
        bin_size = max(1, len(data) // n_bins)
        heatmap  = np.zeros((K, n_bins))
        for b in range(n_bins):
            chunk = prob_matrix[b * bin_size: (b + 1) * bin_size]
            if len(chunk) > 0:
                heatmap[:, b] = chunk.mean(axis=0)

        short_labels = [f"F{i+1}: {l[:18]}" for i, l in enumerate(labels)]
        sns.heatmap(heatmap, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    yticklabels=short_labels,
                    xticklabels=[f"D{i+1}" for i in range(n_bins)],
                    linewidths=0.3, linecolor="white",
                    cbar_kws={"label": "Mean prob."})
        ax.set_title(task, fontsize=15, fontweight="bold", color=color, pad=10)
        ax.set_xlabel("Training decile")
        ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig5_probability_heatmap")


# ===========================================================================
# Fig 6 — Eval-time greedy routing (held-out data)
# ===========================================================================

def fig_eval_routing(evl, figures_dir):
    print("\nGenerating Fig 6: Eval-time routing distribution...")
    if evl is None or evl.empty:
        print("  Skipped — icip_eval_routing.csv not found"); return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, task in zip(axes, TASKS):
        color  = TASK_COLORS[task]
        colorl = TASK_LIGHT[task]
        labels = FUNCTION_LABELS[task]
        K      = len(labels)

        data   = evl[evl["task"] == task]
        counts = data["selected_function"].value_counts().reindex(labels, fill_value=0)
        freqs  = counts.values.astype(float)
        total  = freqs.sum()
        if total > 0:
            freqs /= total

        top3    = set(np.argsort(freqs)[-3:])
        bcolors = [color if i in top3 else colorl for i in range(K)]
        ax.barh(range(K), freqs[::-1], color=list(reversed(bcolors)),
                edgecolor=color, linewidth=0.9, height=0.6)

        for i, (freq, lbl) in enumerate(zip(freqs[::-1], labels[::-1])):
            if (K - 1 - i) in top3:
                ax.text(freq + 0.004, i, f"{freq:.1%}",
                        va="center", fontsize=15, fontweight="bold", color=color)

        ax.set_yticks(range(K))
        ax.set_yticklabels([f"F{K-i}: {labels[K-1-i][:22]}"
                            for i in range(K)], fontsize=14)
        ax.set_xlabel("Selection proportion")
        ax.set_xlim(0, max(freqs.max() * 1.3, 1/K * 2))
        ax.set_title(f"{task}\n(n={int(total)})",
                     fontsize=15, fontweight="bold", color=color, pad=10)
        ax.axvline(1/K, color="#888888", linestyle="--",
                   linewidth=1.2, label=f"Uniform (1/{K})")
        ax.legend(fontsize=14)

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig6_eval_routing")


# ===========================================================================
# Fig 7 — Real vs shuffled encoder comparison (combined 2-panel figure)
# ===========================================================================

def fig_real_vs_shuffled(real_dir, shuffled_dir, task_prior_dir, figures_dir):
    """
    Compact publication-ready figure for the main manuscript.

    Top row: policy entropy trajectories during training.
    Bottom row: held-out greedy routing distributions.

    Improvements:
      - reduced vertical gap between panels
      - wider grouped bars in evaluation plots
      - removed long function labels below the x-axis
      - kept only F1--F6 labels for readability
      - kept one shared legend above the figure
    """
    print("\nGenerating Fig 7: Real vs control routing behavior...")

    real_ent_path = os.path.join(real_dir, "icip_entropy_history.csv")
    real_evl_path = os.path.join(real_dir, "icip_eval_routing.csv")

    if not os.path.exists(real_ent_path) or not os.path.exists(real_evl_path):
        print("  Skipped — real entropy or eval-routing file not found.")
        return

    real_ent = pd.read_csv(real_ent_path)
    real_evl = pd.read_csv(real_evl_path)

    shuf_ent = pd.DataFrame()
    shuf_evl = pd.DataFrame()
    prior_ent = pd.DataFrame()
    prior_evl = pd.DataFrame()

    if shuffled_dir:
        shuf_ent_path = os.path.join(shuffled_dir, "icip_entropy_history.csv")
        shuf_evl_path = os.path.join(shuffled_dir, "icip_eval_routing.csv")
        if os.path.exists(shuf_ent_path):
            shuf_ent = pd.read_csv(shuf_ent_path)
        if os.path.exists(shuf_evl_path):
            shuf_evl = pd.read_csv(shuf_evl_path)

    if task_prior_dir:
        prior_ent_path = os.path.join(task_prior_dir, "icip_entropy_history.csv")
        prior_evl_path = os.path.join(task_prior_dir, "icip_eval_routing.csv")
        if os.path.exists(prior_ent_path):
            prior_ent = pd.read_csv(prior_ent_path)
        if os.path.exists(prior_evl_path):
            prior_evl = pd.read_csv(prior_evl_path)

    # ------------------------------------------------------------------
    # Compact ESWA-style layout
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 3,
        figsize=(20, 9.2),
        gridspec_kw={
            "height_ratios": [1.0, 1.12],
            "hspace": 0.34,
            "wspace": 0.22,
        }
    )

    condition_styles = {
        "ICIF-Router": {
            "entropy_df": real_ent,
            "eval_df": real_evl,
            "color": "black",
            "linestyle": "-",
            "linewidth": 2.3,
            "bar_alpha": 0.92,
        },
        "Shuffled embedding control": {
            "entropy_df": shuf_ent,
            "eval_df": shuf_evl,
            "color": "#8A8A8A",
            "linestyle": "--",
            "linewidth": 2.0,
            "bar_alpha": 0.90,
        },
        "Task-prior control": {
            "entropy_df": prior_ent,
            "eval_df": prior_evl,
            "color": "#D97706",
            "linestyle": ":",
            "linewidth": 2.2,
            "bar_alpha": 0.92,
        },
    }

    top_titles = [
        "(A) Clinical Diagnosis",
        "(B) Clinical Reasoning",
        "(C) Medical Question Answering",
    ]

    bottom_titles = [
        "(D) Clinical Diagnosis",
        "(E) Clinical Reasoning",
        "(F) Medical Question Answering",
    ]

    # ------------------------------------------------------------------
    # Top row: entropy trajectories
    # ------------------------------------------------------------------
    for col, task in enumerate(TASKS):
        ax = axes[0, col]
        task_color = TASK_COLORS[task]

        for condition_name, style in condition_styles.items():
            df = style["entropy_df"]
            if df.empty:
                continue

            mask = df["task"] == task
            if "mode" in df.columns:
                mask &= df["mode"] == "policy"

            d = df[mask].sort_values("step")
            if d.empty or "entropy" not in d.columns:
                continue

            ax.plot(
                d["step"].values,
                smooth(d["entropy"].values),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=condition_name,
                zorder=3,
            )

        ax.axhline(
            np.log(6),
            color="#B5B5B5",
            linestyle="-.",
            linewidth=1.5,
            label="Max entropy / Uniform (1/6)",
            zorder=2,
        )

        ax.set_title(
            top_titles[col],
            fontsize=18,
            fontweight="bold",
            color=task_color,
            pad=8,
        )
        ax.set_xlabel("Training step", fontsize=15)
        if col == 0:
            ax.set_ylabel(r"Policy entropy $H[\pi]$", fontsize=15)
        else:
            ax.set_ylabel("")

        ax.set_ylim(0, 1.9)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.25)

    # ------------------------------------------------------------------
    # Bottom row: held-out greedy routing distributions
    # ------------------------------------------------------------------
    K = 6
    x = np.arange(K)

    # Wider bars than before
    bar_width = 0.28

    for col, task in enumerate(TASKS):
        ax = axes[1, col]
        task_color = TASK_COLORS[task]
        labels = FUNCTION_LABELS[task]

        def routing_freqs(evl_df):
            if evl_df.empty:
                return np.zeros(K)
            data = evl_df[evl_df["task"] == task]
            if data.empty:
                return np.zeros(K)
            counts = data["selected_function"].value_counts().reindex(labels, fill_value=0)
            freqs = counts.values.astype(float)
            total = freqs.sum()
            return freqs / total if total > 0 else freqs

        real_f = routing_freqs(real_evl)
        shuf_f = routing_freqs(shuf_evl)
        prior_f = routing_freqs(prior_evl)

        bar_series = [
            ("ICIF-Router", real_f, -bar_width),
            ("Shuffled embedding control", shuf_f, 0.0),
            ("Task-prior control", prior_f, bar_width),
        ]

        for condition_name, freqs, offset in bar_series:
            style = condition_styles[condition_name]
            ax.bar(
                x + offset,
                freqs,
                width=bar_width,
                color=style["color"],
                alpha=style["bar_alpha"],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

        ax.axhline(
            1 / K,
            color="#5A5A5A",
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )

        ax.set_title(
            bottom_titles[col],
            fontsize=18,
            fontweight="bold",
            color=task_color,
            pad=8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i+1}" for i in range(K)], fontsize=14)
        ax.set_xlabel("Instruction function", fontsize=15)

        if col == 0:
            ax.set_ylabel("Selection proportion", fontsize=15)
        else:
            ax.set_ylabel("")

        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

        # Optional: annotate only large bars to avoid clutter
        for freqs, offset, color in [
            (real_f, -bar_width, "black"),
            (shuf_f, 0.0, "#666666"),
            (prior_f, bar_width, "#B85C00"),
        ]:
            for i, val in enumerate(freqs):
                if val >= 0.35:
                    ax.text(
                        x[i] + offset,
                        val + 0.025,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        color=color,
                        fontweight="bold",
                    )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    legend_handles = [
        plt.Line2D(
            [0], [0],
            color="black",
            linewidth=2.3,
            linestyle="-",
            label="ICIF-Router"
        ),
        plt.Line2D(
            [0], [0],
            color="#8A8A8A",
            linewidth=2.0,
            linestyle="--",
            label="Shuffled embedding control"
        ),
        plt.Line2D(
            [0], [0],
            color="#D97706",
            linewidth=2.2,
            linestyle=":",
            label="Task-prior control"
        ),
        plt.Line2D(
            [0], [0],
            color="#5A5A5A",
            linewidth=1.5,
            linestyle="--",
            label="Uniform reference"
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        fontsize=13,
        bbox_to_anchor=(0.5, 1.01),
    )

    # Tight but safe margins
    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        left=0.06,
        right=0.985,
        hspace=0.34,
        wspace=0.22,
    )

    save_fig(fig, figures_dir, "fig7_real_vs_shuffled")


# ===========================================================================
# Summary table — for paper Table 1
# ===========================================================================

def print_summary_table(sel, evl, analysis, output_dir):
    print("\n" + "="*60)
    print("SUMMARY TABLE — for paper")
    print("="*60)

    lines = ["ICIP-Router: Function Distribution Summary\n"]

    if analysis:
        for task in TASKS:
            if task not in analysis:
                continue
            lines.append(f"\n{task} (training-time policy phase):")
            funcs = analysis[task].get("functions", {})
            for fname, fdata in funcs.items():
                lines.append(f"  {fname:<35} {fdata['selection_pct']:>6.1f}%  "
                             f"perf={fdata['performance_score']:.4f}")
            top3 = analysis[task].get("top_3", [])
            lines.append(f"  Top-3: {', '.join(t['function'] for t in top3)}")

    if evl is not None and not evl.empty:
        lines.append("\n\nEval-time greedy routing (held-out validation set):")
        for task in TASKS:
            data   = evl[evl["task"] == task]
            labels = FUNCTION_LABELS[task]
            counts = data["selected_function"].value_counts().reindex(labels, fill_value=0)
            total  = counts.sum()
            lines.append(f"\n  {task} (n={total}):")
            for lbl, cnt in counts.items():
                pct = 100 * cnt / total if total > 0 else 0
                lines.append(f"    {lbl:<35} {cnt:>5}  ({pct:.1f}%)")

    table_text = "\n".join(lines)
    print(table_text)

    table_path = os.path.join(output_dir, "summary_table.txt")
    with open(table_path, "w") as f:
        f.write(table_text)
    print(f"\nSummary table saved -> {table_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="ICIP-Router post-training analysis")
    parser.add_argument("--output_dir",     required=True,
                        help="Real experiment output directory")
    parser.add_argument("--shuffled_dir",   default=None,
                        help="Shuffled-encoder control output directory (optional)")
    parser.add_argument("--task_prior_dir", default=None,
                        help="Task-prior control output directory (optional)")
    args = parser.parse_args()

    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    sel, ent, rew, evl, stats, analysis = load_all(args.output_dir)

    fig_function_distribution(sel, figures_dir)
    fig_entropy_over_training(ent, figures_dir)
    fig_reward_trajectory(rew, figures_dir)
    fig_policy_confidence(ent, figures_dir)

    try:
        fig_probability_heatmap(sel, figures_dir)
    except ImportError:
        print("  Fig 5 skipped — install seaborn: pip install seaborn")

    fig_eval_routing(evl, figures_dir)

    fig_real_vs_shuffled(args.output_dir, args.shuffled_dir, args.task_prior_dir, figures_dir)

    print_summary_table(sel, evl, analysis, args.output_dir)

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()