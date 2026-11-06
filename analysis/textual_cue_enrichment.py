"""
textual_cue_enrichment.py — v2
================================
Fixes applied:
  1. Uses full input text from val_split_filtered.csv (not raw_input_preview)
  2. CR dominant functions corrected to Differential elimination + Hypothesis-driven
  3. Stop word list reduced — clinical terms retained
  4. Row-order alignment between routing and validation CSVs
  5. Real encoder condition only

Run:
    python textual_cue_enrichment.py \
        --eval_routing /home/share/henry/ICIP_ROUTER/QWEN/real/icip_eval_routing.csv \
        --val_data     /home/share/henry/ICIP_ROUTER/QWEN/real/datasets/val_split_filtered.csv \
        --output_dir   /home/share/henry/ICIP_ROUTER/QWEN/real \
        --top_n        10
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from collections import Counter

# ---------------------------------------------------------------------------
# Minimal stop words — keep clinical terms
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
    "who", "whom", "how", "when", "where", "why", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "as", "if", "than", "then",
    "my", "your", "his", "her", "its", "our", "their", "me", "him", "us",
    "them", "there", "here", "all", "any", "each", "every", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "too", "very",
    "just", "about", "above", "after", "before", "between", "during",
    "into", "through", "under", "up", "down", "over", "out", "also",
    "however", "therefore", "thus",
    # NOTE: clinical terms like diagnosis, treatment, symptoms, signs
    # are intentionally kept — they may be meaningful routing cues
}

# Dominant functions — Qwen real encoder held-out routing
DOMINANT_FUNCTIONS = {
    "Clinical Diagnosis": [
        "Differential reasoning",
        "Severity / acuity",
    ],
    "Clinical Reasoning": [
        "Differential elimination",
        "Hypothesis-driven",
    ],
    "Medical Question Answering": [
        "Direct factual",
        "Mechanistic",
    ],
}


def tokenize(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def get_bigrams(tokens: list) -> list:
    return [
        f"{tokens[i]} {tokens[i+1]}"
        for i in range(len(tokens) - 1)
        if tokens[i] not in STOP_WORDS and tokens[i+1] not in STOP_WORDS
    ]


def log_odds_enrichment(fg_counts, bg_counts, fg_total, bg_total,
                        alpha=0.5):
    all_terms = set(fg_counts) | set(bg_counts)
    scores = {}
    for term in all_terms:
        fg = fg_counts.get(term, 0) + alpha
        bg = bg_counts.get(term, 0) + alpha
        fg_norm = fg / (fg_total + alpha * len(all_terms))
        bg_norm = bg / (bg_total + alpha * len(all_terms))
        scores[term] = float(np.log(fg_norm / bg_norm))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_routing", required=True)
    parser.add_argument("--val_data",     required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--top_n",        type=int, default=10)
    parser.add_argument("--min_fg_count", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and row-order align routing with full input text
    # -----------------------------------------------------------------------
    print(f"Loading eval routing: {args.eval_routing}")
    routing = pd.read_csv(args.eval_routing).reset_index(drop=True)
    print(f"  {len(routing)} rows | columns: {list(routing.columns)}")

    print(f"Loading val data: {args.val_data}")
    val = pd.read_csv(args.val_data).reset_index(drop=True)
    print(f"  {len(val)} rows | columns: {list(val.columns)}")

    if len(routing) != len(val):
        raise ValueError(
            f"Row count mismatch: routing={len(routing)}, val={len(val)}. "
            f"Use filtered versions of both files."
        )

    # Find full input text column
    candidate_cols = ["input_text", "input", "question", "text", "raw_input"]
    input_col = next((c for c in candidate_cols if c in val.columns), None)
    if input_col is None:
        raise ValueError(
            f"No input text column found. Available: {list(val.columns)}"
        )
    print(f"  Using input column: '{input_col}'")

    routing["full_input_text"] = val[input_col].astype(str).values

    # -----------------------------------------------------------------------
    # Compute enrichment per dominant function
    # -----------------------------------------------------------------------
    results  = {}
    latex_rows = []

    for task, functions in DOMINANT_FUNCTIONS.items():
        task_routing = routing[routing["task"] == task].copy()
        print(f"\n{task} (n={len(task_routing)}):")
        results[task] = {}

        for func in functions:
            fg_df = task_routing[task_routing["selected_function"] == func]
            bg_df = task_routing[task_routing["selected_function"] != func]

            print(f"  {func}: fg={len(fg_df)}, bg={len(bg_df)}")

            if len(fg_df) < 5:
                print(f"    Skipped — too few foreground examples ({len(fg_df)})")
                continue

            fg_tokens = Counter()
            bg_tokens = Counter()

            for _, row in fg_df.iterrows():
                text = str(row["full_input_text"])
                toks = tokenize(text)
                fg_tokens.update(toks)
                fg_tokens.update(get_bigrams(toks))

            for _, row in bg_df.iterrows():
                text = str(row["full_input_text"])
                toks = tokenize(text)
                bg_tokens.update(toks)
                bg_tokens.update(get_bigrams(toks))

            fg_total = sum(fg_tokens.values())
            bg_total = sum(bg_tokens.values())

            scores = log_odds_enrichment(
                fg_tokens, bg_tokens, fg_total, bg_total
            )

            # Filter: must appear at least min_fg_count times in foreground
            scores = {
                k: v for k, v in scores.items()
                if fg_tokens.get(k, 0) >= args.min_fg_count
            }

            top_pos = sorted(
                [(k, v) for k, v in scores.items() if v > 0],
                key=lambda x: x[1], reverse=True
            )[:args.top_n]

            results[task][func] = {
                "fg_n":        len(fg_df),
                "bg_n":        len(bg_df),
                "top_enriched": [(t, round(s, 3)) for t, s in top_pos],
            }

            top_str = ", ".join(t for t, _ in top_pos[:8])
            print(f"    Top cues: {top_str}")

            latex_rows.append({
                "task":     task,
                "function": func.replace(" / ", "/"),
                "fg_n":     len(fg_df),
                "top_cues": ", ".join(t for t, _ in top_pos[:8]),
            })

    # Save JSON
    json_path = os.path.join(args.output_dir, "textual_cue_enrichment.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Generate LaTeX
    latex = generate_latex(latex_rows)
    tex_path = os.path.join(args.output_dir, "textual_cue_enrichment_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"Saved: {tex_path}")
    print("\n=== LaTeX table ===")
    print(latex)


def latex_escape(text):
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def generate_latex(rows):
    task_short = {
        "Clinical Diagnosis":         "CD",
        "Clinical Reasoning":         "CR",
        "Medical Question Answering": "MQA",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        (r"\caption{Textual cue enrichment for dominant held-out routed "
         r"instruction functions (Qwen2.5-7B-Instruct, real encoder). "
         r"Terms are ranked by log-odds enrichment relative to other "
         r"validation examples from the same task (minimum 3 occurrences "
         r"in foreground; bigrams included). This analysis is a semantic "
         r"plausibility check and has not been clinician-verified.}"),
        r"\label{tab:semantic_enrichment}",
        r"\begin{tabular}{llcp{9.5cm}}",
        r"\toprule",
        (r"\textbf{Task} & \textbf{Routed function} & "
         r"\textbf{$n$} & \textbf{Top enriched cues (log-odds ranked)} \\"),
        r"\midrule",
    ]
    prev_task = None
    for i, row in enumerate(rows):
        task_cell = task_short.get(row["task"], row["task"]) \
                    if row["task"] != prev_task else ""
        prev_task = row["task"]
        cues = latex_escape(row["top_cues"])
        func = latex_escape(row["function"])
        lines.append(
            f"{task_cell} & {func} & {row['fg_n']} & "
            f"\\textit{{{cues}}} \\\\"
        )
        if i < len(rows) - 1 and rows[i+1]["task"] != row["task"]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()