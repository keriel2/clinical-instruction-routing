"""
extract_case_examples.py — fixed version
=========================================
Fixes vs previous:
  1. select_examples() now returns the DataFrame (was missing return statement)
  2. Inner for loop indentation fixed (was outside the func loop)
  3. raw_data has no example_id — lookup falls back to raw_input_preview correctly
  4. is_acceptable_example() filters non-clinical examples
  5. latex_escape() handles backslash first

Run:
    python extract_case_examples.py \
        --eval_csv /home/share/henry/ICIP_ROUTER/QWEN/real/icip_eval_routing.csv \
        --raw_data /home/share/henry/trainset.csv \
        --output_dir /home/share/henry/ICIP_ROUTER/QWEN/real \
        --n_per_func 2
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd

DOMINANT_FUNCTIONS: Dict[str, List[str]] = {
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

ROUTING_RATIONALE: Dict[Tuple[str, str], str] = {
    ("Clinical Diagnosis", "Differential reasoning"):
        "The selected function frames the case as a diagnostic choice among plausible alternatives.",
    ("Clinical Diagnosis", "Severity / acuity"):
        "The selected function emphasizes clinically significant or acuity-related features before diagnosis.",
    ("Clinical Reasoning", "Differential elimination"):
        "The selected function supports structured comparison and exclusion of competing explanations.",
    ("Clinical Reasoning", "Hypothesis-driven"):
        "The selected function organizes the case around competing hypotheses and evidence evaluation.",
    ("Medical Question Answering", "Direct factual"):
        "The selected function is appropriate for a question seeking a concise factual answer.",
    ("Medical Question Answering", "Mechanistic"):
        "The selected function is appropriate for a question asking how or why a medical process occurs.",
}


def truncate(text: str, n: int = 120) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())
    return text[:n] + "..." if len(text) > n else text


def latex_escape(text: str) -> str:
    # Backslash MUST be first
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    result = []
    for ch in str(text):
        replaced = False
        for old, new in replacements:
            if ch == old:
                result.append(new)
                replaced = True
                break
        if not replaced:
            result.append(ch)
    return "".join(result)


def get_probability(row: pd.Series) -> float:
    if "max_prob" in row and pd.notna(row["max_prob"]):
        try:
            return float(row["max_prob"])
        except Exception:
            pass
    if "prob_vector" in row and pd.notna(row["prob_vector"]):
        try:
            return float(max(json.loads(row["prob_vector"])))
        except Exception:
            pass
    return float("nan")


def is_acceptable(task: str, text: str) -> bool:
    """Filter out non-clinical or trivially short examples."""
    t = str(text).lower()

    # Remove logic puzzles and non-medical content
    blocklist = [
        "tells the truth", "lies", "says", "knight", "knave",
        "michael says", "sherrie", "elaner", "elanor", "millicent",
        "jim tells", "ryan says", "roundtable", "if a then b",
    ]
    if any(b in t for b in blocklist):
        return False

    if len(t.split()) < 6:
        return False

    if task == "Clinical Reasoning":
        clinical = [
            "patient", "presents", "year-old", "diagnosis", "symptom",
            "pain", "fever", "disease", "syndrome", "treatment",
            "history", "examination", "laboratory", "imaging", "physician",
        ]
        return any(c in t for c in clinical)

    if task == "Clinical Diagnosis":
        symptoms = [
            "pain", "fever", "cough", "swelling", "nausea", "vomiting",
            "rash", "bleeding", "headache", "dizziness", "stiffness",
            "fatigue", "weight loss", "palpitations", "shortness of breath",
            "chest", "abdomen", "joint", "skin", "muscle",
        ]
        return any(s in t for s in symptoms)

    if task == "Medical Question Answering":
        return "?" in t or any(
            w in t for w in [
                "what", "how", "why", "treatment", "inherited",
                "caused by", "symptoms of", "diagnosed", "prevent",
            ]
        )

    return True


def select_examples(evl: pd.DataFrame, n_per_func: int,
                    truncate_chars: int) -> pd.DataFrame:
    """
    FIX: now correctly returns a DataFrame.
    FIX: inner loop correctly indented inside func loop.
    """
    rows = []

    for task, functions in DOMINANT_FUNCTIONS.items():
        task_evl = evl[evl["task"] == task].copy()
        task_evl["confidence"] = task_evl.apply(get_probability, axis=1)
        task_evl = task_evl.sort_values("confidence", ascending=False)

        for func in functions:
            func_evl = task_evl[task_evl["selected_function"] == func]
            if func_evl.empty:
                print(f"  WARNING: no examples for {task} / {func}")
                continue

            kept = 0
            for _, row in func_evl.iterrows():     # FIX: correctly inside func loop
                preview = str(row.get("raw_input_preview", ""))
                if not is_acceptable(task, preview):
                    continue

                rows.append({
                    "task":              task,
                    "input_preview":     truncate(preview, truncate_chars),
                    "selected_function": func,
                    "confidence":        f"{row['confidence']:.2f}",
                    "routing_rationale": ROUTING_RATIONALE.get(
                        (task, func),
                        "The selected function reflects the router's highest-probability framing."
                    ),
                })
                kept += 1
                if kept >= n_per_func:
                    break

            if kept == 0:
                print(f"  WARNING: no acceptable clinical examples for {task} / {func}")
            elif kept < n_per_func:
                print(f"  WARNING: only {kept}/{n_per_func} examples found for {task} / {func}")

    return pd.DataFrame(rows)   # FIX: return statement was missing


def generate_latex(df: pd.DataFrame) -> str:
    task_short = {
        "Clinical Diagnosis":         "CD",
        "Clinical Reasoning":         "CR",
        "Medical Question Answering": "MQA",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        (r"\caption{Qualitative examples of ICIP-Router instruction-function routing "
         r"on held-out validation data. Examples are selected from high-confidence "
         r"greedy routing decisions for the two dominant validation-time functions per "
         r"task (Fig.~\ref{fig:eval}). Confidence is the maximum action probability. "
         r"Rationales are interpretive and have not been clinician-verified.}"),
        r"\label{tab:case_examples}",
        r"\begin{tabular}{p{1.2cm}p{5.8cm}p{2.7cm}cp{5.0cm}}",
        r"\toprule",
        (r"\textbf{Task} & \textbf{Input preview} & \textbf{Selected function} "
         r"& \textbf{Conf.} & \textbf{Interpretive routing rationale} \\"),
        r"\midrule",
    ]

    tasks = list(df["task"].drop_duplicates())
    for ti, task in enumerate(tasks):
        trows = df[df["task"] == task]
        for ri, (_, row) in enumerate(trows.iterrows()):
            task_cell = task_short.get(task, task) if ri == 0 else ""
            lines.append(
                f"{task_cell} & "
                f"\\textit{{{latex_escape(row['input_preview'])}}} & "
                f"{latex_escape(row['selected_function'])} & "
                f"{latex_escape(row['confidence'])} & "
                f"{latex_escape(row['routing_rationale'])} \\\\"
            )
        if ti < len(tasks) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv",       required=True)
    parser.add_argument("--raw_data",       required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--n_per_func",     type=int, default=1)
    parser.add_argument("--truncate_chars", type=int, default=120)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading eval routing: {args.eval_csv}")
    evl = pd.read_csv(args.eval_csv)
    print(f"  {len(evl)} rows | columns: {list(evl.columns)}")

    # raw_data used only for context — no example_id in this dataset
    # so we rely on raw_input_preview from the eval CSV
    print(f"\nNote: raw_data has no example_id column — using raw_input_preview from eval CSV")

    df_out = select_examples(evl, args.n_per_func, args.truncate_chars)

    if df_out.empty:
        print("ERROR: No examples selected. Check filters in is_acceptable().")
        return

    csv_path = os.path.join(args.output_dir, "case_examples.csv")
    tex_path = os.path.join(args.output_dir, "table_case_examples.tex")

    df_out.to_csv(csv_path, index=False)
    latex = generate_latex(df_out)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(f"\n{len(df_out)} examples selected\n")
    print("=== PREVIEW ===")
    for _, r in df_out.iterrows():
        print(f"\n[{r['task']}] {r['selected_function']} (conf={r['confidence']})")
        print(f"  Input: {r['input_preview']}")
        print(f"  Rationale: {r['routing_rationale']}")
    print("\n=== LaTeX ===")
    print(latex)


if __name__ == "__main__":
    main()