"""
filter_clinical_data.py — v3
==============================
Minimal blocklist only. Only removes examples that are clearly non-medical.
Default: KEEP everything unless explicitly blocked.

Removes ONLY:
  1. Philosophy questions (Nagel, Kant etc.)
  2. Grammar/adjective order questions
  3. Logic truth-teller/liar puzzles (Jamey tells the truth...)
  4. Gypsum/dental material setting time questions

Keeps everything else including all MedQuAD symptom questions.
"""

import argparse
import re
import pandas as pd


# Exact string fragments that definitively identify non-clinical content
HARD_BLOCK_TERMS = [
    # Philosophy
    "according to nagel",
    "according to kant",
    "according to descartes",
    "according to hume",
    "according to aristotle",
    "according to plato",
    "nagel thinks",
    "nagel argues",
    "epistemology",
    # Grammar puzzles
    "correct adjective order",
    "adjective order",
    "which sentence has the correct",
    "which sentence is grammatically",
    # Gypsum/dental material
    "gypsum product",
    "mixing gypsum",
    "setting time of plaster",
    "setting time if it",
]

# Regex patterns for logic puzzles only
LOGIC_PUZZLE_PATTERNS = [
    # Truth-teller/liar puzzle format: "Name tells the truth. Name2 says Name lies."
    r"(tells the truth|always lies)\.\s+\w+ says \w+ (tells|lies)",
    # Named puzzle characters
    r"\b(jamey|delfina|vina|teressa|crista|bernita|delbert|phoebe|sherrie|elanor|millicent)\b.{0,50}(lies|truth)",
]


def should_remove(input_text: str, task: str) -> bool:
    """Return True only if clearly non-clinical."""
    t = str(input_text).lower().strip()

    # Check hard block terms (exact match)
    for term in HARD_BLOCK_TERMS:
        if term in t:
            return True

    # Check logic puzzle patterns
    for pattern in LOGIC_PUZZLE_PATTERNS:
        if re.search(pattern, t, re.I):
            return True

    # Everything else: KEEP
    return False


def filter_dataset(df: pd.DataFrame) -> tuple:
    df = df.copy()
    input_col = "input" if "input" in df.columns else "raw_input"

    mask    = ~df.apply(lambda r: should_remove(r[input_col], r["task"]), axis=1)
    kept    = df[mask].reset_index(drop=True)
    removed = df[~mask].reset_index(drop=True)
    return kept, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--removed", default=None)
    parser.add_argument("--report",  action="store_true")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Total: {len(df)} | Tasks: {df['task'].value_counts().to_dict()}")

    kept, removed = filter_dataset(df)

    print(f"\nAfter filtering:")
    print(f"  Kept:    {len(kept)} ({100*len(kept)/len(df):.1f}%)")
    print(f"  Removed: {len(removed)} ({100*len(removed)/len(df):.1f}%)")

    print(f"\n  Kept by task:")
    for task, n in kept["task"].value_counts().items():
        orig = len(df[df["task"] == task])
        print(f"    {task}: {n}/{orig} ({100*n/orig:.1f}%)")

    if args.report and len(removed) > 0:
        input_col = "input" if "input" in removed.columns else "raw_input"
        print(f"\n  Removed ({len(removed)} total):")
        for _, row in removed.iterrows():
            print(f"    [{row['task']}] {str(row.get(input_col,''))[:120]}")

    kept.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    if args.removed:
        removed.to_csv(args.removed, index=False)
        print(f"Saved removed: {args.removed}")


if __name__ == "__main__":
    main()