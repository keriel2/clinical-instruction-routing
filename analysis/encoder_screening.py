#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_top1_cosine_six_functions.py
====================================

Purpose:
    Compute Top-1 cosine similarity between clinical inputs and the current
    six ICIP-Router instruction functions per task.

This is a lightweight encoder-selection / encoder-justification analysis.
It does NOT train Qwen and does NOT update the ICIP router.

Outputs:
    1. top1_cosine_by_task.csv
       - one row per embedding model x sample size x task

    2. top1_cosine_summary.csv
       - mean across tasks per embedding model x sample size

    3. top_function_distribution.csv
       - selected F1-F6 instruction-function distribution

    4. top1_cosine_summary_latex.txt
       - LaTeX-ready summary table

Example:
    python3 /home/share/henry/ICIP_ROUTER/compute_top1_cosine_six_functions.py \
        --dataset_path /home/share/henry/trainset.csv \
        --output_dir /home/share/henry/ICIP_ROUTER/QWEN/encoder_screening_sixfunc \
        --sample_sizes 250 750 \
        --seed 2026

If your dataset columns are not automatically detected:
    python3 compute_top1_cosine_six_functions.py \
        --dataset_path /home/share/henry/trainset.csv \
        --input_col input \
        --task_col task \
        --output_col output
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------
# Current ICIP-Router six-function instruction bank
# ---------------------------------------------------------------------

INSTRUCTION_FUNCTIONS: Dict[str, List[Dict[str, str]]] = {
    "Clinical Diagnosis": [
        {
            "function_id": "F1",
            "label": "Direct label",
            "instruction": (
                "Identify the most likely diagnosis from the clinical presentation. "
                "Return the diagnostic label directly and avoid unnecessary explanation."
            ),
        },
        {
            "function_id": "F2",
            "label": "Urgency / triage",
            "instruction": (
                "Assess the clinical presentation from an urgency or triage perspective. "
                "Prioritize whether the case suggests routine, urgent, or emergency-level concern."
            ),
        },
        {
            "function_id": "F3",
            "label": "Differential reasoning",
            "instruction": (
                "Consider the most plausible differential diagnoses and select the condition "
                "that best explains the symptoms."
            ),
        },
        {
            "function_id": "F4",
            "label": "Mechanism / pathophysiology",
            "instruction": (
                "Use the likely underlying mechanism or pathophysiology to infer the diagnosis "
                "from the reported symptoms."
            ),
        },
        {
            "function_id": "F5",
            "label": "Organ-system localization",
            "instruction": (
                "Localize the symptoms to the most relevant organ system and use that localization "
                "to identify the likely diagnosis."
            ),
        },
        {
            "function_id": "F6",
            "label": "Severity / acuity",
            "instruction": (
                "Evaluate the severity and acuity of the clinical presentation and infer the most "
                "likely diagnostic condition."
            ),
        },
    ],

    "Clinical Reasoning": [
        {
            "function_id": "F1",
            "label": "Deductive reasoning",
            "instruction": (
                "Apply deductive clinical reasoning to the case. Move from the clinical facts "
                "to the most defensible conclusion."
            ),
        },
        {
            "function_id": "F2",
            "label": "Hypothesis-driven reasoning",
            "instruction": (
                "Generate and evaluate clinical hypotheses for the case, then select the most "
                "supported explanation."
            ),
        },
        {
            "function_id": "F3",
            "label": "Pathophysiology-first reasoning",
            "instruction": (
                "Begin with the underlying pathophysiology and use it to explain the clinical "
                "findings and final conclusion."
            ),
        },
        {
            "function_id": "F4",
            "label": "Differential elimination",
            "instruction": (
                "Compare possible diagnoses or explanations and eliminate less likely alternatives "
                "before reaching the final conclusion."
            ),
        },
        {
            "function_id": "F5",
            "label": "Red-flag / safety reasoning",
            "instruction": (
                "Focus on red flags, safety-critical findings, and potentially dangerous conditions "
                "when reasoning through the case."
            ),
        },
        {
            "function_id": "F6",
            "label": "Uncertainty-aware reasoning",
            "instruction": (
                "Reason through the case while explicitly accounting for uncertainty, missing "
                "information, and alternative explanations."
            ),
        },
    ],

    "Medical Question Answering": [
        {
            "function_id": "F1",
            "label": "Direct factual",
            "instruction": (
                "Answer the medical question directly and factually using concise medical knowledge."
            ),
        },
        {
            "function_id": "F2",
            "label": "Mechanistic",
            "instruction": (
                "Answer the medical question by explaining the relevant biological or clinical mechanism."
            ),
        },
        {
            "function_id": "F3",
            "label": "Evidence-based",
            "instruction": (
                "Answer the medical question using an evidence-based clinical or biomedical explanation."
            ),
        },
        {
            "function_id": "F4",
            "label": "Patient-facing",
            "instruction": (
                "Answer the medical question in clear, patient-friendly language while preserving accuracy."
            ),
        },
        {
            "function_id": "F5",
            "label": "Risk / safety",
            "instruction": (
                "Answer the medical question with emphasis on risks, safety issues, warnings, or when "
                "medical attention is needed."
            ),
        },
        {
            "function_id": "F6",
            "label": "Management-oriented",
            "instruction": (
                "Answer the medical question with emphasis on management, treatment, monitoring, or "
                "practical next steps."
            ),
        },
    ],
}


# ---------------------------------------------------------------------
# Default embedding models
# ---------------------------------------------------------------------
# I recommend using encoder/sentence-embedding models for this analysis.
# BioMedLM and SciFive are generative/seq2seq models, so they are not included
# by default. You can still pass them manually using --model_paths.

DEFAULT_EMBEDDING_MODEL_PATHS = [
    "/home/share/henry/model_download/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "/home/share/henry/model_download/emilyalsentzerBio_ClinicalBERT",
    "/home/share/henry/model_download/S-BioBert-snli-multinli-stsb",
    "/home/share/henry/model_download/SapBERT-from-PubMedBERT-fulltext",
    "/home/share/henry/model_download/sentence-transformersall-MiniLM-L6-v2",
]


TASK_ALIASES = {
    "clinical diagnosis": "Clinical Diagnosis",
    "diagnosis": "Clinical Diagnosis",
    "cd": "Clinical Diagnosis",

    "clinical reasoning": "Clinical Reasoning",
    "reasoning": "Clinical Reasoning",
    "cr": "Clinical Reasoning",

    "medical question answering": "Medical Question Answering",
    "medical qa": "Medical Question Answering",
    "medqa": "Medical Question Answering",
    "mqa": "Medical Question Answering",
    "qa": "Medical Question Answering",
}


# ---------------------------------------------------------------------
# Helper classes and functions
# ---------------------------------------------------------------------

class HFMeanEncoder:
    """
    Fallback encoder for local Hugging Face models that are not directly
    loadable as SentenceTransformer models.

    Uses mean pooling over last hidden states.
    This is mainly a fallback. SentenceTransformer loading is preferred.
    """

    def __init__(self, model_path: str, device: Optional[str] = None, max_length: int = 256):
        self.model_path = model_path
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 16,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        all_embs = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                # Encoder-decoder models such as T5-style models
                if getattr(self.model.config, "is_encoder_decoder", False) and hasattr(self.model, "encoder"):
                    outputs = self.model.encoder(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                    )
                else:
                    outputs = self.model(**enc)

                hidden = outputs.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                all_embs.append(pooled.detach().cpu().numpy())

        return np.vstack(all_embs)


def clean_name(model_path: str) -> str:
    name = Path(model_path).name
    name = name.replace("sentence-transformers", "")
    name = name.replace("emilyalsentzer", "")
    name = name.strip("/ ")
    return name


def normalize_task_name(task_value: str) -> Optional[str]:
    if pd.isna(task_value):
        return None
    key = str(task_value).strip().lower()
    key = re.sub(r"[_\-]+", " ", key)
    key = re.sub(r"\s+", " ", key)
    return TASK_ALIASES.get(key, str(task_value).strip())


def auto_detect_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not auto-detect {kind} column. "
        f"Tried {candidates}. Found columns: {list(df.columns)}. "
        f"Pass it manually using --{kind}_col."
    )


def load_embedding_model(model_path: str):
    """
    Try loading as SentenceTransformer.
    If that fails, rebuild SentenceTransformer using Transformer + mean pooling.
    If that also fails, use generic Hugging Face mean pooling.
    """
    print(f"\nLoading embedding model: {model_path}")

    try:
        model = SentenceTransformer(model_path)
        print("  Loaded as SentenceTransformer.")
        return model
    except Exception as e:
        print(f"  SentenceTransformer direct load failed: {e}")

    try:
        word_emb = models.Transformer(model_path)
        pooling = models.Pooling(
            word_emb.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[word_emb, pooling])
        print("  Loaded as Transformer + mean pooling.")
        return model
    except Exception as e:
        print(f"  SentenceTransformer fallback failed: {e}")

    try:
        model = HFMeanEncoder(model_path)
        print("  Loaded as generic Hugging Face mean encoder.")
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load model {model_path}: {e}")


def compute_top1_for_task(
    emb_model,
    task: str,
    inputs: List[str],
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cosine similarity between inputs and six task instruction functions.

    Returns:
        per_example_df
        distribution_df
    """

    funcs = INSTRUCTION_FUNCTIONS[task]
    instruction_texts = [f"{x['label']}. {x['instruction']}" for x in funcs]

    # Normalize embeddings, then cosine similarity = dot product.
    inst_emb = emb_model.encode(
        instruction_texts,
        normalize_embeddings=True,
        batch_size=min(batch_size, 16),
        show_progress_bar=False,
    )

    input_emb = emb_model.encode(
        inputs,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    sims = np.matmul(input_emb, inst_emb.T)  # [N, 6]

    # Sort descending for Top-1 and Top-2 margin.
    sorted_idx = np.argsort(-sims, axis=1)
    top1_idx = sorted_idx[:, 0]
    top2_idx = sorted_idx[:, 1]

    rows = []
    for i in range(len(inputs)):
        f = funcs[int(top1_idx[i])]
        rows.append({
            "task": task,
            "example_index": i,
            "top1_function_id": f["function_id"],
            "top1_function_label": f["label"],
            "top1_cosine": float(sims[i, top1_idx[i]]),
            "top2_cosine": float(sims[i, top2_idx[i]]),
            "top1_top2_margin": float(sims[i, top1_idx[i]] - sims[i, top2_idx[i]]),
            "all_cosines_json": json.dumps(
                {funcs[j]["function_id"] + "_" + funcs[j]["label"]: float(sims[i, j]) for j in range(6)},
                ensure_ascii=False,
            ),
        })

    per_example_df = pd.DataFrame(rows)

    dist_rows = []
    total = len(per_example_df)
    for f in funcs:
        count = int((per_example_df["top1_function_id"] == f["function_id"]).sum())
        dist_rows.append({
            "task": task,
            "function_id": f["function_id"],
            "function_label": f["label"],
            "count": count,
            "proportion": count / total if total else 0.0,
        })

    distribution_df = pd.DataFrame(dist_rows)

    return per_example_df, distribution_df


def make_latex_summary(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Create a compact LaTeX summary table with mean Top-1 cosine similarity.
    """
    if summary_df.empty:
        return

    # Keep useful columns
    cols = [
        "embedding_model",
        "sample_size_per_task",
        "mean_top1_cosine_across_tasks",
        "mean_top1_top2_margin_across_tasks",
    ]
    tmp = summary_df[cols].copy()
    tmp["mean_top1_cosine_across_tasks"] = tmp["mean_top1_cosine_across_tasks"].map(lambda x: f"{x:.3f}")
    tmp["mean_top1_top2_margin_across_tasks"] = tmp["mean_top1_top2_margin_across_tasks"].map(lambda x: f"{x:.3f}")

    latex = tmp.to_latex(
        index=False,
        escape=True,
        caption="Preliminary embedding-backbone screening using the current six ICIP-Router instruction functions. Mean Top-1 cosine similarity is averaged across Clinical Diagnosis, Clinical Reasoning, and Medical Question Answering.",
        label="tab:encoder_screening_sixfunc",
        column_format="lccc",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/share/henry/trainset.csv",
        help="CSV file containing task and input columns.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/share/henry/ICIP_ROUTER/QWEN/encoder_screening_sixfunc",
        help="Directory where output CSV files will be saved.",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="*",
        default=DEFAULT_EMBEDDING_MODEL_PATHS,
        help="Local embedding model paths. Defaults to recommended encoder models.",
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[250, 750],
        help="Balanced sample sizes per task. Default: 250 and 750.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--input_col", type=str, default=None)
    parser.add_argument("--task_col", type=str, default=None)
    parser.add_argument("--output_col", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("ICIP-Router encoder screening using six instruction functions")
    print("=" * 90)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output : {args.output_dir}")
    print(f"Sample sizes per task: {args.sample_sizes}")
    print(f"Seed: {args.seed}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(args.dataset_path)

    input_col = args.input_col or auto_detect_column(
        df,
        ["raw_input", "input", "text", "question", "prompt"],
        "input",
    )
    task_col = args.task_col or auto_detect_column(
        df,
        ["task", "task_family", "label_task", "dataset_task"],
        "task",
    )

    print(f"Using input column: {input_col}")
    print(f"Using task column : {task_col}")

    df = df.copy()
    df["__task__"] = df[task_col].apply(normalize_task_name)
    df["__input__"] = df[input_col].astype(str).fillna("").str.strip()

    df = df[df["__task__"].isin(INSTRUCTION_FUNCTIONS.keys())]
    df = df[df["__input__"].str.len() > 0]

    if df.empty:
        raise ValueError("No rows matched the expected task names after normalization.")

    print("\nRows by task after normalization:")
    print(df["__task__"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Run screening
    # ------------------------------------------------------------------
    all_task_rows = []
    all_dist_rows = []
    all_example_rows = []

    for model_path in args.model_paths:
        model_name = clean_name(model_path)

        try:
            emb_model = load_embedding_model(model_path)
        except Exception as e:
            print(f"\nERROR: Skipping model {model_path}")
            print(f"Reason: {e}")
            continue

        for n in args.sample_sizes:
            print(f"\n--- Model: {model_name} | n={n} per task ---")

            sampled_parts = []
            for task, g in df.groupby("__task__"):
                take = min(n, len(g))
                sampled = g.sample(n=take, random_state=args.seed)
                sampled_parts.append(sampled)

            sample_df = pd.concat(sampled_parts, ignore_index=True)

            for task in INSTRUCTION_FUNCTIONS.keys():
                sub = sample_df[sample_df["__task__"] == task].copy()
                if sub.empty:
                    continue

                inputs = sub["__input__"].tolist()

                per_example_df, dist_df = compute_top1_for_task(
                    emb_model=emb_model,
                    task=task,
                    inputs=inputs,
                    batch_size=args.batch_size,
                )

                per_example_df.insert(0, "embedding_model", model_name)
                per_example_df.insert(1, "model_path", model_path)
                per_example_df.insert(2, "sample_size_per_task", n)

                dist_df.insert(0, "embedding_model", model_name)
                dist_df.insert(1, "model_path", model_path)
                dist_df.insert(2, "sample_size_per_task", n)

                all_example_rows.append(per_example_df)
                all_dist_rows.append(dist_df)

                all_task_rows.append({
                    "embedding_model": model_name,
                    "model_path": model_path,
                    "sample_size_per_task": n,
                    "task": task,
                    "actual_n": len(inputs),
                    "mean_top1_cosine": float(per_example_df["top1_cosine"].mean()),
                    "std_top1_cosine": float(per_example_df["top1_cosine"].std(ddof=1)),
                    "median_top1_cosine": float(per_example_df["top1_cosine"].median()),
                    "mean_top1_top2_margin": float(per_example_df["top1_top2_margin"].mean()),
                    "dominant_function_id": dist_df.sort_values("proportion", ascending=False).iloc[0]["function_id"],
                    "dominant_function_label": dist_df.sort_values("proportion", ascending=False).iloc[0]["function_label"],
                    "dominant_function_proportion": float(dist_df["proportion"].max()),
                })

                print(
                    f"{task:30s} | mean Top-1 cosine={per_example_df['top1_cosine'].mean():.3f} "
                    f"| margin={per_example_df['top1_top2_margin'].mean():.3f} "
                    f"| dominant={dist_df.sort_values('proportion', ascending=False).iloc[0]['function_id']} "
                    f"({dist_df['proportion'].max():.2%})"
                )

        # cleanup
        del emb_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    task_df = pd.DataFrame(all_task_rows)

    if not all_dist_rows:
        raise RuntimeError("No results were produced. Check dataset/model paths.")

    dist_df = pd.concat(all_dist_rows, ignore_index=True)
    example_df = pd.concat(all_example_rows, ignore_index=True)

    summary_df = (
        task_df
        .groupby(["embedding_model", "model_path", "sample_size_per_task"], as_index=False)
        .agg(
            mean_top1_cosine_across_tasks=("mean_top1_cosine", "mean"),
            mean_top1_top2_margin_across_tasks=("mean_top1_top2_margin", "mean"),
            mean_dominant_function_proportion=("dominant_function_proportion", "mean"),
            tasks_covered=("task", "nunique"),
        )
        .sort_values(["sample_size_per_task", "mean_top1_cosine_across_tasks"], ascending=[True, False])
    )

    task_csv = os.path.join(args.output_dir, "top1_cosine_by_task.csv")
    summary_csv = os.path.join(args.output_dir, "top1_cosine_summary.csv")
    dist_csv = os.path.join(args.output_dir, "top_function_distribution.csv")
    examples_csv = os.path.join(args.output_dir, "top1_cosine_per_example.csv")
    latex_txt = os.path.join(args.output_dir, "top1_cosine_summary_latex.txt")

    task_df.to_csv(task_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    dist_df.to_csv(dist_csv, index=False)
    example_df.to_csv(examples_csv, index=False)
    make_latex_summary(summary_df, latex_txt)

    print("\n" + "=" * 90)
    print("Saved outputs")
    print("=" * 90)
    print(f"Task-level metrics       : {task_csv}")
    print(f"Summary metrics          : {summary_csv}")
    print(f"Function distribution    : {dist_csv}")
    print(f"Per-example similarities : {examples_csv}")
    print(f"LaTeX summary table      : {latex_txt}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()