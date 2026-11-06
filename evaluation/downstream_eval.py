#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
downstream_sanity_eval.py
=========================

Clean downstream sanity-check evaluation for ICIP-Router.

It compares:
  1. Real encoder final model
  2. Shuffled-embedding-control final model
  3. Task-prior-control final model

Default mode:
  Uses each condition's validation split and matching held-out greedy routing file:
    <run_dir>/datasets/val_split.csv
    <run_dir>/icip_eval_routing.csv

Important:
  For an external test set, do NOT reuse validation routing files.
  You must first export routing decisions for that external test set and then pass
  those route CSVs explicitly.

Metrics kept intentionally simple:
  - Clinical Diagnosis: normalized exact match, token-F1
  - Clinical Reasoning: ROUGE-L, token-F1
  - Medical QA: ROUGE-L, token-F1

Outputs:
  - downstream_sanity_generations.csv
  - downstream_sanity_metrics_long.csv
  - downstream_sanity_metrics_wide.csv

Example validation run:
  python3 /home/share/henry/ICIP_ROUTER/downstream_sanity_eval.py \
      --mode validation \
      --base_model_path /home/share/henry/model_download/Qwen2.5-7B-Instruct \
      --output_dir /home/share/henry/ICIP_ROUTER/QWEN/downstream_sanity_validation \
      --batch_size 2

Example external-test run after exporting external route CSVs:
  python3 /home/share/henry/ICIP_ROUTER/downstream_sanity_eval.py \
      --mode external \
      --external_eval_csv /home/share/henry/ablation_study/testset/standardized_test.csv \
      --real_route_csv /path/to/real_test_routes.csv \
      --shuffled_route_csv /path/to/shuffled_test_routes.csv \
      --taskprior_route_csv /path/to/taskprior_test_routes.csv \
      --base_model_path /home/share/henry/model_download/Qwen2.5-7B-Instruct \
      --output_dir /home/share/henry/ICIP_ROUTER/QWEN/downstream_sanity_external \
      --batch_size 2
"""

import os
import re
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    ROUGE_AVAILABLE = False

try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except Exception:
    BERTSCORE_AVAILABLE = False

# BERTScorer for Clinical Reasoning
_BERT_SCORER = None
def get_bert_scorer():
    global _BERT_SCORER
    if _BERT_SCORER is None and BERTSCORE_AVAILABLE:
        _BERT_SCORER = BERTScorer(
            model_type='/home/share/henry/model_download/emilyalsentzerBio_ClinicalBERT',
            num_layers=12,
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        )
    return _BERT_SCORER


# ---------------------------------------------------------------------
# Condition configuration
# ---------------------------------------------------------------------

RUNS = {
    "Real encoder": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/QWEN/real/final_model",
        "run_dir": "/home/share/henry/ICIP_ROUTER/QWEN/real",
        "val_csv": "/home/share/henry/ICIP_ROUTER/QWEN/real/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/QWEN/real/icip_eval_routing_filtered.csv",
    },
    "Shuffled encoder": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/QWEN/shuffled_control/final_model",
        "run_dir": "/home/share/henry/ICIP_ROUTER/QWEN/shuffled_control",
        "val_csv": "/home/share/henry/ICIP_ROUTER/QWEN/shuffled_control/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/QWEN/shuffled_control/icip_eval_routing_filtered.csv",
    },
    "Task-prior control": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/QWEN/task_prior_control/final_model",
        "run_dir": "/home/share/henry/ICIP_ROUTER/QWEN/task_prior_control",
        "val_csv": "/home/share/henry/ICIP_ROUTER/QWEN/task_prior_control/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/QWEN/task_prior_control/icip_eval_routing_filtered.csv",
    },
    "Example-shuffled control": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/QWEN/example_shuffled_control/final_model",
        "run_dir":   "/home/share/henry/ICIP_ROUTER/QWEN/example_shuffled_control",
        "val_csv":   "/home/share/henry/ICIP_ROUTER/QWEN/example_shuffled_control/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/QWEN/example_shuffled_control/icip_eval_routing_filtered.csv",
    },
}

# LLaMA-3.1-8B-Instruct runs
LLAMA_RUNS = {
    "LLaMA real encoder": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/LLAMA/real/final_model",
        "run_dir":   "/home/share/henry/ICIP_ROUTER/LLAMA/real",
        "val_csv":   "/home/share/henry/ICIP_ROUTER/LLAMA/real/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/LLAMA/real/icip_eval_routing_filtered.csv",
    },
    "LLaMA shuffled encoder": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/LLAMA/shuffled_control/final_model",
        "run_dir":   "/home/share/henry/ICIP_ROUTER/LLAMA/shuffled_control",
        "val_csv":   "/home/share/henry/ICIP_ROUTER/LLAMA/shuffled_control/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/LLAMA/shuffled_control/icip_eval_routing_filtered.csv",
    },
    "LLaMA task-prior": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/LLAMA/task_prior_control/final_model",
        "run_dir":   "/home/share/henry/ICIP_ROUTER/LLAMA/task_prior_control",
        "val_csv":   "/home/share/henry/ICIP_ROUTER/LLAMA/task_prior_control/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/LLAMA/task_prior_control/icip_eval_routing_filtered.csv",
    },
}

QWEN_RANDOM_RUNS = {
    "Random instruction baseline": {
        "model_dir": "/home/share/henry/ICIP_ROUTER/QWEN/random_baseline/final_model",
        "run_dir": "/home/share/henry/ICIP_ROUTER/QWEN/static_baseline",
        "val_csv": "/home/share/henry/ICIP_ROUTER/QWEN/random_baseline/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/QWEN/random_baseline/random_eval_routing_filtered.csv",
    },
}
QWEN_FIXED_RUN ={
    "static_base":{
        "model_dir": "/home/share/henry/ICIP_ROUTER/QWEN/static_baseline/final_model",
        "run_dir": "/home/share/henry/ICIP_ROUTER/QWEN/static_baseline",
        "val_csv": "/home/share/henry/ICIP_ROUTER/QWEN/static_baseline/datasets/val_split_filtered.csv",
        "route_csv": "/home/share/henry/ICIP_ROUTER/QWEN/static_baseline/fixed_eval_routing_filtered.csv",

    },
}
ICIP_TEMPLATES = {
    "Clinical Diagnosis": [
        "Based on these symptoms, provide ONLY the most likely medical diagnosis as a single term:\n{input_text}\nDiagnosis: ",
        "Considering clinical urgency, identify the condition that should be prioritised or ruled out first. State ONLY the condition name:\n{input_text}\nDiagnosis: ",
        "Considering the most plausible differential diagnoses, select the single most likely diagnosis. State ONLY the condition name:\n{input_text}\nDiagnosis: ",
        "Considering the likely underlying pathological process, select the most appropriate diagnosis. Output ONLY the condition name:\n{input_text}\nDiagnosis: ",
        "Considering the organ system most likely affected, select the most appropriate diagnosis. Output ONLY the diagnosis:\n{input_text}\nDiagnosis: ",
        "Considering symptom severity and the most clinically significant features, identify the diagnosis that best accounts for the presentation. Output ONLY the condition name:\n{input_text}\nDiagnosis: ",
    ],
    "Clinical Reasoning": [
        "Analyse this clinical case step by step using deductive reasoning, then provide your final assessment:\n{input_text}\nReasoning: ",
        "Formulate two clinical hypotheses for this case, evaluate the evidence for each, then conclude with the most supported answer:\n{input_text}\nReasoning: ",
        "Begin by explaining the pathophysiological mechanism suggested by this case, then use that mechanism to reach your clinical conclusion:\n{input_text}\nReasoning: ",
        "List the top differential diagnoses for this case, systematically rule each out based on the evidence, and state your final conclusion:\n{input_text}\nReasoning: ",
        "Identify any red-flag features in this case that must not be missed, explain their significance, then provide your overall clinical assessment:\n{input_text}\nReasoning: ",
        "Reason through this case while explicitly acknowledging uncertainty or missing information, then provide your best-supported clinical conclusion:\n{input_text}\nReasoning: ",
    ],
    "Medical Question Answering": [
        "Answer this medical question concisely and accurately based on medical knowledge:\n{input_text}\nAnswer: ",
        "Explain the underlying biological or physiological mechanism relevant to this question, then provide a direct answer:\n{input_text}\nAnswer: ",
        "Answer this question using evidence-based medical knowledge where applicable. Avoid unsupported claims:\n{input_text}\nAnswer: ",
        "Answer this medical question as you would explain it to a patient with no medical background. Use clear, simple language:\n{input_text}\nAnswer: ",
        "Answer this medical question and highlight any important risks, contraindications, or complications that should be considered:\n{input_text}\nAnswer: ",
        "Answer this question with a focus on management, including treatment, monitoring, prevention, or follow-up where relevant:\n{input_text}\nAnswer: ",
    ],
}

ICIP_FUNCTION_LABELS = {
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


ICIP_FUNCTION_LABELS = {
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


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_task(x: str) -> str:
    key = str(x).strip().lower()
    key = re.sub(r"[_\-]+", " ", key)
    key = re.sub(r"\s+", " ", key)

    aliases = {
        "clinical diagnosis": "Clinical Diagnosis",
        "diagnosis": "Clinical Diagnosis",
        "clinical reasoning": "Clinical Reasoning",
        "reasoning": "Clinical Reasoning",
        "medical question answering": "Medical Question Answering",
        "medical qa": "Medical Question Answering",
        "mqa": "Medical Question Answering",
        "question answering": "Medical Question Answering",
    }
    return aliases.get(key, str(x).strip())

def normalize_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = x.lower().strip()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    ref_counts = {}

    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = sum(min(pred_counts.get(t, 0), ref_counts.get(t, 0)) for t in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return float(2 * precision * recall / (precision + recall + 1e-12))


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def rouge_l(pred: str, ref: str) -> float:
    if not ROUGE_AVAILABLE:
        return np.nan

    pred = "" if pred is None else str(pred).strip()
    ref = "" if ref is None else str(ref).strip()

    if not pred or not ref:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(ref, pred)["rougeL"].fmeasure)


def detect_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not detect {label} column. Tried: {candidates}. "
        f"Found: {list(df.columns)}"
    )

def extract_reasoning_conclusion(text: str) -> dict:
    """
    Conservative split for Clinical Reasoning outputs.
    Returns reasoning text and final conclusion text.
    """
    if text is None:
        return {"reasoning": "", "conclusion": ""}

    text = str(text).strip()
    if not text:
        return {"reasoning": "", "conclusion": ""}

    # Look for explicit conclusion markers.
    patterns = [
        r"(?i)\bfinal assessment\s*:",
        r"(?i)\bfinal conclusion\s*:",
        r"(?i)\bconclusion\s*:",
        r"(?i)\banswer\s*:",
        r"(?i)\bassessment\s*:",
    ]

    for pat in patterns:
        parts = re.split(pat, text, maxsplit=1)
        if len(parts) == 2:
            return {
                "reasoning": parts[0].strip(),
                "conclusion": parts[1].strip(),
            }

    # If no marker exists, use the last sentence as an approximate conclusion.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= 2:
        return {
            "reasoning": " ".join(sentences[:-1]).strip(),
            "conclusion": sentences[-1].strip(),
        }

    return {
        "reasoning": text,
        "conclusion": text,
    }
def clean_generation(text: str) -> str:
    """
    Light cleanup only. Avoid aggressive disease-synonym normalization because it can inflate scores.
    """
    text = "" if text is None else str(text)
    text = text.strip()

    # Remove common leading assistant labels only.
    text = re.sub(r"^\s*(assistant|answer|diagnosis|reasoning|conclusion)\s*:\s*", "", text, flags=re.I)
    text = text.strip()

    return text

def extract_diagnosis(text: str) -> str:
    """
    Conservative diagnosis extraction for Clinical Diagnosis outputs.

    It extracts the first diagnosis-like phrase without using synonym expansion
    or disease hierarchy matching. This avoids inflating scores while preventing
    long explanations from unfairly breaking exact match/token-F1.
    """
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    # Remove leading assistant-style labels.
    text = re.sub(
        r"^\s*(assistant|answer|diagnosis|final diagnosis|condition|medical diagnosis)\s*:\s*",
        "",
        text,
        flags=re.I,
    ).strip()

    # Keep first line only.
    text = text.split("\n")[0].strip()

    # Remove common lead-in phrases.
    text = re.sub(
        r"^\s*(the most likely diagnosis is|most likely diagnosis is|likely diagnosis is|this is most consistent with|the diagnosis is)\s+",
        "",
        text,
        flags=re.I,
    ).strip()

    # Case: numbered list, e.g.
    # "1. Lung Cancer 2. Leukemia 3. Lymphoma"
    # Keep only item 1.
    numbered_match = re.match(
        r"^\s*(?:\d+[\.\)]\s*)?(.+?)(?=\s+\d+[\.\)]\s+|$)",
        text,
        flags=re.I,
    )
    if numbered_match:
        text = numbered_match.group(1).strip()

    # Remove a leading bullet or number again if still present.
    text = re.sub(r"^\s*[\-\*\u2022]?\s*\d*[\.\)]?\s*", "", text).strip()

    # If the first item has an explanation after colon, keep the label before colon.
    # Example: "Ingrown Toenail: This condition occurs..."
    if ":" in text:
        left, right = text.split(":", 1)
        if len(left.strip()) >= 3:
            text = left.strip()

    # Cut explanation markers.
    cut_markers = [
        " because ",
        " due to ",
        " based on ",
        " given ",
        " characterized by ",
        " as evidenced by ",
        " which ",
        " that ",
    ]
    lower = text.lower()
    positions = [lower.find(m) for m in cut_markers if lower.find(m) != -1]
    if positions:
        text = text[:min(positions)].strip()

    # If still a sentence, keep only first sentence.
    sentence_parts = re.split(r"[.!?]", text, maxsplit=1)
    if sentence_parts:
        text = sentence_parts[0].strip()

    # If multiple alternatives are given, keep the first.
    for sep in [";", "|", " or ", " and "]:
        if sep in text.lower():
            text = re.split(re.escape(sep), text, maxsplit=1, flags=re.I)[0].strip()
            break

    # Remove bracketed notes and trailing punctuation.
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = text.strip(" .,:;-")

    return text
def get_instruction(task: str, selected_function: str) -> str:
    task = normalize_task(task)
    selected_function = "" if selected_function is None else str(selected_function).strip()

    if task not in ICIP_TEMPLATES:
        raise KeyError(f"Unknown task: {task}")

    templates = ICIP_TEMPLATES[task]
    labels = ICIP_FUNCTION_LABELS[task]

    # Case 1: selected_function is a numeric template index, e.g. "0", "1", ...
    if selected_function.isdigit():
        idx = int(selected_function)
        if 0 <= idx < len(templates):
            return templates[idx]

    # Case 2: selected_function is something like "template_0" or "function_3"
    m = re.search(r"(\d+)", selected_function)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(templates):
            return templates[idx]

    # Case 3: selected_function matches one of the human-readable function labels
    norm_sel = normalize_text(selected_function)
    for idx, label in enumerate(labels):
        if normalize_text(label) == norm_sel:
            return templates[idx]

    # Case 4: selected_function already contains the full template text
    for template in templates:
        if normalize_text(template) == norm_sel:
            return template

    raise KeyError(
        f"No instruction found for task='{task}' and selected_function='{selected_function}'. "
        f"Available labels: {labels}"
    )


def build_prompt(task: str, input_text: str, selected_function: str) -> str:
    task = normalize_task(task)
    template = get_instruction(task, selected_function)

    # Fill the {input_text} placeholder inside the selected ICIP template.
    instruction = template.format(input_text=input_text)

    if task == "Clinical Diagnosis":
        return (
            f"{instruction}\n\n"
            f"Output only the diagnosis."
        )

    if task == "Clinical Reasoning":
        return (
            f"{instruction}\n\n"
            f"Provide the requested clinical reasoning or conclusion."
        )

    if task == "Medical Question Answering":
        return (
            f"{instruction}\n\n"
            f"Answer concisely and accurately."
        )

    return f"{instruction}\n\nOutput:"

def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful medical NLP assistant. "
                "Follow the instruction and produce only the requested output."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        "System: You are a careful medical NLP assistant. "
        "Follow the instruction and produce only the requested output.\n\n"
        f"User: {prompt}\n\nAssistant:"
    )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def prepare_eval_df(
    eval_csv: str,
    id_col: Optional[str],
    task_col: Optional[str],
    input_col: Optional[str],
    output_col: Optional[str],
    force_row_ids: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)

    task_col = task_col or detect_column(
        df,
        ["task", "task_family", "label_task", "dataset_task"],
        "task",
    )
    input_col = input_col or detect_column(
        df,
        ["raw_input", "input", "text", "question", "prompt"],
        "input",
    )
    output_col = output_col or detect_column(
        df,
        ["output", "target", "answer", "label", "reference", "gold"],
        "output/reference",
    )

    out = pd.DataFrame()
    out["task"] = df[task_col].apply(normalize_task)
    out["input_text"] = df[input_col].astype(str)
    out["reference"] = df[output_col].astype(str)

    # Keep only the ICIP tasks, then reset order.
    out = out[out["task"].isin(ICIP_TEMPLATES.keys())].reset_index(drop=True)

    # IMPORTANT:
    # The ICIP routing export uses row-order IDs: ex_0, ex_1, ...
    # Therefore, for validation mode we often need to force row-order IDs
    # instead of using any existing example_id column in val_split.csv.
    if force_row_ids:
        out["example_id"] = [f"ex_{i}" for i in range(len(out))]
    elif id_col and id_col in df.columns:
        out["example_id"] = df.loc[out.index, id_col].astype(str).values
    elif "example_id" in df.columns:
        out["example_id"] = df.loc[out.index, "example_id"].astype(str).values
    else:
        out["example_id"] = [f"ex_{i}" for i in range(len(out))]

    # Reorder columns.
    out = out[["example_id", "task", "input_text", "reference"]]

    return out

def load_routes(route_csv: str) -> pd.DataFrame:
    routes = pd.read_csv(route_csv)

    required = {"example_id", "task", "selected_function", "template_idx"}
    missing = required - set(routes.columns)
    if missing:
        raise ValueError(
            f"Routing file missing {missing}: {route_csv}. "
            f"For accurate ICIP evaluation, route CSV must include template_idx."
        )

    routes = routes.copy()
    routes["example_id"] = routes["example_id"].astype(str)
    routes["route_task"] = routes["task"].apply(normalize_task)
    routes["template_idx"] = routes["template_idx"].astype(int)

    cols = ["example_id", "route_task", "selected_function", "template_idx"]

    for c in ["entropy", "max_prob", "prob_vector", "raw_input_preview"]:
        if c in routes.columns:
            cols.append(c)

    return routes[cols]

def merge_eval_and_routes(
    eval_df: pd.DataFrame,
    routes: pd.DataFrame,
    route_csv: str,
    align_by_order: bool = False,
) -> pd.DataFrame:
    """
    Align evaluation rows with routing decisions.

    For ICIP validation evaluation, row-order alignment is safest because
    the routing export was generated from the validation split in order,
    while example_id values are not globally unique.
    """

    eval_df = eval_df.copy().reset_index(drop=True)
    routes = routes.copy().reset_index(drop=True)

    eval_df["eval_task"] = eval_df["task"].apply(normalize_task)

    if "route_task" not in routes.columns:
        routes["route_task"] = routes["task"].apply(normalize_task)
    else:
        routes["route_task"] = routes["route_task"].apply(normalize_task)

    if align_by_order:
        if len(eval_df) != len(routes):
            raise RuntimeError(
                f"Row-order alignment requested, but row counts differ: "
                f"eval={len(eval_df)}, routes={len(routes)} for {route_csv}"
            )

        merged = pd.concat(
            [
                eval_df.reset_index(drop=True),
                routes.drop(columns=["example_id"], errors="ignore").reset_index(drop=True),
            ],
            axis=1,
        )

        mismatch = merged[merged["eval_task"] != merged["route_task"]].copy()

        if len(mismatch) > 0:
            debug_cols = ["eval_task", "route_task", "input_text"]
            if "raw_input_preview" in mismatch.columns:
                debug_cols.append("raw_input_preview")

            raise RuntimeError(
                f"Task mismatch under row-order alignment for {route_csv}: "
                f"{len(mismatch)}/{len(merged)} rows mismatch.\n\n"
                f"First mismatches:\n{mismatch[debug_cols].head(20).to_string(index=False)}"
            )

        merged["task"] = merged["eval_task"]
        return merged.reset_index(drop=True)

    # Strict ID merge only allowed if IDs are unique.
    if eval_df["example_id"].duplicated().any():
        raise RuntimeError("Evaluation example_id values are duplicated. Use --align_by_order.")

    if routes["example_id"].duplicated().any():
        raise RuntimeError("Routing example_id values are duplicated. Use --align_by_order.")

    merged = eval_df.merge(routes, on="example_id", how="inner")

    if len(merged) != len(eval_df):
        raise RuntimeError(
            f"Matched only {len(merged)}/{len(eval_df)} examples for {route_csv}. "
            f"Use --align_by_order for ICIP validation routing exports."
        )

    mismatch = merged[merged["eval_task"] != merged["route_task"]].copy()
    if len(mismatch) > 0:
        raise RuntimeError(
            f"Task mismatch after example_id merge for {route_csv}: "
            f"{len(mismatch)}/{len(merged)} rows mismatch."
        )

    merged["task"] = merged["eval_task"]
    return merged.reset_index(drop=True)

def build_prompt_from_route(row: pd.Series):
    """
    Reconstruct the exact ICIP training prompt using template_idx.
    This is the only prompt builder that should be used for downstream sanity evaluation.
    """

    task = normalize_task(row["task"])
    input_text = str(row["input_text"])

    if "template_idx" not in row or pd.isna(row["template_idx"]):
        raise ValueError("template_idx missing. Cannot reconstruct exact ICIP training template.")

    idx = int(row["template_idx"])

    if task not in ICIP_TEMPLATES:
        raise KeyError(f"Unknown task: {task}")

    if idx < 0 or idx >= len(ICIP_TEMPLATES[task]):
        raise ValueError(f"Invalid template_idx={idx} for task={task}")

    template = ICIP_TEMPLATES[task][idx]
    function_label = ICIP_FUNCTION_LABELS[task][idx]
    prompt = template.format(input_text=input_text)

    return prompt, idx, function_label
def optionally_sample(df: pd.DataFrame, max_per_task: Optional[int], seed: int) -> pd.DataFrame:
    if max_per_task is None or max_per_task <= 0:
        return df.reset_index(drop=True)

    parts = []
    for task, g in df.groupby("task"):
        take = min(max_per_task, len(g))
        parts.append(g.sample(n=take, random_state=seed))

    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_tokenizer(model_dir: str, base_model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side="left")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side="left")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(model_dir: str, base_model_path: str, use_4bit: bool = True):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    # Case 1: final_model is a full saved causal LM.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model
    except Exception as e:
        print(f"[INFO] Direct model load failed for {model_dir}: {e}")
        print("[INFO] Trying base model + PEFT adapter loading.")

    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not available, and direct model loading failed.")

    # Case 2: final_model is a PEFT/LoRA adapter.
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base,
        model_dir,
        is_trainable=False,
        device_map="auto",
    )
    model.eval()
    return model


def generate_predictions(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    preds = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        # Use raw ICIP templates because training used raw templates.
        formatted = [apply_chat_template(tokenizer, p) for p in batch_prompts]

        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        preds.extend([clean_generation(x) for x in decoded])

        if start == 0 or (start // batch_size) % 50 == 0:
            print(f"  generated {min(start + batch_size, len(prompts))}/{len(prompts)}")

    return preds


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def score_task(task: str, preds: List[str], refs: List[str]) -> List[Dict]:
    """
    Compute conservative downstream sanity-check metrics.

    Clinical Diagnosis:
      - raw_exact_match: exact match on raw generated text
      - diagnosis_exact_match: exact match after conservative diagnosis extraction
      - diagnosis_token_f1: token-F1 after conservative diagnosis extraction

    Clinical Reasoning and Medical QA:
      - rougeL
      - token_f1
    """

    rows = []
    task = normalize_task(task)

    if len(preds) != len(refs):
        raise ValueError(f"Prediction/reference length mismatch for {task}: {len(preds)} vs {len(refs)}")

    if len(refs) == 0:
        return rows

    if task == "Clinical Diagnosis":
        extracted_preds = [extract_diagnosis(p) for p in preds]
        extracted_refs = [extract_diagnosis(r) for r in refs]

        raw_em_values = [exact_match(p, r) for p, r in zip(preds, refs)]
        extracted_em_values = [exact_match(p, r) for p, r in zip(extracted_preds, extracted_refs)]
        extracted_f1_values = [token_f1(p, r) for p, r in zip(extracted_preds, extracted_refs)]

        rows.append({
            "task": task,
            "metric": "raw_exact_match",
            "value": float(np.mean(raw_em_values)),
            "n": len(refs),
        })
        rows.append({
            "task": task,
            "metric": "diagnosis_exact_match",
            "value": float(np.mean(extracted_em_values)),
            "n": len(refs),
        })
        rows.append({
            "task": task,
            "metric": "diagnosis_token_f1",
            "value": float(np.mean(extracted_f1_values)),
            "n": len(refs),
        })

        return rows
    if task == "Clinical Reasoning":
        pred_parts = [extract_reasoning_conclusion(p) for p in preds]
        ref_parts = [extract_reasoning_conclusion(r) for r in refs]

        full_rouge = [rouge_l(p, r) for p, r in zip(preds, refs)]
        full_f1 = [token_f1(p, r) for p, r in zip(preds, refs)]

        pred_conclusions = [x["conclusion"] for x in pred_parts]
        ref_conclusions = [x["conclusion"] for x in ref_parts]

        conclusion_rouge = [
            rouge_l(p, r) for p, r in zip(pred_conclusions, ref_conclusions)
        ]
        conclusion_f1 = [
            token_f1(p, r) for p, r in zip(pred_conclusions, ref_conclusions)
        ]

        rows.append({
            "task": task,
            "metric": "full_rougeL",
            "value": float(np.nanmean(full_rouge)),
            "n": len(refs),
        })
        rows.append({
            "task": task,
            "metric": "full_token_f1",
            "value": float(np.mean(full_f1)),
            "n": len(refs),
        })
        rows.append({
            "task": task,
            "metric": "conclusion_rougeL",
            "value": float(np.nanmean(conclusion_rouge)),
            "n": len(refs),
        })
        rows.append({
            "task": task,
            "metric": "conclusion_token_f1",
            "value": float(np.mean(conclusion_f1)),
            "n": len(refs),
        })

        # BERTScore-F1 on full CR output using Bio-ClinicalBERT
        try:
            from bert_score import BERTScorer
            bs = BERTScorer(
                model_type="/home/share/henry/model_download/emilyalsentzerBio_ClinicalBERT",
                num_layers=12,
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            )
            preds_trunc = [p[:512] for p in preds]
            refs_trunc  = [r[:512] for r in refs]
            _, _, F1 = bs.score(preds_trunc, refs_trunc)
            bert_f1_values = F1.tolist()
            rows.append({
                "task": task,
                "metric": "bertscore_f1",
                "value": float(__import__("numpy").mean(bert_f1_values)),
                "n": len(refs),
            })
            del bs
            __import__("torch").cuda.empty_cache()
        except Exception as e:
            print(f"  BERTScore skipped: {e}")

        return rows
    if task in ["Clinical Reasoning", "Medical Question Answering"]:
        rouge_values = [rouge_l(p, r) for p, r in zip(preds, refs)]
        f1_values = [token_f1(p, r) for p, r in zip(preds, refs)]

        rows.append({
            "task": task,
            "metric": "rougeL",
            "value": float(np.nanmean(rouge_values)),
            "n": len(refs),
        })
        rows.append({
            "task": task,
            "metric": "token_f1",
            "value": float(np.mean(f1_values)),
            "n": len(refs),
        })

        return rows

    raise ValueError(f"Unknown task passed to score_task(): {task}")


def evaluate_condition(
    condition_name: str,
    model_dir: str,
    eval_csv: str,
    route_csv: str,
    base_model_path: str,
    output_dir: str,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print("\n" + "=" * 100)
    print(f"Condition: {condition_name}")
    print(f"Model    : {model_dir}")
    print(f"Eval CSV : {eval_csv}")
    print(f"Routes   : {route_csv}")
    print("=" * 100)

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not Path(eval_csv).exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_csv}")
    if not Path(route_csv).exists():
        raise FileNotFoundError(f"Routing CSV not found: {route_csv}")

    eval_df = prepare_eval_df(
    eval_csv=eval_csv,
    id_col=args.id_col,
    task_col=args.task_col,
    input_col=args.input_col,
    output_col=args.output_col,
    force_row_ids=args.force_row_ids,
)
    routes = load_routes(route_csv)
    data = merge_eval_and_routes(
    eval_df,
    routes,
    route_csv,
    align_by_order=args.align_by_order,
)
    data = optionally_sample(data, args.max_per_task, args.seed)

    print("Rows by task:")
    print(data["task"].value_counts().to_string())

    tokenizer = load_tokenizer(model_dir, base_model_path)
    model = load_model(model_dir, base_model_path, use_4bit=not args.no_4bit)

    pred_parts = []
    metric_rows = []

    max_tokens_by_task = {
        "Clinical Diagnosis": args.max_new_tokens_cd,
        "Clinical Reasoning": args.max_new_tokens_cr,
        "Medical Question Answering": args.max_new_tokens_mqa,
    }

    for task, g in data.groupby("task", sort=False):
        g = g.reset_index(drop=True)
        prompts = []
        resolved_indices = []
        resolved_labels = []

        for _, row in g.iterrows():
            prompt, idx, label = build_prompt_from_route(row)
            prompts.append(prompt)
            resolved_indices.append(idx)
            resolved_labels.append(label)

        g["resolved_template_idx"] = resolved_indices
        g["resolved_function_label"] = resolved_labels
        g["prompt_used"] = prompts

        print(f"\nGenerating {task}: n={len(g)}")
        preds = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=max_tokens_by_task.get(task, 128),
        )

        g["prediction"] = preds
        if task == "Clinical Diagnosis":
            g["extracted_prediction"] = [extract_diagnosis(p) for p in preds]
            g["extracted_reference"] = [extract_diagnosis(r) for r in g["reference"].astype(str).tolist()]
            g["condition"] = condition_name
        if task == "Clinical Reasoning":
            cr_pred_parts = [extract_reasoning_conclusion(p) for p in preds]
            cr_ref_parts = [
                extract_reasoning_conclusion(r)
                for r in g["reference"].astype(str).tolist()
            ]

            g["pred_reasoning"] = [x["reasoning"] for x in cr_pred_parts]
            g["pred_conclusion"] = [x["conclusion"] for x in cr_pred_parts]
            g["ref_reasoning"] = [x["reasoning"] for x in cr_ref_parts]
            g["ref_conclusion"] = [x["conclusion"] for x in cr_ref_parts]
        pred_parts.append(g)

        refs = g["reference"].astype(str).tolist()
        metric_rows.extend(score_task(task, preds, refs))

    pred_df = pd.concat(pred_parts, ignore_index=True)
    metric_df = pd.DataFrame(metric_rows)
    metric_df.insert(0, "condition", condition_name)

    # Save per-condition predictions immediately.
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", condition_name).strip("_").lower()
    pred_path = Path(output_dir) / f"generations_{safe_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved condition generations: {pred_path}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_df, metric_df


def build_external_run_config(args) -> Dict[str, Dict]:
    if not args.external_eval_csv:
        raise ValueError("--external_eval_csv is required when --mode external")

    required_routes = {
        "Real encoder": args.real_route_csv,
        "Shuffled encoder": args.shuffled_route_csv,
        "Task-prior control": args.taskprior_route_csv,
    }

    for name, route in required_routes.items():
        if not route:
            raise ValueError(
                f"External mode requires a route CSV for {name}. "
                f"Do not reuse validation routing files for an external test set."
            )

    return {
        "Real encoder": {
            "model_dir": RUNS["Real encoder"]["model_dir"],
            "eval_csv": args.external_eval_csv,
            "route_csv": args.real_route_csv,
        },
        "Shuffled encoder": {
            "model_dir": RUNS["Shuffled encoder"]["model_dir"],
            "eval_csv": args.external_eval_csv,
            "route_csv": args.shuffled_route_csv,
        },
        "Task-prior control": {
            "model_dir": RUNS["Task-prior control"]["model_dir"],
            "eval_csv": args.external_eval_csv,
            "route_csv": args.taskprior_route_csv,
        },
    }


def build_validation_run_config(
    llama_only: bool = False,

    static_only: bool = False,
) -> Dict[str, Dict]:

    if static_only:
        selected_runs = QWEN_FIXED_RUN
    elif llama_only:
        selected_runs = LLAMA_RUNS
        
    else:
        selected_runs = RUNS

    return {
        name: {
            "model_dir": cfg["model_dir"],
            "eval_csv": cfg["val_csv"],
            "route_csv": cfg["route_csv"],
        }
        for name, cfg in selected_runs.items()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["validation", "external"], default="validation")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_dir", default="/home/share/henry/ICIP_ROUTER/QWEN/downstream_sanity_validation")

    # External test set mode.
    parser.add_argument("--external_eval_csv", default=None)
    parser.add_argument("--real_route_csv", default=None)
    parser.add_argument("--shuffled_route_csv", default=None)
    parser.add_argument("--taskprior_route_csv", default=None)

    # Column overrides.
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--task_col", default=None)
    parser.add_argument("--input_col", default=None)
    parser.add_argument("--output_col", default=None)

    # Runtime controls.
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_per_task", type=int, default=0, help="0 means use all matched examples.")
    parser.add_argument("--seed", type=int, default=42)

    # Generation length.
    parser.add_argument("--max_new_tokens_cd", type=int, default=12)
    parser.add_argument("--max_new_tokens_cr", type=int, default=160)
    parser.add_argument("--max_new_tokens_mqa", type=int, default=128)
    parser.add_argument(
        "--random_only",
        action="store_true",
        help="Evaluate only the Qwen uniform-random instruction baseline.",
    )
    parser.add_argument(
    "--static_only",
    action="store_true",
    help="Evaluate only the Qwen statcic baseline.",
)

    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--llama_only", action="store_true",
                        help="Evaluate LLaMA runs instead of Qwen runs.")
    parser.add_argument(
    "--align_by_order",
    action="store_true",
    help="Align evaluation rows and routing rows by row order. Use this for ICIP validation routing exports.",
)
    parser.add_argument(
    "--force_row_ids",
    action="store_true",
    help="Force validation examples to use row-order IDs ex_0, ex_1, ... to match ICIP routing exports.",
)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.max_per_task <= 0:
        args.max_per_task = None

    if args.mode == "validation":
        run_config = build_validation_run_config(
            llama_only=args.llama_only,
            #random_only=args.random_only,
             static_only=args.static_only,
        )
    else:
        run_config = build_external_run_config(args)
    all_preds = []
    all_metrics = []

    for condition_name, cfg in run_config.items():
        pred_df, metric_df = evaluate_condition(
            condition_name=condition_name,
            model_dir=cfg["model_dir"],
            eval_csv=cfg["eval_csv"],
            route_csv=cfg["route_csv"],
            base_model_path=args.base_model_path,
            output_dir=args.output_dir,
            args=args,
        )
        all_preds.append(pred_df)
        all_metrics.append(metric_df)

    preds_all = pd.concat(all_preds, ignore_index=True)
    metrics_long = pd.concat(all_metrics, ignore_index=True)

    metrics_wide = (
        metrics_long
        .pivot_table(index="condition", columns=["task", "metric"], values="value")
        .reset_index()
    )

    pred_path = Path(args.output_dir) / "downstream_sanity_generations.csv"
    metrics_long_path = Path(args.output_dir) / "downstream_sanity_metrics_long.csv"
    metrics_wide_path = Path(args.output_dir) / "downstream_sanity_metrics_wide.csv"

    preds_all.to_csv(pred_path, index=False)
    metrics_long.to_csv(metrics_long_path, index=False)
    metrics_wide.to_csv(metrics_wide_path, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Generations : {pred_path}")
    print(f"Metrics long: {metrics_long_path}")
    print(f"Metrics wide: {metrics_wide_path}")

    print("\nMetrics:")
    print(metrics_long.to_string(index=False))


if __name__ == "__main__":
    main()