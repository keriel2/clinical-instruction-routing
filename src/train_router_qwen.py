import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import gc
import csv
import json
import logging
import math
import multiprocessing
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Set, Tuple, Union 
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import spacy
import torch
import torch.distributed as dist
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from rouge_score import rouge_scorer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import Dataset
from accelerate import Accelerator
world_size = int(os.environ.get("WORLD_SIZE", "1"))

os.environ["WANDB_DISABLED"] = "true"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
  
    multiprocessing.set_start_method('spawn', force=True)
    logger.info("Set multiprocessing start method to 'spawn'")

def sample_balanced_dataset(df, samples_per_task=10000, random_seed=42):
    """
    Sample a balanced dataset with equal number of examples per task
    
    Args:
        df: DataFrame with your full dataset
        samples_per_task: Number of examples to keep per task
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with balanced samples
    """
   
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    
    tasks = df['task'].unique()
    print(f"Found {len(tasks)} unique tasks")

    sampled_data = []
  
    for task in tasks:
        task_df = df[df['task'] == task]
        print(f"Task '{task}' has {len(task_df)} examples")
      
     
        if len(task_df) > samples_per_task:
            task_sample = task_df.sample(samples_per_task, random_state=random_seed)
            print(f"  Sampled {len(task_sample)} examples")
        else:
           
            task_sample = task_df
            print(f"  Using all {len(task_sample)} examples (fewer than requested {samples_per_task})")
        
        sampled_data.append(task_sample)
    balanced_df = pd.concat(sampled_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"Final balanced dataset has {len(balanced_df)} examples")
    return balanced_df


 


DATALOADER_NUM_WORKERS: int = 0      # collator uses GPU/manager; forks break this
REMOVE_UNUSED_COLUMNS: bool = False  # collator needs raw_input, task, example_id
 
IM_START      = "<|im_start|>"
IM_END        = "<|im_end|>"
SYSTEM_PROMPT = "You are a clinical NLP assistant. Answer concisely and accurately."
 
# ===========================================================================
# 1.  Clinical Encoders
# ===========================================================================
 
class ClinicalEncoder(nn.Module):
    """Frozen Bio-ClinicalBERT -> [CLS] embeddings (768-dim)."""
 
    MODEL_NAME = "/home/share/henry/model_download/emilyalsentzerBio_ClinicalBERT"
 
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"[ClinicalEncoder] Loading: {self.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
 
    @torch.no_grad()
    def encode(self, texts: list, max_length: int = 128) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        return self.encoder(**enc).last_hidden_state[:, 0, :]   # (batch, 768)
 
 
class ClinicalEncoderShuffled(ClinicalEncoder):
    """
    Control encoder: shuffles FEATURE dimension per example.
    Valid at batch_size=1 — torch.randperm(batch) would be a no-op there.
    """
    @torch.no_grad()
    def encode(self, texts: list, max_length: int = 128) -> torch.Tensor:
        cls = super().encode(texts, max_length)
        shuffled = torch.zeros_like(cls)
        for i in range(cls.size(0)):
            perm = torch.randperm(cls.size(1), device=cls.device)
            shuffled[i] = cls[i][perm]
        return shuffled
 

import torch
import torch.nn as nn

TASK_INDEX = {
    "Clinical Diagnosis":         0,
    "Clinical Reasoning":         1,
    "Medical Question Answering": 2,
}


class TaskPriorEncoder(nn.Module):
    """
    Input-agnostic task-prior encoder.

    Returns a fixed 768-dimensional vector per task family. The returned
    representation does not depend on the clinical input text at all.
    All examples from the same task receive the identical vector.

    Design: uses separated block vectors so the three task representations
    are orthogonal and distinct, but contain no input-specific information.
    The vectors are fixed (not trainable) to keep the control clean.
    """

    def __init__(self, device: torch.device = None, dim: int = 768):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dim = dim

        # Fixed deterministic block vectors — one per task.
        # Each task gets a dedicated non-overlapping block of the 768 dims.
        # Not trainable: requires_grad=False (default for register_buffer).
        vectors = torch.zeros(len(TASK_INDEX), dim)
        block   = dim // len(TASK_INDEX)   # 256 per task
        for task_name, idx in TASK_INDEX.items():
            start = idx * block
            end   = dim if idx == len(TASK_INDEX) - 1 else (idx + 1) * block
            vectors[idx, start:end] = 1.0

        self.register_buffer("task_vectors", vectors)
        self._current_task = None

        import logging
        logging.getLogger(__name__).info(
            f"[TaskPriorEncoder] Ready | dim={dim} | "
            f"tasks={list(TASK_INDEX.keys())} | fixed (not trainable)"
        )

    @torch.no_grad()
    def encode(self, texts: list, max_length: int = 128) -> torch.Tensor:
        """
        Ignores texts entirely. Returns the fixed task vector.
        _current_task must be set by the patch before each call.
        """
        task = getattr(self, "_current_task", None)
        if task in TASK_INDEX:
            idx = TASK_INDEX[task]
            emb = self.task_vectors[idx].unsqueeze(0)      # (1, 768)
            return emb.expand(len(texts), -1).to(self.device)
        # Fallback: zero vector if task was not set (should not happen)
        return torch.zeros(len(texts), self.dim, device=self.device)


def patch_icip_manager_for_task_prior(manager):
    """
    Patch ICIPManager so the encoder receives the current task before
    every encode() call, covering both training and evaluation paths.

    Patches:
      - format_prompt_train()  — used during training collation
      - format_prompt()        — used during eval/greedy routing
      - export_eval_routing()  — sets task before each encode() call
    """

    # --- Training path ---
    if hasattr(manager, "format_prompt_train"):
        _orig_train = manager.format_prompt_train

        def _patched_train(task: str, input_text: str, example_id=None):
            manager.encoder._current_task = task
            return _orig_train(task, input_text, example_id=example_id)

        manager.format_prompt_train = _patched_train

    # --- Eval/greedy path ---
    if hasattr(manager, "format_prompt"):
        _orig_eval = manager.format_prompt

        def _patched_eval(task: str, input_text: str):
            manager.encoder._current_task = task
            return _orig_eval(task, input_text)

        manager.format_prompt = _patched_eval

    # --- export_eval_routing path ---
    # This method loops over dataset items and calls self.encoder.encode()
    # via the policy. We wrap it to set task before each item.
    if hasattr(manager, "export_eval_routing"):
        _orig_export = manager.export_eval_routing

        def _patched_export(dataset, output_dir, **kwargs):
            # Pre-set task for each item by wrapping the dataset iterator
            class TaskSettingDataset:
                def __init__(self, ds, enc):
                    self._ds  = ds
                    self._enc = enc

                def __iter__(self):
                    for item in self._ds:
                        self._enc._current_task = item.get("task", None)
                        yield item

                def __len__(self):
                    return len(self._ds) if hasattr(self._ds, "__len__") else 0

            wrapped = TaskSettingDataset(dataset, manager.encoder)
            return _orig_export(wrapped, output_dir, **kwargs)

        manager.export_eval_routing = _patched_export

    print("[TaskPriorEncoder] ICIPManager patched — training, eval, and export paths covered.")
    return manager


if __name__ == "__main__":
    print(__doc__)
# ===========================================================================
# 2.  Policy Network
# ===========================================================================
 
class TemplatePolicy(nn.Module):
    def __init__(self, embed_dim=768, num_templates=6, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_templates),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
 
    def get_distribution(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(x))
 
 
# ===========================================================================
# 3.  6-Function Template Bank
# ===========================================================================
 
ICIP_TEMPLATES: Dict[str, List[str]] = {
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
 
ICIP_FUNCTION_LABELS: Dict[str, List[str]] = {
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
 
 
# ===========================================================================
# ---------------------------------------------------------------------------
# Example-Shuffled Encoder (inline)
# ---------------------------------------------------------------------------
class ExampleShuffledEncoder(nn.Module):
    """
    Precomputed within-task example-shuffled encoder.

    At startup, builds a fixed permutation table: for each example index i,
    stores which other example j's embedding to return (j ≠ i, same task).
    All embeddings are computed once and stored in CPU memory.
    """

    MODEL_NAME = "/home/share/henry/model_download/emilyalsentzerBio_ClinicalBERT"
    BATCH_SIZE  = 64    # encoding batch size for precomputation

    def __init__(self, device: torch.device = None, seed: int = 42):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed

        logger.info("[ExampleShuffledEncoder] Loading Bio-ClinicalBERT")
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._encoder   = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        for p in self._encoder.parameters():
            p.requires_grad = False
        self._encoder.eval()

        # Set after precompute_swap_table() is called
        self._swap_embeddings: torch.Tensor = None   # (N, 768) on CPU
        self._example_id_to_idx: dict       = {}     # example_id -> row index
        self._current_task: str             = None
        self._current_example_id: str       = None
        self._precomputed: bool             = False

        logger.info("[ExampleShuffledEncoder] Ready — call precompute_swap_table() before training")

    @torch.no_grad()
    def _encode_batch(self, texts: list, max_length: int = 128) -> torch.Tensor:
        enc = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(self.device)
        out = self._encoder(**enc)
        return out.last_hidden_state[:, 0, :].cpu()   # (batch, 768)

    def precompute_swap_table(self, df: pd.DataFrame,
                               input_col: str = "raw_input",
                               id_col: str    = "example_id",
                               max_length: int = 128):
        """
        Precompute embeddings for all examples and build the swap permutation.

        Args:
            df:         DataFrame with columns [id_col, input_col, 'task']
            input_col:  column containing the raw input text
            id_col:     column containing the example identifier
            max_length: max token length for Bio-ClinicalBERT
        """
        logger.info(f"[ExampleShuffledEncoder] Precomputing swap table for {len(df)} examples...")
        np.random.seed(self.seed)

        all_embeddings = []
        self._example_id_to_idx = {}

        # Encode all examples in batches
        texts = df[input_col].astype(str).tolist()
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start:start + self.BATCH_SIZE]
            embs  = self._encode_batch(batch, max_length)
            all_embeddings.append(embs)
            if start % 1000 == 0:
                logger.info(f"  Encoded {start}/{len(texts)}")

        all_embeddings = torch.cat(all_embeddings, dim=0)   # (N, 768)

        # Build example_id -> index mapping
        ids = df[id_col].astype(str).tolist()
        for i, eid in enumerate(ids):
            self._example_id_to_idx[eid] = i

        # Build within-task permutation: σ(i) = j where j ≠ i, task_j = task_i
        tasks  = df["task"].tolist()
        N      = len(tasks)
        perm   = list(range(N))   # will be overwritten

        task_indices = {}
        for i, task in enumerate(tasks):
            task_indices.setdefault(task, []).append(i)

        for task, indices in task_indices.items():
            indices_arr = np.array(indices)
            if len(indices_arr) < 2:
                logger.warning(f"  Task '{task}' has only {len(indices_arr)} example — cannot swap")
                continue

            # Create a derangement (permutation with no fixed points)
            shuffled = indices_arr.copy()
            max_attempts = 10
            for attempt in range(max_attempts):
                np.random.shuffle(shuffled)
                if not np.any(shuffled == indices_arr):
                    break
                if attempt == max_attempts - 1:
                    # Force derangement: shift by 1
                    shuffled = np.roll(indices_arr, 1)

            for orig, swap in zip(indices_arr, shuffled):
                perm[orig] = swap

            fixed = sum(1 for i in indices_arr if perm[i] == i)
            logger.info(f"  Task '{task}': {len(indices_arr)} examples, "
                        f"{fixed} fixed points (should be 0)")

        # Build swap embedding table: swap_embeddings[i] = embeddings[σ(i)]
        perm_tensor          = torch.tensor(perm, dtype=torch.long)
        self._swap_embeddings = all_embeddings[perm_tensor]   # (N, 768)
        self._precomputed     = True

        logger.info(f"[ExampleShuffledEncoder] Swap table ready: "
                    f"{self._swap_embeddings.shape}, seed={self.seed}")

    @torch.no_grad()
    def encode(self, texts: list, max_length: int = 128) -> torch.Tensor:
        """
        Returns the precomputed swap embedding for the current example.
        Falls back to real encoding if precomputation has not been run.
        """
        if not self._precomputed:
            # Fallback during any edge case before precomputation
            logger.warning("[ExampleShuffledEncoder] Swap table not ready — using real embedding")
            return self._encode_batch(texts, max_length).to(self.device)

        example_id = getattr(self, "_current_example_id", None)

        if example_id and example_id in self._example_id_to_idx:
            idx  = self._example_id_to_idx[example_id]
            emb  = self._swap_embeddings[idx].unsqueeze(0).to(self.device)
            return emb.expand(len(texts), -1)

        # Fallback: encode normally if example_id not found
        return self._encode_batch(texts, max_length).to(self.device)


def patch_manager_for_example_shuffle(manager):
    """
    Patch ICIPManager so the encoder receives the current example_id
    and task before each encode() call.
    """
    if hasattr(manager, "format_prompt_train"):
        _orig_train = manager.format_prompt_train

        def _patched_train(task: str, input_text: str, example_id=None):
            manager.encoder._current_task       = task
            manager.encoder._current_example_id = str(example_id) if example_id else None
            return _orig_train(task, input_text, example_id=example_id)

        manager.format_prompt_train = _patched_train

    if hasattr(manager, "format_prompt"):
        _orig_eval = manager.format_prompt

        def _patched_eval(task: str, input_text: str, example_id=None):
            manager.encoder._current_task       = task
            manager.encoder._current_example_id = str(example_id) if example_id else None
            # Try passing example_id if the original accepts it
            try:
                return _orig_eval(task, input_text, example_id=example_id)
            except TypeError:
                return _orig_eval(task, input_text)

        manager.format_prompt = _patched_eval

    # Also patch export_eval_routing to set example_id before each encode call
    if hasattr(manager, "export_eval_routing"):
        _orig_export = manager.export_eval_routing

        def _patched_export(dataset, output_dir, **kwargs):
            class ExampleIDSettingDataset:
                def __init__(self, ds, enc):
                    self._ds  = ds
                    self._enc = enc
                def __iter__(self):
                    for item in self._ds:
                        self._enc._current_task       = item.get("task", None)
                        self._enc._current_example_id = str(item.get("example_id", ""))
                        yield item
                def __len__(self):
                    return len(self._ds) if hasattr(self._ds, "__len__") else 0
            return _orig_export(ExampleIDSettingDataset(dataset, manager.encoder),
                                output_dir, **kwargs)
        manager.export_eval_routing = _patched_export

    logger.info("[ExampleShuffledEncoder] ICIPManager patched for precomputed example swap.")
    return manager

import hashlib

class FixedTemplateManager:
    """
    No-policy fixed/static instruction-function baseline.

    This baseline does not use:
      - Bio-ClinicalBERT encoder
      - policy network
      - reward signal
      - entropy regularization
      - router learning

    Each task family is assigned one fixed instruction function throughout
    training and evaluation.
    """

    TEMPLATES = ICIP_TEMPLATES

    def __init__(self, fixed_idx_by_task=None):
        """
        Args:
            fixed_idx_by_task:
                Dictionary mapping each task to a fixed template index.

                If None, the default is index 0 for each task:
                  Clinical Diagnosis -> Direct label
                  Clinical Reasoning -> Deductive
                  Medical QA -> Direct factual
        """

        self.fixed_idx_by_task = fixed_idx_by_task or {
            "Clinical Diagnosis": 0,
            "Clinical Reasoning": 0,
            "Medical Question Answering": 0,
        }

        self.selection_uid = 0
        self.selection_history = []

        # Validate indices
        for task, idx in self.fixed_idx_by_task.items():
            if task not in self.TEMPLATES:
                raise ValueError(f"Unknown task in fixed_idx_by_task: {task}")

            if idx < 0 or idx >= len(self.TEMPLATES[task]):
                raise ValueError(
                    f"Invalid fixed template index {idx} for task '{task}'. "
                    f"Valid range is 0 to {len(self.TEMPLATES[task]) - 1}."
                )

    def _get_fixed_template_idx(self, task: str) -> int:
        if task not in self.fixed_idx_by_task:
            raise ValueError(
                f"No fixed template index provided for task '{task}'. "
                f"Available tasks: {list(self.fixed_idx_by_task.keys())}"
            )

        return int(self.fixed_idx_by_task[task])

    def format_prompt_train(self, task: str, input_text: str, example_id=None):
        templates = self.TEMPLATES[task]
        t_idx = self._get_fixed_template_idx(task)

        sel_id = self.selection_uid
        self.selection_uid += 1

        self._log_selection(
            selection_id=sel_id,
            task=task,
            t_idx=t_idx,
            mode="fixed_train",
            example_id=example_id,
        )

        prompt = templates[t_idx].format(input_text=input_text)
        return prompt, t_idx, sel_id

    def format_prompt(self, task: str, input_text: str, example_id=None):
        templates = self.TEMPLATES[task]
        t_idx = self._get_fixed_template_idx(task)
        return templates[t_idx].format(input_text=input_text)

    def _log_selection(self, selection_id, task, t_idx, mode, example_id=None):
        self.selection_history.append({
            "selection_id": selection_id,
            "task": task,
            "example_id": example_id,
            "template_idx": t_idx,
            "function": ICIP_FUNCTION_LABELS[task][t_idx],
            "mode": mode,
        })

    def save_logs(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, "fixed_selection_history.csv")

        _write_csv(
            path,
            ["selection_id", "task", "example_id", "template_idx", "function", "mode"],
            self.selection_history,
        )

        logger.info(f"[FixedTemplateBaseline] Logs saved -> {path}")

    def get_stats(self) -> dict:
        stats = {
            "strategy": "fixed_static_instruction_assignment",
            "fixed_idx_by_task": self.fixed_idx_by_task,
        }

        for task in self.TEMPLATES:
            rows = [s for s in self.selection_history if s["task"] == task]
            counts = Counter(s["template_idx"] for s in rows)
            total = sum(counts.values()) or 1

            stats[f"{task}_function_distribution"] = {
                ICIP_FUNCTION_LABELS[task][i]: round(100 * counts.get(i, 0) / total, 2)
                for i in range(len(self.TEMPLATES[task]))
            }

            fixed_idx = self._get_fixed_template_idx(task)
            stats[f"{task}_fixed_function"] = ICIP_FUNCTION_LABELS[task][fixed_idx]

        return stats

    def export_eval_routing(self, dataset, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        rows = []

        for item in dataset:
            task = item["task"]
            raw_input = item.get("raw_input", item.get("input", ""))
            ex_id = item.get("example_id", None)

            t_idx = self._get_fixed_template_idx(task)

            rows.append({
                "example_id": ex_id,
                "task": task,
                "raw_input_preview": raw_input[:80].replace("\n", " "),
                "selected_function": ICIP_FUNCTION_LABELS[task][t_idx],
                "template_idx": t_idx,
                "selection_type": "fixed_static_no_policy",
            })

        path = os.path.join(output_dir, "fixed_eval_routing.csv")

        _write_csv(
            path,
            [
                "example_id",
                "task",
                "raw_input_preview",
                "selected_function",
                "template_idx",
                "selection_type",
            ],
            rows,
        )

        logger.info(
            f"[FixedTemplateBaseline] Eval routing saved -> {path} "
            f"({len(rows)} rows)"
        )


class RandomInstructionManager:
    """
    No-policy random instruction-function baseline.

    This baseline does not use:
      - Bio-ClinicalBERT encoder
      - policy network
      - reward signal
      - entropy regularization
      - router learning

    Each example is assigned one of the six task-specific instruction functions
    using a reproducible uniform random hash.
    """

    TEMPLATES = ICIP_TEMPLATES

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.selection_uid = 0
        self.selection_history = []

    def _random_template_idx(self, task: str, input_text: str, example_id=None) -> int:
        templates = self.TEMPLATES[task]

        key = f"{self.seed}|{task}|{example_id}|{str(input_text)[:512]}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest, 16) % len(templates)

    def format_prompt_train(self, task: str, input_text: str, example_id=None):
        templates = self.TEMPLATES[task]
        t_idx = self._random_template_idx(task, input_text, example_id)

        sel_id = self.selection_uid
        self.selection_uid += 1

        self._log_selection(
            selection_id=sel_id,
            task=task,
            t_idx=t_idx,
            mode="random_train",
            example_id=example_id,
        )

        prompt = templates[t_idx].format(input_text=input_text)
        return prompt, t_idx, sel_id

    def format_prompt(self, task: str, input_text: str, example_id=None):
        templates = self.TEMPLATES[task]
        t_idx = self._random_template_idx(task, input_text, example_id)
        return templates[t_idx].format(input_text=input_text)

    def _log_selection(self, selection_id, task, t_idx, mode, example_id=None):
        self.selection_history.append({
            "selection_id": selection_id,
            "task": task,
            "example_id": example_id,
            "template_idx": t_idx,
            "function": ICIP_FUNCTION_LABELS[task][t_idx],
            "mode": mode,
        })

    def save_logs(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, "random_selection_history.csv")
        _write_csv(
            path,
            ["selection_id", "task", "example_id", "template_idx", "function", "mode"],
            self.selection_history,
        )
        logger.info(f"[RandomBaseline] Logs saved -> {path}")

    def get_stats(self) -> dict:
        stats = {
            "strategy": "uniform_random_instruction_assignment",
            "seed": self.seed,
        }

        for task in self.TEMPLATES:
            rows = [s for s in self.selection_history if s["task"] == task]
            counts = Counter(s["template_idx"] for s in rows)
            total = sum(counts.values()) or 1

            stats[f"{task}_function_distribution"] = {
                ICIP_FUNCTION_LABELS[task][i]: round(100 * counts.get(i, 0) / total, 2)
                for i in range(len(self.TEMPLATES[task]))
            }

        return stats

    def export_eval_routing(self, dataset, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        rows = []

        for item in dataset:
            task = item["task"]
            raw_input = item.get("raw_input", item.get("input", ""))
            ex_id = item.get("example_id", None)

            t_idx = self._random_template_idx(task, raw_input, ex_id)

            rows.append({
                "example_id": ex_id,
                "task": task,
                "raw_input_preview": raw_input[:80].replace("\n", " "),
                "selected_function": ICIP_FUNCTION_LABELS[task][t_idx],
                "template_idx": t_idx,
                "selection_type": "uniform_random_no_policy",
            })

        path = os.path.join(output_dir, "random_eval_routing.csv")
        _write_csv(
            path,
            [
                "example_id",
                "task",
                "raw_input_preview",
                "selected_function",
                "template_idx",
                "selection_type",
            ],
            rows,
        )
        logger.info(f"[RandomBaseline] Eval routing saved -> {path}")


class RandomInstructionCollator:
    """
    Data collator for the no-policy random instruction-function baseline.
    It uses the same prompt-output masking logic as ICIF-Router, but the
    instruction function is selected uniformly at random using a fixed seed.
    """

    def __init__(self, tokenizer, random_manager: RandomInstructionManager, max_length: int = 512):
        self.tokenizer = tokenizer
        self.random_manager = random_manager
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict:
        if not features:
            raise ValueError("RandomInstructionCollator received empty batch.")

        input_ids_list, attn_mask_list, labels_list = [], [], []

        for feat in features:
            task = feat["task"]
            raw_input = feat["raw_input"]
            raw_output = feat.get("raw_output", "")
            example_id = feat.get("example_id", None)

            prompt, t_idx, sel_id = self.random_manager.format_prompt_train(
                task=task,
                input_text=raw_input,
                example_id=example_id,
            )

            chatml_prompt_text = (
                f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
                f"{IM_START}user\n{prompt.rstrip()}{IM_END}\n"
                f"{IM_START}assistant\n"
            )

            chatml_full_text = chatml_prompt_text + str(raw_output) + IM_END

            full_enc = self.tokenizer(
                chatml_full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

            full_ids = full_enc["input_ids"]
            attn_mask = full_enc["attention_mask"]

            prompt_enc = self.tokenizer(
                chatml_prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_len = len(prompt_enc["input_ids"])

            labels = list(full_ids)

            for j in range(prompt_len):
                labels[j] = -100

            for j, am in enumerate(attn_mask):
                if am == 0:
                    labels[j] = -100

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attn_mask_list.append(torch.tensor(attn_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels": torch.stack(labels_list),
        }
# 4.  ICIPManager
# ===========================================================================
 
class ICIPManager:
    """
    Input-Conditioned Instruction Policy manager.
    Pass encoder_cls=ClinicalEncoderShuffled for the control experiment.
    """
 
    TEMPLATES = ICIP_TEMPLATES
 
    def __init__(
        self,
        device: torch.device = None,
        encoder_cls=ExampleShuffledEncoder,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        policy_lr: float = 1e-3,
        policy_beta: float = 0.1,
        entropy_coeff: float = 0.01,
        baseline_decay: float = 0.99,
        warmup_steps: int = 200,
        debug: bool = False,
    ):
        self.device        = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_beta   = policy_beta
        self.entropy_coeff = entropy_coeff
        self.debug         = debug
        self.warmup_steps  = warmup_steps
        self.global_step   = 0
        self.selection_uid = 0       # unique ID per routing decision
 
        self.encoder = encoder_cls(device=self.device)
        logger.info(f"[ICIP] Encoder: {encoder_cls.__name__}")
 
        self.policies = nn.ModuleDict({
            task: TemplatePolicy(embed_dim=embed_dim, num_templates=len(templates),
                                 hidden_dim=hidden_dim).to(self.device)
            for task, templates in self.TEMPLATES.items()
        })
        self.policy_optimizer = torch.optim.Adam(self.policies.parameters(), lr=policy_lr)
 
        self.baselines      = {task: 0.0 for task in self.TEMPLATES}
        self.reward_stds    = {task: 1.0 for task in self.TEMPLATES}
        self.baseline_decay = baseline_decay
        self.rr_index       = {task: 0    for task in self.TEMPLATES}
        self._step_buffer: List = []
        self.selection_history: List[Dict] = []
        self.template_performance = {
            task: {i: 0.5 for i in range(len(t))}
            for task, t in self.TEMPLATES.items()
        }
        logger.info(f"[ICIP] Ready | warmup={warmup_steps} | K=6 | encoder={encoder_cls.__name__}")
 
    # ------------------------------------------------------------------ #
    # Prompt selection
    # ------------------------------------------------------------------ #
 
    def format_prompt_train(self, task: str, input_text: str, example_id=None):
        """
        Returns: prompt, log_prob, t_idx, cls_emb, entropy, prob_vector, selection_id
        selection_id: unique integer for exact reward-to-log matching.
        prob_vector:  full K-dim distribution for policy confidence analysis.
        """
        templates = self.TEMPLATES[task]
        n         = len(templates)
        zero_log  = torch.tensor(0.0, device=self.device, requires_grad=False)
        zero_emb  = torch.zeros(1, 768, device=self.device)
        zero_ent  = torch.tensor(0.0, device=self.device)
        uniform   = torch.ones(n, device=self.device) / n
 
        sel_id = self.selection_uid
        self.selection_uid += 1
 
        if self.global_step < self.warmup_steps:
            t_idx = self.rr_index[task] % n
            self.rr_index[task] += 1
            self._log_selection(sel_id, task, t_idx, "warmup",
                                0.0, float(uniform.max()), uniform, example_id)
            return (templates[t_idx].format(input_text=input_text),
                    zero_log, t_idx, zero_emb, zero_ent, uniform, sel_id)
 
        cls_emb  = self.encoder.encode([input_text])
        dist     = self.policies[task].get_distribution(cls_emb)
        probs    = dist.probs.squeeze().detach()
        entropy  = dist.entropy().squeeze()
        t_tensor = dist.sample()
        log_prob = dist.log_prob(t_tensor).squeeze()
        t_idx    = t_tensor.item()
 
        self._log_selection(sel_id, task, t_idx, "policy",
                            entropy.item(), probs.max().item(), probs, example_id)
        return (templates[t_idx].format(input_text=input_text),
                log_prob, t_idx, cls_emb, entropy, probs, sel_id)
 
    def format_prompt(self, task: str, input_text: str) -> str:
        """Greedy argmax — used at eval/inference time, no gradient."""
        templates = self.TEMPLATES[task]
        with torch.no_grad():
            if self.global_step < self.warmup_steps:
                t_idx = self.rr_index[task] % len(templates)
            else:
                cls_emb = self.encoder.encode([input_text])
                t_idx   = self.policies[task](cls_emb).argmax(dim=-1).item()
        return templates[t_idx].format(input_text=input_text)
 
    # ------------------------------------------------------------------ #
    # REINFORCE
    # ------------------------------------------------------------------ #
 
    def record_step(self, task, t_idx, log_prob, lm_loss,
                    cls_emb=None, entropy=None, example_id=None, selection_id=None):
        reward = -lm_loss
        self._step_buffer.append((task, t_idx, log_prob, reward, entropy))
 
        if self.global_step >= self.warmup_steps:
            self.baselines[task] = (self.baseline_decay * self.baselines[task]
                                    + (1 - self.baseline_decay) * reward)
            diff = abs(reward - self.baselines[task])
            self.reward_stds[task] = (self.baseline_decay * self.reward_stds[task]
                                      + (1 - self.baseline_decay) * diff)
 
        self.template_performance[task][t_idx] = (
            0.9 * self.template_performance[task][t_idx] + 0.1 * reward
        )
 
        baseline  = self.baselines[task]
        advantage = (reward - baseline) / (self.reward_stds[task] + 1e-6)
 
        # Match by selection_id for exact per-example reward back-fill
        matched = False
        if selection_id is not None:
            for entry in reversed(self.selection_history):
                if entry.get("selection_id") == selection_id:
                    entry.update({"lm_loss": round(lm_loss, 6), "reward": round(reward, 6),
                                  "baseline": round(baseline, 6), "advantage": round(advantage, 6)})
                    matched = True
                    break
        if not matched:
            for entry in reversed(self.selection_history):
                if entry["task"] == task and entry["lm_loss"] is None:
                    entry.update({"lm_loss": round(lm_loss, 6), "reward": round(reward, 6),
                                  "baseline": round(baseline, 6), "advantage": round(advantage, 6)})
                    break
 
        self.global_step += 1
 
    def compute_policy_loss(self) -> torch.Tensor:
        if not self._step_buffer:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        policy_losses, entropy_losses = [], []
        for task, t_idx, log_prob, reward, entropy in self._step_buffer:
            if not log_prob.requires_grad:
                continue
            advantage = (reward - self.baselines[task]) / (self.reward_stds[task] + 1e-6)
            policy_losses.append(-advantage * log_prob)
            if entropy is not None and isinstance(entropy, torch.Tensor) and entropy.requires_grad:
                entropy_losses.append(entropy)
        if not policy_losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        loss = torch.stack(policy_losses).mean()
        if entropy_losses:
            loss = self.policy_beta * loss - self.entropy_coeff * torch.stack(entropy_losses).mean()
        return loss
 
    def clear_step_buffer(self):
        self._step_buffer.clear()
 
    def sync_policy_gradients(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        world_size = torch.distributed.get_world_size()
        for param in self.policies.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                param.grad /= world_size
 
    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #
 
    def save_policy(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            "policies_state":       self.policies.state_dict(),
            "optimizer_state":      self.policy_optimizer.state_dict(),
            "baselines":            self.baselines,
            "reward_stds":          self.reward_stds,
            "global_step":          self.global_step,
            "selection_uid":        self.selection_uid,
            "template_performance": self.template_performance,
        }, os.path.join(output_dir, "icip_policy.pt"))
        logger.info(f"[ICIP] Policy saved -> {output_dir}/icip_policy.pt")
 
    def load_policy(self, output_dir: str):
        ckpt = torch.load(os.path.join(output_dir, "icip_policy.pt"), map_location=self.device)
        self.policies.load_state_dict(ckpt["policies_state"])
        self.policy_optimizer.load_state_dict(ckpt["optimizer_state"])
        self.baselines            = ckpt["baselines"]
        self.reward_stds          = ckpt.get("reward_stds", self.reward_stds)
        self.global_step          = ckpt["global_step"]
        self.selection_uid        = ckpt.get("selection_uid", self.selection_uid)
        self.template_performance = ckpt["template_performance"]
        logger.info(f"[ICIP] Policy loaded <- {output_dir}/icip_policy.pt")
 
    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
 
    def _log_selection(self, selection_id, task, t_idx, mode,
                       entropy_val, max_prob, probs, example_id):
        self.selection_history.append({
            "selection_id": selection_id,
            "step":         self.global_step,
            "task":         task,
            "example_id":   example_id,
            "template_idx": t_idx,
            "function":     ICIP_FUNCTION_LABELS[task][t_idx],
            "mode":         mode,
            "entropy":      round(entropy_val, 6),
            "max_prob":     round(max_prob, 6),
            "prob_vector":  json.dumps([round(float(p), 6) for p in probs.detach().cpu()]),
            "lm_loss":      None,
            "reward":       None,
            "baseline":     None,
            "advantage":    None,
        })
 
    def save_logs(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        h = self.selection_history
        _write_csv(os.path.join(output_dir, "icip_selection_history.csv"), [
            "selection_id", "step", "task", "example_id", "template_idx",
            "function", "mode", "entropy", "max_prob", "prob_vector",
            "lm_loss", "reward", "baseline", "advantage",
        ], h)
        _write_csv(os.path.join(output_dir, "icip_entropy_history.csv"), [
            "selection_id", "step", "task", "mode", "entropy", "max_prob", "prob_vector",
        ], h)
        _write_csv(os.path.join(output_dir, "icip_reward_history.csv"), [
            "selection_id", "step", "task", "mode", "lm_loss", "reward", "baseline", "advantage",
        ], h)
        logger.info(f"[ICIP] Logs saved -> {output_dir} ({len(h)} entries)")
 
    def get_stats(self) -> dict:
        post  = [s for s in self.selection_history if s["mode"] == "policy"]
        stats = {"strategy": "ICIP", "global_step": self.global_step,
                 "baselines": self.baselines, "reward_stds": self.reward_stds}
        for task in self.TEMPLATES:
            counts = Counter(s["template_idx"] for s in post if s["task"] == task)
            total  = sum(counts.values()) or 1
            stats[f"{task}_function_distribution"] = {
                ICIP_FUNCTION_LABELS[task][i]: round(100 * counts.get(i, 0) / total, 2)
                for i in range(len(self.TEMPLATES[task]))
            }
        return stats
 
    def export_eval_routing(self, dataset, output_dir: str):
        """Greedy routing on held-out data — saves icip_eval_routing.csv."""
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        self.policies.eval()
        with torch.no_grad():
            for item in dataset:
                task      = item["task"]
                raw_input = item.get("raw_input", item.get("input", ""))
                ex_id     = item.get("example_id", None)
                cls_emb   = self.encoder.encode([raw_input])
                dist      = self.policies[task].get_distribution(cls_emb)
                probs     = dist.probs.squeeze().cpu()
                entropy   = dist.entropy().squeeze().item()
                t_idx     = probs.argmax().item()
                rows.append({
                    "example_id":        ex_id,
                    "task":              task,
                    "raw_input_preview": raw_input[:80].replace("\n", " "),
                    "selected_function": ICIP_FUNCTION_LABELS[task][t_idx],
                    "template_idx":      t_idx,
                    "entropy":           round(entropy, 6),
                    "max_prob":          round(probs.max().item(), 6),
                    "prob_vector":       json.dumps([round(float(p), 6) for p in probs]),
                })
        _write_csv(os.path.join(output_dir, "icip_eval_routing.csv"), [
            "example_id", "task", "raw_input_preview", "selected_function",
            "template_idx", "entropy", "max_prob", "prob_vector",
        ], rows)
        logger.info(f"[ICIP] Eval routing saved -> {output_dir}/icip_eval_routing.csv ({len(rows)} rows)")
 
 
def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        import csv; writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
 
 
# ===========================================================================
# 5.  ICIPCallback
# ===========================================================================
 
class ICIPCallback(TrainerCallback):
    """
    Extends (not overwrites) pending buffers across gradient accumulation.
    Updates policy every optimizer step (state.global_step = optimizer steps).
    Carries selection_id for exact per-example reward matching.
    """
 
    def __init__(self, manager: ICIPManager,
                 policy_update_every_n_optimizer_steps: int = 1):
        self.manager      = manager
        self.update_every = policy_update_every_n_optimizer_steps
        self._trainer     = None
        self._clear_pending()
 
    def _is_main_process(self) -> bool:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        return int(os.environ.get("LOCAL_RANK", 0)) == 0
 
    def register_batch(self, tasks, t_idxs, log_probs, cls_embs,
                       entropies, selection_ids, example_ids=None, losses=None):
        n = len(tasks)
        self._pending_tasks.extend(tasks)
        self._pending_t_idxs.extend(t_idxs)
        self._pending_log_probs.extend(log_probs)
        self._pending_cls_embs.extend(cls_embs)
        self._pending_entropies.extend(entropies)
        self._pending_selection_ids.extend(selection_ids)
        self._pending_example_ids.extend(example_ids if example_ids is not None else [None] * n)
        self._pending_losses.extend(losses if losses is not None else [None] * n)
 
    def on_step_end(self, args, state, control, **kwargs):
        if not self._pending_tasks:
            return control
 
        trainer = kwargs.get("trainer") or getattr(self, "_trainer", None)
        per_example_losses = None
        if trainer is not None and hasattr(trainer, "_last_per_example_loss"):
            per_example_losses = trainer._last_per_example_loss
 
        for i, (task, t_idx, log_prob, cls_emb, entropy, sel_id, ex_id) in enumerate(zip(
            self._pending_tasks, self._pending_t_idxs, self._pending_log_probs,
            self._pending_cls_embs, self._pending_entropies,
            self._pending_selection_ids, self._pending_example_ids,
        )):
            if self._pending_losses and i < len(self._pending_losses) \
                    and self._pending_losses[i] is not None:
                lm_loss = self._pending_losses[i]
            elif per_example_losses is not None and i < len(per_example_losses):
                lm_loss = per_example_losses[i].item()
            else:
                lm_loss = state.log_history[-1].get("loss", 1.0) if state.log_history else 1.0
                if self.manager.debug:
                    logger.warning(f"[ICIP] step={state.global_step} i={i}: batch-avg fallback")
 
            self.manager.record_step(task=task, t_idx=t_idx, log_prob=log_prob,
                                     lm_loss=lm_loss, cls_emb=cls_emb,
                                     entropy=entropy, example_id=ex_id,
                                     selection_id=sel_id)
 
        if state.global_step % self.update_every == 0:
            policy_loss = self.manager.compute_policy_loss()
            if policy_loss.requires_grad:
                self.manager.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.manager.sync_policy_gradients()
                self.manager.policy_optimizer.step()
                self.manager.clear_step_buffer()
                if self._is_main_process() and state.global_step % 100 == 0:
                    logger.info(f"[ICIP] step={state.global_step} | "
                                f"policy_loss={policy_loss.item():.4f} | "
                                f"window={len(self._pending_tasks)}")
 
        self._clear_pending()
        return control
 
    def on_evaluate(self, args, state, control, **kwargs):
        if self._is_main_process():
            self.manager.save_policy(args.output_dir)
            self.manager.save_logs(args.output_dir)
            stats = self.manager.get_stats()
            with open(os.path.join(args.output_dir,
                      f"icip_stats_step_{state.global_step}.json"), "w") as f:
                json.dump(stats, f, indent=2, default=str)
        return control
 
    def _clear_pending(self):
        self._pending_tasks         = []
        self._pending_t_idxs        = []
        self._pending_log_probs     = []
        self._pending_cls_embs      = []
        self._pending_entropies     = []
        self._pending_selection_ids = []
        self._pending_example_ids   = []
        self._pending_losses        = []
 
 
# ===========================================================================
# 6.  ICIPDynamicCollator
# ===========================================================================
 
class ICIPDynamicCollator:
    """
    Selects instruction function per example at training time.
    Output budget guard prevents prompt budget going negative for long outputs.
    All tensors returned on CPU — Trainer/Accelerate handles device placement.
    """
 
    def __init__(self, tokenizer, icip_manager: ICIPManager,
                 icip_callback: ICIPCallback, max_length: int = 512):
        self.tokenizer     = tokenizer
        self.icip_manager  = icip_manager
        self.icip_callback = icip_callback
        self.max_length    = max_length
 
    def __call__(self, features: List[Dict]) -> Dict:
        if not features:
            raise ValueError("ICIPDynamicCollator received empty batch.")
 
        input_ids_list, attn_mask_list, labels_list = [], [], []
        b_log_probs, b_t_idxs, b_tasks = [], [], []
        b_cls_embs, b_entropies, b_losses = [], [], []
        b_ex_ids, b_sel_ids = [], []
 
        for feat in features:
            task       = feat["task"]
            raw_input  = feat["raw_input"]
            output_ids = feat["output_ids"]
            example_id = feat.get("example_id", None)
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids.tolist()
 
            # Output budget guard: cap outputs at half sequence length
            max_output_len = min(len(output_ids), self.max_length // 2)
            output_ids     = output_ids[:max_output_len]
            prompt_budget  = max(16, self.max_length - len(output_ids))
 
            prompt, log_prob, t_idx, cls_emb, entropy, _probs, sel_id = \
                self.icip_manager.format_prompt_train(task, raw_input, example_id=example_id)
 
            b_log_probs.append(log_prob); b_t_idxs.append(t_idx)
            b_tasks.append(task);         b_cls_embs.append(cls_emb)
            b_entropies.append(entropy);  b_losses.append(None)
            b_ex_ids.append(example_id);  b_sel_ids.append(sel_id)
 
            # Tokenize full sequence together (prompt + output) then mask prompt.
            # Tokenizing separately and concatenating breaks subword boundaries
            # and is the root cause of loss ~85-96 in Qwen2.5.
            raw_output = feat.get("raw_output", "")
            chatml_prompt_text = (
                f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
                f"{IM_START}user\n{prompt.rstrip()}{IM_END}\n"
                f"{IM_START}assistant\n"
            )
            chatml_full_text = chatml_prompt_text + raw_output + IM_END
 
            full_enc = self.tokenizer(
                chatml_full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            full_ids  = full_enc["input_ids"]
            attn_mask = full_enc["attention_mask"]
 
            # Find where output starts by tokenizing prompt alone
            prompt_enc = self.tokenizer(
                chatml_prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_len = len(prompt_enc["input_ids"])
 
            # Mask prompt tokens and padding positions in labels
            labels = list(full_ids)
            for j in range(prompt_len):
                labels[j] = -100
            for j, am in enumerate(attn_mask):
                if am == 0:
                    labels[j] = -100
 
            input_ids_list.append(torch.tensor(full_ids,   dtype=torch.long))
            attn_mask_list.append(torch.tensor(attn_mask,  dtype=torch.long))
           
            labels_list.append(torch.tensor(labels,        dtype=torch.long))
 
        batch = {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels":         torch.stack(labels_list),
        }
 
        self.icip_callback.register_batch(
            tasks=b_tasks, t_idxs=b_t_idxs, log_probs=b_log_probs,
            cls_embs=b_cls_embs, entropies=b_entropies,
            selection_ids=b_sel_ids, example_ids=b_ex_ids, losses=b_losses,
        )
        return batch
 

# ===========================================================================
# 7.  ICIPTrainer
# ===========================================================================
 
class ICIPTrainer(Trainer):
    """
    Computes real per-example LM loss and injects into callback._pending_losses.
    Accepts **kwargs for newer Transformers versions (num_items_in_batch etc).
    """
 
    def __init__(self, *args, icip_callback: ICIPCallback = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.icip_callback          = icip_callback
        self._last_per_example_loss = None
 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels   = inputs.get("labels")
        outputs  = model(**inputs)
 
        loss_fct     = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = outputs.logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss   = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
 
        mask             = shift_labels != -100
        per_example_loss = (token_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        self._last_per_example_loss = per_example_loss.detach().cpu()
 
        if self.icip_callback is not None and hasattr(self.icip_callback, "_pending_losses"):
            pending = self.icip_callback._pending_losses
            n, start = len(per_example_loss), max(0, len(pending) - len(per_example_loss))
            for i, loss_val in enumerate(per_example_loss.detach().cpu()):
                idx = start + i
                if idx < len(pending):
                    pending[idx] = loss_val.item()
 
        return (per_example_loss.mean(), outputs) if return_outputs else per_example_loss.mean()
 
 
# ===========================================================================
# 8.  Data helpers
# ===========================================================================
 
def sample_balanced_dataset(df, samples_per_task=10000, random_seed=42):
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    sampled = []
    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        sampled.append(
            task_df.sample(samples_per_task, random_state=random_seed)
            if len(task_df) > samples_per_task else task_df
        )
    out = pd.concat(sampled, ignore_index=True)
    return out.sample(frac=1, random_state=random_seed).reset_index(drop=True)
 
 
def preprocess_raw(examples, tokenizer, max_length):
    """
    Keep raw text for dynamic collation. Adds example_id for routing analysis.
    Stores raw_output as text so the collator can tokenize the full sequence
    (prompt + output) together, which is required for correct subword alignment.
    """
    n = len(examples["input"])
    return {
        "task":       examples["task"],
        "raw_input":  examples["input"],
        "raw_output": [str(o) for o in examples["output"]],
        "output_ids": [[] for _ in range(n)],   # unused; kept for schema compat
        "example_id": [f"ex_{i}" for i in range(n)],
    }
 
 
# ===========================================================================
# 9.  MemoryEfficientDataProcessor
# ===========================================================================


class MemoryEfficientDataProcessor:
    """
    Loads model, preprocesses data, and runs ICIP training.
    Supports Qwen2.5-7B-Instruct and LLaMA-3-8B.
    Set encoder_cls=ClinicalEncoderShuffled for the control experiment.
    """
 
    def __init__(self, model_path: str, tokenizer_path: str, output_dir: str,
                 max_length: int = 512, encoder_cls=ClinicalEncoder):
        self.model_path     = model_path
        self.tokenizer_path = tokenizer_path
        self.output_dir     = output_dir
        self.max_length     = max_length
        self.encoder_cls    = encoder_cls
        self.prompt_manager = None
        os.makedirs(output_dir, exist_ok=True)
        self._setup_environment()
        self._setup_tokenizer()
        self._setup_model()
 
    def _is_qwen(self) -> bool:
        return "qwen" in self.model_path.lower()
 
    def _setup_environment(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            torch.multiprocessing.set_sharing_strategy("file_system")
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
 
    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, padding_side="left", truncation_side="left",
            use_fast=True, trust_remote_code=True,
        )
        if self._is_qwen():
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token    = "<|endoftext|>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
                logger.info(f"[Qwen2.5] pad_token='<|endoftext|>' id={self.tokenizer.pad_token_id}")
        else:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
 
    def _setup_model(self):
        local_rank   = int(os.environ.get("LOCAL_RANK", "0"))
        device_map   = {"": local_rank} if torch.cuda.device_count() > 1 else "auto"
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
 
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        )
        logger.info(f"Loading model: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=compute_dtype, low_cpu_mem_usage=True,
            quantization_config=nf4_config, use_cache=False,
            device_map=device_map, trust_remote_code=True,
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if self._is_qwen() and len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self._setup_lora()
        torch.cuda.empty_cache()
        gc.collect()
 
    def _setup_lora(self):
        self.model = prepare_model_for_kbit_training(self.model)
        if self._is_qwen():
            # lm_head tied to embed_tokens in Qwen2.5 — must not appear in both lists
            target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"]
            modules_to_save = ["norm"]
        else:
            target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj", "lm_head"]
            modules_to_save = ["norm", "embed_tokens"]
 
        lora_config = LoraConfig(
            r=64, lora_alpha=128, target_modules=target_modules,
            lora_dropout=0.05, task_type="CAUSAL_LM",
            fan_in_fan_out=False, bias="lora_only", modules_to_save=modules_to_save,
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA | trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
 
    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
 
    def _extract_diagnosis(self, text):
        text = text.lower().strip()
        for m in ["diagnosis:", "condition:", "answer:", "output:"]:
            if m in text:
                text = text.split(m)[-1].strip()
        return re.sub(r"[^\w\s-]", "", text).split("\n")[0].split(".")[0].strip()
 
    def _split_reasoning(self, text):
        text = text.strip()
        for m in ["conclusion:", "final answer:", "assessment:", "therefore,"]:
            if m in text.lower():
                idx = text.lower().rfind(m)
                return text[:idx].strip(), text[idx:].strip()
        s = text.split(".")
        return (".".join(s[:-1]).strip(), s[-1].strip()) if len(s) > 1 else (text, text)
 
    def compute_metrics(self, eval_preds):
        predictions, labels = eval_preds
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        labels      = np.where(labels      != -100, labels,      self.tokenizer.pad_token_id)
        try:
            preds  = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            refs   = self.tokenizer.batch_decode(labels,      skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            return {"eval_loss": 0.0}
 
        tp, tl = defaultdict(list), defaultdict(list)
        for p, l in zip(preds, refs):
            key = ("Clinical Diagnosis"   if "diagnosis:" in p.lower() or "diagnosis:" in l.lower() else
                   "Clinical Reasoning"   if "reasoning:" in p.lower() or "reasoning:" in l.lower() else
                   "Medical Question Answering")
            tp[key].append(p); tl[key].append(l)
 
        metrics = {}
        if tp["Clinical Diagnosis"]:
            correct = sum(1 for p, r in zip(tp["Clinical Diagnosis"], tl["Clinical Diagnosis"])
                          if self._extract_diagnosis(p) == self._extract_diagnosis(r))
            metrics["CD_accuracy"] = correct / len(tp["Clinical Diagnosis"])
        if tp["Medical Question Answering"]:
            scores = [scorer.score(r, p)["rougeL"].fmeasure
                      for p, r in zip(tp["Medical Question Answering"], tl["Medical Question Answering"])
                      if p.strip() and r.strip()]
            metrics["MQA_rougeL"] = float(np.mean(scores)) if scores else 0.0
        if tp["Clinical Reasoning"]:
            correct = 0
            for p, r in zip(tp["Clinical Reasoning"], tl["Clinical Reasoning"]):
                _, pc = self._split_reasoning(p); _, rc = self._split_reasoning(r)
                if pc.strip() and rc.strip() and scorer.score(rc, pc)["rougeL"].fmeasure >= 0.3:
                    correct += 1
            metrics["CR_conclusion_accuracy"] = correct / len(tp["Clinical Reasoning"])
 
        values = [v for v in metrics.values() if isinstance(v, float)]
        metrics["eval_score"] = float(np.mean(values)) if values else 0.0
        logger.info(f"[ICIP] Eval: {metrics}")
        del preds, refs, predictions, labels
        gc.collect()
        return metrics
 
    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
 
    def train(self, data_path_or_df):
        logger.info("=== ICIP TRAINING START ===")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
 
        # ---- 1. ICIP objects ---- #
        icip_manager = ICIPManager(
            device=device, encoder_cls=self.encoder_cls,
            policy_lr=1e-3, policy_beta=0.1, entropy_coeff=0.01,
            warmup_steps=200, debug=False,
        )
        if hasattr(icip_manager.encoder, '_current_task') and not isinstance(icip_manager.encoder, ExampleShuffledEncoder):
            from task_prior_encoder import patch_icip_manager_for_task_prior
            icip_manager = patch_icip_manager_for_task_prior(icip_manager)
        icip_callback = ICIPCallback(manager=icip_manager,
                                     policy_update_every_n_optimizer_steps=1)
        collator = ICIPDynamicCollator(tokenizer=self.tokenizer, icip_manager=icip_manager,
                                       icip_callback=icip_callback, max_length=self.max_length)
        self.prompt_manager = icip_manager
        logger.info(f"[ICIP] Manager ready | encoder={self.encoder_cls.__name__} | K=6")
        logger.info(f"[ICIP] Sanity: {icip_manager.format_prompt('Clinical Reasoning','test')[:80]}")
 
        # ---- 2. Load data ---- #
        if isinstance(data_path_or_df, pd.DataFrame):
            df = data_path_or_df
        elif isinstance(data_path_or_df, str):
            df = pd.read_csv(data_path_or_df)
        else:
            raise TypeError("data_path_or_df must be a path string or DataFrame")
        logger.info(f"Loaded: {df.shape} | {df['task'].value_counts().to_dict()}")
 
        # ---- 3. Balance + split ---- #
        datasets_dir = os.path.join(self.output_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        balanced_df = sample_balanced_dataset(df, samples_per_task=10000)
        train_df, val_df = train_test_split(
            balanced_df, test_size=0.1, random_state=42, stratify=balanced_df["task"])
        train_df.to_csv(os.path.join(datasets_dir, "train_split.csv"), index=False)
        val_df.to_csv(os.path.join(datasets_dir, "val_split.csv"), index=False)
        logger.info(f"Split | train={len(train_df)} | val={len(val_df)}")
 
        # ---- 4. HuggingFace datasets in RAW mode ---- #
        tok_ref, ml_ref = self.tokenizer, self.max_length
 
        def _raw_batch(ex):
            return preprocess_raw(ex, tok_ref, ml_ref)
 
        train_dataset = HFDataset.from_pandas(train_df.reset_index(drop=True)) \
            .map(_raw_batch, batched=True, remove_columns=train_df.columns.tolist(),
                 desc="Train (raw)")
        val_dataset = HFDataset.from_pandas(val_df.reset_index(drop=True)) \
            .map(_raw_batch, batched=True, remove_columns=val_df.columns.tolist(),
                 desc="Val (raw)")
        train_dataset.set_format(type=None)
        val_dataset.set_format(type=None)
 
        sample = train_dataset[0]
        assert "raw_input" in sample and "task" in sample and "example_id" in sample and "raw_output" in sample, \
            f"Missing keys: {list(sample.keys())}"
        logger.info(f"Datasets | train={len(train_dataset)} | val={len(val_dataset)}")

        # Precompute example-swap table after datasets are ready
        if isinstance(icip_manager.encoder, ExampleShuffledEncoder):
            logger.info("[ICIP] Precomputing example-swap table...")
            swap_df = pd.DataFrame({
                "example_id": train_dataset["example_id"],
                "raw_input":  train_dataset["raw_input"],
                "task":       train_dataset["task"],
            })
            icip_manager.encoder.precompute_swap_table(
                swap_df, input_col="raw_input", id_col="example_id"
            )
            icip_manager = patch_manager_for_example_shuffle(icip_manager)
            logger.info("[ICIP] Example-swap table ready and manager patched")
 
        # ---- 5. DeepSpeed config ---- #
        # Use DeepSpeed only when running with multiple GPUs via deepspeed launcher.
        # For single-GPU runs (CUDA_VISIBLE_DEVICES=N python ...), set to None.
        num_gpus_visible = torch.cuda.device_count()
        use_deepspeed    = num_gpus_visible > 1
 
        if use_deepspeed:
            ds_config = {
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": 48,
                "steps_per_print": 1,
                "train_batch_size": "auto",
                "zero_optimization": {
                    "stage": 2,
                    "reduce_bucket_size": 5e7,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 5e7,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                },
                "zero_allow_untested_optimizer": True,
            }
            logger.info(f"[ICIP] Multi-GPU mode: DeepSpeed enabled ({num_gpus_visible} GPUs)")
        else:
            ds_config = None
            logger.info(f"[ICIP] Single-GPU mode: DeepSpeed disabled")
 
    
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            gradient_accumulation_steps=48,
            gradient_checkpointing=True,
            fp16=True,
            learning_rate=2e-5,
            lr_scheduler_type="cosine_with_restarts",
            warmup_ratio=0.05,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            
            remove_unused_columns=REMOVE_UNUSED_COLUMNS,
            ddp_find_unused_parameters=False,
            report_to="none",
            local_rank=local_rank,
            deepspeed=ds_config,   
        )
 
    
        trainer = ICIPTrainer(
            model=self.model, args=training_args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=collator, compute_metrics=None,
            callbacks=[icip_callback],
            icip_callback=icip_callback,
        )
        icip_callback._trainer = trainer
        logger.info("[ICIP] Trainer ready")
 
        # ---- 8. Train ---- #
        try:
            trainer.train()
            logger.info("[ICIP] Training complete")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            
            torch.cuda.empty_cache()
            gc.collect()
 
        # ---- 9. Save (rank 0 only) ---- #
        if local_rank == 0:
            try:
                model_path = os.path.join(self.output_dir, "final_model")
                trainer.save_model(model_path)
                self.tokenizer.save_pretrained(model_path)
                logger.info(f"Model saved -> {model_path}")
            except Exception as e:
                logger.error(f"Model save failed: {e}")
 
            try:
                icip_manager.save_policy(self.output_dir)
            except Exception as e:
                logger.error(f"Policy save failed: {e}")
 
            try:
                icip_manager.save_logs(self.output_dir)
            except Exception as e:
                logger.error(f"Log save failed: {e}")
 
            try:
                stats = icip_manager.get_stats()
                with open(os.path.join(self.output_dir, "icip_final_stats.json"), "w") as f:
                    json.dump(stats, f, indent=2, default=str)
 
                post = [s for s in icip_manager.selection_history if s["mode"] == "policy"]
                analysis = {}
                for task in icip_manager.TEMPLATES:
                    perf   = icip_manager.template_performance[task]
                    counts = Counter(s["template_idx"] for s in post if s["task"] == task)
                    total  = sum(counts.values()) or 1
                    analysis[task] = {
                        "baseline_reward": round(icip_manager.baselines[task], 4),
                        "reward_std":      round(icip_manager.reward_stds[task], 4),
                        "functions": {
                            ICIP_FUNCTION_LABELS[task][i]: {
                                "selection_pct":   round(100 * counts.get(i, 0) / total, 2),
                                "performance_score": round(perf.get(i, 0.0), 4),
                            }
                            for i in range(len(icip_manager.TEMPLATES[task]))
                        },
                        "top_3": [
                            {"function": ICIP_FUNCTION_LABELS[task][i], "pct": round(100 * c / total, 2)}
                            for i, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        ],
                    }
                with open(os.path.join(self.output_dir, "icip_function_analysis.json"), "w") as f:
                    json.dump(analysis, f, indent=2, default=str)
                logger.info("Stats + analysis saved")
            except Exception as e:
                logger.error(f"Stats save failed: {e}")
 
            try:
                val_list = [{"example_id": val_dataset[i].get("example_id", f"val_{i}"),
                             "task": val_dataset[i]["task"],
                             "raw_input": val_dataset[i]["raw_input"]}
                            for i in range(len(val_dataset))]
                icip_manager.export_eval_routing(val_list, self.output_dir)
            except Exception as e:
                logger.error(f"Eval routing export failed: {e}")
 
            logger.info("\n[ICIP] === TOP FUNCTIONS PER TASK ===")
            for task in icip_manager.TEMPLATES:
                ranked = sorted(icip_manager.template_performance[task].items(),
                                key=lambda x: x[1], reverse=True)
                logger.info(f"  {task}: " + " | ".join(
                    f"{ICIP_FUNCTION_LABELS[task][i]}={s:.3f}" for i, s in ranked[:3]))
 
        logger.info("=== ICIP TRAINING END ===")
        return trainer
    
    def train_random_baseline(self, data_path_or_df, seed: int = 42):
        """
        Uniform random instruction-function baseline.

        Same:
        - dataset
        - train/validation split
        - QLoRA configuration
        - six instruction functions
        - target model
        - training schedule

        Different:
        - no clinical encoder
        - no policy network
        - no reward
        - no entropy regularization
        - no learned router
        """

        logger.info("=== RANDOM INSTRUCTION BASELINE TRAINING START ===")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        random_manager = RandomInstructionManager(seed=seed)

        collator = RandomInstructionCollator(
            tokenizer=self.tokenizer,
            random_manager=random_manager,
            max_length=self.max_length,
        )

        # ---- Load data ---- #
        if isinstance(data_path_or_df, pd.DataFrame):
            df = data_path_or_df
        elif isinstance(data_path_or_df, str):
            df = pd.read_csv(data_path_or_df)
        else:
            raise TypeError("data_path_or_df must be a path string or DataFrame")

        logger.info(f"Loaded: {df.shape} | {df['task'].value_counts().to_dict()}")

        # ---- Balance + split ---- #
        datasets_dir = os.path.join(self.output_dir, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        balanced_df = sample_balanced_dataset(df, samples_per_task=10000)

        train_df, val_df = train_test_split(
            balanced_df,
            test_size=0.1,
            random_state=42,
            stratify=balanced_df["task"],
        )

        train_df.to_csv(os.path.join(datasets_dir, "train_split.csv"), index=False)
        val_df.to_csv(os.path.join(datasets_dir, "val_split.csv"), index=False)

        logger.info(f"Split | train={len(train_df)} | val={len(val_df)}")

        # ---- HuggingFace datasets in raw mode ---- #
        tok_ref, ml_ref = self.tokenizer, self.max_length

        def _raw_batch(ex):
            return preprocess_raw(ex, tok_ref, ml_ref)

        train_dataset = HFDataset.from_pandas(train_df.reset_index(drop=True)).map(
            _raw_batch,
            batched=True,
            remove_columns=train_df.columns.tolist(),
            desc="Train random baseline raw",
        )

        val_dataset = HFDataset.from_pandas(val_df.reset_index(drop=True)).map(
            _raw_batch,
            batched=True,
            remove_columns=val_df.columns.tolist(),
            desc="Val random baseline raw",
        )

        train_dataset.set_format(type=None)
        val_dataset.set_format(type=None)

        logger.info(f"Datasets | train={len(train_dataset)} | val={len(val_dataset)}")

        # ---- DeepSpeed config ---- #
        num_gpus_visible = torch.cuda.device_count()
        use_deepspeed = num_gpus_visible > 1

        if use_deepspeed:
            ds_config = {
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": 48,
                "steps_per_print": 1,
                "train_batch_size": "auto",
                "zero_optimization": {
                    "stage": 2,
                    "reduce_bucket_size": 5e7,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 5e7,
                    #"offload_optimizer": {"device": "cpu", "pin_memory": True},
                },
                "fp16": {
                    "enabled": True,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                },
                "zero_allow_untested_optimizer": True,
            }
            logger.info(f"[RandomBaseline] Multi-GPU mode: DeepSpeed enabled ({num_gpus_visible} GPUs)")
        else:
            ds_config = None
            logger.info("[RandomBaseline] Single-GPU mode: DeepSpeed disabled")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=1,
            prediction_loss_only=True,
            gradient_accumulation_steps=48,
            gradient_checkpointing=True,
            fp16=True,
            learning_rate=2e-5,
            lr_scheduler_type="cosine_with_restarts",
            warmup_ratio=0.05,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
            remove_unused_columns=REMOVE_UNUSED_COLUMNS,
            ddp_find_unused_parameters=False,
            report_to="none",
            local_rank=local_rank,
            deepspeed=ds_config,
        )

        trainer = ICIPTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=None,
            callbacks=[],
            icip_callback=None,  # no policy, just the loss computation
        )
        logger.info("[RandomBaseline] Trainer ready")

        try:
            trainer.train()
            logger.info("[RandomBaseline] Training complete")
        except Exception as e:
            logger.error(f"[RandomBaseline] Training failed: {e}")
            raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        # ---- Save rank 0 only ---- #
        if local_rank == 0:
            try:
                model_path = os.path.join(self.output_dir, "final_model")
                trainer.save_model(model_path)
                self.tokenizer.save_pretrained(model_path)
                logger.info(f"[RandomBaseline] Model saved -> {model_path}")
            except Exception as e:
                logger.error(f"[RandomBaseline] Model save failed: {e}")

            try:
                random_manager.save_logs(self.output_dir)
            except Exception as e:
                logger.error(f"[RandomBaseline] Log save failed: {e}")

            try:
                stats = random_manager.get_stats()
                with open(os.path.join(self.output_dir, "random_final_stats.json"), "w") as f:
                    json.dump(stats, f, indent=2, default=str)

                logger.info("[RandomBaseline] Final stats saved")
            except Exception as e:
                logger.error(f"[RandomBaseline] Stats save failed: {e}")

            try:
                val_list = [
                    {
                        "example_id": val_dataset[i].get("example_id", f"val_{i}"),
                        "task": val_dataset[i]["task"],
                        "raw_input": val_dataset[i]["raw_input"],
                    }
                    for i in range(len(val_dataset))
                ]

                random_manager.export_eval_routing(val_list, self.output_dir)
            except Exception as e:
                logger.error(f"[RandomBaseline] Eval routing export failed: {e}")

        logger.info("=== RANDOM INSTRUCTION BASELINE TRAINING END ===")
        return trainer
 
# ===========================================================================
# 10.  Entry point
# ===========================================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    processor = MemoryEfficientDataProcessor(
        model_path="/home/share/henry/model_download/Qwen2.5-7B-Instruct",
        tokenizer_path="/home/share/henry/model_download/Qwen2.5-7B-Instruct",
        output_dir="/home/share/henry/ICIP_ROUTER/QWEN/random_baseline",
        max_length=512,
    )

    processor.train_random_baseline(
        "/home/share/henry/trainset.csv",
        seed=42,
    )
#if __name__ == "__main__":
    #import multiprocessing
    #multiprocessing.set_start_method("spawn", force=True)
 

   # processor = MemoryEfficientDataProcessor(
   #model_path="/home/share/henry/model_download/Qwen2.5-7B-Instruct",
   # tokenizer_path="/home/share/henry/model_download/Qwen2.5-7B-Instruct",
   # output_dir="/home/share/henry/ICIP_ROUTER/QWEN/example_shuffled_control",
   # encoder_cls=ExampleShuffledEncoder, 
#)
#processor.train("/home/share/henry/trainset.csv")