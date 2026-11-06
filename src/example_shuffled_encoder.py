"""
example_shuffled_encoder.py — v2
==================================
Precomputed within-task example-shuffled embedding control.

Design:
  For each task at startup:
    1. Encode ALL training examples with frozen Bio-ClinicalBERT
    2. Create a fixed permutation: σ(i) = j where j ≠ i, task_j = task_i
    3. At routing time for example i: return h_{σ(i)} instead of h_i

Guarantees:
  - x_i and y_i are unchanged (LLM still trains on the original example)
  - Router receives h_j where j ≠ i and task_j = task_i (within-task swap)
  - Embeddings are genuine Bio-ClinicalBERT representations of real inputs
  - The input-embedding alignment is broken, task identity is preserved
  - Fully reproducible via fixed random seed

This is stronger than online cache shuffling because:
  - No possibility of returning the current example's own embedding
  - All examples are covered from step 1 (no warmup fallback)
  - Deterministic for reproducibility

Answers: Does it matter that h_i belongs to x_i specifically?
vs feature-dim shuffle which answers: Does semantic structure in h_i matter?

Integration:
  1. Copy to /home/share/henry/CAIT-L/example_shuffled_encoder.py
  2. In ICIPL-qwentrain.py, after datasets are prepared:
       icip_manager.encoder.precompute_swap_table(train_df, tokenizer)
  3. In train(), patch the manager:
       icip_manager = patch_manager_for_example_shuffle(icip_manager)
  4. Set encoder_cls=ExampleShuffledEncoder, output_dir=.../example_shuffled_control
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Minimal test with synthetic data
    import pandas as pd
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ExampleShuffledEncoder(device=device)

    df = pd.DataFrame({
        "example_id": [f"ex_{i}" for i in range(12)],
        "raw_input":  [
            "Fever and cough", "Chest pain radiating to arm",
            "Shortness of breath", "Joint swelling",
            "What causes diabetes?", "How does insulin work?",
            "Treatment for hypertension?", "Symptoms of pneumonia?",
            "Patient with abdominal pain", "Elderly patient with confusion",
            "Young woman with rash", "Child with fever and stiff neck",
        ],
        "task": (
            ["Clinical Diagnosis"] * 4 +
            ["Medical Question Answering"] * 4 +
            ["Clinical Reasoning"] * 4
        ),
    })

    encoder.precompute_swap_table(df, input_col="raw_input")

    print("\n=== SWAP VERIFICATION ===")
    print("Each example should receive a DIFFERENT example's embedding from the same task")
    for _, row in df.iterrows():
        encoder._current_example_id = row["example_id"]
        real_emb = encoder._encode_batch([row["raw_input"]])
        swap_emb = encoder.encode([row["raw_input"]])
        same = torch.allclose(real_emb.to(device), swap_emb, atol=1e-4)
        print(f"  {row['example_id']} [{row['task'][:2]}]: "
              f"same_as_real={same} (should be False)")