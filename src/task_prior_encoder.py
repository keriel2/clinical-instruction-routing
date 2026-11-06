"""
task_prior_encoder.py
=====================
Input-agnostic task-prior control for ICIP-Router.

This encoder removes example-specific clinical input information. Each task
receives a fixed 768-dimensional vector, and all examples from the same task
receive the same vector. This tests whether the router learns only task-level
instruction-function priors rather than input-conditioned routing behavior.

Fixes vs previous version:
  - No Bio-ClinicalBERT needed — uses fixed deterministic block vectors
  - Not trainable (correct — the control should be fixed, not learned)
  - Patches both format_prompt_train() and format_prompt() (eval)
  - Also patches export_eval_routing() via a pre-encode hook
  - Wording corrected: "fixed" not "trainable"

Integration:
  1. Copy to /home/share/henry/CAIT-L/task_prior_encoder.py
  2. Add to ICIPL-qwentrain.py imports:
       from task_prior_encoder import TaskPriorEncoder, patch_icip_manager_for_task_prior
  3. Change __main__ block:
       encoder_cls=TaskPriorEncoder
       output_dir="/home/share/henry/ICIP_ROUTER/QWEN/task_prior_control"
  4. In train(), after icip_manager = ICIPManager(...), add:
       icip_manager = patch_icip_manager_for_task_prior(icip_manager)
  5. Run with same deepspeed command as real experiment.

What to look for in results:
  - Task-prior entropy stays HIGH (~1.79) throughout training:
    input embeddings matter — task label alone is not enough to specialize
  - Task-prior entropy drops similarly to real encoder:
    router is mainly learning task-level priors, input content adds little
  - Task-prior eval-time function distributions differ from real encoder:
    input content influences which specific functions are selected
"""

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