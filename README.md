# Clinical Instruction Routing

Code and result files for the paper:

> **Reward-Guided, Representation-Conditioned Instruction Routing for Clinical LLM Fine-Tuning**
> Henry Mukalazi Serugunda, Ouyang Jianquan, Jacob Katende.
> IEEE BIBM 2026.

During fine-tuning, a small policy network reads a frozen Bio-ClinicalBERT embedding of each training example and chooses one of six instruction functions for it. The chosen instruction is used to build the training example, and the policy is updated with a reward equal to the negative target-output loss of the language model. The repository contains the router, five comparison conditions, the evaluation pipeline and the analysis scripts used in the paper.

## Main findings

- Learned routing produced task-specific routing preferences that differed from all controls, while validation losses were nearly identical (0.911 to 0.912).
- Random instruction assignment reduced Clinical Diagnosis exact match from 0.611 (single direct-answer instruction) to 0.523.
- The learned router avoided this loss (0.620) but did not significantly exceed the single instruction (+0.008, 95% CI [-0.014, 0.031], p = 0.55).
- Differences on Clinical Reasoning and Medical Question Answering were about 0.01 or less.
- Llama-3.1-8B-Instruct learned different routing preferences and did not reproduce the diagnosis result.

## Repository layout

```
clinical-instruction-routing/
├── src/
│   ├── train_router_qwen.py        # router training; also contains the random-baseline
│   │                               # and shuffled-embedding encoder classes
│   ├── train_router_llama.py       # same pipeline for Llama-3.1-8B-Instruct
│   ├── train_static_baseline.py    # single fixed instruction (F0) baseline
│   ├── task_prior_encoder.py       # task-prior control
│   └── example_shuffled_encoder.py # example-shuffled control
├── data/
│   └── filter_clinical_data.py     # removes non-clinical items from the evaluation split
├── evaluation/
│   ├── downstream_eval.py          # generation and metrics for every condition
│   ├── paired_tests.py             # McNemar and Wilcoxon tests, learned vs random
│   └── paired_baseline_tests.py    # learned vs static vs random, bootstrap intervals
├── analysis/
│   ├── routing_stats.py            # Jensen-Shannon distances, chi-square tests
│   ├── encoder_screening.py        # cosine-similarity encoder screen (Table I)
│   ├── textual_cue_enrichment.py   # input terms associated with each route
│   ├── example_table.py            # routed examples
│   ├── analysis.py                 # routing distributions and entropy curves
│   └── make_fig7.py                # control-comparison figure
├── results/                        # metrics, routing logs and statistics per condition
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Conditions

| Condition | What the router receives | Instruction at evaluation |
|---|---|---|
| Learned router | Bio-ClinicalBERT embedding of the example | Greedy route from its policy |
| Static baseline | No router | F0 (direct answer) for every example |
| Random baseline | No router | One function per example from a seeded hash, fixed across epochs |
| Task-prior control | A fixed vector per task | Greedy route from its policy |
| Shuffled-embedding control | The embedding with a new random feature permutation for every example | Greedy route from its policy |
| Example-shuffled control | The embedding of another example from the same task | Greedy route from its policy |

Each condition fine-tunes a separate model under the same settings, and each model is evaluated with the instructions of its own condition.

## Setup

```bash
conda create -n icif python=3.10 -y
conda activate icif
pip install -r requirements.txt
```

Models (download from Hugging Face):
- Target models: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`
- Encoder: `emilyalsentzer/Bio_ClinicalBERT`

Model and data paths are set at the top of each script. Edit them to match your machine before running.

## Data

The corpus combines three public datasets:

| Task | Source |
|---|---|
| Clinical Diagnosis | [SymptomsDisease246k](https://huggingface.co/datasets/fhai50032/SymptomsDisease246k) |
| Clinical Reasoning | [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) |
| Medical Question Answering | MedQuAD (Ben Abacha and Demner-Fushman, 2019) |

From 37,723 preprocessed examples, Medical Question Answering and Clinical Reasoning are capped at 10,000 each (seed 42), giving 29,874 examples. A stratified 90/10 split gives 26,886 training and 2,988 held-out examples. Ten non-clinical Clinical Reasoning items are removed from the held-out split by `data/filter_clinical_data.py`, leaving 2,978 for downstream evaluation. Datasets are not redistributed here; follow each source's licence.

## Training settings

| Setting | Value |
|---|---|
| Fine-tuning | QLoRA, 4-bit NF4, rank 64, alpha 128, dropout 0.05 |
| Epochs, learning rate | 3, 2e-5, cosine with restarts, warmup ratio 0.05 |
| Batch | 1 per device, 48 accumulation steps, 2 GPUs (effective 96) |
| Sequence length | 512 tokens (encoder input truncated to 128) |
| Precision and parallelism | FP16, gradient checkpointing, DeepSpeed ZeRO-2 with CPU optimizer offload |
| Policy | MLP, 768 to 256 to 6, dropout 0.1, Adam 1e-3 |
| Policy objective | REINFORCE, beta 0.1, entropy coefficient 0.01, 200-decision round-robin warmup |
| Seed | 42, one run per condition |

## Reproducing the paper

1. Train each condition with the scripts in `src/` (two 24 GB GPUs, DeepSpeed). Conditions other than the learned router are selected in the `__main__` block of the training script (encoder class and output directory).
2. Run `evaluation/downstream_eval.py` to generate outputs and metrics for every condition.
3. Run `evaluation/paired_tests.py` and `evaluation/paired_baseline_tests.py` for the significance tests.
4. Run `analysis/routing_stats.py` for routing-distribution statistics.

The `results/` folder holds the metric files and routing logs behind the tables, so steps 3 and 4 can be rerun without training.

## Limitations

The six-function instruction bank was defined by the authors and has not been validated by clinicians. Each condition was trained once, so variation across seeds is not measured. Results come from two models of 7 to 8 billion parameters and secondary text datasets. This code is for research only and must not be used for clinical decisions.

## Related work by the authors

- CAIT: *A Context Adaptive Instruction Tuning Framework for Diverse Clinical NLP Tasks*, IEEE BIBM 2025. doi:10.1109/BIBM66473.2025.11356497
- PAITA: *Performance-Adaptive Instruction-Template Assignment for Clinical Reasoning in Large Language Models*, ACM CIKM 2026. doi:10.1145/3799682.3840978. Code: [keriel2/PAITA](https://github.com/keriel2/PAITA)

## Citation

```bibtex
@inproceedings{serugunda2026routing,
  author    = {Serugunda, Henry Mukalazi and Ouyang, Jianquan and Katende, Jacob},
  title     = {Reward-Guided, Representation-Conditioned Instruction Routing for Clinical LLM Fine-Tuning},
  booktitle = {2026 IEEE International Conference on Bioinformatics and Biomedicine},
  year      = {2026}
}
```

## Licence

MIT. See `LICENSE`.
