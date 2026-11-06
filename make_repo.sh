#!/usr/bin/env bash
# Assembles the clinical-instruction-routing release from the training server.
# Run from anywhere:  bash make_repo.sh
# Then review the printed report before pushing.
set -euo pipefail

SRC=/home/share/henry/ICIP_ROUTER          # existing experiment folder
DST=/home/share/henry/clinical-instruction-routing          # release folder (created)
KIT=$(cd "$(dirname "$0")" && pwd)         # folder holding README.md etc.

mkdir -p "$DST"/{src,data,evaluation,analysis,results}

# ---- code (renamed for clarity) ----
cp "$SRC/ICIPL-qwentrain.py"           "$DST/src/train_router_qwen.py"
cp "$SRC/ICIPL-llamatrain.py"          "$DST/src/train_router_llama.py"
cp "$SRC/staticbaseline.py"            "$DST/src/train_static_baseline.py"
cp "$SRC/task_prior_encoder.py"        "$DST/src/task_prior_encoder.py"
cp "$SRC/example_shuffled_encoder.py"  "$DST/src/example_shuffled_encoder.py"
cp "$SRC/filter_clinical_data.py"      "$DST/data/filter_clinical_data.py"
cp "$SRC/downstream_sanity_eval.py"    "$DST/evaluation/downstream_eval.py"
cp "$SRC/paired_tests.py"              "$DST/evaluation/paired_tests.py"
cp "$SRC/routing_stats.py"             "$DST/analysis/routing_stats.py"
cp "$SRC/encoderablation.py"           "$DST/analysis/encoder_screening.py"
cp "$SRC/textual_cue_enrichment.py"    "$DST/analysis/textual_cue_enrichment.py"
cp "$SRC/exampletable.py"              "$DST/analysis/example_table.py"
cp "$SRC/analysis.py"                  "$DST/analysis/analysis.py"
cp "$SRC/generate_fig7_combined.py"    "$DST/analysis/make_fig7.py"
# debug_eval.py is intentionally left out.

# ---- kit files ----
cp "$KIT"/{README.md,LICENSE,.gitignore,CITATION.cff} "$DST/"
cp "$KIT/evaluation/paired_baseline_tests.py" "$DST/evaluation/"

# ---- results: small metric, routing and statistics files only ----
for model in QWEN LLAMA; do
  [ -d "$SRC/$model" ] || continue
  for d in "$SRC/$model"/*/; do
    name=$(basename "$d")
    case "$name" in *quick*|*bertscore_test*) continue;; esac
    out="$DST/results/$(echo $model | tr A-Z a-z)/$name"
    mkdir -p "$out"
    find "$d" -maxdepth 1 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.txt" \) \
         -size -50M -exec cp {} "$out/" \;
    [ -n "$(ls -A "$out")" ] || rmdir "$out"
  done
done

# ---- exact package versions ----
pip freeze 2>/dev/null | grep -iE "^(torch|transformers|peft|bitsandbytes|deepspeed|accelerate|datasets|pandas|numpy|scipy|scikit-learn|rouge-score|rouge_score|bert-score|bert_score|nltk|spacy|matplotlib|seaborn|psutil)==" \
  > "$DST/requirements.txt" || true

# ---- report ----
echo "=== Imports between moved files (must still resolve) ==="
grep -n "^from \|^import " "$DST"/src/*.py | grep -E "encoder|ICIP|static|filter" || echo "none"
echo
echo "=== Hard-coded absolute paths (edit to relative or config values) ==="
grep -rn "/home/\|/root/" "$DST" --include=*.py | cut -c1-160 || echo "none"
echo
echo "=== Possible secrets (must be empty before pushing) ==="
grep -rnE "hf_[A-Za-z0-9]{20,}|api[_-]?key|token\s*=\s*['\"][A-Za-z0-9]" "$DST" --include=*.py || echo "none found"
echo
echo "=== Size ==="
du -sh "$DST"; find "$DST" -type f -size +20M -exec ls -lh {} \;
echo
echo "Release assembled in $DST"
