#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python sentence_alignment/train_model_script.py \
    --train_dataset "${TRAIN_DATASET:-dataset/train.json}" \
    --max_evi "${MAX_EVIDENCE_COUNT:-8}" \
    --train_epoch "${TRAIN_EPOCHS:-5}" \
    --batch_size "${TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"

CHECKPOINT_DIR="$(find sentence_alignment/results-model -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sort -t- -k2,2n | tail -n 1)"
if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "No training checkpoint was created in sentence_alignment/results-model" >&2
    exit 1
fi

python sentence_alignment/predict_model_script.py \
    --model_dir "sentence_alignment/results-model/$CHECKPOINT_DIR" \
    --test_data "${TEST_DATASET:-dataset/test.json}" \
    --output_file "${ALIGNMENT_OUTPUT:-test_alignment.json}" \
    --max_evidence_count "${PREDICT_EVIDENCE_COUNT:-4}"