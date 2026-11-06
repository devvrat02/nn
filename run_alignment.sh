#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" && -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
fi
PYTHON="${PYTHON:-python}"
MODEL_DIR="${ALIGNMENT_MODEL_DIR:-sentence_alignment/results-model}"

"$PYTHON" sentence_alignment/train_model_script.py \
    --train_dataset "${TRAIN_DATASET:-dataset/train.json}" \
    --max_evi "${MAX_EVIDENCE_COUNT:-8}" \
    --train_epoch "${TRAIN_EPOCHS:-5}" \
    --batch_size "${TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --output_dir "$MODEL_DIR"

CHECKPOINT_DIR="$(find "$MODEL_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sort -t- -k2,2n | tail -n 1)"
if [[ -z "$CHECKPOINT_DIR" ]]; then
    echo "No training checkpoint was created in $MODEL_DIR" >&2
    exit 1
fi

PREDICT_ARGS=(
    --model_dir "$MODEL_DIR/$CHECKPOINT_DIR"
    --test_data "${TEST_DATASET:-dataset/test.json}"
    --output_file "${ALIGNMENT_OUTPUT:-test_alignment.json}"
    --max_evidence_count "${PREDICT_EVIDENCE_COUNT:-4}"
    --start_index "${PREDICT_START_INDEX:-0}"
    --do_eval
)
if [[ -n "${PREDICT_END_INDEX:-}" ]]; then
    PREDICT_ARGS+=(--end_index "$PREDICT_END_INDEX")
fi
"$PYTHON" sentence_alignment/predict_model_script.py "${PREDICT_ARGS[@]}"