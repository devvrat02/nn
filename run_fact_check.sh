#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LLM_MODEL_PATH="${LLM_MODEL_PATH:-/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B}"
if [[ ! -f "$LLM_MODEL_PATH/config.json" && -f "$LLM_MODEL_PATH/FP16/config.json" ]]; then
    LLM_MODEL_PATH="$LLM_MODEL_PATH/FP16"
    export LLM_MODEL_PATH
fi

if [[ ! -f "$LLM_MODEL_PATH/config.json" ]]; then
    echo "No config.json found under LLM_MODEL_PATH=$LLM_MODEL_PATH" >&2
    exit 1
fi

python -m method.claim_verification_hiss \
    --datapath "${DATASET:-dataset/test.json}"
