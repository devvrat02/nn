#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATAFILE="test_alignment.json" # the output of evidence alignment
LITERAL="method/results/literal/log.jsonl"
: "${INTENT_MODEL:?Set INTENT_MODEL to your accessible intent model ID}"

python -m method.reassessment --datafile "$DATAFILE" --literal "$LITERAL" --intent_model "$INTENT_MODEL"
