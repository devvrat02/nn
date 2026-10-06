#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -m method.claim_verification_hiss --datapath dataset/test.json
