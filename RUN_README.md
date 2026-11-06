# Quick benchmark

From `nn/`, initialize the project environment:

```bash
source .venv/bin/activate
```

Then compare HiSS, CoT, and TRACER (HiSS followed by reassessment) on the first
200 test facts:

```bash
bash run_fact_check_benchmark.sh
```

The benchmark reports accuracy and F1 for all three methods, plus the number
of distinct CHE evidence items found by TRACER. Results are saved under
`method/result/<timestamp>_benchmark/`. It runs one GPU process at a time.

## Compare final classifier prompts

To compare the original final-classification prompt against the structured
component-wise prompt on the same first 200 examples and saved HiSS/alignment
outputs, run:

```bash
LLM_MODEL_PATH=/mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16 \
AUXILIARY_MODEL_DEVICE=cuda \
.venv/bin/python -m method.compare_tracer_filtering \
  --comparison-mode classifier \
  --dataset dataset/test.json \
  --alignment-file test_alignment_evidence_1_200_fixed.json \
  --hiss-log method/result/20261104_213400_650571_benchmark/hiss/log.jsonl \
  --model /mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16 \
  --start-index 0 --end-index 200
```

Each run saves separate metrics, confusion matrices, predictions, and raw final
classifier responses under a new timestamped directory. The `--classifier-prompt`
option on `method.reassessment` can select `original` (the default) or
`structured` for an individual run. Use the same instruction-tuned checkpoint
for both configurations.

To compare the opt-in Evidence Consistency Check against original TRACER on all
200 examples, use:

```bash
LLM_MODEL_PATH=/mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16 \
AUXILIARY_MODEL_DEVICE=cuda \
.venv/bin/python -m method.compare_tracer_filtering \
  --comparison-mode evidence-consistency \
  --dataset dataset/test.json \
  --alignment-file test_alignment_evidence_1_200_fixed.json \
  --hiss-log method/result/20261104_213400_650571_benchmark/hiss/log.jsonl \
  --model /mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16 \
  --start-index 0 --end-index 200
```

This mode checks every claim against the supplied evidence; it does not use
ground-truth labels as model input. The original mode remains unchanged and is
the default for `method.reassessment`.

To select another range, use zero-based indexes; the end is exclusive:

```bash
FACT_CHECK_START=200 FACT_CHECK_END=400 bash run_fact_check_benchmark.sh
```

## Decision Consistency Guard (development before locked evaluation)

The optional Decision Consistency Guard is disabled by default. It runs after
the original TRACER + HiSS prediction, analyzes the claim components against the
same supplied evidence, and validates the prediction. It only makes a
reconsideration call when its consistency rules identify a conflict; malformed
or ambiguous responses escalate and preserve the pre-guard prediction.
In guard logs, `original_prediction` is the output of the unchanged original
TRACER + HiSS path immediately before guard evaluation (including its existing
True-only TRACER reassessment when applicable); the raw HiSS label remains in
`old_pred`.
`--classifier-prompt evidence-consistency` and
`--decision-consistency-guard` are mutually exclusive.

The repository includes a 1,000-example `dataset/dev.json`, but the comparison
runner also requires matching alignment and HiSS prediction files. Prepare
those files for the development subset using the existing alignment checkpoint
and HiSS pipeline. For example, with the established group-size-1 alignment
checkpoint:

```bash
.venv/bin/python sentence_alignment/predict_model_script.py \
  --model_dir sentence_alignment/results-evidence-1-fixed/checkpoint-3745 \
  --test_data dataset/dev.json \
  --output_file method/result/dev_alignment_0_100.json \
  --max_evidence_count 1 \
  --start_index 0 --end_index 100 \
  --do_eval

FACT_CHECK_START=0 FACT_CHECK_LIMIT=100 \
FACT_CHECK_RESULT_DIR=method/result/dev_hiss \
LLM_MODEL_PATH=/mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16 \
.venv/bin/python -m method.claim_verification_hiss \
  --datapath dataset/dev.json --method hiss
```

Use the resulting HiSS log path under `method/result/dev_hiss/` in the
comparison command below. Refine and freeze the guard prompt/configuration only
with development data. Do not select or tune it using the locked test outputs.

After freezing the development-set configuration, run the three independent
arms on the locked 200-example set once:

```bash
LLM_MODEL_PATH=/mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16 \
AUXILIARY_MODEL_DEVICE=cuda \
.venv/bin/python -m method.compare_tracer_filtering \
  --comparison-mode decision-guard \
  --dataset dataset/test.json \
  --alignment-file test_alignment_evidence_1_200_fixed.json \
  --hiss-log method/result/20261104_213400_650571_benchmark/hiss/log.jsonl \
  --model /mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16 \
  --start-index 0 --end-index 200
```

The runner writes a new timestamped
`method/result/*_decision_consistency_guard_comparison/` directory, containing
separate ORIGINAL, EXTENSION, and GUARD result JSON/logs, an
`experiment_comparison.json` with per-class metrics and confusion matrices,
`three_configuration_predictions.csv`, and
`decision_consistency_guard_results.csv` with each guard analysis, action,
evidence status, prompt/configuration ID, and prediction change. Guard action
counts, changed-prediction counts, regressions, and corrected errors are
reported separately. Never point `--output-dir` to an existing results folder.

The default model path is the local Llama 3.1 8B base checkpoint. It has no
chat template and is not instruction-tuned; a smoke test echoed its input rather
than returning a verdict. The runner warns about this and rejects exact prompt
echoes, so do not treat it as benchmark-ready unless it produces valid labels.
Set `LLM_MODEL_PATH` to an Instruct/chat-tuned checkpoint for reliable scoring.
If CUDA is unavailable, check `nvidia-smi` and
`python -c 'import torch; print(torch.cuda.is_available())'`.
