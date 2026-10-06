# Run TRACER locally, one step at a time

This is the manual for a **fully local Qwen adaptation** of the paper. No OpenAI
API key, hosted inference, or paid API calls are used. Internet is needed once to
download public model files. Every execution command after downloading runs offline.

The initial configuration uses Qwen2.5-3B-Instruct in BF16 on the RTX 5080 Laptop
GPU (16 GB). It replaces verification, intent generation, question/assumption
generation, counterfactual checking, and final reassessment. MiniLM and DeBERTa NLI
run on CPU to preserve GPU memory. RoBERTa alignment training/inference run in
separate processes, so their GPU memory is released before Qwen starts.

The paper used hosted GPT models and a fine-tuned intent model. This first local
experiment uses **prompted Qwen intent, without intent fine-tuning**. Record that
distinction when presenting results. The aim is to measure how well this local
adaptation works and whether reassessment improves the same local baseline.

Setup has already downloaded all models and passed the real five-claim execution
check. See [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md) for exactly what was tested.
You can repeat the checks or continue the full experiment at the first unfinished
step. Check saved training metadata and each stage's `local_run.json` for current
completion status; the historical validation report is not a live run-status report.

## 0. Open PowerShell in the right directory

Copy this first. All later commands assume this directory:

```powershell
Set-Location A:\TTU\NN\Researchpaper\TRACER
```

Use the explicit Python path below; no environment activation or PowerShell
execution-policy changes are required. Run each command separately and wait until
it finishes. If a command fails, stop and resolve that error before the next step.
The project's Python environment is already installed.

To rebuild only the additional dependencies if needed:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

This environment inherits the machine's installed CUDA-enabled PyTorch. Do not
replace it with the old `torch==2.4.1` from the historical requirements file.

## 1. Check the environment and run software tests

```powershell
..\.venv\Scripts\python.exe local_workflow.py doctor
..\.venv\Scripts\python.exe -m unittest tests.test_regressions tests.test_local_workflow tests.test_generation_recovery tests.test_cot -v
..\.venv\Scripts\python.exe run_pipeline.py --mode smoke
```

Expected: CUDA/BF16 check passes, tests say `OK`, and the smoke test writes
`outputs/smoke/smoke_report.json`. Smoke mode uses a tiny model and scripted
responses; its scores must never be used as research results.

The doctor writes package/GPU information to `outputs/local_environment.json`.

## 2. Download the four real models

```powershell
..\.venv\Scripts\python.exe local_workflow.py download
```

This downloads several GB into:

| Folder | Model | Function |
|---|---|---|
| `models/qwen` | Qwen/Qwen2.5-3B-Instruct | All language-model stages |
| `models/roberta` | FacebookAI/roberta-large | Alignment encoder |
| `models/minilm` | sentence-transformers/all-MiniLM-L6-v2 | Semantic similarity |
| `models/nli` | cross-encoder/nli-deberta-v3-large | Entailment/contradiction |

Model revision hashes are pinned in `models/model_revisions.json`. Interrupted
downloads can be resumed by repeating the command. Keep adequate disk space for
downloaded weights plus optimizer checkpoints; reserve at least 30 GB for this
workflow. The machine had more than 180 GB free at setup.

Verify the actual language model loads and generates:

```powershell
..\.venv\Scripts\python.exe local_workflow.py llm-check
```

Expected response: `LOCAL_MODEL_OK`. It may also print model-loading notices.
If model files are missing, rerun download. There is no automatic hosted fallback.

Exercise real MiniLM/NLI ranking and final Qwen reassessment with a synthetic example:

```powershell
..\.venv\Scripts\python.exe local_workflow.py components-check
```

This supplies an assumption explicitly to test the downstream components, since
causal filtering on a small natural sample may retain none. Its output is labeled
synthetic in `outputs/local-check/components.json`; it is not benchmark evidence.

## 3. Validate the data

```powershell
..\.venv\Scripts\python.exe run_pipeline.py --mode prepare
```

Expected counts: train 11,994; dev 1,000; test 2,000. This also exports intent
training JSONL, which is useful for a later fine-tuning experiment but is **not used
to fine-tune Qwen in this workflow**. Test annotations are used only for evaluation;
inference uses the alignment model's predictions.

## 4. Check real-model execution on five development examples

Do this before committing to full training. Use development data to debug settings;
keep the test split for the final fixed experiment.

Create the small input:

The subset alternates reference classes to exercise more branches. It is a debugging
sample, not a representative sample for reporting accuracy.

```powershell
..\.venv\Scripts\python.exe local_workflow.py subset --data dataset/dev.json --limit 5 --output outputs/local-check/input.json
```

Train the actual RoBERTa-large architecture for just two optimizer steps on training
data. This verifies memory and checkpointing; it does not produce a useful research
model:

```powershell
..\.venv\Scripts\python.exe local_workflow.py train --data dataset/train.json --steps 2 --output outputs/alignment-check
```

Predict evidence alignment for the five development claims:

```powershell
..\.venv\Scripts\python.exe local_workflow.py align --data outputs/local-check/input.json --checkpoint outputs/alignment-check --output outputs/local-check/alignment.json
```

Run real local verification:

```powershell
..\.venv\Scripts\python.exe local_workflow.py verify --data outputs/local-check/input.json --output outputs/local-check/literal
```

Run real local intent generation, questions, assumptions, causal filtering,
evidence ranking, and reassessment. These are internal parts of this one stage;
the intermediate results are saved for inspection:

```powershell
..\.venv\Scripts\python.exe local_workflow.py reassess --data outputs/local-check/alignment.json --literal outputs/local-check/literal/log.jsonl --output outputs/local-check/reassessment
```

Only claims initially predicted `true` enter intent/causal reassessment. If the
sample has no such predictions, unchanged results are expected and do not show
that those model stages were exercised. Inspect `hidden_info.jsonl` for actual
intent/argument outputs.

Create a pipeline-check report:

```powershell
..\.venv\Scripts\python.exe local_workflow.py compare --data outputs/local-check/input.json --literal outputs/local-check/literal/log.jsonl --reassessed outputs/local-check/reassessment/log.jsonl --output outputs/local-check/comparison
```

Read `outputs/local-check/comparison/comparison.md`. It is prominently marked as a
subset/development run. Neither five examples nor a two-step alignment checkpoint
supports comparison claims against the paper's benchmark.

## 5. Train the real alignment checkpoint

Once the execution check passes:

```powershell
..\.venv\Scripts\python.exe local_workflow.py train --data dataset/train.json --epochs 5 --output outputs/alignment-full
```

Settings: RoBERTa-large; learning rate 1e-5; 512 tokens; up to eight consecutive
evidence sentences; BF16; gradient checkpointing; microbatch one with accumulation
eight. The effective batch size is eight, although accumulation and mixed precision
are implementation differences from the historical setup. Training duration depends
on laptop thermals and GPU load; it is not instantaneous.

Successful completion writes `model.safetensors`, `config.json`, tokenizer files,
and `training_run.json` in `outputs/alignment-full`. Checkpoints are saved each epoch.
Do not substitute the `alignment-check` or `outputs/smoke` checkpoint here.

If interrupted after an epoch checkpoint was saved, find the most recent checkpoint:

```powershell
Get-ChildItem outputs/alignment-full -Directory -Filter 'checkpoint-*'
```

Then rerun the same training command with `--resume-checkpoint` and the actual
checkpoint directory printed above. Example syntax (replace the placeholder):

```text
..\.venv\Scripts\python.exe local_workflow.py train --data dataset/train.json --epochs 5 --output outputs/alignment-full --resume-checkpoint outputs/alignment-full/checkpoint-NUMBER
```

If there is no checkpoint yet, rerun training to restart. A completed model is
reused when the saved settings exactly match the same command. Different settings
cannot overwrite it accidentally; use a new output folder for another experiment.

## 6. Optional: measure development results before fixing settings

Repeat steps 7–10 below with `dataset/dev.json` and output folders under
`outputs/local-dev/`. Choose settings using development performance only. Once
settings are fixed, run the test commands exactly once for your main reported run.
Do not train on development or test claims.

## 7. Align all 2,000 test claims

Check that the literal prompts fit the configured local context before a long run:

```powershell
..\.venv\Scripts\python.exe local_workflow.py context-check --data dataset/test.json
```

The supplied test split passed this check at setup: longest literal prompt 4,430
tokens, plus a 1,536-token output allowance, within the 16,384-token limit.

```powershell
..\.venv\Scripts\python.exe local_workflow.py align --data dataset/test.json --checkpoint outputs/alignment-full --output outputs/local-test/alignment.json
```

This prints alignment evaluation and saves predicted labels under `prediction`.
`alignment.provenance.json` records the input hash and checkpoint training settings.
Gold `annotation` is retained for analysis but does not feed local intent generation.

## 8. Get the local HiSS baseline

```powershell
..\.venv\Scripts\python.exe local_workflow.py verify --data dataset/test.json --output outputs/local-test/literal
```

Output: `outputs/local-test/literal/log.jsonl`. Each record contains an ID, predicted
label, reference label, and model-generated justification. This is the baseline to
compare against local reassessment. All GPT-named call sites are redirected to
Qwen in this command; no GPT requests are sent.

## 9. Apply local TRACER reassessment

```powershell
..\.venv\Scripts\python.exe local_workflow.py reassess --data outputs/local-test/alignment.json --literal outputs/local-test/literal/log.jsonl --output outputs/local-test/reassessment
```

Outputs:

- `log.jsonl`: original and revised predictions for all claims.
- `hidden_info.jsonl`: generated intents, questions, assumptions, and backed arguments
  for claims that entered reassessment.
- `local_run.json`: model, prompt limits, package versions, input hashes, and completion status.

This stage can take longer than literal verification because eligible claims need
several local language-model calls and CPU NLI inference. If no critical hidden
evidence is found, the original label is preserved. Failed generation raises an
error instead of silently inventing a prediction.

## 10. Generate the final comparison with the paper

```powershell
..\.venv\Scripts\python.exe local_workflow.py compare --data dataset/test.json --literal outputs/local-test/literal/log.jsonl --reassessed outputs/local-test/reassessment/log.jsonl --output outputs/local-test/comparison
```

Open `outputs/local-test/comparison/comparison.md`. The report contains:

| Row | Where the numbers come from |
|---|---|
| Paper CoT / CoT + TRACER | Published values reproduced in the original README |
| Paper HiSS / HiSS + TRACER | Published values reproduced in the original README |
| Local HiSS | Your completed Qwen baseline predictions |
| Local HiSS + TRACER | Your completed Qwen reassessment predictions |

Columns are accuracy, macro-F1, half-true precision, half-true recall, and half-true
F1, all as percentages. The JSON also records the local before/after gains in
percentage points. The script checks IDs, reference labels, and completion status;
it refuses missing or duplicate predictions. Full test-set comparison is labeled
only when the evaluation data matches the supplied test records exactly.

Interpret the comparison as **published GPT-based TRACER versus a local Qwen
adaptation**. The local baseline-to-reassessment difference is the cleaner measure
of whether TRACER helps your selected local model. Different LLMs, prompted rather
than fine-tuned intent, training arithmetic, and repaired code prevent claims of
exact replication.

See [Why local benchmark results can differ](README.md#why-local-benchmark-results-can-differ)
for the model, intent-training, prompt, and alignment differences, including why
microbatch one with accumulation eight preserves the effective training batch.

## 11. Add local CoT and CoT + TRACER

CoT is a separate **zero-shot** baseline. It asks Qwen to assess claim and evidence
step by step, give a brief justification, and produce a final bracketed verdict.
It does not use HiSS's few-shot question/answer examples. The released repository
did not supply its exact CoT prompt, so our version is explicitly named `cot-v1`
in `method/claim_verification_cot.py`. This is a documented local implementation,
not a claim to reproduce the authors' exact prompt.

Reuse the existing full alignment predictions; no retraining is needed. Run these
commands **one at a time** from `TRACER`, using separate CoT output directories:

```powershell
..\.venv\Scripts\python.exe local_workflow.py verify --verifier cot --data dataset/test.json --output outputs/local-test/cot-literal --max-new-tokens 4096
```

Then feed the **CoT** justifications and predictions into TRACER:

```powershell
..\.venv\Scripts\python.exe local_workflow.py reassess --data outputs/local-test/alignment.json --literal outputs/local-test/cot-literal/log.jsonl --output outputs/local-test/cot-reassessment --max-new-tokens 4096
```

The reassessment stage infers `cot` from the baseline manifest. It reassesses only
CoT's own `true` predictions. Both runs support resume by repeating the same command,
and use the existing local response cache and bounded-recovery mechanism. HiSS
results cannot be resumed into CoT directories or relabeled as CoT predictions.

After **all four local runs** finish, generate the combined report:

```powershell
..\.venv\Scripts\python.exe local_workflow.py compare --data dataset/test.json --literal outputs/local-test/literal/log.jsonl --reassessed outputs/local-test/reassessment/log.jsonl --cot-literal outputs/local-test/cot-literal/log.jsonl --cot-reassessed outputs/local-test/cot-reassessment/log.jsonl --output outputs/local-test/comparison-all
```

Read `outputs/local-test/comparison-all/comparison.md`. It contains the four paper
rows and four measured local rows: HiSS, HiSS + TRACER, CoT, CoT + TRACER. It
reports reassessment gains separately for each baseline. The JSON preserves input
hashes, model settings, token-cap history, prompt version, and recovery counts.
The report verifies that CoT reassessment actually used the supplied CoT log.

For a short execution check, use the same commands with the corresponding files
under `outputs/local-check/` instead of the full test files. A five-claim check
cannot establish performance; full test results are needed for the final table.
For fair interpretation, note any different token allowances or recoveries across
HiSS and CoT in the run manifests. Both use the same local model and evidence.

## Restarting, memory, and errors

Reassessment now also recovers from an overlong or malformed response in intent,
questions, assumptions, counterfactual choices, and the final verdict. It keeps the
original input and retries once with a concise, stage-specific output instruction.
It does not continue an unfinished repetitive answer, reset a global quota, change
models, or guess a label. A second failure stops with the failing stage's name.
Recovered claims are marked `generation_recovery: bounded-reassessment-v1` and list
the recovered stages in `generation_recoveries`; the comparison report counts them.
Resume reassessment with its existing command and token allowance. You do not need
to increase the allowance just because a repetition loop reached it.

- **Resume verification/reassessment:** repeat the exact same command. Completed
  example IDs are skipped. Successful model responses are cached under
  `outputs/local-response-cache`, avoiding repeated generation after interruption.
- **Changed settings:** choose new `--output` directories. Input/model/settings
  checks prevent mixing different runs when resuming.
- **GPU out of memory:** close other GPU-heavy applications and run one stage at a
  time. The local language model already uses BF16 and CPU evidence rankers.
  Lower `--context` only if the actual inputs fit; this code refuses silent input
  truncation. Full training memory was checked separately from inference.
- **Context error:** the default is 16,384 tokens including output allowance. A
  longer prompt fails explicitly. You can try `--context 32768` with a fresh output
  folder, but it uses more GPU memory. Keep the setting fixed for the experiment.
- **Output limit:** the default is 1,536 generated tokens per call. If the local
  HiSS verifier exhausts that allowance, it retries with the original claim/evidence
  plus instructions for at most three questions, 200 words, and a final verdict.
  This handles observed repetition loops. Recovered rows contain
  `generation_recovery: bounded-hiss-v1`; the comparison report counts these cases.
  CoT uses its own shorter format instruction and records `bounded-cot-v1`.
  Failed partial generations are saved as `*.truncated.json` for diagnosis and are
  never treated as predictions. Identical failed greedy requests are not regenerated.
  If the recovery also runs out of tokens, stop the old process and rerun the same command and output
  folder with `--max-new-tokens 4096`. Increasing only this cap is now allowed on resume;
  completed records are preserved, and `budget_changes.jsonl` records the old settings
  and how many records were completed before the change. Subsequent resumes must use
  the new cap. The comparison JSON includes this history. Other settings changes still
  require a new folder. Identical greedy retries are no longer attempted for this error.
- **Malformed model output:** the pipeline reports the failed stage instead of
  assigning an arbitrary label. A successful setup test cannot guarantee that an
  LLM will obey every format on all 2,000 claims; record any required prompt change
  and repeat comparable stages with fixed settings.
- **Older run stopped with blank `Attempt ... failed:` messages:** an empty trailing
  `<>` after a valid verdict triggered the old parser's assertion. The parser now
  ignores empty tags while requiring a valid final nonempty verdict. Rerun the same
  verification command with the same output folder; completed records are preserved
  and the saved response can be parsed without regenerating it. Do not delete the
  output log or response cache. This is a parsing repair, not a model/prompt change.
- **Missing models or download/network error:** rerun `download` with internet
  access. Model execution does not contact the network.
- **Training interrupted within the first epoch:** no saved optimizer checkpoint
  may exist yet. Restart training; do not use a partially written final model.

## Sources and scope

- [Qwen model card and Transformers usage](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-large)
- The supplied paper and `PAPER_EXPLAINED.md` for task definition and reported results.

This guide sets up local HiSS and CoT experiments with and without TRACER. Local intent fine-tuning,
larger/quantized LLM comparisons, the paper's remaining baselines, and statistical
significance analysis are separate experiments, not silently included here.
