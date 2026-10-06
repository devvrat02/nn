# Running the repaired TRACER repository

For the new fully local Qwen workflow and step-by-step paper comparison, use
[RUN_LOCAL.md](RUN_LOCAL.md). The API-backed workflow below is retained as an alternative.

The local environment is `../.venv`. It uses Python 3.14 and inherits the machine's
existing CUDA-enabled PyTorch 2.12.0.dev20260408+cu128. CUDA and the RTX 5080 were
detected. Additional libraries were installed inside the virtual environment;
the system PyTorch installation was not replaced.

## Run now, without API charges

From `A:\TTU\NN\Researchpaper` in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\TRACER\run_pipeline.py --mode smoke
.\.venv\Scripts\python.exe .\TRACER\run_pipeline.py --mode prepare
```

Alternatively, from the `TRACER` directory:

```powershell
.\run_local.cmd --mode smoke
.\run_local.cmd --mode prepare
```

The `.cmd` launcher works with this machine's restricted PowerShell script policy.
An optional `run_local.ps1` wrapper is also supplied, but this machine blocks direct
`.ps1` execution. No execution-policy setting was changed.

Smoke mode trains a tiny randomly initialized RoBERTa model for two steps, saves
and reloads it, and runs alignment inference. It then tests verification through
evaluation using scripted LLM responses and scripted embedding/NLI objects.
Explicit alignment fixtures exercise reassessment branches independently of the
tiny model's predictions. Network calls are forbidden during those mocked stages.
This verifies execution and interfaces; it is **not a trained TRACER experiment**.

Outputs:

- `outputs/smoke/smoke_report.json`: check list and clearly labeled fixture metrics.
- `outputs/smoke/alignment_model/`: tiny test checkpoint, unsuitable for research.
- `outputs/smoke/alignment.json`: actual tiny-model predictions.
- `outputs/smoke/scripted_alignment.json`: deterministic branch-test inputs.
- `outputs/smoke/literal/log.jsonl` and `reassessment/log.jsonl`: fixture results.
- `outputs/smoke/reassessment/hidden_info.jsonl`: intermediate fixture arguments.
- `outputs/prepare/dataset_report.json`: counts, labels, and ID overlap checks.
- `outputs/prepare/intent_train/finetune_data.jsonl`: 10,277 training examples.
- `outputs/prepare/intent_dev/finetune_data.jsonl`: 713 development examples.

Both modes completed successfully locally. No paid API calls, hosted fine-tuning,
full RoBERTa-large training, or published-result reproduction were performed.
Eight regression tests also passed, along with dependency consistency, Python
compilation, standalone evaluation, and a check that research mode refuses to run
without credentials. Run the regression tests from `TRACER` with
`..\.venv\Scripts\python.exe -m unittest tests.test_regressions -v`.
Running again writes into the same output directories; use `--output-dir` to keep
separate runs.

## Environment setup on this machine

The environment has already been created. To rebuild it with the installed
Python/PyTorch combination, run from the parent research folder:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r .\TRACER\requirements-local.txt
```

This setup intentionally inherits existing PyTorch. On another computer, install
a suitable PyTorch build separately before using this recipe. `requirements.txt`
preserves the original historical versions and is not the dependency set tested
under Python 3.14. `requirements-local.txt` records the main local dependency pins;
it is not a fully isolated cross-platform lockfile.

## Train the real alignment model later

From `TRACER`, using the parent environment:

```powershell
..\.venv\Scripts\python.exe -m sentence_alignment.train_model_script --train_epoch 5 --batch_size 1 --gradient_accumulation_steps 8 --gradient_checkpointing --learning_rate 1e-5
```

This downloads RoBERTa-large if absent and performs genuine training. Batch size
one with accumulation eight is a memory-conscious starting point for the available
16 GB GPU; it is not a measured guarantee that full training fits or finishes in a
particular time. Default batch size eight matches the paper. The final model and
tokenizer are saved directly under `sentence_alignment/results-model`, so no
hard-coded checkpoint number is needed. A quick training debug run can use
`--max_steps 2`, but such a checkpoint is not sufficient for research evaluation.

## Future API-backed experiment

This was not executed, following the request for local testing only. It requires
an accessible alignment checkpoint, an OpenAI API key configured locally, an intent
model accessible to that account, and internet access for the neural rankers.

Intent JSONL files are already prepared. The paper used a fine-tuned
GPT-4o-mini-2024-07-18 model with three epochs and batch size four. The preparation
script does not submit a hosted training job. A base `gpt-4o-mini` ID is allowed
only as an explicit exploratory substitution; its results are not a reproduction.

After configuring `OPENAI_API_KEY` in your local environment, from `TRACER`:

```powershell
..\.venv\Scripts\python.exe run_pipeline.py --mode research --model-dir sentence_alignment/results-model --intent-model YOUR_ACCESSIBLE_MODEL_ID --limit 5 --output-dir outputs/research-small
```

Remove `--limit 5` to process all 2,000 test records. The default verifier is
`gpt-3.5-turbo`; use `--verifier-model` to explicitly choose another accessible
model. Actual hosted model availability and account access have not been tested.

Research mode runs alignment → literal verification → reassessment → evaluation.
It records `input.json`, `alignment.json`, `literal/log.jsonl`,
`reassessment/log.jsonl`, `reassessment/hidden_info.jsonl`, `metrics.json`, and
`run_config.json`. Model failures raise errors rather than becoming blank labels.
Logs are flushed after each claim, but automatic resume is not implemented; use
separate output folders and retain partial logs if a run fails.

The API helpers use the Chat Completions interface described in
[official OpenAI documentation](https://developers.openai.com/api/reference/resources/chat).
Credentials are read from the environment and are never written into run metadata.
The local SDK was updated because the released 1.69.0 version warned about Python
3.14 compatibility. Actual network calls remain untested.

## Standalone stages

From `TRACER`:

```powershell
..\.venv\Scripts\python.exe -m sentence_alignment.predict_model_script --model_dir sentence_alignment/results-model --test_data dataset/test.json --output_file test_alignment.json --max_evidence_count 4 --do_eval
..\.venv\Scripts\python.exe -m intent_generation.intent_finetune --datapath dataset/train.json --output_dir outputs/intent-training
..\.venv\Scripts\python.exe -m method.claim_verification_hiss --datapath dataset/test.json --output_dir method/results/literal
..\.venv\Scripts\python.exe -m method.reassessment --datafile test_alignment.json --literal method/results/literal/log.jsonl --intent_model YOUR_ACCESSIBLE_MODEL_ID --output_dir method/results/reassessment
..\.venv\Scripts\python.exe -m method.eval --datapath method/results/reassessment/log.jsonl --output method/results/metrics.json
```

## Repairs and reproducibility differences

- Fixed undefined dependencies in the model wrappers and made API client creation lazy.
- Replaced account-specific model IDs, GPU index 3, dated log paths, and checkpoint-7500 assumptions.
- Added stable output locations, command-line evaluation, UTF-8 file access, and genuine JSONL writing.
- Made modules importable without starting fine-tuning-data export or requiring an API key.
- Persisted encoder configuration and tokenizer with checkpoints for local reload.
- Corrected truncated-sentence masking and inference; fixed device placement of indexing tensors.
- Exposed training memory/settings controls and set default learning rate to the paper's 1e-5.
- Used predicted hidden evidence for intent generation instead of ignoring alignment output.
- Raised errors on failed verification/mining or malformed reassessment instead of silently returning incomplete results.
- Added local preparation, an offline integration test, and a cross-platform pipeline runner.

These corrections and updated dependencies mean an eventual run is a repaired
implementation, not a bit-for-bit copy of the authors' original environment.
