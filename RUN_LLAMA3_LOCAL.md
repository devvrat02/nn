# Run Llama 3 locally on this Windows laptop

Model: **`meta-llama/Meta-Llama-3-8B`**, the base pretrained model.
These are PowerShell commands for this machine. No Slurm, REPACSS, or paid API is
needed. Internet is needed for model downloads; inference runs offline.

Run each command separately and wait for success before continuing. The commands
use the existing project Python directly, so environment activation is optional.
The model has loaded successfully on this laptop, but the full local benchmark
has not been validated. Repetition controls and format recovery do not guarantee
that every base-model response is usable.

## 1. Open the project

```powershell
Set-Location A:\TTU\NN\Researchpaper\TRACER
..\.venv\Scripts\python.exe local_workflow.py doctor --model-profile llama3-8b
```

The existing environment already contains CUDA PyTorch and the dependencies.
If only the additional packages need reinstalling:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

Do not install the historical `requirements.txt` over this environment.

## 2. Model access and download — skip if already complete

Use the Hugging Face account approved for the exact model above. Enter your token
only at the local login prompt:

```powershell
..\.venv\Scripts\hf.exe auth login
..\.venv\Scripts\python.exe local_workflow.py download --model-profile llama3-8b
```

The weights are already downloaded on this machine. Repeating download reuses
completed files and resumes interrupted transfers. The workflow respects `HF_HOME`
and can reference an existing default CLI login without copying its token.
A browser login alone does not authenticate the command line.

## 3. Choose GPU placement

The verification/reassessment commands below explicitly use **`--device-map cuda`**
to match your current `cot-repeat-control` run. Full BF16 loading has used nearly
all 16 GB of GPU memory on this laptop. Close other GPU-heavy workloads and run
one model process at a time. Successful loading does not guarantee enough memory
for every later prompt.

If full GPU loading runs out of memory, use **`--device-map auto`** for GPU/CPU
offload. Change it on every verification/reassessment command and use new output
folders, such as `outputs/llama3-auto-check/` and `outputs/llama3-auto-test/`.
Placement is tracked; do not switch it within an existing output folder. Offload
needs free system RAM and may be much slower. It previously passed a short loading
check but took over five minutes without finishing a single CoT claim.

Optional short loading check using offload:

```powershell
..\.venv\Scripts\python.exe local_workflow.py llm-check --model-profile llama3-8b --device-map auto --max-new-tokens 64
```

Inspect whether the answer is `LOCAL_MODEL_OK`; this is not an accuracy test.

## 4. Validate software and data

```powershell
..\.venv\Scripts\python.exe -m unittest tests.test_regressions tests.test_local_workflow tests.test_generation_recovery tests.test_cot tests.test_llama_profile tests.test_generation_progress -q
..\.venv\Scripts\python.exe run_pipeline.py --mode prepare
```

Expected dataset sizes: train 11,994; dev 1,000; test 2,000. Preparation does not
fine-tune Llama. This experiment uses prompted intent generation.

## 5. Five-claim development check

Skip the next three commands if these same development fixtures already exist.
The two-step training is only an execution check, not a research checkpoint:

```powershell
..\.venv\Scripts\python.exe local_workflow.py subset --data dataset/dev.json --limit 5 --output outputs/llama3-check/input.json
..\.venv\Scripts\python.exe local_workflow.py train --data dataset/train.json --steps 2 --output outputs/llama3-alignment-check
..\.venv\Scripts\python.exe local_workflow.py align --data outputs/llama3-check/input.json --checkpoint outputs/llama3-alignment-check --output outputs/llama3-check/alignment.json
```

**To continue your current CoT check, start here.** Both repetition flags and the
new output folder are required. The old `cot-literal` run has different settings:

```powershell
..\.venv\Scripts\python.exe -u local_workflow.py verify --model-profile llama3-8b --device-map cuda --verifier cot --data outputs/llama3-check/input.json --output outputs/llama3-check/cot-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

After all five predictions finish, apply TRACER to that exact CoT log:

```powershell
..\.venv\Scripts\python.exe -u local_workflow.py reassess --model-profile llama3-8b --device-map cuda --data outputs/llama3-check/alignment.json --literal outputs/llama3-check/cot-repeat-control/log.jsonl --output outputs/llama3-check/cot-reassessment-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

Also check HiSS and its reassessment before a full HiSS experiment:

```powershell
..\.venv\Scripts\python.exe -u local_workflow.py verify --model-profile llama3-8b --device-map cuda --verifier hiss --data outputs/llama3-check/input.json --output outputs/llama3-check/hiss-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
..\.venv\Scripts\python.exe -u local_workflow.py reassess --model-profile llama3-8b --device-map cuda --data outputs/llama3-check/alignment.json --literal outputs/llama3-check/hiss-repeat-control/log.jsonl --output outputs/llama3-check/hiss-reassessment-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

Only baseline `true` predictions enter reassessment. Unchanged labels may be
expected when no critical hidden evidence is found. Five claims are not enough
to establish benchmark performance. Resolve failed stages before proceeding.

## 6. Train or reuse the full alignment model

If a completed five-epoch `outputs/alignment-full` checkpoint already exists,
reuse it and skip this command. Otherwise:

```powershell
..\.venv\Scripts\python.exe local_workflow.py train --data dataset/train.json --epochs 5 --output outputs/alignment-full
```

Training uses RoBERTa-large, BF16, gradient checkpointing, microbatch 1, and
accumulation 8 (effective batch 8). It saves final weights, tokenizer, config,
and `training_run.json`. Do not use the two-step checkpoint for full evaluation.

For interrupted training, find an actual saved epoch checkpoint:

```powershell
Get-ChildItem outputs/alignment-full -Directory -Filter 'checkpoint-*'
```

Rerun training with `--resume-checkpoint outputs/alignment-full/checkpoint-NUMBER`,
replacing `NUMBER` with the saved checkpoint number. If none exists, restart
training. Do not overwrite an unrelated completed model.

## 7. Check full-test context and align evidence

```powershell
..\.venv\Scripts\python.exe local_workflow.py context-check --model-profile llama3-8b --verifier hiss --data dataset/test.json
..\.venv\Scripts\python.exe local_workflow.py context-check --model-profile llama3-8b --verifier cot --data dataset/test.json
..\.venv\Scripts\python.exe local_workflow.py align --data dataset/test.json --checkpoint outputs/alignment-full --output outputs/llama3-local-repeat-test/alignment.json
```

Both context checks must report `over_budget: 0`. They previously passed all 2,000
claims at context 8,192 and output allowance 1,536. Intermediate reassessment
prompts are checked during execution. GPU capacity does not extend native context.
Choose experiment settings on development data before the full test run.

## 8. Run all four full-test methods

These outputs are separate from Qwen, earlier decoding settings, and REPACSS runs.
Run the following four commands **one at a time**; wait for each to finish:

```powershell
..\.venv\Scripts\python.exe -u local_workflow.py verify --model-profile llama3-8b --device-map cuda --verifier hiss --data dataset/test.json --output outputs/llama3-local-repeat-test/literal --repetition-penalty 1.1 --no-repeat-ngram-size 8
..\.venv\Scripts\python.exe -u local_workflow.py reassess --model-profile llama3-8b --device-map cuda --data outputs/llama3-local-repeat-test/alignment.json --literal outputs/llama3-local-repeat-test/literal/log.jsonl --output outputs/llama3-local-repeat-test/reassessment --repetition-penalty 1.1 --no-repeat-ngram-size 8
..\.venv\Scripts\python.exe -u local_workflow.py verify --model-profile llama3-8b --device-map cuda --verifier cot --data dataset/test.json --output outputs/llama3-local-repeat-test/cot-literal --repetition-penalty 1.1 --no-repeat-ngram-size 8
..\.venv\Scripts\python.exe -u local_workflow.py reassess --model-profile llama3-8b --device-map cuda --data outputs/llama3-local-repeat-test/alignment.json --literal outputs/llama3-local-repeat-test/cot-literal/log.jsonl --output outputs/llama3-local-repeat-test/cot-reassessment --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

Long runtime is possible on this laptop. Passing a small check does not guarantee
completion of all 2,000 claims. Do not skip failures and report a partial run as
the full benchmark.

## 9. Compare with the paper

After all four runs finish:

```powershell
..\.venv\Scripts\python.exe local_workflow.py compare --model-profile llama3-8b --data dataset/test.json --literal outputs/llama3-local-repeat-test/literal/log.jsonl --reassessed outputs/llama3-local-repeat-test/reassessment/log.jsonl --cot-literal outputs/llama3-local-repeat-test/cot-literal/log.jsonl --cot-reassessed outputs/llama3-local-repeat-test/cot-reassessment/log.jsonl --output outputs/llama3-local-repeat-test/comparison-all
```

Read `outputs/llama3-local-repeat-test/comparison-all/comparison.md` and its JSON.
Report absolute scores and each baseline's gain from TRACER. Preserve model and
training metadata, decoding settings, token-cap history, and recovery annotations.
This is a base-Llama local adaptation with prompted intent, not an exact paper
reproduction. Published scores are reference values, not measured local results.

## Resume and common errors

- **Interrupted process:** press Ctrl+C once if needed, wait for PowerShell, then
  repeat the same command with the same flags and folder. Completed records are
  preserved; an unfinished generation starts again. Never run two writers to the
  same folder.
- **Cached failure:** repeating identical greedy settings reproduces the same
  failure. The repetition-control commands above use different cache entries from
  the old run. Keep both flags on every resume; do not delete the whole cache.
- **Output limit:** inspect `*.truncated.json` in `outputs/local-response-cache`.
  More tokens may prolong a repetition loop. If increasing `--max-new-tokens`,
  prompt plus output must still fit 8,192 tokens. Only an increase is allowed on
  resume; retain the new cap thereafter. Other setting changes need fresh folders.
- **No visible claim progress:** the claim bar advances only after completion.
  Token counts and elapsed time print approximately every 15 seconds as decoding
  advances. Before the first token the model may still be processing the prompt;
  these messages are not an independent heartbeat.
- **Settings mismatch:** use a fresh folder for changed model, placement,
  repetition controls, or inputs. Do not manually relabel old predictions.
- **Formatting errors:** CoT can accept an audited explicit opening verdict;
  HiSS can accept a standalone label, optionally followed by an `### Evidence`
  heading and URL-only lines (`explicit-hiss-verdict-v1`). This reads the model's
  label without validating the citation or its factual correctness. Other local
  HiSS format failures get one distinct bounded prompt (`bounded-hiss-format-v1`)
  instead of repeatedly parsing the same cached answer. Repeat the same command
  after updating the code; existing completed responses can be reparsed from cache.
  choice parsing can safely inspect one literal `print("A")` code wrapper without
  executing it. Conflicting, unsupported, or truncated answers still fail. These
  repairs do not make the base model instruction-tuned.

For cluster execution only, see [RUN_LLAMA3.md](RUN_LLAMA3.md). Earlier Qwen
instructions remain in [RUN_LOCAL.md](RUN_LOCAL.md).
