# Meta Llama 3 8B on REPACSS

For a separate Windows-only guide, use [RUN_LLAMA3_LOCAL.md](RUN_LLAMA3_LOCAL.md).

This guide targets **REPACSS Linux GPU jobs**, not the Windows laptop. It selects
`meta-llama/Meta-Llama-3-8B`, the base pretrained Llama 3 8B model, using
`--model-profile llama3-8b`. This is now the default model profile. To resume historical Qwen runs, explicitly
add `--model-profile qwen` to those commands. The repetition-control experiment below uses `outputs/llama3-repeat-test/`.

The REPACSS job script requests **one H100 NVL**, 8 CPU cores, 64 GB host RAM,
and 12 hours. It loads Llama in **BF16 entirely on the allocated GPU**. No
quantization or CPU weight offload is needed for this target. MiniLM/NLI remain on
CPU to preserve the existing workflow. Alignment keeps microbatch 1 and accumulation
8 (effective batch 8); moving machines does not silently change training settings.

**Status:** Llama weights are installed on the Windows laptop, and the short
`LOCAL_MODEL_OK` generation succeeded. All 2,000 HiSS and CoT baseline prompts
passed context checks at 8,192 context / 1,536 output tokens. The base model has
produced repetitive or malformed responses in the development check. The controls
below are software-tested, but the five-claim and full repetition-control runs
have not been validated as successful. REPACSS execution remains untested.

**Model variant:** this exact repository is a base completion model, not Instruct.
The workflow uses the experimental `base-v1` plain-text task/response wrapper,
recorded in manifests and separated in the response cache. It does not add
instruction fine-tuning. Run the small check first: malformed answers remain errors
and are not silently converted into predictions.

## Windows: retry the current five-claim CoT check

Use this section for your current laptop run; sections 1?8 below are for REPACSS.
Stop any earlier verification process before starting another. From the activated
project environment in PowerShell, run:

```powershell
Set-Location A:\TTU\NN\Researchpaper\TRACER
python -u local_workflow.py verify --model-profile llama3-8b --verifier cot --data outputs/llama3-check/input.json --output outputs/llama3-check/cot-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

This uses the existing five-claim input and the same full-GPU placement as your
last command. If that input is missing, create it first:

```powershell
python local_workflow.py subset --data dataset/dev.json --limit 5 --output outputs/llama3-check/input.json
```

**Do not rerun the old `cot-literal` command to fix the cached failure.** The new
command includes both repetition flags and writes to `cot-repeat-control`.
It starts a separate experiment; previous predictions remain in `cot-literal`.
Do not copy those predictions into the new folder or delete the response cache.
The new decoding settings automatically use different cache entries.

To resume this new experiment, repeat the exact command above, including both
flags and the same output folder. A settings-mismatch error means the chosen
folder belongs to a different experiment; use another fresh folder.

After all five CoT claims finish, and the matching development alignment file
exists, run reassessment with the same controls:

```powershell
python -u local_workflow.py reassess --model-profile llama3-8b --data outputs/llama3-check/alignment.json --literal outputs/llama3-check/cot-repeat-control/log.jsonl --output outputs/llama3-check/cot-reassessment-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

The alignment file is created in section 5; on Windows use those Python commands
in your activated environment, without Slurm. Do not substitute test alignment
for development alignment. Repetition controls may reduce loops but do not ensure
valid answers. Stop and inspect failures before launching the full benchmark.
For memory pressure, see **Windows laptop execution check** below; switching to
`--device-map auto` requires a different output folder and may be much slower.

## 1. Transfer the repaired project

Use your assigned REPACSS project directory with sufficient quota. In Windows
PowerShell, replace `YOUR_USER` and `/YOUR/PROJECT/DIRECTORY` below with your actual
username and existing destination. Package the edited source, not a fresh upstream
clone that lacks these repairs:

```powershell
Set-Location A:\TTU\NN\Researchpaper
tar -czf tracer-repacss.tar.gz --exclude=TRACER/models --exclude=TRACER/outputs --exclude=TRACER/.cache --exclude=TRACER/.git --exclude=__pycache__ TRACER
scp tracer-repacss.tar.gz YOUR_USER@repacss.ttu.edu:/YOUR/PROJECT/DIRECTORY/
ssh YOUR_USER@repacss.ttu.edu
```

On REPACSS, extract into a directory that does not already contain a different
TRACER checkout:

```bash
cd /YOUR/PROJECT/DIRECTORY
tar -xzf tracer-repacss.tar.gz
cd TRACER
mkdir -p logs outputs
```

Do not copy the Windows `.venv`. Build a Linux environment instead. The archive
excludes model weights, results, and cached Hugging Face credentials. Optionally
transfer your completed `outputs/alignment-full/` checkpoint separately, including
its tokenizer, config, weights, and `training_run.json`; then skip retraining and
run alignment on the cluster. Do not copy the two-step `alignment-check` as your
research model. Saved Qwen predictions are not Llama predictions.

## 2. Allocate a GPU and create the environment

REPACSS uses Slurm. Run model execution and substantial setup work in an allocated
compute session. The official guide currently documents the `h100` partition and
`gpu:nvidia_h100_nvl:1` resource. Check your account's access and limits; if needed,
add your assigned `--account=...` to `srun` and `sbatch` commands. Adjust the wall
time to your allocation's policy.

```bash
srun --partition=h100 --gres=gpu:nvidia_h100_nvl:1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
nvidia-smi
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n tracer-llama3 python=3.11 -y
conda activate tracer-llama3
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-repacss.txt
python -m pip check
mkdir -p outputs
python -m pip freeze > outputs/repacss-environment.txt
```

These commands assume Miniforge at `~/miniforge3`; use your actual installation
path if different. If it is not installed, follow REPACSS's Miniforge guide linked
below. The CUDA PyTorch wheel includes runtime libraries; a compatible NVIDIA
driver is still required. Do not install the laptop's nightly PyTorch build or
historical `requirements.txt`. This Linux environment is a proposed setup, not an
already tested REPACSS installation. If you already installed the earlier environment,
reuse it with `export TRACER_CONDA_ENV=tracer-llama2` before submitting jobs and
activate that environment for interactive commands; no reinstall is required. Keep its frozen package list with results.

Ensure the working directory is the uploaded `TRACER` folder inside the allocation.
Run the software and GPU checks:

```bash
python local_workflow.py doctor --model-profile llama3-8b
python -m unittest tests.test_regressions tests.test_local_workflow tests.test_generation_recovery tests.test_cot tests.test_llama_profile tests.test_generation_progress -q
python run_pipeline.py --mode smoke
```

`doctor` should identify the allocated GPU, selected model, and shared models,
and pass BF16 arithmetic. Smoke mode uses scripted responses and is not benchmark evidence.

## 3. Authenticate and download the model

Request access on the [official Meta Llama 3 8B model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
using your own Hugging Face account. Once access is granted, authenticate locally:

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf auth login
python local_workflow.py download --model-profile llama3-8b
```

Enter a token with access to the gated model at the login prompt. Do not put it in
the job script or source files. A 401/403 means account/token access must be fixed.
Downloads need network access and may exceed the initial allocation's wall time;
repeat the command to resume. If outbound downloads are restricted, use the
cluster's supported transfer method. Reserve space for roughly 16 GB of Llama
weights plus the shared models, training checkpoints, caches, and outputs.

The download stores weights in `models/llama3-8b` and the shared models in
`models/roberta`, `models/minilm`, and `models/nli`. Revisions are pinned in
`models/model_revisions.json`. Later inference runs offline without paid APIs.

## 4. Check the model and its context limit

```bash
python local_workflow.py llm-check --model-profile llama3-8b
python local_workflow.py context-check --model-profile llama3-8b --data dataset/test.json
python local_workflow.py context-check --model-profile llama3-8b --verifier cot --data dataset/test.json
```

The first command asks for `LOCAL_MODEL_OK`; inspect the actual answer. Each
context check must report `over_budget: 0` before starting its full baseline.

**A larger GPU does not increase Llama 3's native 8,192-token context.** The
profile reserves 1,536 output tokens, leaving at most 6,656 for the formatted prompt.
Some existing HiSS/evidence prompts may not fit. The code rejects an oversized
context setting and does not drop evidence. Do not use the earlier 16K/128K Llama 3.1 settings. A 4,096-token output allowance
leaves only 4,096 tokens for the input; check whether each prompt fits before
increasing the allowance.

If a context check fails, stop that baseline. Reducing the output allowance helps
only if the input itself fits and can increase output-limit failures. Otherwise a
separately documented shorter-prompt/evidence-selection experiment or a longer-context
model is required. Do not skip claims and present the remainder as a full benchmark.
The check covers baseline prompts; generated intermediate TRACER prompts are also
checked at execution time and may exceed the limit. Thus the unchanged full
benchmark is conditional on context compatibility even on REPACSS.

## 5. Small development execution check

Still in the GPU allocation, run each command and wait for success:

```bash
python run_pipeline.py --mode prepare
python local_workflow.py subset --data dataset/dev.json --limit 5 --output outputs/llama3-check/input.json
python local_workflow.py train --data dataset/train.json --steps 2 --output outputs/llama3-alignment-check
python local_workflow.py align --data outputs/llama3-check/input.json --checkpoint outputs/llama3-alignment-check --output outputs/llama3-check/alignment.json
python local_workflow.py context-check --model-profile llama3-8b --verifier cot --data outputs/llama3-check/input.json
python local_workflow.py verify --model-profile llama3-8b --verifier cot --data outputs/llama3-check/input.json --output outputs/llama3-check/cot-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
python local_workflow.py reassess --model-profile llama3-8b --data outputs/llama3-check/alignment.json --literal outputs/llama3-check/cot-repeat-control/log.jsonl --output outputs/llama3-check/cot-reassessment-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

The two-step alignment model and five-claim sample test execution only. Only
baseline `true` predictions enter reassessment, so this sample may not exercise
all reasoning stages. Resolve context, memory, or format errors before a full run.

## 6. Submit each full stage manually

Finish the interactive session with `exit`. From the REPACSS login shell, enter
the uploaded TRACER directory and create `logs` before submitting jobs. The supplied
`scripts/repacss_llama3.slurm` activates `tracer-llama3` and adds the Llama profile
and `--device-map cuda` to every command. It automatically checks all input prompt
lengths before a verification job, so that stage stops before partial predictions
if its baseline cannot fit.

If Miniforge is elsewhere, set `TRACER_CONDA_INIT` to its `etc/profile.d/conda.sh`
path before submission. To use another environment, set `TRACER_CONDA_ENV`.
No GPU IDs are hardcoded; the script respects Slurm's allocation.

```bash
mkdir -p logs
sbatch scripts/repacss_llama3.slurm train --data dataset/train.json --epochs 5 --output outputs/alignment-full
```

**Wait for each job to finish successfully before submitting the next command.**
If you transferred a completed full alignment checkpoint, skip the training job.
Then submit alignment:

```bash
sbatch scripts/repacss_llama3.slurm align --data dataset/test.json --checkpoint outputs/alignment-full --output outputs/llama3-repeat-test/alignment.json
```

Next run HiSS and its reassessment, one at a time, only if HiSS passed preflight:

```bash
sbatch scripts/repacss_llama3.slurm verify --data dataset/test.json --output outputs/llama3-repeat-test/literal --repetition-penalty 1.1 --no-repeat-ngram-size 8
sbatch scripts/repacss_llama3.slurm reassess --data outputs/llama3-repeat-test/alignment.json --literal outputs/llama3-repeat-test/literal/log.jsonl --output outputs/llama3-repeat-test/reassessment --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

Then run CoT and its reassessment, one at a time, only if CoT passed preflight:

```bash
sbatch scripts/repacss_llama3.slurm verify --verifier cot --data dataset/test.json --output outputs/llama3-repeat-test/cot-literal --repetition-penalty 1.1 --no-repeat-ngram-size 8
sbatch scripts/repacss_llama3.slurm reassess --data outputs/llama3-repeat-test/alignment.json --literal outputs/llama3-repeat-test/cot-literal/log.jsonl --output outputs/llama3-repeat-test/cot-reassessment --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

These commands use the base model with repetition penalty 1.1 and no-repeat
n-gram size 8 for both baselines and both reassessment runs, without shortening
the inputs. The Slurm script forwards these explicit flags; it does not add them
automatically. Use these settings for a full experiment only after the development
check succeeds. Its outputs are separate from the original decoding experiment.
They are not a promise that every 2,000-claim stage fits or every generation succeeds.

## 7. Monitor and resume

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 50 logs/tracer-llama3-JOB_ID.out
tail -n 50 logs/tracer-llama3-JOB_ID.err
```

Replace `JOB_ID` with the number printed by `sbatch`. Logs use the Slurm job name
and ID. If needed, override runtime at submission with `sbatch --time=...` within
your allocation's limits. Use `scancel JOB_ID` only for the specific job you intend
to stop; do not run two jobs writing the same output directory.

For interrupted verification/reassessment, retain `--repetition-penalty 1.1`
and `--no-repeat-ngram-size 8` and resubmit the exact same command after
the old job exits. Completed JSONL records and cached successful responses are
reused. After a hard kill, inspect the last JSONL line if resume reports a parsing
error; do not delete the entire run. For interrupted training, select an actual
saved epoch checkpoint and pass it explicitly:

```bash
ls -d outputs/alignment-full/checkpoint-*
# Replace NUMBER with a saved checkpoint's number:
sbatch scripts/repacss_llama3.slurm train --data dataset/train.json --epochs 5 --output outputs/alignment-full --resume-checkpoint outputs/alignment-full/checkpoint-NUMBER
```

Llama's default output allowance is 1,536 tokens. Bounded format recovery is enabled
and audited, but can still fail. Any output-cap increase must fit input plus output
within 8,192 tokens and be retained on subsequent resumes. New models, changed
inputs, or other settings require new output folders. Do not mix Qwen and Llama logs.

## 8. Produce the comparison

After all four full stages complete:

```bash
sbatch scripts/repacss_llama3.slurm compare --data dataset/test.json --literal outputs/llama3-repeat-test/literal/log.jsonl --reassessed outputs/llama3-repeat-test/reassessment/log.jsonl --cot-literal outputs/llama3-repeat-test/cot-literal/log.jsonl --cot-reassessed outputs/llama3-repeat-test/cot-reassessment/log.jsonl --output outputs/llama3-repeat-test/comparison-all
```

The comparison itself does not need a GPU; the generic script is used here for
consistent environment activation and scheduling. Read
`outputs/llama3-repeat-test/comparison-all/comparison.md` and its JSON. Compare with the
separate Qwen report, reporting model, prompt versions, token budgets, recovery
counts, and environment differences. This remains a local adaptation with prompted
intent, not the paper's fine-tuned intent model. Larger model size or GPU capacity
does not guarantee better benchmark scores.

## Windows laptop execution check

For the 16 GB laptop GPU, use `--device-map auto` to permit CPU weight offload.
This is slower than the REPACSS configuration and needs free system RAM too.
From `A:\TTU\NN\Researchpaper\TRACER` in PowerShell:

```powershell
..\.venv\Scripts\hf.exe auth login
..\.venv\Scripts\python.exe local_workflow.py download --model-profile llama3-8b
..\.venv\Scripts\python.exe local_workflow.py llm-check --model-profile llama3-8b --device-map auto --max-new-tokens 64
```

The workflow now respects a configured `HF_HOME` and otherwise references an
existing default Hugging Face login when the project cache has no token. A browser
login alone is insufficient. No token is copied into model files or run metadata.
If access still fails, verify that the CLI account has approval for this exact
model and that its token can read gated repositories.

The short check only tests loading and generation. Use the development fixture
before a full run; the base model may fail to follow the requested answer format.

CoT also accepts a normally completed response starting with the explicit sentence
`The claim is true.`, `The claim is half-true.`, or `The claim is false.` when
there are no conflicting label mentions, uncertainty/negation markers, or malformed
angle tags. This conservative parser fallback is audited as
`explicit-opening-verdict-v1` in `generation_recoveries`; rows identify the parser
as `cot-explicit-opening-v1`. Truncated generations remain invalid. A cached
completed response can be reparsed on resume without regenerating it. Include this
parser policy when reporting benchmark results; it changes which formats qualify
as valid predictions.

During generation the console now prints the input/output budget, the first
generated token, and token counts with elapsed time approximately every 15 seconds
as decoding advances. The claim progress bar advances only after a complete claim.
If no first token appears, the model may still be processing the prompt. These
messages are driven by token generation, not a background timer, so a stalled
forward pass does not produce a heartbeat. Cached answers are identified too.
An already running Python process must be restarted to use this logging change.
Press Ctrl+C once, wait for the prompt, and repeat the same command/output folder
to preserve completed claims. The unfinished generation starts again. Logging
does not change prompts, token limits, or predictions.

On this laptop, full BF16 `--device-map cuda` loading can leave very little VRAM
for generation. If trying `--device-map auto`, use a new output folder because
placement is part of the tracked settings. CPU offload may be substantially slower;
it is a memory option, not a speed improvement.

## What the repetition controls change

Counterfactual/final reassessment choice parsing also accepts a single fenced
Python `print("A")` (or another allowed letter) as a formatting wrapper. The
parser inspects syntax without executing it; multiple statements, expressions,
and unsupported choices remain errors. This fallback is recorded as
`literal-print-choice-v1` with a response hash in `generation_recoveries`.
Resume with the same command to reuse cached responses after this parser repair.
Include formatting recoveries when reporting results; they are not model training.

If the base model repeatedly exhausts its output allowance, inspect the saved
`*.truncated.json` response. A repeating answer is not fixed merely by increasing
the token allowance. The diagnostic commands in this guide apply repetition controls:

```powershell
python -u local_workflow.py verify --model-profile llama3-8b --verifier cot --data outputs/llama3-check/input.json --output outputs/llama3-check/cot-repeat-control --repetition-penalty 1.1 --no-repeat-ngram-size 8
```

These controls discourage reused tokens and block repeated eight-token sequences;
they do not guarantee valid formatting, a correct label, or normal completion.
They can also affect legitimate repeated phrases, so report them as a decoding
change. The Python defaults remain unchanged; the guide passes the controls explicitly. New settings use distinct response-cache
keys, and manifests prevent mixing them into an existing run. Use the same flags
on resume and when intentionally applying this policy to reassessment. Truncated
responses are still rejected. This option is software-tested, not yet validated
as a successful full Llama benchmark.

## References

- [REPACSS jobs and Slurm templates](https://www.repacss.org/user-guide/running-jobs/)
- [REPACSS GPU resource instructions](https://guide.repacss.org/running-jobs/basics.html)
- [REPACSS Miniforge setup](https://guide.repacss.org/software/miniforge.html)
- [PyTorch version-specific CUDA installation](https://pytorch.org/get-started/previous-versions/)
- [Meta Llama 3 8B model card and access](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
