# Llama 2 7B Chat on REPACSS

This guide targets **REPACSS Linux GPU jobs**, not the Windows laptop. It selects
`meta-llama/Llama-2-7b-chat-hf`, the chat-tuned Llama 2 7B model, using
`--model-profile llama2-7b`. Qwen stays the default for older commands so existing
runs remain resumable. New Llama results use `outputs/llama2-test/`.

The REPACSS job script requests **one H100 NVL**, 8 CPU cores, 64 GB host RAM,
and 12 hours. It loads Llama in **BF16 entirely on the allocated GPU**. No
quantization or CPU weight offload is needed for this target. MiniLM/NLI remain on
CPU to preserve the existing workflow. Alignment keeps microbatch 1 and accumulation
8 (effective batch 8); moving machines does not silently change training settings.

**Status:** local offline regression tests pass. This environment has no REPACSS
connection or Llama weights; cluster installation, real Llama inference, and the
full benchmark still need validation with the checks below.

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
conda create -n tracer-llama2 python=3.11 -y
conda activate tracer-llama2
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
already tested REPACSS installation. Keep its frozen package list with results.

Ensure the working directory is the uploaded `TRACER` folder inside the allocation.
Run the software and GPU checks:

```bash
python local_workflow.py doctor --model-profile llama2-7b
python -m unittest tests.test_regressions tests.test_local_workflow tests.test_generation_recovery tests.test_cot tests.test_llama_profile -q
python run_pipeline.py --mode smoke
```

`doctor` should identify the allocated GPU and pass BF16 arithmetic. Its general
model inventory also includes Qwen; missing Qwen is expected for a Llama-only
installation. Smoke mode uses scripted responses and is not benchmark evidence.

## 3. Authenticate and download the model

Request access on the [official Llama 2 7B Chat model page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
using your own Hugging Face account. Once access is granted, authenticate locally:

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf auth login
python local_workflow.py download --model-profile llama2-7b
```

Enter a token with access to the gated model at the login prompt. Do not put it in
the job script or source files. A 401/403 means account/token access must be fixed.
Downloads need network access and may exceed the initial allocation's wall time;
repeat the command to resume. If outbound downloads are restricted, use the
cluster's supported transfer method. Reserve space for roughly 14 GB of Llama
weights plus the shared models, training checkpoints, caches, and outputs.

The download stores weights in `models/llama2-7b-chat` and the shared models in
`models/roberta`, `models/minilm`, and `models/nli`. Revisions are pinned in
`models/model_revisions.json`. Later inference runs offline without paid APIs.

## 4. Check the model and its context limit

```bash
python local_workflow.py llm-check --model-profile llama2-7b
python local_workflow.py context-check --model-profile llama2-7b --data dataset/test.json
python local_workflow.py context-check --model-profile llama2-7b --verifier cot --data dataset/test.json
```

The first command asks for `LOCAL_MODEL_OK`; inspect the actual answer. Each
context check must report `over_budget: 0` before starting its full baseline.

**A larger GPU does not increase Llama 2's native 4,096-token context.** The
profile reserves 512 output tokens, leaving at most 3,584 for the formatted prompt.
Some existing HiSS/evidence prompts may not fit. The code rejects an oversized
context setting and does not drop evidence. Do not use Qwen's
`--max-new-tokens 4096`: that leaves no room for Llama's input.

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
python local_workflow.py subset --data dataset/dev.json --limit 5 --output outputs/llama2-check/input.json
python local_workflow.py train --data dataset/train.json --steps 2 --output outputs/llama2-alignment-check
python local_workflow.py align --data outputs/llama2-check/input.json --checkpoint outputs/llama2-alignment-check --output outputs/llama2-check/alignment.json
python local_workflow.py context-check --model-profile llama2-7b --verifier cot --data outputs/llama2-check/input.json
python local_workflow.py verify --model-profile llama2-7b --verifier cot --data outputs/llama2-check/input.json --output outputs/llama2-check/cot-literal
python local_workflow.py reassess --model-profile llama2-7b --data outputs/llama2-check/alignment.json --literal outputs/llama2-check/cot-literal/log.jsonl --output outputs/llama2-check/cot-reassessment
```

The two-step alignment model and five-claim sample test execution only. Only
baseline `true` predictions enter reassessment, so this sample may not exercise
all reasoning stages. Resolve context, memory, or format errors before a full run.

## 6. Submit each full stage manually

Finish the interactive session with `exit`. From the REPACSS login shell, enter
the uploaded TRACER directory and create `logs` before submitting jobs. The supplied
`scripts/repacss_llama2.slurm` activates `tracer-llama2` and adds the Llama profile
and `--device-map cuda` to every command. It automatically checks all input prompt
lengths before a verification job, so that stage stops before partial predictions
if its baseline cannot fit.

If Miniforge is elsewhere, set `TRACER_CONDA_INIT` to its `etc/profile.d/conda.sh`
path before submission. To use another environment, set `TRACER_CONDA_ENV`.
No GPU IDs are hardcoded; the script respects Slurm's allocation.

```bash
mkdir -p logs
sbatch scripts/repacss_llama2.slurm train --data dataset/train.json --epochs 5 --output outputs/alignment-full
```

**Wait for each job to finish successfully before submitting the next command.**
If you transferred a completed full alignment checkpoint, skip the training job.
Then submit alignment:

```bash
sbatch scripts/repacss_llama2.slurm align --data dataset/test.json --checkpoint outputs/alignment-full --output outputs/llama2-test/alignment.json
```

Next run HiSS and its reassessment, one at a time, only if HiSS passed preflight:

```bash
sbatch scripts/repacss_llama2.slurm verify --data dataset/test.json --output outputs/llama2-test/literal
sbatch scripts/repacss_llama2.slurm reassess --data outputs/llama2-test/alignment.json --literal outputs/llama2-test/literal/log.jsonl --output outputs/llama2-test/reassessment
```

Then run CoT and its reassessment, one at a time, only if CoT passed preflight:

```bash
sbatch scripts/repacss_llama2.slurm verify --verifier cot --data dataset/test.json --output outputs/llama2-test/cot-literal
sbatch scripts/repacss_llama2.slurm reassess --data outputs/llama2-test/alignment.json --literal outputs/llama2-test/cot-literal/log.jsonl --output outputs/llama2-test/cot-reassessment
```

These commands change the language model without shortening the original inputs.
They are not a promise that every 2,000-claim stage fits or every generation succeeds.

## 7. Monitor and resume

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 50 logs/tracer-llama2-JOB_ID.out
tail -n 50 logs/tracer-llama2-JOB_ID.err
```

Replace `JOB_ID` with the number printed by `sbatch`. Logs use the Slurm job name
and ID. If needed, override runtime at submission with `sbatch --time=...` within
your allocation's limits. Use `scancel JOB_ID` only for the specific job you intend
to stop; do not run two jobs writing the same output directory.

For interrupted verification/reassessment, resubmit the exact same command after
the old job exits. Completed JSONL records and cached successful responses are
reused. After a hard kill, inspect the last JSONL line if resume reports a parsing
error; do not delete the entire run. For interrupted training, select an actual
saved epoch checkpoint and pass it explicitly:

```bash
ls -d outputs/alignment-full/checkpoint-*
# Replace NUMBER with a saved checkpoint's number:
sbatch scripts/repacss_llama2.slurm train --data dataset/train.json --epochs 5 --output outputs/alignment-full --resume-checkpoint outputs/alignment-full/checkpoint-NUMBER
```

Llama's default output allowance is 512 tokens. Bounded format recovery is enabled
and audited, but can still fail. Any output-cap increase must fit input plus output
within 4,096 tokens and be retained on subsequent resumes. New models, changed
inputs, or other settings require new output folders. Do not mix Qwen and Llama logs.

## 8. Produce the comparison

After all four full stages complete:

```bash
sbatch scripts/repacss_llama2.slurm compare --data dataset/test.json --literal outputs/llama2-test/literal/log.jsonl --reassessed outputs/llama2-test/reassessment/log.jsonl --cot-literal outputs/llama2-test/cot-literal/log.jsonl --cot-reassessed outputs/llama2-test/cot-reassessment/log.jsonl --output outputs/llama2-test/comparison-all
```

The comparison itself does not need a GPU; the generic script is used here for
consistent environment activation and scheduling. Read
`outputs/llama2-test/comparison-all/comparison.md` and its JSON. Compare with the
separate Qwen report, reporting model, prompt versions, token budgets, recovery
counts, and environment differences. This remains a local adaptation with prompted
intent, not the paper's fine-tuned intent model. Larger model size or GPU capacity
does not guarantee better benchmark scores.

## References

- [REPACSS jobs and Slurm templates](https://www.repacss.org/user-guide/running-jobs/)
- [REPACSS GPU resource instructions](https://guide.repacss.org/running-jobs/basics.html)
- [REPACSS Miniforge setup](https://guide.repacss.org/software/miniforge.html)
- [PyTorch version-specific CUDA installation](https://pytorch.org/get-started/previous-versions/)
- [Meta Llama 2 7B Chat model card and access](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
