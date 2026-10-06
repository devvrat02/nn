# Llama 3.1 8B on REPACSS

This guide replaces the Llama 2 model steps with **Llama 3.1 8B**. Use one H100
with BF16 weights fully on GPU. The Linux environment and shared RoBERTa/MiniLM/NLI
models can be reused. Do not mix model predictions in the same output folder.

## 1. Select the exact model

| Profile | Hugging Face model | Prompt handling |
|---|---|---|
| `llama31-8b` | `meta-llama/Llama-3.1-8B` | Base pretrained model; experimental plain-text `base-v1` task/response format |
| `llama31-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | Instruction-tuned model; tokenizer's official chat template |

The model you linked is the **base model**. It can generate text, but its ability
to obey TRACER's exact verdict and structured-answer formats is not established.
The **Instruct** model is recommended for these stages if your account can access
it. Base access alone is not treated as confirmation of Instruct access. A plain
task/response wrapper does not turn a base model into an instruction-tuned model.

Llama 3.1 supports 128K context according to Meta's model cards. This workflow starts
with **16,384 total tokens and a 1,536-token output allowance** for both profiles.
Run preflight checks; increase the configured context only if needed, using new
output folders for changed settings. More GPU RAM alone does not change context.

## 2. Transfer and install on REPACSS

Follow **sections 1 and 2 only** of [RUN_LLAMA2.md](RUN_LLAMA2.md) for transferring
the repaired project, allocating an H100, and installing `requirements-repacss.txt`.
The environment name remains `tracer-llama2` for reuse; it can run all supported
models. Do not run that guide's Llama 2 download or benchmark commands.

The following commands run in the allocated GPU session from the uploaded TRACER
directory, with that environment active. They select the exact base model you
linked. If Instruct access is confirmed, change the profile assignment before
running any download or experiment commands:

```bash
export TRACER_LLAMA_PROFILE=llama31-8b
# For confirmed Instruct access, use this INSTEAD:
# export TRACER_LLAMA_PROFILE=llama31-8b-instruct
export HF_HOME="$PWD/.cache/huggingface"
hf auth login
python local_workflow.py download --model-profile "$TRACER_LLAMA_PROFILE"
```

Enter your Hugging Face token privately at the login prompt. A gated-access error
must be resolved with the account/token that has permission for the chosen model.
The selected model gets its own `models/` directory and pinned revision. Allow
roughly 16 GB for its BF16 weights plus space for shared models and outputs.
Downloads need internet; subsequent model execution is offline without paid APIs.

## 3. Check execution and context

```bash
python -m unittest tests.test_regressions tests.test_local_workflow tests.test_generation_recovery tests.test_cot tests.test_llama_profile -q
python local_workflow.py llm-check --model-profile "$TRACER_LLAMA_PROFILE"
python local_workflow.py context-check --model-profile "$TRACER_LLAMA_PROFILE" --data dataset/test.json
python local_workflow.py context-check --model-profile "$TRACER_LLAMA_PROFILE" --verifier cot --data dataset/test.json
```

Inspect whether the loading check actually replies `LOCAL_MODEL_OK`; it prints
the answer rather than certifying instruction following. Both context checks must
report zero over-budget prompts. Intermediate reassessment prompts are checked at
runtime too. The base model may ignore formatting even when the context fits.

For a small development check, first create a five-claim input and two-step
alignment fixture if you do not already have them:

```bash
python local_workflow.py subset --data dataset/dev.json --limit 5 --output outputs/llama31-check/input.json
python local_workflow.py train --data dataset/train.json --steps 2 --output outputs/llama31-alignment-check
python local_workflow.py align --data outputs/llama31-check/input.json --checkpoint outputs/llama31-alignment-check --output outputs/llama31-check/alignment.json
CHECK="outputs/${TRACER_LLAMA_PROFILE}-check"
python local_workflow.py verify --model-profile "$TRACER_LLAMA_PROFILE" --verifier cot --data outputs/llama31-check/input.json --output "$CHECK/cot-literal"
python local_workflow.py reassess --model-profile "$TRACER_LLAMA_PROFILE" --data outputs/llama31-check/alignment.json --literal "$CHECK/cot-literal/log.jsonl" --output "$CHECK/cot-reassessment"
```

Resolve failures before benchmarking. Five claims do not establish performance.
If the base model repeatedly produces invalid answers, use Instruct or conduct a
separate fine-tuning experiment rather than interpreting parser failures as labels.

## 4. Submit the full stages one at a time

Exit the interactive allocation. In the REPACSS login shell, enter the TRACER
directory and set the same profile again; exports in the allocation's shell do not
automatically propagate back to the login shell:

```bash
export TRACER_LLAMA_PROFILE=llama31-8b
# Use llama31-8b-instruct here instead if that is the profile you tested.
RUN="outputs/${TRACER_LLAMA_PROFILE}-test"
mkdir -p logs
```

The new script `scripts/repacss_llama31.slurm` reads this exported selection and
defaults to the base model if unset. It requests one H100, 64 GB RAM, 8 CPU cores,
and 12 hours. Add your assigned Slurm account or adjust time limits if required.
`TRACER_CONDA_INIT` and `TRACER_CONDA_ENV` can override the Miniforge path and
environment, as in the earlier guide.

If you have no fully trained alignment model, submit training and wait for success:

```bash
sbatch scripts/repacss_llama31.slurm train --data dataset/train.json --epochs 5 --output outputs/alignment-full
```

Reuse a transferred full checkpoint if available. The model change does not require
retraining alignment. Then run alignment, followed by verification and reassessment.
**Wait for each job to finish successfully before submitting the next command.**

```bash
sbatch scripts/repacss_llama31.slurm align --data dataset/test.json --checkpoint outputs/alignment-full --output "$RUN/alignment.json"
sbatch scripts/repacss_llama31.slurm verify --data dataset/test.json --output "$RUN/literal"
sbatch scripts/repacss_llama31.slurm reassess --data "$RUN/alignment.json" --literal "$RUN/literal/log.jsonl" --output "$RUN/reassessment"
sbatch scripts/repacss_llama31.slurm verify --verifier cot --data dataset/test.json --output "$RUN/cot-literal"
sbatch scripts/repacss_llama31.slurm reassess --data "$RUN/alignment.json" --literal "$RUN/cot-literal/log.jsonl" --output "$RUN/cot-reassessment"
```

Verification jobs first check prompt lengths. The script respects Slurm GPU
assignment; it does not set physical GPU IDs. Model weights stay on GPU while
evidence rankers stay on CPU. RoBERTa training retains effective batch size 8.

## 5. Monitor, resume, and compare

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 50 logs/tracer-llama31-JOB_ID.out
tail -n 50 logs/tracer-llama31-JOB_ID.err
```

Replace `JOB_ID` with the submitted job number. After an interrupted verification
or reassessment job exits, resubmit its same command with the same profile/settings.
Do not launch simultaneous writers to one output folder. Training resumes from
saved epoch checkpoints using `--resume-checkpoint`, as in the Llama 2 guide.

Unlike Llama 2, Llama 3.1 has room for a larger output allowance when required.
An increase such as `--max-new-tokens 4096` is permitted on resume if prompt plus
output fits the configured context. It is audited and must be retained on later
resumes. Larger allowances do not cure all repetition/format failures.

After all four stages complete:

```bash
sbatch scripts/repacss_llama31.slurm compare --data dataset/test.json --literal "$RUN/literal/log.jsonl" --reassessed "$RUN/reassessment/log.jsonl" --cot-literal "$RUN/cot-literal/log.jsonl" --cot-reassessed "$RUN/cot-reassessment/log.jsonl" --output "$RUN/comparison-all"
```

Read `$RUN/comparison-all/comparison.md` and its JSON. Report the exact model
variant, prompt style, token budgets, recovery counts, and package versions. This
is a local adaptation with prompted intent, not the paper's fine-tuned model.

**Validation scope:** software tests run locally; real Llama 3.1 weights and
REPACSS execution have not been tested here. Access is supplied by your account.

Sources: [Meta base model](https://huggingface.co/meta-llama/Llama-3.1-8B),
[Meta Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
and the REPACSS setup references in [RUN_LLAMA2.md](RUN_LLAMA2.md#references).
