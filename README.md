# TRACER for Half-Truth Detection

**Fully local, manual experiment:** follow [RUN_LOCAL.md](RUN_LOCAL.md) to download
models, train alignment, run Qwen verification/reassessment, and compare measured
results with the paper. No paid API is required for this workflow.
Section 11 adds local CoT and CoT + TRACER and produces a combined four-method report.

**Llama 2 7B Chat on REPACSS:** see [RUN_LLAMA2.md](RUN_LLAMA2.md) for Linux setup,
model access, the `--model-profile llama2-7b` switch, and Slurm benchmark commands.
The supplied job script loads BF16 weights fully on one H100 GPU. Its native
4,096-token context requires preflight checks; long prompts may prevent an
unchanged full benchmark. Qwen remains the default for existing commands.

Alternative API-backed setup and smoke checks: [RUNNING.md](RUNNING.md).
Paper walkthrough and implementation details: [PAPER_EXPLAINED.md](PAPER_EXPLAINED.md).
On this Windows machine, run `./run_local.cmd --mode smoke` for an offline integration test
or `./run_local.cmd --mode prepare` to validate the dataset and export intent training data.
Use the local run guide for the repaired paths and this machine's Python 3.14 environment.

This repository contains the code and resources for our EMNLP paper:
**"The Missing Parts: Augmenting Fact Verification with Half-Truth Detection"**.

We introduce **POLITIFACT-HIDDEN**, a benchmark of \~15k political claims annotated with **sentence-level evidence alignment**. Building on this dataset, we present **TRACER** (Truth Re-Assessment with Critical hidden Evidence Reasoning), a modular framework designed to detect **omission-based misinformation**, i.e., claims that are factually correct yet misleading due to missing critical context.

## Dataset: POLITIFACT-HIDDEN

POLITIFACT-HIDDEN extends the original [PolitiFact](https://www.politifact.com/) corpus with **fine-grained omission-aware annotations**:

Each example contains:

* **claim:** the statement to be verified.
* **Presented Evidence (PE):** Sentences explicitly stated or implied in the claim.
* **Hidden Evidence (HE):** Relevant sentences not mentioned in the claim.
* **Intent:** The implied conclusion the claim conveys.
* **rating:** True / False / Half-truth label.

### Label Mapping

PolitiFact’s original six-level ratings are consolidated into three coarse labels:

| Original Rating(s)                 | Consolidated Label |
| ---------------------------------- | ------------------ |
| True                               | True               |
| Mostly True, Half True             | Half-True          |
| Mostly False, False, Pants on Fire | False              |

### Dataset Statistics

| Split     | True      | Half-True | False     | Total      |
| --------- | --------- | --------- | --------- | ---------- |
| Train     | 1,352     | 4,564     | 6,078     | 11,994     |
| Dev       | 64        | 195       | 741       | 1,000      |
| Test      | 93        | 406       | 1,501     | 2,000      |

Additionally, the test set (2020–2025) is **temporally disjoint** from training data to assess generalization and robustness.

## TRACER Framework Overview

Half-truths exploit **omission** rather than outright falsehood.
TRACER complements traditional fact verification pipelines by re-assessing claims through three stages:

1. **Evidence Alignment** – classify retrieved evidence into *Presented* vs. *Hidden* evidence.
2. **Intent Generation** – infer the implied conclusion of the claim.
3. **Causality Analysis** – identify **Critical Hidden Evidence (CHE)** that undermines the claim’s intent.

This re-assessment improves detection of misleading claims that would otherwise be labeled “True” by standard FV systems.

![framework](pics/overall_framework.png)


## Run the local experiment

Follow [RUN_LOCAL.md](RUN_LOCAL.md) in order. Run commands from the `TRACER`
directory with `..\.venv\Scripts\python.exe`; no API key is required. The guide
covers environment checks, model downloads, data validation, a small execution
check, full alignment training, and all four benchmark runs.

| Step | Guide section | Output |
|---|---|---|
| Train RoBERTa-large for five epochs | 5 | `outputs/alignment-full` |
| Align the 2,000 test claims with that checkpoint | 7 | `outputs/local-test/alignment.json` |
| Run local HiSS | 8 | `outputs/local-test/literal/log.jsonl` |
| Apply TRACER to HiSS | 9 | `outputs/local-test/reassessment/log.jsonl` |
| Run local CoT and apply TRACER to CoT | 11 | `outputs/local-test/cot-literal/log.jsonl`, `outputs/local-test/cot-reassessment/log.jsonl` |
| Compare all four completed runs | 11 | `outputs/local-test/comparison-all/comparison.md` and `comparison.json` |

Reuse the full alignment predictions for both baselines. Only claims predicted
`true` by the corresponding baseline enter reassessment. A two-step debug
checkpoint or five-claim execution check is not a benchmark result.

Verification and reassessment resume when the same command is repeated: completed
records are preserved. If you increased `--max-new-tokens` in an existing run,
include that same increased value on subsequent resumes. See the guide's
**Restarting, memory, and errors** section for output-limit recovery and audit logs.
The shorter recovery prompt can still fail; setup tests do not guarantee every
model response will be valid.

The original hosted workflow is documented separately in [RUNNING.md](RUNNING.md).
It requires hosted GPT access and an intent model; it is not needed for local runs.

## Why local benchmark results can differ

**The largest expected source of difference is replacing the paper's GPT models
with Qwen2.5-3B-Instruct.** This is a likely explanation, not a measured attribution:
isolating each cause would require controlled experiments.

| Difference | Effect on the comparison |
|---|---|
| Language model | The paper uses GPT-3.5-turbo for HiSS and GPT-4o-mini for other baseline/reasoning stages. The local workflow uses Qwen2.5-3B-Instruct for all language-model stages, changing verification, reasoning, and instruction following. |
| Intent training | The paper fine-tunes GPT-4o-mini for intent generation. Local Qwen uses prompting without intent fine-tuning, which can change the inferred intent and downstream assumptions. |
| Prompts and output limits | Local CoT uses our zero-shot `cot-v1` prompt because the exact authors' CoT prompt was not supplied in the repository. Output limits and additional brevity/format recovery instructions can also change answers. |
| Alignment training | Training duration, random seed, checkpoint selection, and numerical precision can change RoBERTa's evidence predictions. Use the fully trained checkpoint, not `alignment-check` or a smoke-test model. |
| Implementation repairs | The repaired pipeline uses predicted hidden evidence during intent inference; the released code discarded this filter. That behavioral correction can change results. |

### RoBERTa-large batch size

The local training wrapper reduces the per-device microbatch from **8 to 1** and
uses **8 gradient accumulation steps**. On this single-GPU setup, the effective
batch remains **1 x 8 = 8 examples per optimizer update**. BF16 and gradient
checkpointing also reduce GPU memory use. These settings do not guarantee
numerically identical training, but reducing the microbatch while preserving the
effective batch is expected to matter less than changing the language model and
omitting intent fine-tuning. This batch setting applies to alignment training,
not Qwen generation.

### How to report the comparison

Describe the experiment as a **local adaptation benchmark**, rather than an exact
reproduction. Report both absolute scores and the change in percentage points for
each matching pair: **local HiSS to local HiSS + TRACER**, and **local CoT to local
CoT + TRACER**. These paired comparisons measure whether reassessment helps the
selected local baseline; improvement is not guaranteed.

Use the same full test split and label mapping for all four runs. Report accuracy,
macro-F1, and half-true precision, recall, and F1. The test split is imbalanced
(1,501 of 2,000 claims are false), so accuracy alone is insufficient. Choose settings
on development data and disclose any changes made after inspecting test failures.

Keep `local_run.json`, alignment provenance, `training_run.json`, any
`budget_changes.jsonl`, and recovery annotations with the results. The comparison
JSON includes run manifests, token-cap history, and recovery counts. Disclose
unequal token allowances or recovery policies across methods. Timing comparisons
also depend on hardware and placement: local Qwen runs on GPU, while MiniLM and
DeBERTa ranking run on CPU.

## Published paper results

These are the published reference values preserved from the released README,
**not measured local results**. Values are percentages; F1 is macro-F1 and H denotes
the half-true class. Generate local scores from completed logs using the comparison
command in [RUN_LOCAL.md](RUN_LOCAL.md#11-add-local-cot-and-cot--tracer).

| Model        | Accuracy | Macro-F1 | Precision(H) | Recall(H) | F1(H)  |
|-------------|----------|-------|-------------|----------|--------|
| CoT         | 76.30    | 64.25 | 44.97       | 63.79    | 52.75  |
| CoT + RA    | 78.50    | **68.00** | 48.49       | **79.31** | 60.19  |
| HiSS        | 78.25    | 59.36 | 53.66       | 37.93    | 44.44  |
| HiSS + RA   | **81.85** | 65.74 | **55.31** | 66.75 | **60.49** |

**RA = TRACER re-assessment module.** These published improvements do not guarantee the same gains with the local model.


## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{conf/emnlp/TRACER,
  author       = {Yixuan Tang and Jincheng Wang and Anthony Kum Hoe Tung},
  title        = {The Missing Parts: Augmenting Fact Verification with Half Truth Detection},
  booktitle    = {{EMNLP}},
  year         = {2025}
}
```
