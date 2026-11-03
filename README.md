# TRACER for Half-Truth Detection

This repository contains the code and resources for our EMNLP paper:
**"The Missing Parts: Augmenting Fact Verification with Half-Truth Detection"**.

For server setup and instructions to run with a local Llama checkpoint, see
[RUN_README.md](RUN_README.md).

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


## Installation

```bash
conda create -n phidden python=3.9
conda activate phidden
pip install -r requirements.txt
```


## Evidence Alignment

Train the sentence alignment model and predict the test split:

```bash
bash run_alignment.sh
```

This script trains a RoBERTa-large alignment model, selects the latest saved
checkpoint, and writes predictions to `test_alignment.json`. See
[RUN_README.md](RUN_README.md) for server configuration and memory options.


## Intent Generation

The original implementation prepares data for GPT-4o-mini fine-tuning with
`python intent_generation/intent_finetune.py`. For running the pipeline with
the local Llama model, no fine-tuning is required; set `LLM_MODEL_PATH` and
follow [RUN_README.md](RUN_README.md).


## Claim Verification & Re-Assessment

First, verify claims literally with the configured local model:

```bash
bash run_fact_check.sh
```

Results are saved under a timestamped directory in `method/result/`.

Then, re-assess samples predicted as **True**:

```bash
LITERAL=method/result/<timestamp>/log.jsonl bash run_reassessment.sh
```

Re-assessment results are saved under a new timestamped directory in
`method/result/`. See [RUN_README.md](RUN_README.md) for the full run order.


## Evaluation

Run evaluation with `method/eval.py`, specify `datapath`.


| Model        | Accuracy | F1    | Precision(H) | Recall(H) | F1(H)  |
|-------------|----------|-------|-------------|----------|--------|
| CoT         | 76.30    | 64.25 | 44.97       | 63.79    | 52.75  |
| CoT + RA    | 78.50    | **68.00** | 48.49       | **79.31** | 60.19  |
| HiSS        | 78.25    | 59.36 | 53.66       | 37.93    | 44.44  |
| HiSS + RA   | **81.85** | 65.74 | **55.31** | 66.75 | **60.49** |

**RA = TRACER re-assessment module.** Gains are most pronounced in the *Half-True* class, showing TRACER’s ability to capture omission-based manipulation.


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
