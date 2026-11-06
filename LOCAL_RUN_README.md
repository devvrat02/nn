# Run the 200-Example TRACER Comparison on Your Computer

This guide is written as a step-by-step checklist. It explains how to run the
comparison saved in
[`method/result/20261105_030023_892399_assumption_aware_comparison/`](method/result/20261105_030023_892399_assumption_aware_comparison/):

- **ORIGINAL**: original TRACER + HiSS.
- **IMPROVED**: TRACER + HiSS with the assumption-aware evidence filter.

The comparison uses the first 200 examples from `dataset/test.json`. It does
not run the whole dataset. The Decision Consistency Guard is a separate
experiment and is not enabled by these instructions.

## What you need before starting

You need:

1. This project folder on a Linux computer (or Linux under WSL).
2. Python 3.9.
3. An NVIDIA GPU, CUDA-enabled PyTorch, and enough free GPU memory to load your
   local instruction-tuned model. This project does not run its fact-checking
   language model on CPU.
4. A local Hugging Face model folder. It must contain `config.json` and the
   model weights. For the saved experiment, the model was Llama 3.2 3B
   Instruct in FP16. You must obtain the model separately and follow its
   license; it is not included in this repository.
5. These matching input files:
   - `dataset/test.json`
   - `test_alignment_evidence_1_200_fixed.json`
   - `method/result/20261104_213400_650571_benchmark/hiss/log.jsonl`

The last two files may be generated artifacts and might not be present in a
fresh copy of the project. Instructions to generate them are below. The
comparison runner checks that the example IDs and claims in these files match
the selected dataset rows.

## Step 1: Open a terminal in the project folder

Change directory to the folder containing `README.md`, `requirements.txt`, and
`method/`:

```bash
cd /path/to/nn
```

Replace `/path/to/nn` with the actual location on your computer. For example:

```bash
cd "$HOME/nn"
```

## Step 2: Create and activate the Python environment

Run these commands once:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch must be able to use your NVIDIA GPU. Check it:

```bash
nvidia-smi
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
```

The second command must print `CUDA available: True`. If it prints `False`,
install a CUDA-enabled PyTorch version compatible with your NVIDIA driver
before continuing. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Step 3: Point the project to your local model

Set `MODEL_DIR` to the folder containing your model's `config.json`. Do not
leave the example path unchanged:

```bash
export MODEL_DIR="$HOME/models/Llama-3.2-3B-Instruct"
export LLM_MODEL_PATH="$MODEL_DIR"
export AUXILIARY_MODEL_DEVICE=cuda
```

Check that the folder is correct:

```bash
test -f "$LLM_MODEL_PATH/config.json" && echo "Model folder looks OK"
```

If your model has an `FP16` subfolder, point `MODEL_DIR` to that subfolder
instead. The model path must be available to the Python process.

## Step 4: Check the prepared input files

For the saved 200-example comparison, run:

```bash
test -f dataset/test.json && echo "Dataset found"
test -f test_alignment_evidence_1_200_fixed.json && echo "Alignment found"
test -f method/result/20261104_213400_650571_benchmark/hiss/log.jsonl \
  && echo "HiSS predictions found"
```

If all three files are found, continue to Step 5. Otherwise, follow Step 4A.

### Step 4A: Generate missing alignment or HiSS files

Skip this section if you already have the matching files.

To create an alignment file for exactly the first 200 test examples, train an
alignment model and then predict the alignment. This can take a while:

```bash
MAX_EVIDENCE_COUNT=1 \
PREDICT_EVIDENCE_COUNT=1 \
PREDICT_START_INDEX=0 \
PREDICT_END_INDEX=200 \
ALIGNMENT_MODEL_DIR=sentence_alignment/results-local \
ALIGNMENT_OUTPUT=test_alignment_evidence_1_200_local.json \
bash run_alignment.sh
```

This trains the alignment model on the training data before predicting the
first 200 test examples. To match the saved experiment exactly, use its
matching alignment checkpoint and alignment file rather than training a new
checkpoint.

To create HiSS predictions for the first 200 test examples, run:

```bash
FACT_CHECK_START=0 \
FACT_CHECK_LIMIT=200 \
FACT_CHECK_RESULT_DIR=method/result/local_hiss_200 \
LLM_MODEL_PATH="$LLM_MODEL_PATH" \
python -m method.claim_verification_hiss \
  --datapath dataset/test.json \
  --method hiss
```

The HiSS log is written under a newly timestamped folder inside
`method/result/local_hiss_200/`. Find it with:

```bash
find method/result/local_hiss_200 -type f -name log.jsonl
```

Use the printed path in Step 5 as the `--hiss-log` value. Also use the
alignment file created above as `--alignment-file`. The data and both files
must refer to the same 200 examples in the same order.

> **Important:** Generating new alignment or HiSS files is not an exact
> reproduction of the saved experiment. The saved metrics used particular
> existing files, model checkpoint, and model version.

## Step 5: Run ORIGINAL and assumption-aware TRACER

This command runs two configurations, sequentially, on examples 0 through 199.
It should not be run with an existing output folder name:

```bash
python -m method.compare_tracer_filtering \
  --comparison-mode evidence-filter \
  --dataset dataset/test.json \
  --alignment-file test_alignment_evidence_1_200_fixed.json \
  --hiss-log method/result/20261104_213400_650571_benchmark/hiss/log.jsonl \
  --model "$LLM_MODEL_PATH" \
  --start-index 0 \
  --end-index 200 \
  --output-dir method/result/my_assumption_aware_200
```

The end index is **not included**, so `0` to `200` means 200 examples.
To run just 100 examples, use a new output folder and change `--end-index 200`
to `--end-index 100`. Do not set `--end-index` above 200 unless you intend to
run more examples and have prepared matching alignment and HiSS files.

The runner checks that the input IDs, order, and claim text agree. If the check
fails, stop and use alignment/HiSS files made for the same dataset range.

## Step 6: Read the results

When the command finishes, open:

```text
method/result/my_assumption_aware_200/experiment_comparison.json
```

Other useful files in that folder:

- `improved_tracer_hiss_predictions.csv` — example-by-example predictions.
- `original_tracer_hiss_results.json` and `improved_tracer_hiss_results.json` —
  metrics for each configuration.
- `original_raw_log.jsonl` and `improved_raw_log.jsonl` — detailed model
  outputs.

The supplied comparison reports 52.00% accuracy and 33.21% Macro-F1 for
ORIGINAL, and 51.00% accuracy and 32.50% Macro-F1 for IMPROVED. Its
Half-True F1 is 37.58% and 36.60%, respectively. Those are reference results,
not a promise that a locally rerun experiment will be identical.

## Common problems

- **`No config.json found`:** `LLM_MODEL_PATH` points to the wrong folder.
- **CUDA is unavailable:** activate the correct virtual environment and install
  a CUDA-enabled PyTorch build. This project's language-model helper refuses
  to run the fact-checking model on CPU.
- **CUDA out of memory:** close other GPU jobs or use a smaller compatible
  instruction-tuned model. Changing the model means the experiment is no longer
  directly comparable to the supplied result.
- **Alignment or HiSS ID mismatch:** those inputs were made for a different
  dataset range, order, or split. Generate or obtain matching inputs; do not
  ignore the validation error.
- **The run seems slow:** model loading happens separately for each comparison
  arm. The supplied run took about 8 minutes total, but your hardware and model
  loading time may differ. The command above is limited to 200 examples.

For the separate Decision Consistency Guard workflow, see
[`RUN_README.md`](RUN_README.md).
