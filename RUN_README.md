# Running TRACER on the Repacss server

These steps use a local Hugging Face causal language model. The same model is
used for literal claim verification and reassessment; no OpenAI API key or
fine-tuning job is needed.

## 1. Check the model directory

The server has several checkpoints under `/mnt/SHARED-AREA/Llama-series`. For
the current 3B setup, use `Llama-3.2-3B-Instruct`. The run scripts use this as
their default. The model path below assumes its checkpoint files are nested in
an `FP16` directory, matching the layout you reported for another checkpoint:

```bash
export LLM_MODEL_PATH=/mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16
ls -lah "$LLM_MODEL_PATH"
test -f "$LLM_MODEL_PATH/config.json"
```

If your shell already has `LLM_MODEL_PATH` set to the model's parent directory
(`.../Llama-3.2-3B-Instruct`), either update it with the export above or leave
it as-is: the Python loader and run scripts automatically use its `FP16`
subdirectory when that is where `config.json` is found.

If the 3B checkpoint has a different subdirectory layout, set
`LLM_MODEL_PATH` to the directory that actually contains the model files.
Other available choices can be selected by changing the model directory:

```bash
export LLM_MODEL_PATH=/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B-Instruct/FP16
```

Instruct variants are recommended because the prompts expect chat-style
responses. The selected model must be a Transformers-compatible
`AutoModelForCausalLM` checkpoint with a usable tokenizer. Confirm the
directory layout on the server with `ls "$LLM_MODEL_PATH"` and ensure
`config.json` and tokenizer files are directly inside the selected model
directory. If they are nested further, point the variable to the directory
that contains them.

The local model is loaded with `device_map="auto"` and `torch_dtype="auto"`.
Set `CUDA_VISIBLE_DEVICES` before running if the server requires selecting
specific GPUs. Local LLM mode puts the sentence-embedding and NLI reranker
models on CPU by default to avoid competing with Llama for GPU memory. To use
GPU for those models when there is enough memory, set
`AUXILIARY_MODEL_DEVICE=cuda`.

## 2. Create the Python environment

Use Python 3.9 or newer, then install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run also needs access to the Hugging Face model cache or the internet
to download `FacebookAI/roberta-large`, `all-MiniLM-L6-v2`, and
`cross-encoder/nli-deberta-v3-large`. The Llama checkpoint itself is loaded from `LLM_MODEL_PATH`.

## 3. Verify local Llama inference

From `nn/`, run this before starting the full jobs:

```bash
python -c 'from utils.utils import call_gpt; print(call_gpt("Reply with READY only."))'
```

Confirm that the model responds and that the command exits successfully. If
loading fails, verify that `LLM_MODEL_PATH` points to the actual model
checkpoint, that the server account can read it, and that enough GPU/CPU memory
is available.

## 4. Train alignment and generate aligned evidence

```bash
bash run_alignment.sh
```

This trains the RoBERTa-large alignment model on `dataset/train.json`, selects
the latest saved training checkpoint, and writes `test_alignment.json` using
`dataset/test.json`. Defaults use per-GPU batch size 1 with gradient
accumulation 8 to reduce VRAM use. They can be adjusted, for example:

```bash
TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 TRAIN_EPOCHS=5 \
  bash run_alignment.sh
```

The script honors `TRAIN_DATASET`, `TEST_DATASET`, `ALIGNMENT_OUTPUT`,
`MAX_EVIDENCE_COUNT`, and `PREDICT_EVIDENCE_COUNT` if set. It uses all visible
GPUs unless `CUDA_VISIBLE_DEVICES` is set.

## 5. Run literal claim verification

```bash
bash run_fact_check.sh
```

This reads `dataset/test.json` by default and creates a timestamped
`method/result/<timestamp>/log.jsonl`. To use another input file:

```bash
DATASET=dataset/dev.json bash run_fact_check.sh
```

Record the `log.jsonl` path printed/created for the next step. To locate recent
logs:

```bash
find method/result -mindepth 2 -maxdepth 2 -name log.jsonl -printf '%T@ %p\n' \
  | sort -nr | head
```

## 6. Run TRACER reassessment

Use the alignment output and the literal-verification log from the previous
step:

```bash
LITERAL=method/result/<timestamp>/log.jsonl \
  bash run_reassessment.sh
```

The default alignment input is `test_alignment.json`. To change it:

```bash
DATAFILE=path/to/alignment.json \
LITERAL=method/result/<timestamp>/log.jsonl \
  bash run_reassessment.sh
```

The result is written to a new timestamped directory under `method/result/`.

## Notes

- The scripts default to
  `/mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16`; override
  `LLM_MODEL_PATH` if the server stores the checkpoint elsewhere.
- The sentence-alignment model is still RoBERTa-large. Llama replaces the
  GPT/OpenAI calls in claim verification and reassessment, not this classifier.
- The sentence-embedding and NLI checkpoints are separate from Llama and must
  be available locally or downloadable.
- For a low-memory server, reduce `TRAIN_BATCH_SIZE` and/or use a smaller
  compatible Llama Instruct checkpoint. Training/inference throughput depends
  on the server hardware.
