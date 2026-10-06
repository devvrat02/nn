> Historical validation of the earlier Windows/Qwen setup. The current REPACSS
> model is `meta-llama/Meta-Llama-3-8B`; see [RUN_LLAMA3.md](RUN_LLAMA3.md).
> These earlier real-model checks do not validate the new model.

# Local setup validation — October 3, 2026

Use `RUN_LOCAL.md` for the ordered manual commands. This file records what was
actually executed during setup, separately from future benchmark work.

## Environment

- Windows; Python 3.14.0 in the parent `.venv`.
- NVIDIA GeForce RTX 5080 Laptop GPU, about 16 GB VRAM.
- PyTorch 2.12.0.dev20260408+cu128; Transformers 4.57.6.
- CUDA matrix multiplication in BF16 passed.
- All four models downloaded into `models/`; exact revisions recorded in
  `models/model_revisions.json`.
- No paid API requests were sent. One legacy helper initially raised an API
  authentication error before a request could be sent; it was corrected to use
  the local dispatcher and regression-tested before rerunning successfully.

## Completed checks

| Check | Result |
|---|---|
| Existing regression suite | 8 passed |
| Local backend, resume, comparison safeguards | 7 passed |
| Original offline integration smoke test | Passed |
| Dependency consistency and Python compilation | Passed |
| Real Qwen generation | Returned `LOCAL_MODEL_OK` |
| Real RoBERTa-large training | Two optimizer steps completed with BF16, gradient checkpointing, accumulation eight |
| Real alignment checkpoint reload/inference | Completed on five development claims |
| Real local HiSS verification | Completed on the same five claims |
| Real local intent/questions/causal reassessment | Completed; three claims entered mining |
| Real MiniLM/DeBERTa and final Qwen reassessment | Passed separate synthetic component check |
| Verification/reassessment resume | Completed IDs skipped; no duplicate records |
| Repeating completed training-check command | Reused matching checkpoint |
| Comparison report generation | Completed with explicit subset warning |
| All 2,000 test literal-prompt lengths | All fit; maximum 4,430 input tokens under configured 16,384 context |

The five development IDs were 12006, 12007, 11994, 12030, and 12019. They were
selected by alternating reference classes for branch coverage, not randomly sampled
for estimating accuracy. Local HiSS got four of five correct. Reassessment retained
all labels because no assumptions survived Qwen's causal filter on the three
eligible claims. This is the measured outcome, not evidence for an accuracy gain.

The synthetic component check explicitly supplied an assumption about permanent
jobs and contradictory evidence about temporary jobs. The real NLI ranker selected
the contradictory evidence, and real Qwen reassessment returned `half-true`. This
checks downstream execution independently; it is not included in dataset metrics.

## Artifacts

- `outputs/alignment-check/`: real RoBERTa architecture trained for only two steps.
- `outputs/local-check/input.json`: five development records.
- `outputs/local-check/alignment.json`: learned alignment predictions.
- `outputs/local-check/literal/log.jsonl`: real local baseline outputs.
- `outputs/local-check/reassessment/log.jsonl`: real local reassessment outputs.
- `outputs/local-check/reassessment/hidden_info.jsonl`: generated intermediate analyses.
- `outputs/local-check/components.json`: explicitly synthetic component check.
- `outputs/local-check/comparison/comparison.md`: demonstration comparison report.
- `outputs/local_environment.json`: environment check.

## Not executed

Full five-epoch alignment training, the complete 2,000-claim test run, local intent
fine-tuning, and reproduction of the published paper scores were not performed.
The two-step checkpoint is only an execution check. Start at step 5 of `RUN_LOCAL.md`
for full training, then follow steps 7–10 for the final local experiment.

These successful checks establish that the configured paths work on this machine.
They cannot guarantee that every future LLM response is correctly formatted, that
hardware never runs out of memory, or that local accuracy matches the paper.

## CoT extension validation — October 4, 2026

Added a separate zero-shot CoT baseline and reuse of TRACER on its own predictions.
Thirty regression tests passed, including prevention of mixed HiSS/CoT resume,
exclusion of gold fields from prompts, and four-method comparison with separate
gains. Real local CoT and CoT + TRACER completed on the same five development
records under `outputs/local-check/cot-literal` and `cot-reassessment`. These
remain execution checks with a two-step alignment checkpoint, not benchmark scores.
Section 11 of `RUN_LOCAL.md` contains the full-test commands. The full CoT
benchmark was not launched during this setup turn.
