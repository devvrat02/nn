"""Cross-platform TRACER entry point. Smoke mode never calls external services."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def validate_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = set()
    for item in data:
        idx = item["example_id"]
        if idx in ids:
            raise ValueError(f"Duplicate example_id: {idx}")
        ids.add(idx)
        if item["veracity"] not in {"true", "half-true", "false"}:
            raise ValueError(f"Invalid label: {idx}")
        if not isinstance(item["claim"], str) or not item["claim"].strip():
            raise ValueError(f"Empty claim: {idx}")
        if len(item["evidence"]) != len(item["annotation"]):
            raise ValueError(f"Misaligned evidence labels: {idx}")
        if any(a not in {0, 1} for a in item["annotation"]):
            raise ValueError(f"Invalid alignment label: {idx}")
    return data


def prepare(output):
    from intent_generation.intent_finetune import finetune_gpt_mini
    stats = {}
    split_ids = {}
    for split in ("train", "dev", "test"):
        data = validate_data(ROOT / "dataset" / f"{split}.json")
        split_ids[split] = {x["example_id"] for x in data}
        stats[split] = {"claims": len(data), "labels": dict(Counter(x["veracity"] for x in data)),
                        "evidence_sentences": sum(len(x["evidence"]) for x in data),
                        "valid_intents": sum(x["intent_valid"] != 0 for x in data)}
    stats["id_overlap"] = {f"{a}_{b}": len(split_ids[a] & split_ids[b])
                           for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    finetune_gpt_mini(ROOT / "dataset/train.json", str(output / "intent_train"))
    finetune_gpt_mini(ROOT / "dataset/dev.json", str(output / "intent_dev"))
    print(json.dumps(stats, indent=2))


def research(args):
    # Check all external prerequisites before model loading or paid requests.
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY locally. Use --mode smoke for an offline integration test.")
    if not args.model_dir or not args.intent_model:
        raise ValueError("Research mode requires --model-dir and --intent-model (fine-tuned ID, or explicit base-model experiment).")
    from sentence_alignment.predict_model_script import predict, find_model_file
    from method.claim_verification_hiss import main as verify
    from method.reassessment import main as reassess
    from method.eval import evaluate
    find_model_file(args.model_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = validate_data(args.data)
    if args.limit:
        data = data[:args.limit]
    selected = output / "input.json"
    selected.write_text(json.dumps(data), encoding="utf-8")
    aligned = output / "alignment.json"
    predict(str(args.model_dir), str(selected), str(aligned))
    verify(SimpleNamespace(datapath=str(selected), output_dir=str(output / "literal"), model=args.verifier_model))
    reassess(SimpleNamespace(datafile=str(aligned), literal=str(output / "literal/log.jsonl"),
                             intent_model=args.intent_model, output_dir=str(output / "reassessment")))
    metrics = {stage: evaluate(output / stage / "log.jsonl") for stage in ("literal", "reassessment")}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "smoke", "research"), default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/test.json")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--intent-model")
    parser.add_argument("--verifier-model", default="gpt-3.5-turbo")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.output_dir = args.output_dir or ROOT / "outputs" / args.mode
    if args.mode == "prepare":
        prepare(args.output_dir)
    elif args.mode == "smoke":
        from tests.offline_smoke import run
        run(args.output_dir)
    else:
        try:
            research(args)
        except ValueError as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
