import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B"
)
LABELS = ("false", "half-true", "true")


def _metrics(records, predictions):
    if len(records) != len(predictions):
        raise RuntimeError(
            f"Expected {len(records)} predictions, got {len(predictions)}."
        )
    y_true = [record["veracity"].strip().lower() for record in records]
    y_pred = [item["pred"].strip().lower() for item in predictions]
    if any(label not in LABELS for label in y_pred):
        raise ValueError(f"Unknown prediction labels: {sorted(set(y_pred) - set(LABELS))}")

    accuracy = sum(true == pred for true, pred in zip(y_true, y_pred)) / len(records)
    class_f1 = {}
    for label in LABELS:
        true_positive = sum(
            true == label and pred == label for true, pred in zip(y_true, y_pred)
        )
        false_positive = sum(
            true != label and pred == label for true, pred in zip(y_true, y_pred)
        )
        false_negative = sum(
            true == label and pred != label for true, pred in zip(y_true, y_pred)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1[label] = (
            2 * true_positive / denominator if denominator else 0.0
        )

    half_true_tp = sum(
        true == "half-true" and pred == "half-true"
        for true, pred in zip(y_true, y_pred)
    )
    half_true_fp = sum(
        true != "half-true" and pred == "half-true"
        for true, pred in zip(y_true, y_pred)
    )
    half_true_fn = sum(
        true == "half-true" and pred != "half-true"
        for true, pred in zip(y_true, y_pred)
    )
    return {
        "examples": len(records),
        "accuracy": accuracy,
        "macro_f1": sum(class_f1.values()) / len(LABELS),
        "half_true_precision": (
            half_true_tp / (half_true_tp + half_true_fp)
            if half_true_tp + half_true_fp
            else 0.0
        ),
        "half_true_recall": (
            half_true_tp / (half_true_tp + half_true_fn)
            if half_true_tp + half_true_fn
            else 0.0
        ),
        "half_true_f1": class_f1["half-true"],
    }


def _read_predictions(path):
    with path.open(encoding="utf-8") as log_file:
        return [json.loads(line) for line in log_file if line.strip()]


def _validate_prefix(records, predictions, method):
    expected_ids = [record["example_id"] for record in records]
    predicted_ids = [item["example_id"] for item in predictions]
    if predicted_ids != expected_ids[: len(predicted_ids)]:
        raise RuntimeError(
            f"{method.upper()} log is not a prefix of the selected benchmark range. "
            f"Check or remove its stale output before resuming."
        )


def _read_hidden_info(path, expected_ids):
    with path.open(encoding="utf-8") as hidden_file:
        hidden_events = json.load(hidden_file)
    if not isinstance(hidden_events, list):
        raise ValueError(f"Expected a JSON list in CHE output: {path}")

    che_by_example = {}
    expected_ids = set(expected_ids)
    for event in hidden_events:
        example_id = event.get("example_id")
        if example_id not in expected_ids:
            continue
        argument = event.get("argument") or {}
        che_keys = {
            key
            for assumption in argument.get("assumption", [])
            for key in assumption.get("refuted_by", [])
        }
        che_by_example[example_id] = sorted(che_keys)
    return che_by_example


def _run_tracer(output_dir, dataset_path, model_path, records, hiss_log):
    tracer_dir = output_dir / "tracer"
    tracer_dir.mkdir(parents=True, exist_ok=True)
    completion_file = tracer_dir / "complete.json"
    expected_ids = [record["example_id"] for record in records]

    if completion_file.exists():
        completion = json.loads(completion_file.read_text(encoding="utf-8"))
        result_log = tracer_dir / completion["log"]
        hidden_info_path = tracer_dir / completion["hidden_info"]
        if result_log.is_file() and hidden_info_path.is_file():
            return _read_predictions(result_log), _read_hidden_info(
                hidden_info_path, expected_ids
            )

    previous_logs = set(tracer_dir.glob("*/log.jsonl"))
    env = os.environ.copy()
    env["REASSESS_RESULT_DIR"] = str(tracer_dir)
    env["LLM_MODEL_PATH"] = str(model_path)
    env.setdefault("AUXILIARY_MODEL_DEVICE", "cuda")
    print(
        "\nRunning TRACER reassessment on HiSS predictions; CHE is counted from "
        "distinct hidden evidence keys in each assumption's refuted_by list.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "method.reassessment",
            "--datafile",
            str(dataset_path),
            "--literal",
            str(hiss_log),
            "--intent_model",
            str(model_path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    new_logs = set(tracer_dir.glob("*/log.jsonl")) - previous_logs
    if len(new_logs) != 1:
        raise RuntimeError(
            f"Expected one new TRACER log under {tracer_dir}, found {len(new_logs)}."
        )
    result_log = new_logs.pop()
    hidden_info_path = result_log.parent / "hidden_info.jsonl"
    if not hidden_info_path.is_file():
        raise FileNotFoundError(
            f"TRACER did not save its CHE details at {hidden_info_path}."
        )

    predictions = _read_predictions(result_log)
    if [item["example_id"] for item in predictions] != expected_ids:
        raise RuntimeError("TRACER output IDs do not match the selected records.")
    che_by_example = _read_hidden_info(hidden_info_path, expected_ids)
    completion = {
        "log": str(result_log.relative_to(tracer_dir)),
        "hidden_info": str(hidden_info_path.relative_to(tracer_dir)),
    }
    completion_file.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    return predictions, che_by_example


def main():
    dataset_path = Path(os.environ.get("DATASET", "dataset/test.json"))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    with dataset_path.open(encoding="utf-8") as dataset_file:
        all_records = json.load(dataset_file)
    if not isinstance(all_records, list):
        raise ValueError(f"Dataset must contain a JSON list: {dataset_path}")

    start_index = int(os.environ.get("FACT_CHECK_START", "0"))
    end_value = os.environ.get("FACT_CHECK_END")
    limit_value = os.environ.get("FACT_CHECK_LIMIT")
    if start_index < 0:
        raise ValueError("FACT_CHECK_START must be zero or a positive integer.")
    if end_value and limit_value:
        raise ValueError("Set only one of FACT_CHECK_END or FACT_CHECK_LIMIT.")
    if end_value:
        end_index = int(end_value)
    elif limit_value:
        limit = int(limit_value)
        if limit < 1:
            raise ValueError("FACT_CHECK_LIMIT must be a positive integer.")
        end_index = start_index + limit
    else:
        end_index = start_index + 200
    if end_index < start_index:
        raise ValueError("FACT_CHECK_END must be greater than or equal to FACT_CHECK_START.")
    records = all_records[start_index:end_index]
    if not records:
        raise ValueError(
            f"No records in {dataset_path} range [{start_index}, {end_index})."
        )

    model_path = Path(os.environ.get("LLM_MODEL_PATH", DEFAULT_MODEL)).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not (model_path / "config.json").is_file() and (
        model_path / "FP16" / "config.json"
    ).is_file():
        model_path /= "FP16"
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"No config.json found under LLM_MODEL_PATH={model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(
        os.environ.get("FACT_CHECK_BENCHMARK_DIR", f"method/result/{timestamp}_benchmark")
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok="FACT_CHECK_BENCHMARK_DIR" in os.environ)

    selected_dataset = output_dir / "selected_dataset.json"
    serialized_records = json.dumps(records, ensure_ascii=False)
    if selected_dataset.exists():
        if json.loads(selected_dataset.read_text(encoding="utf-8")) != records:
            raise RuntimeError(
                f"The selected dataset in {output_dir} does not match this run."
            )
    else:
        selected_dataset.write_text(serialized_records, encoding="utf-8")

    metrics_by_method = {}
    for method in ("hiss", "cot"):
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        canonical_log = method_dir / "log.jsonl"
        if canonical_log.exists():
            predictions = _read_predictions(canonical_log)
        else:
            prior_logs = sorted(
                method_dir.glob("*/log.jsonl"),
                key=lambda path: path.stat().st_mtime,
            )
            latest_predictions = {}
            for prior_log in prior_logs:
                for prediction in _read_predictions(prior_log):
                    latest_predictions[prediction["example_id"]] = prediction
            expected_ids = [record["example_id"] for record in records]
            predictions = [
                latest_predictions[example_id]
                for example_id in expected_ids
                if example_id in latest_predictions
            ]
            _validate_prefix(records, predictions, method)

        _validate_prefix(records, predictions, method)
        if len(predictions) < len(records):
            remaining_records = records[len(predictions) :]
            remaining_dataset = output_dir / f".{method}_remaining.json"
            remaining_dataset.write_text(
                json.dumps(remaining_records, ensure_ascii=False),
                encoding="utf-8",
            )
            previous_logs = set(method_dir.glob("*/log.jsonl"))
            env = os.environ.copy()
            env["FACT_CHECK_METHOD"] = method
            env["FACT_CHECK_RESULT_DIR"] = str(method_dir)
            env["FACT_CHECK_START"] = "0"
            env.pop("FACT_CHECK_END", None)
            env.pop("FACT_CHECK_LIMIT", None)
            env["LLM_MODEL_PATH"] = str(model_path)
            print(
                f"\nRunning {method.upper()} on {len(remaining_records)} remaining "
                f"examples ({len(predictions)}/{len(records)} already complete)...",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "method.claim_verification_hiss",
                    "--datapath",
                    str(remaining_dataset),
                    "--method",
                    method,
                ],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )
            remaining_dataset.unlink()

            new_logs = set(method_dir.glob("*/log.jsonl")) - previous_logs
            if len(new_logs) != 1:
                raise RuntimeError(
                    f"Expected one new {method.upper()} result log under "
                    f"{method_dir}, found {len(new_logs)}."
                )
            new_predictions = _read_predictions(new_logs.pop())
            expected_remaining_ids = [
                record["example_id"] for record in remaining_records
            ]
            if [item["example_id"] for item in new_predictions] != expected_remaining_ids:
                raise RuntimeError(
                    f"{method.upper()} output IDs do not match the remaining "
                    "benchmark records; partial logs were retained."
                )
            predictions.extend(new_predictions)
            with canonical_log.open("w", encoding="utf-8") as log_file:
                for prediction in predictions:
                    log_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        if not canonical_log.exists():
            with canonical_log.open("w", encoding="utf-8") as log_file:
                for prediction in predictions:
                    log_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        if [item["example_id"] for item in predictions] != [
            item["example_id"] for item in records
        ]:
            raise RuntimeError(
                f"{method.upper()} output IDs do not match the selected dataset range."
            )
        metrics_by_method[method] = _metrics(records, predictions)

    tracer_predictions, che_by_example = _run_tracer(
        output_dir,
        selected_dataset,
        model_path,
        records,
        output_dir / "hiss" / "log.jsonl",
    )
    if [item["example_id"] for item in tracer_predictions] != [
        item["example_id"] for item in records
    ]:
        raise RuntimeError("TRACER output IDs do not match the selected dataset range.")
    metrics_by_method["tracer_hiss"] = _metrics(records, tracer_predictions)

    che_counts = {
        str(record["example_id"]): len(che_by_example.get(record["example_id"], []))
        for record in records
    }
    total_che = sum(che_counts.values())
    che_summary = {
        "definition": (
            "Distinct hidden evidence keys in assumption.refuted_by per example; "
            "an example with no generated CHE contributes zero."
        ),
        "total_distinct_che_evidence_items": total_che,
        "examples_with_che": sum(count > 0 for count in che_counts.values()),
        "average_che_per_example": total_che / len(records),
        "per_example": che_counts,
    }
    (output_dir / "che_counts.json").write_text(
        json.dumps(che_summary, indent=2) + "\n", encoding="utf-8"
    )

    selected_dataset.unlink()
    summary = {
        "dataset": str(dataset_path),
        "start_index": start_index,
        "end_index_exclusive": end_index,
        "model": str(model_path),
        "metrics": metrics_by_method,
        "che": che_summary,
    }
    summary_path = output_dir / "benchmark.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nBenchmark summary")
    print("| Method | Examples | Accuracy | Macro-F1 | Half-true P | Half-true R | Half-true F1 | CHE |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method, metrics in metrics_by_method.items():
        che_column = str(total_che) if method == "tracer_hiss" else "—"
        print(
            f"| {method.replace('_', ' + ').upper()} | {metrics['examples']} | "
            f"{metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['half_true_precision']:.4f} | "
            f"{metrics['half_true_recall']:.4f} | "
            f"{metrics['half_true_f1']:.4f} | {che_column} |"
        )
    print(
        f"\nCHE: {total_che} distinct hidden evidence items across "
        f"{che_summary['examples_with_che']} examples."
    )
    print(f"\nDetailed results: {output_dir}")
    print(f"Metrics: {summary_path}")
    print(f"CHE counts: {output_dir / 'che_counts.json'}")


if __name__ == "__main__":
    main()
