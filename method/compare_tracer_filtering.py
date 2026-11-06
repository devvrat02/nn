import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from method.run_fact_check_benchmark import _metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/mnt/SHARED-AREA/Llama-series/Llama-3.2-3B-Instruct/FP16"
LABELS = ("false", "half-true", "true")


def _read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _evaluate(records, predictions, elapsed_seconds, config):
    metrics = _metrics(records, predictions)
    y_true = [record["veracity"].strip().lower() for record in records]
    y_pred = [item["pred"].strip().lower() for item in predictions]
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    result = {
        "configuration": config,
        "metrics": {
            **metrics,
            "ground_truth_distribution": dict(Counter(y_true)),
            "predicted_class_distribution": dict(Counter(y_pred)),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=list(LABELS)
            ).tolist(),
        },
        "wall_time_seconds": elapsed_seconds,
        "time_includes_model_loading": True,
        "predictions": [
            {
                "example_id": prediction["example_id"],
                "gold": record["veracity"].strip().lower(),
                "pred": prediction["pred"].strip().lower(),
                "correct": (
                    record["veracity"].strip().lower()
                    == prediction["pred"].strip().lower()
                ),
            }
            for record, prediction in zip(records, predictions)
        ],
    }
    guard_predictions = [
        item for item in predictions if item.get("guard_action") is not None
    ]
    if guard_predictions:
        action_counts = Counter(item["guard_action"] for item in guard_predictions)
        changed = [
            item for item in guard_predictions
            if item.get("prediction_changed_by_guard") is True
        ]
        result["decision_guard"] = {
            "action_counts": {
                action: action_counts.get(action, 0)
                for action in ("KEEP", "RECONSIDER", "ESCALATE")
            },
            "predictions_changed": len(changed),
            "correct_changed_to_incorrect": sum(
                item["original_prediction"] == item["veracity"]
                and item["final_prediction"] != item["veracity"]
                for item in guard_predictions
            ),
            "incorrect_corrected": sum(
                item["original_prediction"] != item["veracity"]
                and item["final_prediction"] == item["veracity"]
                for item in guard_predictions
            ),
            "analysis_parse_valid": sum(
                item.get("guard_analysis_parse_valid") is True
                for item in guard_predictions
            ),
            "analysis_parse_invalid": sum(
                item.get("guard_analysis_parse_valid") is not True
                for item in guard_predictions
            ),
        }
    return result


def _run_experiment(
    name,
    evidence_filter,
    records,
    aligned_records,
    literal_predictions,
    model_path,
    output_dir,
    classifier_prompt="original",
    decision_consistency_guard=False,
):
    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    result_root = experiment_dir / "tracer"
    result_root.mkdir()

    with tempfile.TemporaryDirectory(prefix="tracer-filter-comparison-") as temp:
        temp_dir = Path(temp)
        data_path = temp_dir / "aligned_test.json"
        literal_path = temp_dir / "hiss_predictions.jsonl"
        data_path.write_text(
            json.dumps(aligned_records, ensure_ascii=False), encoding="utf-8"
        )
        literal_path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False) + "\n"
                for item in literal_predictions
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["LLM_MODEL_PATH"] = str(model_path)
        env["REASSESS_RESULT_DIR"] = str(result_root)
        env.setdefault("AUXILIARY_MODEL_DEVICE", "cuda")
        command = [
            sys.executable,
            "-m",
            "method.reassessment",
            "--datafile",
            str(data_path),
            "--literal",
            str(literal_path),
            "--intent_model",
            str(model_path),
            "--evidence-filter",
            evidence_filter,
            "--classifier-prompt",
            classifier_prompt,
        ]
        if decision_consistency_guard:
            command.append("--decision-consistency-guard")
        print(f"\nRunning {name}: {' '.join(command)}", flush=True)
        started = time.perf_counter()
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        elapsed_seconds = time.perf_counter() - started

    logs = list(result_root.glob("*/log.jsonl"))
    hidden_info = list(result_root.glob("*/hidden_info.jsonl"))
    if len(logs) != 1 or len(hidden_info) != 1:
        raise RuntimeError(
            f"{name} expected one log and hidden_info output; found "
            f"{len(logs)} logs and {len(hidden_info)} hidden_info files."
        )
    predictions = _read_jsonl(logs[0])
    expected_ids = [record["example_id"] for record in records]
    if [item["example_id"] for item in predictions] != expected_ids:
        raise RuntimeError(f"{name} output IDs do not match selected records.")

    report = _evaluate(
        records,
        predictions,
        elapsed_seconds,
        {
            "evidence_filter": evidence_filter,
            "classifier_prompt": classifier_prompt,
            "decision_consistency_guard": decision_consistency_guard,
            "prompt_configuration": (
                "decision-consistency-guard-v1"
                if decision_consistency_guard
                else classifier_prompt
            ),
            "alignment_group_size": 1,
            "examples": len(records),
            "model": str(model_path),
        },
    )
    if classifier_prompt == "evidence-consistency":
        valid_responses = sum(
            item.get("classifier_parse_valid") is True
            for item in predictions
        )
        report["classification_response_status"] = {
            "valid": valid_responses,
            "invalid_or_fallback": len(predictions) - valid_responses,
        }
    if decision_consistency_guard:
        guard_metrics = report.get("decision_guard", {})
        if sum(guard_metrics.get("action_counts", {}).values()) != len(records):
            raise RuntimeError(
                f"{name} did not record one guard action per selected example."
            )
    report["hidden_info_file"] = str(hidden_info[0].relative_to(experiment_dir))
    return report, predictions, logs[0], hidden_info[0]


def main():
    parser = argparse.ArgumentParser(
        description="Compare TRACER configurations on one fixed subset."
    )
    parser.add_argument(
        "--comparison-mode",
        choices=(
            "evidence-filter",
            "classifier",
            "evidence-consistency",
            "decision-guard",
        ),
        default="evidence-filter",
        help=(
            "Compare evidence filtering, classifier prompts, evidence consistency, "
            "or the independent decision-consistency guard."
        ),
    )
    parser.add_argument("--dataset", default="dataset/test.json")
    parser.add_argument(
        "--alignment-file", default="test_alignment_evidence_1_200_fixed.json"
    )
    parser.add_argument(
        "--hiss-log",
        default="method/result/20261104_213400_650571_benchmark/hiss/log.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=200)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    def resolve_input(value):
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()

    dataset_path = resolve_input(args.dataset)
    alignment_path = resolve_input(args.alignment_file)
    hiss_path = resolve_input(args.hiss_log)

    if args.start_index < 0 or args.end_index <= args.start_index:
        raise ValueError("Require 0 <= start-index < end-index.")
    if not Path(args.model).expanduser().is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model}")
    model_path = Path(args.model).expanduser().resolve()
    records = json.loads(dataset_path.read_text(encoding="utf-8"))[
        args.start_index : args.end_index
    ]
    aligned_records = json.loads(alignment_path.read_text(encoding="utf-8"))
    hiss_predictions = _read_jsonl(hiss_path)
    expected_ids = [record["example_id"] for record in records]
    if not records or len(records) != args.end_index - args.start_index:
        raise ValueError("Dataset does not contain the full requested subset.")
    for label, rows in (
        ("alignment", aligned_records),
        ("HiSS", hiss_predictions),
    ):
        if [row["example_id"] for row in rows] != expected_ids:
            raise ValueError(f"{label} IDs do not match the selected dataset range.")
    for index, (record, aligned) in enumerate(zip(records, aligned_records)):
        if record["claim"] != aligned.get("claim"):
            raise ValueError(f"Alignment claim mismatch at selected row {index}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    default_suffix = (
        {
            "classifier": "classifier_comparison",
            "evidence-consistency": "evidence_consistency_comparison",
            "decision-guard": "decision_consistency_guard_comparison",
            "evidence-filter": "assumption_aware_comparison",
        }[args.comparison_mode]
    )
    output_dir = Path(
        args.output_dir
        or f"method/result/{timestamp}_{default_suffix}"
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    reports = {}
    prediction_rows = {}
    if args.comparison_mode == "classifier":
        configurations = (
            ("original", "original", "original", False),
            ("structured", "original", "structured", False),
        )
        comparison_name = "structured_minus_original"
        prediction_filename = "structured_tracer_hiss_predictions.csv"
        alternative_name = "structured"
    elif args.comparison_mode == "evidence-consistency":
        configurations = (
            ("original", "original", "original", False),
            ("consistency", "original", "evidence-consistency", False),
        )
        comparison_name = "consistency_minus_original"
        prediction_filename = "evidence_consistency_tracer_hiss_predictions.csv"
        alternative_name = "consistency"
    elif args.comparison_mode == "decision-guard":
        configurations = (
            ("original", "original", "original", False),
            ("extension", "original", "evidence-consistency", False),
            ("guard", "original", "original", True),
        )
        comparison_name = "decision_guard_comparison"
        prediction_filename = None
        alternative_name = None
    else:
        configurations = (
            ("original", "original", "original", False),
            ("improved", "assumption-aware", "original", False),
        )
        comparison_name = "improved_minus_original"
        prediction_filename = "improved_tracer_hiss_predictions.csv"
        alternative_name = "improved"
    for name, evidence_filter, classifier_prompt, decision_guard in configurations:
        report, predictions, log_path, hidden_path = _run_experiment(
            name,
            evidence_filter,
            records,
            aligned_records,
            hiss_predictions,
            model_path,
            output_dir,
            classifier_prompt=classifier_prompt,
            decision_consistency_guard=decision_guard,
        )
        reports[name] = report
        prediction_rows[name] = predictions
        results_name = f"{name}_tracer_hiss_results.json"
        (output_dir / results_name).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / f"{name}_hidden_info.json").write_text(
            hidden_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (output_dir / f"{name}_raw_log.jsonl").write_text(
            log_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    if args.comparison_mode == "decision-guard":
        guard_predictions = prediction_rows["guard"]
        with (output_dir / "three_configuration_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            fields = (
                "example_id", "ground_truth", "original_prediction",
                "extension_prediction", "guard_original_prediction",
                "guard_final_prediction", "guard_action",
                "consistency_check", "consistency_reason",
            )
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for record, original, extension, guard in zip(
                records,
                prediction_rows["original"],
                prediction_rows["extension"],
                guard_predictions,
            ):
                writer.writerow({
                    "example_id": record["example_id"],
                    "ground_truth": record["veracity"],
                    "original_prediction": original["pred"],
                    "extension_prediction": extension["pred"],
                    "guard_original_prediction": guard["original_prediction"],
                    "guard_final_prediction": guard["final_prediction"],
                    "guard_action": guard["guard_action"],
                    "consistency_check": guard["consistency_check"],
                    "consistency_reason": guard["consistency_reason"],
                })
        with (output_dir / "decision_consistency_guard_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            fields = (
                "example_id", "ground_truth", "original_prediction",
                "central_claim_status", "supported_components",
                "contradicted_components", "insufficient_components",
                "consistency_check", "consistency_reason", "guard_action",
                "final_prediction", "prediction_changed_by_guard",
                "relevant_evidence", "evidence_statuses", "model_name",
                "prompt_configuration", "guard_analysis_parse_valid",
                "guard_analysis_response", "guard_correction_response",
            )
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            for item in guard_predictions:
                writer.writerow({
                    field: (
                        json.dumps(item.get(field), ensure_ascii=False)
                        if field in {
                            "supported_components", "contradicted_components",
                            "insufficient_components", "relevant_evidence",
                            "evidence_statuses",
                        }
                        else item.get(field)
                    )
                    for field in fields
                })
    else:
        with (output_dir / prediction_filename).open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "example_id",
                    "gold",
                    "original_pred",
                    f"{alternative_name}_pred",
                ),
            )
            writer.writeheader()
            for record, original, alternative in zip(
                records, prediction_rows["original"], prediction_rows[alternative_name]
            ):
                writer.writerow({
                    "example_id": record["example_id"],
                    "gold": record["veracity"],
                    "original_pred": original["pred"],
                    f"{alternative_name}_pred": alternative["pred"],
                })

    differences = {}
    comparison_metrics = {
        "accuracy": lambda m: m["accuracy"],
        "macro_f1": lambda m: m["macro_f1"],
    }
    for label, prefix in (
        ("false", "false"),
        ("half-true", "half_true"),
        ("true", "true"),
    ):
        for metric in ("precision", "recall", "f1-score"):
            output_name = f"{prefix}_{'f1' if metric == 'f1-score' else metric}"
            comparison_metrics[output_name] = (
                lambda m, label=label, metric=metric:
                    m["classification_report"][label][metric]
            )
    comparison_arms = (
        ("extension", "guard")
        if args.comparison_mode == "decision-guard"
        else (alternative_name,)
    )
    for arm in comparison_arms:
        arm_differences = {}
        for metric, get_value in comparison_metrics.items():
            original_value = get_value(reports["original"]["metrics"])
            improved_value = get_value(reports[arm]["metrics"])
            arm_differences[metric] = {
                "difference_percentage_points": 100 * (
                    improved_value - original_value
                ),
                "absolute_difference_percentage_points": 100 * abs(
                    improved_value - original_value
                ),
            }
        differences[f"{arm}_minus_original"] = arm_differences
    if args.comparison_mode != "decision-guard":
        differences = differences[f"{alternative_name}_minus_original"]
    comparison = {
        "comparison_mode": args.comparison_mode,
        "dataset": str(dataset_path),
        "start_index": args.start_index,
        "end_index_exclusive": args.end_index,
        "examples": len(records),
        "model": str(model_path),
        "alignment_group_size": 1,
        "experiments": reports,
        comparison_name: differences,
    }
    comparison_path = output_dir / "experiment_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nComparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
