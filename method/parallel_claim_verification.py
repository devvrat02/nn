import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B"
)


def _read_worker_log(path):
    log_files = list(path.glob("*/log.jsonl"))
    if len(log_files) != 1:
        raise RuntimeError(
            f"Expected one worker log under {path}, found {len(log_files)}."
        )
    return log_files[0]


def main():
    dataset_path = Path(os.environ.get("DATASET", "dataset/test.json"))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    with dataset_path.open() as dataset_file:
        records = json.load(dataset_file)
    if not isinstance(records, list):
        raise ValueError(f"Dataset must contain a JSON list: {dataset_path}")

    start_index = int(os.environ.get("FACT_CHECK_START", "0"))
    end_value = os.environ.get("FACT_CHECK_END")
    limit = int(os.environ.get("FACT_CHECK_LIMIT", "0"))
    if start_index < 0:
        raise ValueError("FACT_CHECK_START must be zero or a positive integer.")
    if end_value and limit:
        raise ValueError("Set only one of FACT_CHECK_END or FACT_CHECK_LIMIT.")
    if end_value:
        end_index = int(end_value)
    elif limit:
        if limit < 1:
            raise ValueError("FACT_CHECK_LIMIT must be a positive integer.")
        end_index = start_index + limit
    else:
        end_index = len(records)
    if end_index < start_index:
        raise ValueError("FACT_CHECK_END must be greater than or equal to FACT_CHECK_START.")
    records = records[start_index:end_index]
    if not records:
        raise ValueError(
            f"No records in {dataset_path} range [{start_index}, {end_index})."
        )

    requested_workers = int(os.environ.get("FACT_CHECK_WORKERS", "10"))
    if requested_workers < 1:
        raise ValueError("FACT_CHECK_WORKERS must be at least 1.")
    worker_count = min(requested_workers, len(records))

    method_name = os.environ.get("FACT_CHECK_METHOD", "hiss").lower()
    if method_name not in {"hiss", "cot"}:
        raise ValueError("FACT_CHECK_METHOD must be 'hiss' or 'cot'.")

    model_path = os.environ.get("LLM_MODEL_PATH", DEFAULT_MODEL)
    model_path = Path(model_path).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not (model_path / "config.json").is_file() and (
        model_path / "FP16" / "config.json"
    ).is_file():
        model_path = model_path / "FP16"
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"No config.json found under LLM_MODEL_PATH={model_path}"
        )

    devices = [
        item.strip()
        for item in os.environ.get(
            "CUDA_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
        ).split(",")
        if item.strip()
    ]
    if not devices:
        raise ValueError("CUDA_DEVICES must list at least one visible GPU.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(
        os.environ.get("FACT_CHECK_OUTPUT_DIR", f"method/result/{timestamp}_parallel")
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    work_dir = output_dir / ".workers"
    work_dir.mkdir()

    chunk_size = math.ceil(len(records) / worker_count)
    workers = []
    started = time.monotonic()
    try:
        for worker_index in range(worker_count):
            start = worker_index * chunk_size
            chunk = records[start : start + chunk_size]
            if not chunk:
                continue

            worker_dir = work_dir / f"worker-{worker_index:02d}"
            worker_dir.mkdir()
            input_path = worker_dir / "input.json"
            input_path.write_text(json.dumps(chunk), encoding="utf-8")
            output_root = worker_dir / "result"
            worker_log = worker_dir / "worker.log"
            log_handle = worker_log.open("w", encoding="utf-8")

            env = os.environ.copy()
            env["LLM_MODEL_PATH"] = str(model_path)
            env["CUDA_VISIBLE_DEVICES"] = devices[worker_index % len(devices)]
            env["FACT_CHECK_RESULT_DIR"] = str(output_root)
            env["FACT_CHECK_START"] = "0"
            env.pop("FACT_CHECK_END", None)
            env.pop("FACT_CHECK_LIMIT", None)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")

            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "method.claim_verification_hiss",
                    "--datapath",
                    str(input_path),
                    "--method",
                    method_name,
                ],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            workers.append(
                {
                    "index": worker_index,
                    "start": start,
                    "count": len(chunk),
                    "process": process,
                    "log_handle": log_handle,
                    "worker_log": worker_log,
                    "output_root": output_root,
                }
            )
            print(
                f"Started worker {worker_index + 1}/{worker_count}: "
                f"{len(chunk)} {method_name.upper()} records on GPU "
                f"{env['CUDA_VISIBLE_DEVICES']}",
                flush=True,
            )

        pending = set(range(len(workers)))
        while pending:
            for position in list(pending):
                worker = workers[position]
                code = worker["process"].poll()
                if code is None:
                    continue
                pending.remove(position)
                worker["log_handle"].close()
                if code != 0:
                    tail = worker["worker_log"].read_text(
                        encoding="utf-8", errors="replace"
                    )[-4000:]
                    raise RuntimeError(
                        f"Worker {worker['index']} failed with exit code {code}. "
                        f"See {worker['worker_log']}.\n{tail}"
                    )
                worker["result_log"] = _read_worker_log(worker["output_root"])
                completed = sum(
                    item["count"]
                    for item in workers
                    if "result_log" in item
                )
                print(
                    f"Worker {worker['index'] + 1}/{worker_count} finished "
                    f"({completed}/{len(records)} records complete).",
                    flush=True,
                )
            if pending:
                time.sleep(1)

        merged_log = output_dir / "log.jsonl"
        with merged_log.open("w", encoding="utf-8") as destination:
            for worker in workers:
                with worker["result_log"].open(encoding="utf-8") as source:
                    shutil.copyfileobj(source, destination)

        with merged_log.open(encoding="utf-8") as merged_file:
            result_count = sum(1 for line in merged_file if line.strip())
        if result_count != len(records):
            raise RuntimeError(
                f"Merged {result_count} results for {len(records)} input records; "
                f"see worker outputs under {work_dir}."
            )

        for worker in workers:
            (work_dir / f"worker-{worker['index']:02d}" / "input.json").unlink()
            shutil.rmtree(worker["output_root"])

        print(
            f"Completed {result_count} records in {time.monotonic() - started:.1f}s. "
            f"Combined results: {merged_log}",
            flush=True,
        )
    except Exception:
        for worker in workers:
            if worker["process"].poll() is None:
                worker["process"].terminate()
        for worker in workers:
            if worker["process"].poll() is None:
                worker["process"].wait()
            if not worker["log_handle"].closed:
                worker["log_handle"].close()
        print(f"Partial outputs and worker logs retained in {work_dir}.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
