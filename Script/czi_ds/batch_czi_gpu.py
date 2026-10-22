"""Run resumable GPU directory jobs from a CSV input_dir/output_dir table."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import fcntl
import logging
import math
import multiprocessing
import subprocess
import sys
from pathlib import Path

from benchmark_readers import write_json
from process_czi import configure_logging
from process_czi_gpu import collect_sources, process_gpu


LOG = logging.getLogger("czi_table")


def load_jobs(table):
    jobs, seen, outputs = [], set(), {}
    with table.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["input_dir", "output_dir"]:
            raise ValueError("CSV header must be exactly: input_dir,output_dir")
        for line, row in enumerate(rows, 2):
            if None in row or any(not row.get(key, "").strip() for key in rows.fieldnames):
                raise ValueError(f"CSV line {line}: expected two nonempty directory paths")
            paths = []
            for key in rows.fieldnames:
                path = Path(row[key].strip()).expanduser()
                paths.append((path if path.is_absolute() else table.parent / path).resolve())
            source_dir, output = paths
            if not source_dir.is_dir():
                raise ValueError(f"CSV line {line}: input directory does not exist: {source_dir}")
            if source_dir == output:
                raise ValueError(f"CSV line {line}: input and output directories must differ")
            for source in collect_sources([source_dir]):
                key = (source, output)
                if key in seen:
                    continue
                collision = (output, source.stem)
                if collision in outputs and outputs[collision] != source:
                    raise ValueError(f"Output collision: {source} and {outputs[collision]} -> {output}")
                outputs[collision] = source
                seen.add(key)
                jobs.append(key)
    if not jobs:
        raise ValueError("CSV contains no jobs")
    return jobs


def run_worker(jobs, target, batch_rows, device):
    results = []
    for source, output in jobs:
        handler = None
        try:
            output.mkdir(parents=True, exist_ok=True)
            log_dir = output / "logs"
            log_dir.mkdir(exist_ok=True)
            handler = logging.FileHandler(log_dir / f"{source.stem}.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(handler)
            with (output / f".{source.stem}.processing.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                reports = process_gpu([source], output, target, batch_rows, device, resume=True)
            results.append({"source": str(source), "output": str(output), "status": "completed",
                            "skipped_channels": sum(item.get("status") == "skipped"
                                                    for item in reports[0]["outputs"])})
        except Exception as error:
            LOG.exception("File failed; continuing with remaining files: %s", source)
            results.append({"source": str(source), "output": str(output), "status": "failed",
                            "error": f"{type(error).__name__}: {error}"})
        finally:
            if handler is not None:
                logging.getLogger().removeHandler(handler)
                handler.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="CSV with input_dir,output_dir")
    parser.add_argument("--report", type=Path, help="JSON batch status report")
    parser.add_argument("--submit", action="store_true", help="Submit using nohup and return immediately")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", type=int, nargs="+", default=None)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--target-um", type=float, default=10.0)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_rows < 1 or not math.isfinite(args.target_um) or args.target_um <= 0:
        parser.error("workers, batch rows and target pixel size must be positive and finite")
    if args.devices is not None and (len(set(args.devices)) != len(args.devices)
                                     or any(index < 0 for index in args.devices)):
        parser.error("devices must be unique nonnegative CUDA indices")
    try:
        jobs = load_jobs(args.jobs.resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    configure_logging()
    if args.submit:
        log_dir = jobs[0][1] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = log_dir / f"batch_{stamp}"
        report = args.report or base.with_suffix(".json")
        command = ["nohup", sys.executable, "-u", str(Path(__file__).resolve()),
                   "--jobs", str(args.jobs.resolve()), "--report", str(report.resolve()),
                   "--workers", str(args.workers), "--batch-rows", str(args.batch_rows),
                   "--target-um", str(args.target_um)]
        if args.devices is not None:
            command.extend(["--devices", *map(str, args.devices)])
        with base.with_suffix(".log").open("ab") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        base.with_suffix(".pid").write_text(f"{child.pid}\n")
        print(f"已提交，PID：{child.pid}\n日志：{base.with_suffix('.log')}\n报告：{report}")
        return
    if args.report is None:
        args.report = jobs[0][1] / "logs" / "batch_report.json"
    import cupy as cp

    count = cp.cuda.runtime.getDeviceCount()
    devices = list(range(count)) if args.devices is None else args.devices
    if not devices or any(index >= count for index in devices):
        parser.error("Selected CUDA GPUs are unavailable")
    active = min(args.workers, len(jobs))
    assignments = [devices[index % len(devices)] for index in range(active)]
    LOG.warning("Concurrent files=%s; GPU assignments=%s. Host and GPU memory multiply per worker.",
                active, assignments)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    summary = {"table": str(args.jobs.resolve()), "total_files": len(jobs),
               "worker_devices": assignments, "results": [], "status": "running"}
    write_json(args.report, summary)
    with ProcessPoolExecutor(max_workers=active, mp_context=multiprocessing.get_context("spawn"),
                             initializer=configure_logging) as executor:
        futures = {executor.submit(run_worker, jobs[index::active], args.target_um,
                                   args.batch_rows, device): jobs[index::active]
                   for index, device in enumerate(assignments)}
        for future in as_completed(futures):
            try:
                summary["results"].extend(future.result())
            except Exception as error:
                LOG.exception("Worker failed")
                summary["results"].extend(
                    {"source": str(source), "output": str(output), "status": "failed",
                     "error": f"Worker failed: {error}"} for source, output in futures[future])
            write_json(args.report, summary)
    failed = sum(item["status"] == "failed" for item in summary["results"])
    summary.update(status="failed" if failed else "completed", failed_files=failed)
    write_json(args.report, summary)
    LOG.info("Batch finished: total=%s failed=%s report=%s", len(jobs), failed, args.report)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
