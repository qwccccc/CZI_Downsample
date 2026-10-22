"""Compare CPU and bounded-VRAM GPU area averaging on complete CZI channels."""

import argparse
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

from benchmark_readers import compare_outputs, write_json
from benchmark_tiles import full_plane_tile
from process_czi import DEFAULT_INPUT, area_weights, configure_logging, process, timed


CASES = ("cpu_2048", "cpu_full", "cpu_full_batched", "gpu_full_batched")
LOG = logging.getLogger("czi_gpu")
CUDA_SOURCE = r'''
extern "C" __global__
void vertical(const unsigned short* input, const int* ptr, const int* indices,
              const double* weights, double* output, int width, int rows) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= (long long)width * rows) return;
    int y = i / width;
    int x = i % width;
    double sum = 0.0;
    for (int k = ptr[y]; k < ptr[y + 1]; ++k)
        sum += weights[k] * (double)input[(long long)indices[k] * width + x];
    output[i] = sum;
}
extern "C" __global__
void horizontal(const double* input, const int* ptr, const int* indices,
                const double* weights, double* output,
                int input_width, int output_width, int rows) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= (long long)output_width * rows) return;
    int y = i / output_width;
    int x = i % output_width;
    double sum = 0.0;
    for (int k = ptr[x]; k < ptr[x + 1]; ++k)
        sum += weights[k] * input[(long long)y * input_width + indices[k]];
    output[i] = sum;
}
'''


class BatchedAreaMean:
    def __init__(self, gpu=False, batch_rows=256, device=0):
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        self.gpu = gpu
        self.batch_rows = batch_rows
        self.stats = dict.fromkeys(("host_to_device", "gpu_kernels", "device_to_host",
                                   "cpu_batch_compute", "cpu_output_conversion"), 0.0)
        self.gpu_peak_pool_mib = 0.0
        self.device_info = None
        if gpu:
            import cupy as cp

            self.cp = cp
            cp.cuda.Device(device).use()
            info = cp.cuda.runtime.getDeviceProperties(device)
            self.device_info = {"index": device, "name": info["name"].decode(),
                                "total_vram_mib": info["totalGlobalMem"] / 1024**2,
                                "cupy_version": cp.__version__}
            self.module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11", "--fmad=false"))
            self.module.compile()
            self.vertical = self.module.get_function("vertical")
            self.horizontal = self.module.get_function("horizontal")

    def _gpu_average(self, source, wx, wy):
        cp = self.cp
        started = perf_counter()
        data = cp.asarray(np.ascontiguousarray(source))
        xp, xi, xw = (cp.asarray(array) for array in (wx.indptr, wx.indices, wx.data))
        yp, yi, yw = (cp.asarray(array) for array in (wy.indptr, wy.indices, wy.data))
        cp.cuda.get_current_stream().synchronize()
        self.stats["host_to_device"] += perf_counter() - started
        started = perf_counter()
        rows, width = wy.shape[0], wx.shape[0]
        intermediate = cp.empty((rows, source.shape[1]), dtype=cp.float64)
        output = cp.empty((rows, width), dtype=cp.float64)
        count = intermediate.size
        self.vertical(((count + 255) // 256,), (256,),
                      (data, yp, yi, yw, intermediate, np.int32(source.shape[1]), np.int32(rows)))
        count = output.size
        self.horizontal(((count + 255) // 256,), (256,),
                        (intermediate, xp, xi, xw, output, np.int32(source.shape[1]),
                         np.int32(width), np.int32(rows)))
        cp.cuda.get_current_stream().synchronize()
        self.stats["gpu_kernels"] += perf_counter() - started
        self.gpu_peak_pool_mib = max(self.gpu_peak_pool_mib,
                                    cp.get_default_memory_pool().total_bytes() / 1024**2)
        started = perf_counter()
        result = cp.asnumpy(output)
        self.stats["device_to_host"] += perf_counter() - started
        return result

    def __call__(self, reader, plane, scene, rect, bounds, ratios, timings):
        rx, ry = ratios
        width, height = math.ceil(rect.w / rx), math.ceil(rect.h / ry)
        if bounds != (0, 0, width, height):
            raise ValueError("Batched full-plane processing requires one whole-plane tile")
        with timed(timings, "read"):
            source = reader.read(roi=(rect.x, rect.y, rect.w, rect.h), plane=plane, scene=scene)
        if source.dtype != np.uint16 or source.shape != (rect.h, rect.w, 1):
            raise ValueError("This GPU experiment supports scalar uint16 channels only")
        result = np.empty((height, width, 1), dtype=np.uint16)
        with timed(timings, "weights"):
            _, _, wx = area_weights(0, width, rx, rect.w)
        for y in range(0, height, self.batch_rows):
            end = min(y + self.batch_rows, height)
            with timed(timings, "weights"):
                top, bottom, wy = area_weights(y, end, ry, rect.h)
            with timed(timings, "resample"):
                band = source[top:bottom, :, 0]
                if self.gpu:
                    averaged = self._gpu_average(band, wx, wy)
                else:
                    started = perf_counter()
                    averaged = (wx @ (wy @ band).T).T
                    self.stats["cpu_batch_compute"] += perf_counter() - started
                started = perf_counter()
                result[y:end, :, 0] = np.clip(np.rint(averaged), 0, 65535).astype(np.uint16)
                self.stats["cpu_output_conversion"] += perf_counter() - started
        return result


def run_one(source, output, case, target=10.0, batch_rows=256, device=0):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = {"case": case, "source": str(source), "target_um": target,
              "batch_output_rows": batch_rows, "status": "running",
              "reader": "pylibCZIrw", "reader_version": importlib.metadata.version("pylibCZIrw")}
    try:
        tile = 2048 if case == "cpu_2048" else full_plane_tile(source, target)
        extra = {}
        processor = None
        setup_started = perf_counter()
        if case.endswith("_batched"):
            processor = BatchedAreaMean(gpu=case.startswith("gpu"), batch_rows=batch_rows, device=device)
            extra["tile_processor"] = processor
        report["compute_setup_seconds"] = perf_counter() - setup_started
        manifest = process(source, output, target, tile, **extra)
        if processor:
            report["batch_detail_seconds"] = processor.stats
            report["gpu_peak_cupy_pool_mib"] = processor.gpu_peak_pool_mib
            report["device"] = processor.device_info
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      wall_seconds=perf_counter() - started)
        write_json(output / "run.json", report)
        LOG.exception("GPU/CPU case %s failed; no fallback used", case)
        raise
    report.update(status="completed", wall_seconds=perf_counter() - started, tile_pixels=tile,
                  peak_host_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  manifest=manifest)
    write_json(output / "run.json", report)
    return report


def run_suite(source, output, target=10.0, batch_rows=256, device=0):
    output.mkdir(parents=True, exist_ok=False)
    summary = {"source": str(source), "target_um": target, "workers": 1,
               "case_order": list(CASES), "batch_output_rows": batch_rows,
               "cache_policy": "Fixed-order serial fresh processes; OS/network caches not flushed",
               "gpu_precision": "float64 separable CSR accumulation; CUDA FMA disabled",
               "runs": []}
    for case in CASES:
        directory = output / case
        command = [sys.executable, str(Path(__file__).resolve()), "--input", str(source),
                   "--output", str(directory), "--case", case, "--target-um", str(target),
                   "--batch-rows", str(batch_rows), "--device", str(device)]
        LOG.info("Starting complete-file case: %s", case)
        started = perf_counter()
        child = subprocess.run(command, check=False)
        path = directory / "run.json"
        report = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
                  else {"status": "failed", "error": "Child did not write run.json"})
        report.update(case=case, exit_code=child.returncode,
                      process_wall_seconds=perf_counter() - started)
        if child.returncode:
            report["status"] = "failed"
            LOG.error("Case %s failed: %s", case, report.get("error"))
        summary["runs"].append(report)
        write_json(output / "summary.json", summary)
    reference = summary["runs"][0]
    if reference["status"] == "completed":
        for run in summary["runs"]:
            if run["status"] == "completed":
                run["comparison"] = compare_outputs(reference, run)
                LOG.info("Case %s pixel-equivalent: %s", run["case"], run["comparison"]["equivalent"])
        eligible = [run for run in summary["runs"] if run.get("comparison", {}).get("equivalent")]
        summary["equivalent_speed_ranking"] = [
            run["case"] for run in sorted(eligible, key=lambda run: run["wall_seconds"])]
    else:
        summary["comparison_error"] = "CPU reference failed"
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT / "sc-C042-149-O4213.czi")
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--case", choices=CASES, help="Run one case in this process")
    parser.add_argument("--target-um", type=float, default=10.0)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.input.is_file() or args.input.suffix.lower() != ".czi":
        parser.error("--input must be an existing CZI file")
    if args.batch_rows < 1 or args.device < 0 or not math.isfinite(args.target_um) or args.target_um <= 0:
        parser.error("Invalid batch rows, device index or target pixel size")
    configure_logging()
    if args.case:
        run_one(args.input.resolve(), args.output.resolve(), args.case,
                args.target_um, args.batch_rows, args.device)
    else:
        summary = run_suite(args.input.resolve(), args.output.resolve(),
                            args.target_um, args.batch_rows, args.device)
        if any(run["status"] != "completed" or not run.get("comparison", {}).get("equivalent")
               for run in summary["runs"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
