"""Split scalar uint16 CZI channels and area-downsample XY on CUDA GPUs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
import multiprocessing
from pathlib import Path
from time import perf_counter

from pylibCZIrw import czi

from benchmark_gpu import BatchedAreaMean
from benchmark_readers import write_json
from benchmark_tiles import full_plane_tile
from process_czi import DEFAULT_INPUT, configure_logging, process


LOG = logging.getLogger("czi_gpu")


def validate_inputs(sources, target, batch_rows):
    if not sources:
        raise ValueError("No input CZI files found")
    if not math.isfinite(target) or target <= 0 or batch_rows < 1:
        raise ValueError("Target and batch rows must be positive")
    if len({source.stem for source in sources}) != len(sources):
        raise ValueError("Input filenames must have unique stems to avoid output collisions")


def collect_sources(inputs):
    sources = []
    for path in inputs:
        if path.is_file() and path.suffix.lower() == ".czi":
            sources.append(path.resolve())
        elif path.is_dir():
            sources.extend(sorted(p.resolve() for p in path.iterdir()
                                  if p.is_file() and p.suffix.lower() == ".czi"))
        else:
            raise ValueError(f"Not a CZI file or directory: {path}")
    validate_inputs(sources, 10.0, 1)
    return sources


def process_gpu(sources, output, target=10.0, batch_rows=4096, device=0, resume=False):
    validate_inputs(sources, target, batch_rows)
    if device < 0:
        raise ValueError("Device index must be nonnegative")
    setup_started = perf_counter()
    processor = BatchedAreaMean(gpu=True, batch_rows=batch_rows, device=device)
    setup_seconds = perf_counter() - setup_started
    LOG.info("GPU ready: %s; setup=%.3fs; batch output rows=%s",
             processor.device_info, setup_seconds, batch_rows)
    reports = []
    for index, source in enumerate(sources, 1):
        started = perf_counter()
        LOG.info("Processing file %s/%s: %s", index, len(sources), source)
        with czi.open_czi(str(source)) as reader:
            if any(pixel_type != "Gray16" for pixel_type in reader.pixel_types.values()):
                raise ValueError(f"{source}: GPU pipeline supports scalar uint16 (Gray16) channels only")
        tile = full_plane_tile(source, target)
        processor.stats = dict.fromkeys(processor.stats, 0.0)
        processor.gpu_peak_pool_mib = 0.0
        manifest = process(source, output, target, tile, tile_processor=processor, resume=resume)
        manifest["gpu"] = {
            "device": processor.device_info,
            "batch_output_rows": batch_rows,
            "session_setup_seconds": setup_seconds,
            "file_wall_seconds": perf_counter() - started,
            "timings_seconds": dict(processor.stats),
            "peak_reserved_cupy_pool_mib": processor.gpu_peak_pool_mib,
        }
        write_json(output / f"{source.stem}_manifest.json", manifest)
        LOG.info("GPU file complete: %s; wall=%.3fs; transfer=%.3fs; kernels=%.3fs",
                 source.name, manifest["gpu"]["file_wall_seconds"],
                 processor.stats["host_to_device"] + processor.stats["device_to_host"],
                 processor.stats["gpu_kernels"])
        reports.append(manifest)
    return reports


def run_gpu_batch(sources, output, target=10.0, batch_rows=4096, workers=1,
                  device=None, devices=None):
    validate_inputs(sources, target, batch_rows)
    if workers < 1:
        raise ValueError("workers must be positive")
    if device is not None and devices is not None:
        raise ValueError("Specify device or devices, not both")
    if devices is not None and not devices:
        raise ValueError("devices must not be empty")
    selected = [device] if device is not None else devices
    if selected is not None and (any(index < 0 for index in selected)
                                 or len(set(selected)) != len(selected)):
        raise ValueError("GPU indices must be nonnegative and unique")
    import cupy as cp

    count = cp.cuda.runtime.getDeviceCount()
    if count == 0:
        raise RuntimeError("No CUDA GPUs available; CPU fallback is disabled")
    selected = list(range(count)) if selected is None else list(selected)
    if any(index >= count for index in selected):
        raise ValueError(f"GPU index out of range; {count} CUDA device(s) are visible")
    active = min(workers, len(sources))
    if active != workers:
        LOG.warning("Requested %s workers; using %s because only %s files were supplied",
                    workers, active, len(sources))
    assignments = [selected[index % len(selected)] for index in range(active)]
    LOG.info("GPU batch: files=%s workers=%s worker GPU assignments=%s",
             len(sources), active, assignments)
    if active > len(selected):
        LOG.warning("Multiple workers share GPUs; each process has independent host buffers, "
                    "CUDA context and VRAM pool. Memory usage increases with workers.")
    if active == 1:
        return process_gpu(sources, output, target, batch_rows, selected[0])
    reports = []
    with ProcessPoolExecutor(max_workers=active, mp_context=multiprocessing.get_context("spawn"),
                             initializer=configure_logging) as executor:
        futures = [executor.submit(process_gpu, sources[index::active], output,
                                   target, batch_rows, gpu)
                   for index, gpu in enumerate(assignments)]
        for future in as_completed(futures):
            reports.extend(future.result())
    order = {str(source): index for index, source in enumerate(sources)}
    reports.sort(key=lambda report: order[report["source"]])
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", default=[DEFAULT_INPUT],
                        help="One or more CZI files or directories; directory scanning is nonrecursive")
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT / "results" / "gpu")
    parser.add_argument("--target-um", type=float, default=10.0)
    parser.add_argument("--batch-rows", type=int, default=4096,
                        help="Output rows per GPU batch (default: 4096); smaller uses less VRAM")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent files; workers may share GPUs (default: 1)")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--device", type=int, help="Restrict processing to this single CUDA GPU")
    gpu_group.add_argument("--devices", type=int, nargs="+",
                           help="Allowed CUDA GPUs, e.g. --devices 0 1; default: auto-detect")
    args = parser.parse_args()
    if not math.isfinite(args.target_um) or args.target_um <= 0:
        parser.error("--target-um must be finite and positive")
    if args.batch_rows < 1 or args.workers < 1:
        parser.error("--batch-rows and --workers must be positive")
    if args.device is not None and args.device < 0:
        parser.error("--device must be nonnegative")
    if args.devices is not None and (any(index < 0 for index in args.devices)
                                     or len(set(args.devices)) != len(args.devices)):
        parser.error("--devices must contain unique nonnegative indices")
    try:
        sources = collect_sources(args.input)
    except ValueError as error:
        parser.error(str(error))
    configure_logging()
    run_gpu_batch(sources, args.output.resolve(), args.target_um, args.batch_rows,
                  args.workers, args.device, args.devices)


if __name__ == "__main__":
    main()
