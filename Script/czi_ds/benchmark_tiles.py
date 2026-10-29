"""Compare full-file processing with different tile sizes in fresh processes."""

import argparse
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from pylibCZIrw import czi

from benchmark_readers import compare_outputs, write_json
from process_czi import DEFAULT_INPUT, configure_logging, metadata


LOG = logging.getLogger("czi_tiles")


def full_plane_tile(source, target):
    with czi.open_czi(str(source)) as reader:
        scales, _ = metadata(reader)
        if target < max(scales["X"], scales["Y"]):
            raise ValueError("Target must not upsample either XY axis")
        scenes = reader.scenes_bounding_rectangle or {None: reader.total_bounding_rectangle}
        ratios = target / scales["X"], target / scales["Y"]
        return max(max(math.ceil(rect.w / ratios[0]),
                       math.ceil(rect.h / ratios[1]))
                   for rect in scenes.values())


def run_suite(source, output, target=10.0, cases=(512, 1024, 2048, "full")):
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Tile cases must be nonempty and unique")
    if any(case != "full" and (not isinstance(case, int) or case < 1) for case in cases):
        raise ValueError("Tile cases must be positive integers or 'full'")
    full = full_plane_tile(source, target)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"source": str(source), "source_bytes": source.stat().st_size,
               "backend": "pylibCZIrw", "target_um": target, "workers": 1,
               "full_plane_tile": full, "case_order": list(cases),
               "cache_policy": "No cache flushing; fixed-order serial runs in fresh processes",
               "runs": []}
    started = perf_counter()
    script = Path(__file__).with_name("benchmark_readers.py")
    for case in cases:
        tile = full if case == "full" else case
        directory = output / f"tile_{case}"
        command = [sys.executable, str(script), "--input", str(source),
                   "--output", str(directory), "--backend", "pylibCZIrw",
                   "--target-um", str(target), "--tile", str(tile)]
        LOG.info("Starting entire-file case %s: tile=%s, workers=1", case, tile)
        case_started = perf_counter()
        completed = subprocess.run(command, check=False)
        report = directory / "run.json"
        if report.exists():
            result = json.loads(report.read_text(encoding="utf-8"))
        else:
            result = {"status": "failed", "error": "Child did not write run.json"}
        result.update(case=case, tile_pixels=tile, exit_code=completed.returncode,
                      process_wall_seconds=perf_counter() - case_started)
        if completed.returncode != 0:
            result["status"] = "failed"
            LOG.error("Case %s failed: exit=%s error=%s", case, completed.returncode, result.get("error"))
        summary["runs"].append(result)
        write_json(output / "summary.json", summary)
    summary["processing_wall_seconds"] = perf_counter() - started
    reference = summary["runs"][0]
    if reference["status"] == "completed":
        for run in summary["runs"]:
            if run["status"] == "completed":
                run["comparison"] = compare_outputs(reference, run)
                LOG.info("Case %s pixel-equivalent: %s", run["case"], run["comparison"]["equivalent"])
                write_json(output / "summary.json", summary)
        eligible = [run for run in summary["runs"] if run.get("comparison", {}).get("equivalent")]
        summary["equivalent_speed_ranking"] = [
            run["case"] for run in sorted(eligible, key=lambda run: run["wall_seconds"])]
    else:
        summary["comparison_error"] = "First case failed; no equivalence claims possible"
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT / "sc-C042-149-O4213.czi")
    parser.add_argument("--output", type=Path, required=True, help="New benchmark directory")
    parser.add_argument("--target-um", type=float, default=10.0)
    args = parser.parse_args()
    if not args.input.is_file() or args.input.suffix.lower() != ".czi":
        parser.error("--input must be an existing CZI file")
    if not math.isfinite(args.target_um) or args.target_um <= 0:
        parser.error("--target-um must be finite and positive")
    configure_logging()
    summary = run_suite(args.input.resolve(), args.output.resolve(), args.target_um)
    if any(run["status"] != "completed" or not run.get("comparison", {}).get("equivalent")
           for run in summary["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
