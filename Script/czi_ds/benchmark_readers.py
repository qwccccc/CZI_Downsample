"""Run equivalent full-file CZI reader benchmarks in isolated serial processes."""

import argparse
from contextlib import ExitStack, contextmanager
from functools import partial
import importlib.metadata
import itertools
import json
import logging
import math
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
from pylibCZIrw import czi

from process_czi import DEFAULT_INPUT, configure_logging, metadata, process


BACKENDS = ("pylibCZIrw", "aicspylibczi", "czifile", "bioio-czi")
LOG = logging.getLogger("czi_benchmark")


def write_json(path, data):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ReaderAdapter:
    def __init__(self, source, backend, canonical, stack):
        self.backend = backend
        self.raw_metadata = canonical.raw_metadata
        self.total_bounding_box = canonical.total_bounding_box
        self.total_bounding_rectangle = canonical.total_bounding_rectangle
        self.scenes_bounding_rectangle = canonical.scenes_bounding_rectangle
        self.pixel_types = canonical.pixel_types
        self.native_bounds = {}
        self.arrays = {}
        self.empty_regions = 0
        if backend == "pylibCZIrw":
            self.reader = canonical
        elif backend == "czifile":
            import czifile

            self.reader = stack.enter_context(czifile.CziFile(source))
            for scene, image in self.reader.scenes.items():
                dims = dict(zip(image.dims, image.start))
                self.native_bounds[scene] = (dims["X"], dims["Y"], image.sizes["X"], image.sizes["Y"])
        else:
            from aicspylibczi import CziFile

            stream = stack.enter_context(open(source, "rb"))
            native = CziFile(stream)
            self.native_dims = set(native.dims)
            scenes = self.scenes_bounding_rectangle or {0: self.total_bounding_rectangle}
            for scene in scenes:
                box = (native.get_mosaic_scene_bounding_box(scene) if native.is_mosaic()
                       else native.get_scene_bounding_box(scene))
                self.native_bounds[scene] = (box.x, box.y, box.w, box.h)
            if backend == "aicspylibczi":
                if native.is_mosaic() and len(scenes) != 1:
                    raise NotImplementedError("This aicspylibczi adapter requires a single-scene mosaic")
                self.reader = native
            elif backend == "bioio-czi":
                from bioio_czi import Reader

                self.reader = Reader(source, use_aicspylibczi=True)
                for index, scene in enumerate(scenes):
                    self.reader.set_scene(index)
                    array = (self.reader.mosaic_xarray_dask_data if "M" in self.reader.dims.order
                             else self.reader.xarray_dask_data)
                    self.arrays[scene] = array
            else:
                raise ValueError(f"Unknown backend: {backend}")

    def read(self, roi, plane, scene):
        if self.backend == "pylibCZIrw":
            return self.reader.read(roi=roi, plane=plane, scene=scene, zoom=1.0)
        scene = 0 if scene is None else scene
        x, y, width, height = roi
        bx, by, bw, bh = self.native_bounds[scene]
        left, top = max(x, bx), max(y, by)
        right, bottom = min(x + width, bx + bw), min(y + height, by + bh)
        pixel_type = self.pixel_types[plane["C"]]
        dtype = {"Gray8": np.uint8, "Gray16": np.uint16, "Gray32Float": np.float32}[pixel_type]
        output = np.zeros((height, width, 1), dtype=dtype)
        if right <= left or bottom <= top:
            self.empty_regions += 1
            return output
        region = (left, top, right - left, bottom - top)
        if self.backend == "czifile":
            image = self.reader.scenes[scene]
            selection = self._selection(plane, set(image.dims))
            try:
                data = self.reader.asarray(scene=scene, roi=region, maxworkers=1, **selection)
            except ValueError as error:
                if str(error) != "roi matches no subblocks":
                    raise
                self.empty_regions += 1
                LOG.debug("czifile ROI %s contains no subblocks; using declared zero background", roi)
                return output
        elif self.backend == "aicspylibczi":
            selection = self._selection(plane, self.native_dims)
            if self.reader.is_mosaic():
                data = self.reader.read_mosaic(region=region, scale_factor=1.0, **selection)
            else:
                if "S" in self.native_dims:
                    selection["S"] = scene
                data, _ = self.reader.read_image(region=region, cores=1, **selection)
        else:
            array = self.arrays[scene]
            selection = self._selection(plane, set(array.dims))
            indices = {axis: value - self.total_bounding_box[axis][0]
                       for axis, value in selection.items()}
            selected = array.isel(**indices, X=slice(left-bx, right-bx), Y=slice(top-by, bottom-by))
            extra = [axis for axis in selected.dims if axis not in {"Y", "X"}]
            if any(selected.sizes[axis] != 1 for axis in extra):
                raise ValueError(f"Unselected BioIO dimensions: {selected.sizes}")
            selected = selected.squeeze(dim=extra).transpose("Y", "X")
            data = selected.data.compute(scheduler="synchronous")
        data = np.asarray(data)
        if data.size != (bottom - top) * (right - left) or data.dtype != output.dtype:
            raise ValueError(f"{self.backend}: unexpected ROI data shape/dtype {data.shape}/{data.dtype}")
        output[top-y:bottom-y, left-x:right-x] = data.reshape(bottom-top, right-left, 1)
        return output

    @staticmethod
    def _selection(plane, available):
        for axis, value in plane.items():
            if axis not in available and value != 0:
                raise ValueError(f"Reader cannot select nonzero {axis}={value}")
        return {axis: value for axis, value in plane.items() if axis in available}


@contextmanager
def open_reader(source, backend):
    with ExitStack() as stack:
        canonical = stack.enter_context(czi.open_czi(source))
        adapter = ReaderAdapter(source, backend, canonical, stack)
        try:
            yield adapter
        finally:
            LOG.info("%s empty/background ROI count: %s", backend, adapter.empty_regions)


def run_one(source, output, backend, target, tile):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    result = {"backend": backend, "version": importlib.metadata.version(backend),
              "source": str(source), "target_um": target, "tile_pixels": tile,
              "thread_policy": "czifile maxworkers=1; BioIO synchronous; direct libCZI defaults",
              "status": "running"}
    try:
        manifest = process(source, output, target, tile,
                           reader_factory=partial(open_reader, backend=backend))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      wall_seconds=perf_counter() - started)
        write_json(output / "run.json", result)
        LOG.exception("%s full-file benchmark failed", backend)
        raise
    result.update(status="completed", wall_seconds=perf_counter() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  manifest=manifest)
    write_json(output / "run.json", result)
    return result


def compare_outputs(reference, candidate):
    comparison = {"equivalent": True, "channels": []}
    expected_channels = {item["source_channel"]: item for item in reference["manifest"]["outputs"]}
    actual_channels = {item["source_channel"]: item for item in candidate["manifest"]["outputs"]}
    if expected_channels.keys() != actual_channels.keys():
        return {"equivalent": False, "error": "Channel sets differ", "channels": []}
    for channel, item in actual_channels.items():
        ref = expected_channels[channel]
        with czi.open_czi(ref["path"]) as expected, czi.open_czi(item["path"]) as actual:
            dimensions_match = expected.total_bounding_box == actual.total_bounding_box
            calibration_match = metadata(expected) == metadata(actual)
            scenes_match = expected.scenes_bounding_rectangle == actual.scenes_bounding_rectangle
            record = {"channel": channel, "dimensions_match": dimensions_match,
                      "calibration_and_names_match": calibration_match, "scenes_match": scenes_match,
                      "mismatched_pixels": 0, "compared_pixels": 0, "max_absolute_difference": 0.0}
            if dimensions_match and scenes_match:
                box = expected.total_bounding_box
                axes = {axis: range(*extent) for axis, extent in box.items()
                        if axis not in {"X", "Y", "S"}}
                scenes = expected.scenes_bounding_rectangle or {None: expected.total_bounding_rectangle}
                for indices in itertools.product(*axes.values()):
                    plane = dict(zip(axes, indices))
                    for scene, rect in scenes.items():
                        roi = (rect.x, rect.y, rect.w, rect.h)
                        a = expected.read(roi=roi, plane=plane, scene=scene)
                        b = actual.read(roi=roi, plane=plane, scene=scene)
                        if a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
                            raise ValueError("Output dtype mismatch or nonfinite pixels")
                        record["compared_pixels"] += a.size
                        record["mismatched_pixels"] += int(np.count_nonzero(a != b))
                        difference = float(np.max(np.abs(a.astype(np.float64) - b), initial=0))
                        record["max_absolute_difference"] = max(record["max_absolute_difference"], difference)
            record["equivalent"] = (dimensions_match and calibration_match and scenes_match
                                    and record["mismatched_pixels"] == 0)
            comparison["equivalent"] &= record["equivalent"]
            comparison["channels"].append(record)
    return comparison


def run_suite(source, output, target=10.0, tile=512):
    output.mkdir(parents=True, exist_ok=False)
    summary = {"source": str(source), "source_bytes": source.stat().st_size,
               "target_um": target, "tile_pixels": tile, "backend_order": list(BACKENDS),
               "cache_policy": "OS/network caches not flushed; readers run serially in fresh processes",
               "writer": "pylibCZIrw zstd1:ExplicitLevel=1", "runs": []}
    for backend in BACKENDS:
        directory = output / backend
        command = [sys.executable, str(Path(__file__).resolve()), "--input", str(source),
                   "--output", str(directory), "--backend", backend,
                   "--target-um", str(target), "--tile", str(tile)]
        started = perf_counter()
        LOG.info("Starting complete-file benchmark: %s", backend)
        completed = subprocess.run(command, check=False)
        report = directory / "run.json"
        if report.exists():
            result = json.loads(report.read_text(encoding="utf-8"))
        else:
            result = {"backend": backend, "status": "failed", "error": "Child did not write run.json"}
        result["process_wall_seconds"] = perf_counter() - started
        result["exit_code"] = completed.returncode
        if completed.returncode != 0:
            result["status"] = "failed"
            LOG.error("%s failed with exit code %s: %s", backend, completed.returncode, result.get("error"))
        summary["runs"].append(result)
        write_json(output / "summary.json", summary)
    reference = next(run for run in summary["runs"] if run["backend"] == "pylibCZIrw")
    if reference["status"] == "completed":
        for run in summary["runs"]:
            if run["status"] == "completed":
                run["comparison"] = compare_outputs(reference, run)
                LOG.info("%s matches reference: %s", run["backend"], run["comparison"]["equivalent"])
        eligible = [run for run in summary["runs"] if run.get("comparison", {}).get("equivalent")]
        summary["equivalent_speed_ranking"] = [
            run["backend"] for run in sorted(eligible, key=lambda run: run["wall_seconds"])]
    else:
        summary["comparison_error"] = "Reference backend failed; no equivalence claims possible"
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT / "sc-C042-149-O4213.czi")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--backend", choices=BACKENDS, help="Run only this backend (also used by child processes)")
    parser.add_argument("--target-um", type=float, default=10.0)
    parser.add_argument("--tile", type=int, default=512, help="Same output tile size for all readers")
    args = parser.parse_args()
    if not args.input.is_file() or args.input.suffix.lower() != ".czi":
        parser.error("--input must be an existing CZI file")
    if args.tile < 1 or not math.isfinite(args.target_um) or args.target_um <= 0:
        parser.error("--tile and --target-um must be positive and finite")
    configure_logging()
    if args.backend:
        run_one(args.input.resolve(), args.output.resolve(), args.backend, args.target_um, args.tile)
    else:
        summary = run_suite(args.input.resolve(), args.output.resolve(), args.target_um, args.tile)
        if any(run["status"] != "completed" for run in summary["runs"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
