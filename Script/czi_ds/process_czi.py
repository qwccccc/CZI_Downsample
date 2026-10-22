"""Split CZI channels and area-downsample XY using bounded-memory tiles."""

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import itertools
import json
import logging
import math
import multiprocessing
import re
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from pylibCZIrw import czi
from scipy.sparse import csr_matrix


DEFAULT_INPUT = Path("/mnt/90-connectome/Personal/CQW/datasets/czi_ds")
LOG = logging.getLogger("czi_ds")
STAGES = ("weights", "read", "resample", "write", "metadata", "writer_open_close", "verify")


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_receipt(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def reusable_channel(path, receipt, identity):
    if not path.exists() or not receipt.exists():
        return None
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record["identity"] != identity or record["sha256"] != file_digest(path):
            raise ValueError("Source, parameters or output checksum changed")
        item = record["output"]
        verify(path, item["pixel_size_um"],
               {key: tuple(value) for key, value in item["bounding_box"].items()}, [])
        return item
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        LOG.warning("Cannot reuse %s: %s; regenerating", path, error)
        return None


@contextmanager
def timed(timings, stage):
    started = perf_counter()
    try:
        yield
    finally:
        timings[stage] += perf_counter() - started


def configure_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(processName)s %(levelname)s %(message)s")


def metadata(reader):
    root = ET.fromstring(reader.raw_metadata)
    scales = {}
    for item in root.findall(".//Scaling/Items/Distance"):
        value = float(item.findtext("Value", "nan")) * 1e6
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid physical pixel size: {item.attrib}")
        scales[item.attrib["Id"]] = value
    if not {"X", "Y"} <= scales.keys():
        raise ValueError("CZI metadata must contain physical X and Y pixel sizes")
    channels = root.findall(".//Information/Image/Dimensions/Channels/Channel")
    names = {int(el.attrib["Id"].split(":")[-1]): el.get("Name", el.attrib["Id"])
             for el in channels}
    return scales, names


def area_weights(start, stop, ratio, limit):
    """Integrate source pixel areas, normalizing partial edge pixels."""
    left = math.floor(start * ratio)
    right = min(math.ceil(stop * ratio), limit)
    rows, cols, values = [], [], []
    for row, index in enumerate(range(start, stop)):
        a, b = index * ratio, min((index + 1) * ratio, limit)
        for source in range(math.floor(a), math.ceil(b)):
            rows.append(row)
            cols.append(source - left)
            values.append((min(b, source + 1) - max(a, source)) / (b - a))
    return left, right, csr_matrix((values, (rows, cols)), shape=(stop - start, right - left))


def downsample_tile(reader, plane, scene, rect, bounds, ratios, timings):
    x, y, end_x, end_y = bounds
    rx, ry = ratios
    with timed(timings, "weights"):
        left, right, wx = area_weights(x, end_x, rx, rect.w)
        top, bottom, wy = area_weights(y, end_y, ry, rect.h)
    with timed(timings, "read"):
        data = reader.read(roi=(rect.x + left, rect.y + top, right - left, bottom - top),
                           plane=plane, scene=scene)
    with timed(timings, "resample"):
        if data.shape[-1] != 1:
            raise ValueError("Only scalar grayscale channels are supported")
        averaged = (wx @ (wy @ data[..., 0]).T).T
        if np.issubdtype(data.dtype, np.integer):
            limits = np.iinfo(data.dtype)
            averaged = np.clip(np.rint(averaged), limits.min, limits.max)
        result = averaged.astype(data.dtype)[..., None]
    return result


def tiles(width, height, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield x, y, min(x + size, width), min(y + size, height)


def verify(path, scales, expected_box, samples):
    with czi.open_czi(str(path)) as reader:
        actual, _ = metadata(reader)
        for axis, value in scales.items():
            if not math.isclose(actual[axis], value, rel_tol=1e-8):
                raise RuntimeError(f"{path}: incorrect {axis} pixel size")
        if reader.total_bounding_box != expected_box:
            raise RuntimeError(f"{path}: unexpected dimensions {reader.total_bounding_box}, expected {expected_box}")
        for plane, scene, roi, expected in samples:
            observed = reader.read(roi=roi, plane=plane, scene=scene)
            if not np.array_equal(observed, expected):
                raise RuntimeError(f"{path}: pixel verification failed at {roi}")


def process(source, output, target=10.0, reduced_tile=512, reader_factory=czi.open_czi,
            tile_processor=downsample_tile, resume=False):
    file_started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with reader_factory(str(source)) as reader:
        scales, names = metadata(reader)
        if target < max(scales["X"], scales["Y"]):
            raise ValueError(f"{source}: target must not upsample either XY axis")
        ratios = target / scales["X"], target / scales["Y"]
        box = reader.total_bounding_box
        axes = {axis: range(*extent) for axis, extent in box.items()
                if axis not in {"X", "Y", "C", "S"}}
        scenes = reader.scenes_bounding_rectangle or {None: reader.total_bounding_rectangle}
        manifest = {"source": str(source), "source_pixel_size_um": scales,
                    "target_xy_um": target, "output_tile_pixels": reduced_tile, "outputs": []}
        for channel in range(*box["C"]):
            channel_started = perf_counter()
            timings = dict.fromkeys(STAGES, 0.0)
            if reader.pixel_types[channel] not in {"Gray8", "Gray16", "Gray32Float"}:
                raise ValueError(f"Unsupported pixel type: {reader.pixel_types[channel]}")
            name = names.get(channel, f"C{channel}")
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "channel"
            path = output / f"{source.stem}_C{channel}_{safe_name}_{target:g}um.czi"
            temporary = path.with_suffix(".partial.czi")
            receipt = path.with_suffix(".complete.json")
            stat = source.stat()
            identity = {"schema": 1, "algorithm": "area-mean-xy-v1",
                        "source": str(source.resolve()), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "target_um": target, "channel": channel}
            if resume:
                existing = reusable_channel(path, receipt, identity)
                if existing is not None:
                    LOG.info("Skipping verified complete channel: %s", path)
                    existing["status"] = "skipped"
                    existing["timings_seconds"] = dict.fromkeys((*STAGES, "other", "total"), 0.0)
                    manifest["outputs"].append(existing)
                    if temporary.exists():
                        temporary.unlink()
                    continue
                for stale in (path, temporary, receipt):
                    if stale.exists():
                        LOG.warning("Removing unverified/incomplete result before regeneration: %s", stale)
                        stale.unlink()
            if path.exists() or temporary.exists():
                raise FileExistsError(f"Refusing to overwrite {path} or {temporary}")
            samples = []
            min_x = min_y = math.inf
            max_x = max_y = -math.inf
            LOG.info("Writing %s", path)
            writer_started = perf_counter()
            with czi.create_czi(str(temporary), compression_options="zstd1:ExplicitLevel=1") as writer:
                timings["writer_open_close"] += perf_counter() - writer_started
                for indices in itertools.product(*axes.values()):
                    plane = dict(zip(axes, indices))
                    for scene, rect in scenes.items():
                        width = math.ceil(rect.w / ratios[0])
                        height = math.ceil(rect.h / ratios[1])
                        ox, oy = round(rect.x / ratios[0]), round(rect.y / ratios[1])
                        min_x, min_y = min(min_x, ox), min(min_y, oy)
                        max_x, max_y = max(max_x, ox + width), max(max_y, oy + height)
                        for bounds in tiles(width, height, reduced_tile):
                            x, y, ex, ey = bounds
                            data = tile_processor(reader, {**plane, "C": channel},
                                                  scene, rect, bounds, ratios, timings)
                            output_plane = {**plane, "C": 0}
                            with timed(timings, "write"):
                                if not writer.write(data, location=(ox + x, oy + y),
                                                    plane=output_plane, scene=scene if scene is not None else 0):
                                    raise RuntimeError(f"Failed writing {path} at {bounds}")
                            if (x == 0 and y == 0) or (ex == width and ey == height):
                                sample = data[:min(16, ey-y), :min(16, ex-x)].copy()
                                samples.append((output_plane, scene,
                                                (ox+x, oy+y, sample.shape[1], sample.shape[0]), sample))
                        LOG.info("  channel=%s scene=%s plane=%s size=%sx%s", channel, scene, plane, width, height)
                out_scales = {**scales, "X": target, "Y": target}
                with timed(timings, "metadata"):
                    writer.write_metadata(document_name=path.stem, channel_names={0: name},
                                          scale_x=target * 1e-6, scale_y=target * 1e-6,
                                          scale_z=scales["Z"] * 1e-6 if "Z" in scales else None,
                                          custom_attributes={"SourceFile": source.name,
                                                             "SourceChannel": channel,
                                                             "Resampling": "area mean XY"})
                writer_started = perf_counter()
            timings["writer_open_close"] += perf_counter() - writer_started
            expected_box = {**box, "C": (0, 1), "X": (int(min_x), int(max_x)),
                            "Y": (int(min_y), int(max_y))}
            with timed(timings, "verify"):
                verify(temporary, out_scales, expected_box, samples)
                if resume:
                    with czi.open_czi(str(temporary)) as check:
                        for indices in itertools.product(*axes.values()):
                            plane = {**dict(zip(axes, indices)), "C": 0}
                            for scene, rect in check.scenes_bounding_rectangle.items():
                                for x, y, ex, ey in tiles(rect.w, rect.h, 512):
                                    pixels = check.read(roi=(rect.x+x, rect.y+y, ex-x, ey-y),
                                                        plane=plane, scene=scene)
                                    if pixels.shape != (ey-y, ex-x, 1) or not np.isfinite(pixels).all():
                                        raise RuntimeError(f"Invalid output pixels in {temporary}")
            temporary.rename(path)
            elapsed = perf_counter() - channel_started
            timings["other"] = elapsed - sum(timings.values())
            timings["total"] = elapsed
            LOG.info("Channel timings %s: %s", path.name,
                     " ".join(f"{key}={value:.3f}s" for key, value in timings.items()))
            manifest["outputs"].append({"path": str(path), "source_channel": channel,
                                        "channel_name": name, "pixel_size_um": out_scales,
                                        "bounding_box": expected_box, "verified": True,
                                        "timings_seconds": timings})
            if resume:
                save_receipt(receipt, {"identity": identity, "sha256": file_digest(path),
                                       "output": manifest["outputs"][-1]})
    totals = {stage: sum(item["timings_seconds"][stage] for item in manifest["outputs"])
              for stage in STAGES}
    elapsed = perf_counter() - file_started
    totals["other"] = elapsed - sum(totals.values())
    totals["total"] = elapsed
    manifest["timings_seconds"] = totals
    report = output / f"{source.stem}_manifest.json"
    report.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("File timings %s: %s", source.name,
             " ".join(f"{key}={value:.3f}s" for key, value in totals.items()))
    return manifest


def run_batch(sources, output, target=10.0, workers=2, tile=512):
    if workers < 1:
        raise ValueError("workers must be positive")
    if tile < 1:
        raise ValueError("tile must be positive")
    if not sources:
        raise ValueError("No input CZI files found")
    if len({source.stem for source in sources}) != len(sources):
        raise ValueError("Input filenames must have unique stems to avoid output collisions")
    started = perf_counter()
    workers = min(workers, len(sources))
    LOG.info("Processing %s files with %s worker(s)", len(sources), workers)
    if workers == 1:
        reports = [process(source, output, target, tile) for source in sources]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=configure_logging) as executor:
            futures = [executor.submit(process, source, output, target, tile) for source in sources]
            reports = [future.result() for future in as_completed(futures)]
    elapsed = perf_counter() - started
    LOG.info("Batch completed: files=%s workers=%s wall=%.3fs summed_file_time=%.3fs",
             len(sources), workers, elapsed, sum(r["timings_seconds"]["total"] for r in reports))
    return {"workers": workers, "wall_seconds": elapsed, "files": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", default=[DEFAULT_INPUT],
                        help="One or more CZI files or directories")
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT / "results")
    parser.add_argument("--target-um", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=2, help="Concurrent files (default: 2)")
    parser.add_argument("--tile", type=int, default=512,
                        help="Output tile edge in pixels; larger tiles use more memory (default: 512)")
    args = parser.parse_args()
    if not math.isfinite(args.target_um) or args.target_um <= 0:
        parser.error("--target-um must be finite and positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.tile < 1:
        parser.error("--tile must be positive")
    sources = []
    for path in args.input:
        sources.extend([path] if path.is_file() else sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".czi"))
    if not sources or any(p.suffix.lower() != ".czi" for p in sources):
        parser.error("No input CZI files found")
    configure_logging()
    run_batch(sources, args.output, args.target_um, args.workers, args.tile)


if __name__ == "__main__":
    main()
