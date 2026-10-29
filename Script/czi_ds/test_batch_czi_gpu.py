import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
import sys

import numpy as np
from pylibCZIrw import czi

from batch_czi_gpu import load_jobs, run_worker
from batch_czi_gpu import main
from process_czi import process
from benchmark_tiles import full_plane_tile
from benchmark_gpu import BatchedAreaMean


class TableResumeTests(unittest.TestCase):
    def test_full_plane_tile_matches_processor_rounding(self):
        reader = Mock()
        rect = Mock(x=0, y=0, w=29600, h=20416)
        reader.scenes_bounding_rectangle = {0: rect}
        with patch("benchmark_tiles.czi.open_czi") as open_czi, \
             patch("benchmark_tiles.metadata", return_value=({"X": 0.65, "Y": 0.65}, {})):
            open_czi.return_value.__enter__.return_value = reader
            tile = full_plane_tile(Path("source.czi"), 10.0)
        self.assertEqual(tile, 1925)
        reader.read.side_effect = RuntimeError("whole-plane bounds accepted")
        with self.assertRaisesRegex(RuntimeError, "whole-plane bounds accepted"):
            BatchedAreaMean()(reader, {"C": 0}, 0, rect, (0, 0, tile, 1328),
                              (10.0 / 0.65, 10.0 / 0.65), {"read": 0.0})

    def test_submit_places_logs_in_first_output_without_starting_real_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            (source / "test.czi").touch()
            output = root / "output"
            table = root / "jobs.csv"
            table.write_text(f"input_dir,output_dir\n{source},{output}\n")
            with patch.object(sys, "argv", ["batch_czi_gpu.py", "--jobs", str(table), "--submit"]), \
                 patch("batch_czi_gpu.subprocess.Popen", return_value=Mock(pid=12345)) as launch:
                main()
            self.assertEqual(len(list((output / "logs").glob("*.log"))), 1)
            self.assertEqual(next((output / "logs").glob("*.pid")).read_text(), "12345\n")
            self.assertEqual(launch.call_args.args[0][0], "nohup")
            self.assertNotIn("--submit", launch.call_args.args[0])
            self.assertTrue(launch.call_args.kwargs["start_new_session"])

    def test_csv_relative_paths_and_duplicate_names_in_separate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                (root / name).mkdir()
                (root / name / "same.CZI").touch()
            table = root / "jobs.csv"
            with table.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerows([["input_dir", "output_dir"], ["a", "out/a"],
                                  ["b", "out/b"], ["a", "out/a"]])
            self.assertEqual(len(load_jobs(table)), 2)
            table.write_text("input_dir,output_dir\na,out\nb,out\n")
            with self.assertRaises(ValueError):
                load_jobs(table)

    def test_skip_verified_and_regenerate_only_corrupt_or_unfinished_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.czi", root / "out"
            with czi.create_czi(str(source)) as writer:
                for channel in range(2):
                    writer.write(np.full((53, 67, 1), channel + 100, np.uint16),
                                 plane={"C": channel})
                writer.write_metadata(scale_x=0.65e-6, scale_y=0.65e-6, scale_z=None)
            first = process(source, output, resume=True)
            paths = [Path(item["path"]) for item in first["outputs"]]
            mtimes = [path.stat().st_mtime_ns for path in paths]
            second = process(source, output, resume=True)
            self.assertTrue(all(item["status"] == "skipped" for item in second["outputs"]))
            self.assertEqual(mtimes, [path.stat().st_mtime_ns for path in paths])
            paths[1].write_bytes(b"corrupt")
            third = process(source, output, resume=True)
            self.assertEqual(third["outputs"][0]["status"], "skipped")
            self.assertNotIn("status", third["outputs"][1])
            with czi.open_czi(str(paths[1])) as reader:
                np.testing.assert_array_equal(reader.read(), 101)
            paths[1].with_suffix(".complete.json").unlink()
            paths[1].with_suffix(".partial.czi").write_bytes(b"interrupted")
            fourth = process(source, output, resume=True)
            self.assertNotIn("status", fourth["outputs"][1])
            self.assertFalse(paths[1].with_suffix(".partial.czi").exists())
            changed = process(source, output, target=10.0, resume=True)
            self.assertTrue(all(item["status"] == "skipped" for item in changed["outputs"]))
            source.touch()
            changed = process(source, output, resume=True)
            self.assertTrue(all(item.get("status") != "skipped" for item in changed["outputs"]))

    def test_worker_continues_after_file_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("batch_czi_gpu.process_gpu", side_effect=[
                RuntimeError("broken input"), [{"outputs": [{"status": "skipped"}]}]]):
                results = run_worker([(root / "bad.czi", root), (root / "good.czi", root)], 10, 256, 0)
            self.assertEqual([item["status"] for item in results], ["failed", "completed"])
            self.assertEqual(results[1]["skipped_channels"], 1)


if __name__ == "__main__":
    unittest.main()
