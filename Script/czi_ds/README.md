# CZI channel splitting and XY downsampling

## CSV directory queue with automatic channel resume

Use `batch_czi_gpu.py` (or `run_gpu_table.sh`) for this new workflow.
Existing single-directory entry points retain their previous overwrite policy.
Create a CSV using `jobs.example.csv` as a template:

```csv
input_dir,output_dir
/data/experiment_A,/results/experiment_A
/data/experiment_B,/results/experiment_B
```

One row maps one input directory to its corresponding output directory.
Only immediate `.czi`/`.CZI` files are scanned. Relative paths are resolved
relative to the CSV, not the shell's current directory. Quote paths containing
commas using standard CSV quoting. Duplicate identical jobs are deduplicated;
same file stems targeting the same output directory are rejected before work.
Every row must contain an existing, nonempty input directory.

Foreground (Python entry point):

```bash
python Script/czi_ds/batch_czi_gpu.py \
  --jobs /path/to/jobs.csv --report /path/to/batch_report.json \
  --workers 4 --devices 0 1 --batch-rows 256 --target-um 10
```

后台提交，只需运行：

```bash
bash Script/czi_ds/run_gpu_table.sh
```

运行前修改 `run_gpu_table.sh` 顶部的 `JOBS_CSV`、`WORKERS`、
`DEVICES`、`BATCH_ROWS` 和 `TARGET_UM`。任务表默认是脚本旁的 `jobs.csv`，
需要自行创建填写；`DEVICES=()` 表示自动选卡。命令行参数仍可覆盖脚本配置。
脚本自动使用 nohup，立即返回终端。每个输出目录的 logs 保存对应文件日志；
任务表第一个输出目录的 logs 保存按时间命名的整批日志、报告和 PID。
提交后按终端显示的日志路径使用 `tail -f` 查看。日志不会默认写入代码目录。
No job is submitted by installing these scripts.
Normal terminal disconnection does not stop nohup jobs, but host reboot,
OOM termination and system cleanup policies can still terminate them.

Workers process their file queues across all CSV rows, round-robin over the
selected GPUs. Multiple workers may share a GPU. A file failure is logged
with a traceback and does not stop subsequent files. The final JSON report
lists completed/failed files and skipped-channel counts; the command exits
nonzero if any file failed. The aggregate report updates when a worker queue
returns; per-channel receipts and logs provide progress during long queues.
Linux advisory locks prevent this entry point from concurrently writing the
same source stem/output pair; they do not protect against older launchers
that do not participate in locking. Lock files intentionally remain on disk;
the kernel releases their locks when processes exit.

### How resume decides whether to skip a channel

- Successful channels are reopened and fully decoded in bounded output tiles,
  in addition to dimension/calibration and source-to-output sample checks.
- A per-channel `.complete.json` receipt is written atomically after output
  publication, recording the source's resolved path, size and nanosecond mtime,
  target resolution, algorithm version and output SHA-256.
- On rerun, matching source identity/parameters, a matching output checksum
  and successful CZI dimension/calibration validation allow that channel to
  be skipped. A change only to workers or GPU batch rows does not invalidate
  equivalent results.
- Missing receipts, changed checksums, changed source identity or invalid
  metadata trigger **direct deletion and regeneration of that channel's
  output/partial file**, as explicitly selected for this workflow.
- Earlier outputs made before receipt support cannot be trusted by presence
  alone and are regenerated once. Do not remove receipt files if you want
  verified skipping. Interrupted source files can resume at channel boundaries,
  not midway through a channel.

The source identity uses size/mtime rather than hashing multi-GB source files;
it does not detect deliberate source edits that preserve both size and mtime.
Do not modify source data during processing. Output checksums detect subsequent
changes, not a guarantee of biological correctness or complete acquisition
metadata preservation.

Run from the project root:

```bash
uv sync
uv run python Script/czi_ds/process_czi.py
```

The default input is `/mnt/90-connectome/Personal/CQW/datasets/czi_ds`.
The default output is its `results` directory. Only CZI files directly inside
the input directory are processed; results are not recursively processed.

To process one file:

```bash
uv run python Script/czi_ds/process_czi.py \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds/sc-C042-149-O4213.czi \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results \
  --target-um 10
```

## Parallel processing and timing

`--workers` selects the maximum number of files processed concurrently
(default: **2**). Use `--workers 1` for sequential processing. A single input
file always uses one process; this is file-level, not channel-level parallelism.
Each worker opens its own CZI reader and writer using the multiprocessing
`spawn` start method. Worker errors propagate to the main process.

Multiple input files or directories can be supplied:

```bash
uv run python Script/czi_ds/process_czi.py \
  --input /path/to/first.czi /path/to/second.czi \
  --workers 2
```

Choose unprocessed inputs: already completed files are not automatically
skipped. Input filename stems must be unique to avoid output collisions.
On shared/network storage, more workers can increase contention rather than
speed up processing. Start with 2; memory usage grows with active workers.

`--tile` controls the output tile edge (default 128). For example,
`--workers 2 --tile 512` uses fewer, larger source reads while retaining the
same output grid and area-average algorithm. Memory grows approximately with
the square of this edge and with the number of active workers. A 512-pixel
output tile at 0.65 um source calibration covers approximately 7877 source
pixels per axis; source arrays and floating-point intermediates require
substantially more memory than the small output tile alone.

The logs and each source's JSON manifest include `timings_seconds` for every
channel and the whole source:

| Field | Measured elapsed time |
| --- | --- |
| `weights` | Constructing area-average weights |
| `read` | CZI ROI read, including library decompression and scene rendering |
| `resample` | Area averaging, rounding and dtype conversion |
| `write` | Writing/compressing output tiles |
| `metadata` | Generating/writing output metadata |
| `writer_open_close` | Opening/finalizing the output CZI writer |
| `verify` | Reopening and checking the output |
| `other` | Remaining setup, bookkeeping, logging and related overhead |
| `total` | Channel or source wall-clock duration |

Source timing includes input open/close but excludes writing its JSON manifest.
The batch log reports overall wall time, including process startup and manifest
writes, and summed per-file time. Concurrent file durations overlap: do not
interpret their sum as batch wall time. Timings are not CPU utilization
measurements, and `read` does not separate network I/O from decompression.

### Real-file timing test (2026-10-15)

Two files were processed concurrently with `--workers 2` on the network mount.
Each file contained four channels, with original XY calibration 0.65 um/pixel.
All eight outputs passed reopening checks and an independent area-average
check against source pixels at each output's brightest pixel.

| Stage (seconds, all four channels) | sc-C042-147-O4213 | sc-C042-148-O4213 |
| --- | ---: | ---: |
| Weight construction | 10.045 | 11.038 |
| Read/decompress/render | 717.908 | 751.311 |
| Resample/dtype conversion | 10.805 | 11.886 |
| Tile write/compression | 0.759 | 1.013 |
| Metadata write | 0.005 | 0.006 |
| Writer open/finalize | 0.030 | 0.049 |
| Reopen verification | 0.101 | 0.076 |
| Other | 0.360 | 0.366 |
| Per-file total | 740.014 | 775.745 |

Batch wall time was **776.134 seconds**, not the sum of the two file times.
Reading accounted for 97.01% and 96.85% of each file's time; the current
bottleneck is the reader stage, not area-average arithmetic. These measurements
do not distinguish network, decompression and scene-rendering costs, and are not
a controlled serial-versus-parallel speedup benchmark on the same inputs.
GPU resampling alone would therefore save little time in this workload.

## Four-reader full-file benchmark

Install the optional reader dependencies:

```bash
uv sync --extra benchmark
uv run --extra benchmark python Script/czi_ds/benchmark_readers.py \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds/sc-C042-149-O4213.czi \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/reader_benchmark_20261015
```

The output directory must be new. Four fresh Python processes run serially in
this order: pylibCZIrw, aicspylibczi, czifile, bioio-czi. Each processes the
**entire input, all channels**, not a sampled ROI, using the same area-average
algorithm and pylibCZIrw CZI writer. `--tile` sets the common output tile size
(default 512, versus the production script's existing 128). No pyramid
downsampling or approximate library scaling is used. The benchmark does not
change the production default reader.

Direct mosaic reads use native resolution. czifile uses `maxworkers=1` for
deterministic overlap compositing; BioIO uses its default aicspylibczi backend
and synchronous Dask computation. Direct aicspylibczi mosaic reading does not
expose a core-count argument. This compares single-file reader paths, not every
possible multithreaded tuning. The direct aicspylibczi adapter currently rejects
multi-scene mosaics rather than risking cross-scene blending.

All adapters use common metadata/calibration and output geometry from
pylibCZIrw. Some readers report a smaller native image box because they exclude
pyramid-level extents. Regions outside native bounds and uncovered mosaic
regions are explicitly zero-filled, matching the declared background policy.
Any actual pixel differences are reported, not normalized away.

Each backend writes CZI results, a source manifest and `run.json`. The root
`summary.json` is updated after each backend and contains versions, process
wall time, stage timings, peak process RSS, errors and complete output pixel
comparisons against the freshly generated pylibCZIrw reference. Only
pixel-identical, metadata-equivalent completed runs are ranked as equivalent.
Failures retain diagnostic partial outputs and do not silently substitute
another reader. Complete comparisons happen after all runs and are outside
the timed processing path.

OS/network caches are **not flushed** on this shared machine. Later readers
may benefit from warm caches; one fixed-order run is not a definitive cold-cache
ranking. Peak RSS includes import/runtime overhead and library caches.

### Full-file results (2026-10-15)

Input: `sc-C042-149-O4213.czi` (5,244,824,896 bytes), all four channels.
Each backend generated four 4439 x 3602 uint16 single-channel CZIs calibrated
at 10 um/pixel. All 16 outputs completed successfully. For each backend,
63,957,112 output pixels were compared with the reference, with **zero
mismatches**; output geometry, channel names and physical calibration also
matched. A separate full comparison with the earlier 128-tile output found
zero pixel differences.

| Reader | Version | Reader stage (s) | Processing total (s) | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: |
| pylibCZIrw | 6.1.0 | 305.040 | 335.145 | 817.0 |
| czifile | 2026.10.14 | 315.722 | 342.495 | 708.9 |
| aicspylibczi | 4.0.1 | 340.336 | 366.815 | 704.2 |
| bioio-czi | 3.0.1 | 1199.326 | 1228.595 | 1053.4 |

Here the processing total includes writing the source manifest, but excludes
interpreter startup and the later full output comparison. More detailed
stage timings are in each source manifest and `run.json`; all results and
comparisons are under `results/reader_benchmark_20261015/summary.json` in the
dataset directory.

For this access pattern, retain pylibCZIrw and consider `--tile 512`.
The pylibCZIrw/czifile difference is only about 2.2%, too small to claim a
robust general winner from a single fixed-order trial. The BioIO result
reflects synchronous Dask mosaic-window reads, which can rebuild/read
overlapping native chunks across successive windows; it is not evidence that
every BioIO usage is intrinsically slower. The earlier 128-tile run took
approximately 12.4 minutes versus 5.6 minutes here, but that was at a different
time/cache state, not a controlled tile-size speedup experiment.

Each source channel produces one genuine, losslessly ZSTD-compressed CZI file:
`<source>_C0_<channel>_10um.czi`, with XY pixel size exactly 10 micrometers.
No original-resolution single-channel copies are generated.

## Tile-size full-file benchmark

```bash
uv run python Script/czi_ds/benchmark_tiles.py \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds/sc-C042-149-O4213.czi \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/tile_benchmark_20261015
```

The output directory must be new. This runs 512, 1024, 2048 and full-plane
cases sequentially, each in a fresh Python process with the same pylibCZIrw
reader, writer and area-average algorithm. Each case processes the complete
file and all channels at 10 um/pixel. Full-plane uses a tile edge large enough
to cover every scene's complete output plane; for the example file the edge
is 4439, so each channel is read as one full-resolution ROI.

Each case writes four single-channel CZIs plus stage timings and peak process
RSS. The root `summary.json` is updated after each case, then compares every
output pixel against the new 512 case and ranks only equivalent results.
Complete comparisons are outside the processing timers.
Caches are not flushed; these are fixed-order observations, not a repeated
cold-cache benchmark. Very large tiles can require tens of GiB of temporary
memory even though the output images are small. Do not use full-plane mode
with multiple files concurrently without checking memory headroom.

### Tile-size results (2026-10-15)

The same complete 5.24 GB, four-channel input was processed once per case.
All cases generated four 4439 x 3602 uint16 CZIs at 10 um/pixel.
Each case's 63,957,112 output pixels matched the new 512 reference exactly;
geometry, channel names and physical calibration matched as well.

| Output tile edge | Reader stage (s) | Resample (s) | Processing total (s) | Peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 512 | 299.472 | 23.131 | 330.143 | 0.798 |
| 1024 | 249.361 | 26.565 | 281.536 | 2.984 |
| 2048 | 224.388 | 26.971 | 256.554 | 11.668 |
| Full plane (4439) | 214.269 | 103.195 | 322.196 | 44.240 |

Here processing total includes source manifest writing, but excludes Python
startup and the subsequent full output comparison. Peak RSS is for each fresh
processing child, not the comparison process.

The fastest observed configuration was **2048**, reducing elapsed time by
22.3% relative to 512 (1.29x throughput). Full-plane reading saved about 10
seconds in the reader stage versus 2048, but spent about 76 seconds more in
resampling and consumed much more memory. Bigger tiles are not monotonically
faster with this algorithm.

For similar inputs, consider `--tile 2048 --workers 1` when approximately
12 GiB per processing worker plus safety headroom is available; 1024 is a
lower-memory compromise. Multiple workers multiply the approximate memory
requirement, and different source pixel calibration or dtype can change it
substantially. No production defaults were changed by this experiment.
This is one fixed-order pass with unflushed caches on a shared machine, not a
universal optimum. Raw results and full comparisons are in
`results/tile_benchmark_20261015/summary.json` under the dataset directory.

## GPU processing (daily use)

Run from the project root:

```bash
uv sync --extra gpu
uv run --extra gpu python Script/czi_ds/process_czi_gpu.py
```

This processes all CZI files directly inside the default dataset directory,
sequentially on GPU 0, and writes only downsampled single-channel CZIs to
`/mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/gpu`.
It does **not** run the CPU benchmark cases. Existing output CZIs or partial
outputs cause an error; files are not overwritten or automatically skipped.

To process one file:

```bash
uv run --extra gpu python Script/czi_ds/process_czi_gpu.py \
  --input /path/to/input.czi \
  --output /path/to/gpu_results \
  --target-um 10 --device 0 --batch-rows 256
```

`--input` also accepts multiple files or directories. Directory scanning is
nonrecursive and accepts both `.czi` and `.CZI`.

For a directory with multiple files, use two GPUs concurrently:

```bash
uv run --extra gpu python Script/czi_ds/process_czi_gpu.py \
  --input /path/to/czi_directory \
  --output /path/to/gpu_results \
  --workers 2 --batch-rows 256
```

`--workers` defaults to 1 and controls concurrent **files**, independently of
GPU count. The effective count is `min(workers, number_of_input_files)`.
GPUs are automatically detected and assigned to worker slots round-robin.
For example, four workers and two GPUs use assignments `[0, 1, 0, 1]`.
Each worker processes its assigned files/channels sequentially and reuses its
own GPU context. File assignment is round-robin, not dynamic load balancing.
Reductions due to too few input files are logged, and sharing GPUs emits a
memory-use warning. Worker failures propagate without automatic CPU fallback.

`--devices 0 1` restricts automatic assignment to those two CUDA GPUs.
`--device 1` restricts all workers to the second GPU, but does not reduce the
worker count. Thus `--device 0 --workers 4` allows four files concurrently
sharing GPU 0. The two selection flags are mutually exclusive. GPU indices
refer to CUDA-visible devices.

`--batch-rows` controls compute-band size, not output resolution; reduce it if
VRAM is limited. The current script default is 4096; the validated timing
example used an explicit 256. Multiple workers increase total host-memory and
VRAM requirements, and larger batches also increase upload-buffer memory.
Every process has independent buffers, a CUDA context and a CuPy memory pool,
even when GPUs are shared. Start by comparing 2 and 4 workers with
`--batch-rows 256`; shared network storage can saturate, so additional readers
may stop helping or reduce throughput. GPU sharing correctness is tested, but
no multi-worker real-file throughput speedup is assumed.

This uses the same validated `BatchedAreaMean` implementation as the GPU
experiment below, reusing the existing helper scripts in this directory.
Keep those files alongside this entry-point script.
Only scalar uint16/Gray16 inputs are currently supported by the GPU pipeline.
It reads a complete native plane into host memory once, then resamples in
bounded-VRAM bands to 10 um/pixel by default. Z spacing and other plane
dimensions are preserved as in the CPU pipeline. There is no automatic CPU
fallback if CUDA or its dependencies fail.

For the tested input, peak host RSS was approximately 15.6 GiB, while peak
reserved CuPy device pool was approximately 1.14 GiB. Allow additional headroom;
host memory scales with original plane size, not only GPU batch size.
Each source manifest includes device details, transfer/kernel timings and peak
reserved CuPy pool size. `session_setup_seconds` is the one shared GPU setup
cost repeated for context in each manifest, not an extra setup per file.

### Running with nohup

The `run_gpu.sh` launcher uses the project's `.venv/bin/python`, independent
of the current directory or shell activation. Install dependencies once with
`uv sync --extra gpu`. Its defaults are 4 file workers and 256 output rows per
GPU batch; additional CLI arguments override these defaults.
Python output is unbuffered and the launcher uses `exec`, so the recorded PID
belongs to the processing parent rather than an extra wrapper shell.

For example, from the project root:

```bash
mkdir -p logs
nohup bash Script/czi_ds/run_gpu.sh \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/gpu \
  --workers 4 --devices 0 1 --batch-rows 256 \
  > logs/czi_gpu.log 2>&1 < /dev/null &
echo $! > logs/czi_gpu.pid
```

Use a different log/PID filename for each submission to preserve earlier logs.
Do not run two jobs against the same output files concurrently.
Existing completed or partial output files still cause an error rather than
being overwritten or skipped.

Check startup and follow progress:

```bash
ps -p "$(cat logs/czi_gpu.pid)" -o pid,etime,stat,args
tail -n 50 logs/czi_gpu.log
tail -f logs/czi_gpu.log
```

Ctrl-C stops `tail -f`, not the processing job. `nohup` permits the job to
continue after SSH disconnects, but does not protect it against a machine
reboot, OOM termination or administrator/session cleanup policies.

## GPU full-channel experiment

```bash
uv sync --extra gpu
uv run --extra gpu python Script/czi_ds/benchmark_gpu.py \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds/sc-C042-149-O4213.czi \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/gpu_benchmark_20261015 \
  --batch-rows 256 --device 0
```

The output directory must be new. Four complete-file cases run sequentially
in fresh processes: CPU tile 2048, CPU full-plane, CPU full-plane with batched
resampling, and GPU full-plane with batched resampling. The extra CPU batched
case separates the effect of batching from the effect of the GPU.
The same reader, writer, output grid and area-average weights are used.
The experimental batched processors support scalar uint16 channels only;
unsupported data fails explicitly, without a CPU fallback.

The batched cases read each original-resolution plane into host memory once.
They then compute bands of 256 output rows (configurable), keeping the complete
source plane on the host. GPU processing uploads only the source rows needed
for each band. Two custom CUDA kernels apply the existing vertical and
horizontal CSR weights in float64 with fused multiply-add disabled; final
rounding/clipping uses the same CPU implementation. No approximate image
resize, normalization, projection or pyramid scaling is used.

GPU timings explicitly synchronize CUDA work. Reports include host-to-device
transfer, GPU kernel execution, device-to-host transfer, output conversion,
initialization, process peak host RSS and peak reserved CuPy memory-pool size.
The pool size excludes CUDA context and other driver allocations; it is not
the GPU's total board-wide memory consumption. Initialization includes optional
GPU imports and kernel compilation/cache loading and is included in total case
wall time. Full output comparisons occur after all processing timers.

The CPU script remains the default. This experiment uses one GPU and one file
at a time, does not overlap reading with GPU computation, and does not change
GPU clocks or flush shared system/network caches.

### CPU/GPU results (2026-10-15)

The complete same four-channel 5.24 GB input was processed in each case.
The GPU was an NVIDIA RTX A6000, device 0, using CuPy 14.2.0 and 256 output
rows per compute band. All 16 resulting single-channel CZIs passed checks;
each case's 63,957,112 output pixels exactly matched the fresh CPU tile-2048
reference, including output geometry, channel names and physical calibration.

| Case | Read (s) | Resample including transfers (s) | Total case wall (s) | Peak host RSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| CPU tile 2048 | 225.406 | 26.463 | 256.453 | 11.668 |
| CPU whole plane | 213.967 | 30.257 | 248.373 | 44.243 |
| CPU whole plane, batched compute | 212.024 | 26.964 | 242.978 | 14.224 |
| GPU whole plane, batched compute | 213.480 | 7.297 | 225.388 | 15.570 |

GPU detail across all four channels:

- Host-to-device uploads: 6.841 seconds.
- CUDA kernels: 0.219 seconds.
- Device-to-host downloads: 0.065 seconds.
- CPU output rounding/conversion: 0.162 seconds.
- GPU import/context/compile setup: 0.382 seconds, included in case wall time.
- Peak reserved CuPy device pool: 1169.091 MiB (1.142 GiB), excluding context
  and driver allocations.

The GPU case saved 31.1 seconds (12.1%) versus CPU tile 2048 and 17.6 seconds
(7.2%) versus the CPU full-plane batched control. Reading still dominates.
The whole-plane CPU resampling time in this fresh round was approximately
30 seconds rather than the earlier 103 seconds. The measured variability
means the earlier whole-plane slowdown is not a stable hardware limit;
shared-machine load/cache/memory conditions may affect results. Compare cases
within this round, and do not treat this one fixed-order pass as a universal
speedup guarantee. No CPU defaults were replaced with GPU processing.

Results and complete pixel comparisons are in
`results/gpu_benchmark_20261015/summary.json` under the dataset directory.
To run only the tested GPU pipeline on a new file, use `--case gpu_full_batched`
with a new `--output` directory; omit `--case` to repeat the four-way experiment.

The output channel index is always zero. Channel names, grayscale dtype,
Z spacing (when present), time points, other plane dimensions and scene indices
are retained. Source mosaics are rendered into rectangular scene images with
zero background in uncovered regions. Original tile layout and pyramid levels
are not copied. Acquisition metadata, stage positions, attachments and display
settings are not copied; these are derived images, not archival substitutes.
Scene XY origins are rounded to the nearest target pixel.
Each scene is resampled on its own local grid.

Area-weighted averaging is applied separately to every XY plane, including
noninteger scale factors. Integer intensities are rounded, not normalized.
Output size is `ceil(input_pixels * input_pixel_size / target_pixel_size)`.
The last row/column averages only the available input area; it is not padded
with black. Z is not resampled, projected or removed. Missing/invalid XY
calibration and requests requiring XY upsampling cause an explicit error.
Scalar Gray8, Gray16 and Gray32Float channels are supported.

Processing uses bounded-memory tiles, but reading full-resolution source
images can still take considerable time. Existing outputs are never
overwritten. Incomplete or failed outputs remain as `.partial.czi` for
inspection; remove only those specific files before retrying. A completed file
is published only after reopening it to check dimensions, pixel sizes and
pixel samples at both ends of every plane/scene. A per-source JSON manifest is
written once all its outputs pass verification.

Tests (standard-library unittest; no separate test dependency):

```bash
uv run python -m unittest discover -s Script/czi_ds -p 'test_*.py' -v
```
