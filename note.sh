uv run python Script/czi_ds/process_czi.py \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/test \
  --target-um 10 \
  --tile-size 2048 \
  --workers 4

uv run --extra gpu python Script/czi_ds/process_czi_gpu.py \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/test-1 \
  --workers 4 \
  --devices 0 1 \
  --batch-rows 4096



mkdir -p /home/chaiqw/project/multi-test/logs

nohup bash /home/chaiqw/project/multi-test/Script/czi_ds/run_gpu.sh \
  > /home/chaiqw/project/multi-test/logs/czi_gpu.log 2>&1 < /dev/null &

echo $! > /home/chaiqw/project/multi-test/logs/czi_gpu.pid



uv run --extra gpu python /home/chaiqw/project/multi-test/Script/czi_ds/process_czi_gpu.py \
--input /mnt/99-raw-data9-2/C100 \
--output /mnt/93-zeiss3/20231222-low-resolution/C100-downSample \
--workers 8 \
--devices 0 1 \
--batch-rows 4096