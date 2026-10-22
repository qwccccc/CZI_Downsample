uv run --extra gpu python Script/czi_ds/process_czi_gpu.py \
  --input /mnt/90-connectome/Personal/CQW/datasets/czi_ds \
  --output /mnt/90-connectome/Personal/CQW/datasets/czi_ds/results/test-1 \
  --workers 4 \
  --devices 0 1 \
  --batch-rows 4096