# CZI 通道拆分与降采样

读取多通道 CZI 文件，将每个通道分别保存为一张 **10 µm/pixel** 的 CZI。
只调整 XY 分辨率，保留 Z 层和时间点，不保存原始分辨率副本。

## 1. 安装环境

在项目目录执行，需要 NVIDIA GPU 和兼容的驱动：

```bash
uv sync --extra gpu
```

当前 GPU 流程支持 uint16（Gray16）通道。

## 2. 填写任务表

复制并修改 [任务表示例](Script/czi_ds/jobs.example.csv)，例如保存为项目根目录下的 `jobs.csv`：

```csv
input_dir,output_dir
/data/C100,/results/C100
/data/C101,/results/C101
```

第一行不变，每行填写一组输入文件夹和对应的输出文件夹。
程序处理输入文件夹第一层的全部 `.czi` 文件，不扫描子文件夹。
不同输入文件夹中的同名文件应使用不同的输出文件夹。

## 3. 后台运行

先打开 [提交脚本](Script/czi_ds/run_gpu_table.sh)，修改顶部的参数：

```bash
JOBS_CSV="$SCRIPT_DIR/jobs.csv"  # 改成自己的 CSV 路径
WORKERS=8
DEVICES=(0 1)
BATCH_ROWS=4096
TARGET_UM=10
```

然后只需执行：

```bash
bash /home/chaiqw/project/multi-test/Script/czi_ds/run_gpu_table.sh
```

- `--workers 8`：最多同时处理 4 个 CZI 文件。
- `--devices 0 1`：使用两张 GPU，多个 worker 可以共用一张 GPU。
- `--batch-rows 4096`：每批计算的输出行数，不影响最终分辨率。
- `--target-um 10`：输出 XY 像素尺寸为 10 µm/pixel。

默认使用 8 个 worker、每批 4096 行、10 µm/pixel，使用 GPU 0 和 1。
只有一张 GPU 时设置 `DEVICES=(0)`；自动选卡设置 `DEVICES=()`。
任务表默认位置为脚本旁的 `jobs.csv`，可复制任务表示例后填写，
或将 `JOBS_CSV` 改成现有任务表的绝对路径。
也可在提交命令后追加参数，覆盖脚本中的同名设置。
脚本自动使用 nohup 后台运行，提交后立即返回终端，无需再写 nohup 或 `&`。

确认日志没有报错后，可以正常断开终端。重启、断电或内存不足仍可能中断任务。
不要重复提交写入相同输出文件的任务。

## 4. 查看进度

```bash
tail -f /输出目录/logs/日志文件名.log
```

按 `Ctrl+C` 只退出日志查看，不会停止后台任务。
每个输出文件夹的 `logs` 子目录保存对应文件的处理日志（追加写入）。
整批日志、JSON 报告和 PID 放在任务表第一个输出文件夹的 `logs` 中，
按提交时间自动命名，不会写到代码目录。提交时会显示实际路径。

## 5. 中断后继续运行

重新执行相同任务即可：

- 已有结果通过完成凭据、校验和、尺寸和分辨率检查后，跳过对应通道。
- 损坏、不完整或无法确认的结果，直接删除后重新生成对应通道。
- 单个文件失败会记录错误并继续处理其他文件，最后汇总失败情况。

请保留输出旁边的 `.complete.json` 文件，这是判断通道已完成的凭据。
以前没有此凭据的结果会重新生成一次。续跑以通道为单位，不支持从通道中间继续。

**资源提醒：**类似测试片子每个 worker 可能需要十几 GiB 主机内存；
同时增加 worker 和批次行数会进一步增加内存及显存占用。

更多说明见 [脚本目录文档](Script/czi_ds/README.md)。