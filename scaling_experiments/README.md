# EEGConformer 预训练被试数量实验

本目录保存数据量实验的调度、流式加载和结果分析；模型、预处理、checkpoint 与回调统一复用 `../bci/`，不再维护重复副本。历史实验结果保持原样。

- 数据根目录：`/Dataset2/dataset_bci_decode_kxy/data`
- 远端结果：`/home/kxy/LYC/experiments/conformer_scaling_S21_20261021`
- 预训练：S01/S02/S05（复用 seed=42）、S01–S05、S01–S10、S01–S20。
- 四组统一在 S21 Session 2 的 2class_Base、3class_Base 上微调。
- 仅宽带 EEGConformer，depth=1、3 秒、ME 拇指/小指、seed=42，不重复。

后续 `run_scaling_residual.py` 使用同样四组和 S21 划分运行 weighted filter-bank + gated broadband residual，对比由 `compare_scaling_models.py`、`finish_residual_results.py` 生成。两轮均已完成，轻量指标与曲线归档在 `../reports/results/`。脚本保留服务器路径、GPU 编号和既有 checkpoint 依赖，新环境运行前需调整。

`streaming_data.py` 逐 MAT 文件读取、预处理，按被试写入 float32 窗口文件。
`WindowStore` 使用内存映射按索引读取 batch，不建立整份训练集的 CPU/GPU Tensor。
每个 epoch 对完整训练窗口索引做随机排列，batch=32，预取一个 batch，保留末尾不足 32 的批次。
先按 trial 做 80/20 划分，重叠窗口不会跨训练/验证集。

`python run_experiments.py scaling suite` 准备数据并安排训练。每个阶段保存 checkpoint、config、history、split、progress 和 verification。
缓存使用完成标记，失败的缓存不会供训练使用。中断后重新执行 suite 会复用完成阶段并接管仍运行的 worker。
不要同时启动多个 suite；文件锁会阻止重复调度。

远端继续或核验：

```bash
cd /home/kxy/LYC/Finger-BCI-Decoding-main
OPENBLAS_NUM_THREADS=1 /home/kxy/anaconda3/bin/python -m unittest discover -s tests -p test_streaming.py
OPENBLAS_NUM_THREADS=1 /home/kxy/anaconda3/bin/python run_experiments.py scaling suite
```

单独训练阶段需设置空闲 GPU、线程数和项目原有 cuDNN 路径，见 `run_scaling.py::env_for`。
监督器会在 GPU 0–2 运行，TensorFlow 每个训练进程显存上限 16 GiB，CUDA 上下文另计。
开始新一轮之前应检查 GPU 是否仍空闲。

信号转换保留原 CAR、重采样、4–40 Hz 滤波、逐窗口 z-score 数值流程。
原转换结果为 float64，但旧模型也将输入转为 float32，因此磁盘缓存采用 float32。
`python -m scaling_experiments.verify_streaming_real` 在真实文件上逐元素比较新旧结果。

本轮发现 S09 Session 1 3class_Finetune R07 有一个无对应 Target 的 TrialEnd。
按事件顺序匹配 trial 后忽略并记录该孤立结束标记，避免原按序号匹配造成后续 trial 边界错位。
所有文件的事件扫描结果保存在 `event_pairing_audit.json`，逐文件处理记录在缓存 manifest 中。

报告由 `summarize_scaling.py` 生成，包含四组 loss/acc 叠加曲线、数据量对比图和单次原始结果。
本轮没有独立测试集，也没有重复实验方差。预训练数据量、被试构成和同等 epoch 下的优化步数同时改变。

## 整理后的调用方式

所有命令从仓库根目录运行。包内脚本使用 `python -m`，不再进入本目录直接运行 `.py` 文件。

```bash
python run_experiments.py scaling --help
python run_experiments.py scaling-residual --help
python run_experiments.py summarize --help
python run_experiments.py summarize-residual --help
python run_experiments.py compare --help
# 以下为实际数据核验 / 监控 / 打包操作，不属于合成数据回归测试
python -m scaling_experiments.verify_streaming_real
python -m scaling_experiments.preflight_residual
python -m scaling_experiments.monitor_residual
python -m scaling_experiments.finish_residual_results
```

调度器将 `bci/`、`experiment_runners/` 和 `scaling_experiments/` 一起保存到实验 `source/`，子进程在该快照中按模块启动。`source_sha256.json` 记录相对路径与哈希，更新快照时旧版本保留到 `source_revisions/`。原先只有平铺脚本的历史快照仍保留在实验归档中。
