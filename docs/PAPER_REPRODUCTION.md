# 原论文离线 EEGNet 复现

对应 Ding et al., Nature Communications 16, 5401 (2025)，DOI：10.1038/s41467-025-61064-x。
本实现对齐作者公开的 `Functions.py` / EEGNet-8,2，并重建 Methods 中的离线五折实验。
作者未公开 Fig. 6 折索引、分组细节、随机种子和完整分析脚本；这里按 trial 分组以避免相互重叠的窗口跨折泄漏，因此不能宣称逐项完全一致。

## 管线修复

| 初版问题 | 修正版行为 |
|---|---|
| 100 Hz 上将 12.5 点取整为 12 点，得到 120 ms | 在 1024 Hz 上移动 128 点，精确为 125 ms |
| 整段重采样，NaN 尾部污染全部数据 | 先切 1024 点窗口，删除无效窗口，再重采样至 100 点 |
| filtfilt 双向滤波 | 四阶 Butterworth b/a、100 点零填充、lfilter、裁剪 |
| 保存最高 val_accuracy 却评估最低 val_loss 权重 | 加载最高 val_accuracy 权重，验证重新评估值与历史一致 |
| 部分 trial 静默消失 | 保留源文件 SHA256、trial ID、原始窗口起点、无效 trial 清单与折划分 |
| 一次性加载全部数据 | 每次读取一个 MAT，磁盘缓存，内存映射，batch=16，单批预取 |

完整 5 秒 trial 为 33 个窗口；3 秒有效数据加 NaN 尾部保留 17 个窗口。
没有有效窗口的 trial 列入 `data_audit.json`，不以零生成伪样本。

## 实验设置

- 仅 `OfflineMovement` / `OfflineImagery`，不混入在线 session。
- 128 通道、1024 Hz、CAR、1 秒输入、125 ms 步长、100 Hz 模型采样率、4–40 Hz。
- 原始 EEGNet-8,2：kernLength=32，F1=8，D=2，F2=16，dropout=0.5。
- Adam legacy(lr=0.001)，batch=16，最多 300 epochs。
- EarlyStopping(val_loss, patience=80)，ReduceLROnPlateau(val_loss, factor=0.5, patience=30)。
- 加载 ModelCheckpoint(val_accuracy, max) 保存的权重后评估。
- StratifiedKFold 在有效 trial 上五折划分，shuffle=True，seed=42；fold 初始化 seed=42–46。
- 二分类 1/4，三分类 1/2/4，四分类 1/2/3/4；1=拇指，2=食指，3=中指，4=小指。
- 同时记录窗口准确率和 trial 多数投票准确率；平票固定选编码较小者并记录数量。

**评估限制：** held-out fold 也用于回调和 checkpoint 选择，是 CV 验证成绩，无独立测试集或嵌套 CV。
作者未完整公开离线模型选择与窗口/trial 汇总细节，因此同时输出两种指标。
不得将离线 ME 的准确率直接与论文 80.56% 的在线 MI 二分类成绩比较。

## 运行

```bash
python run_experiments.py paper \
  --data-root /Dataset2/dataset_bci_decode_kxy/data \
  --cache /path/to/reproduction/cache --out /path/to/reproduction/S01_ME_4-40 \
  --subject 1 --task ME --labels 1 4 --band 4 40
python -m experiment_runners.paper_report /path/to/reproduction
python -m unittest discover -s tests -p test_paper_reproduction.py -v
```

训练需要 TensorFlow、NumPy、SciPy、scikit-learn；绘图另需 matplotlib。
本次训练使用远端 `bci_wjd_20261022` 环境（TensorFlow 2.15.1、SciPy 1.10.1、scikit-learn 1.3.2），
绘图使用已有 base 环境，通过 `PAPER_REPORT_PYTHON` 指定，未改动共享环境。

Linux 队列 `python -m experiment_runners.paper_suite` 接受 `--gpus` 物理 GPU UUID，
仅在显存占用不超过 500 MiB 时派发；文件锁防止重复启动 controller。
`--scope binary` 为 42 条件，`--scope core` 为 126 条件，`--scope frequency` 增加四个单频带。
本次实际启动 **core**：21 名被试 × ME/MI × 二、三、四分类，全部 4–40 Hz。
频带扫描、FBCSP、ERD 拓扑和在线闭环图表未包含在本阶段。

远端目录：`/home/kxy/LYC/experiments/paper_reproduction_fixed_20261024`。
旧 pilot 保留，不混入修正版汇总。

## 输出与核验

- `suite.plan.json`、`suite.status.json`、`resources.jsonl`：队列、进程、资源。
- 各条件 `data_audit.json`：源文件、短 trial、无窗口 trial、实际样本数量。
- 各 fold：`split.json`、`history.csv`、`best.weights.h5`、`predictions.npz`、`trial_predictions.csv`。
- `summary.csv`、`verification.json`：已完成结果，权重 SHA256、窗口起点、OOF 覆盖和投票重算核验。
- `paper_comparison.csv`：匹配同一被试、任务、手指组合和频带的原文数值，以及 trial/window 两种差值（百分点）。
- `figures/`：修正版 pilot 箱线图、学习曲线、按实际已完成 n 更新的 Fig. 6 风格图。

Fig. 6 风格图的统计单位为被试五折平均；pilot 箱线图为同一被试的五个 fold，两者不能混为独立重复。
所有不完整条件标实际 n，不将部分被试标为 n=21。

图表生成前还会核验每个 trial 的投票覆盖、每个窗口恰好一次的 OOF 覆盖、checkpoint 轮次与日志、
汇总均值和标准差与逐窗口预测的一致性。导出的完整快照包含权重及预测，逐文件 SHA256 校验；不包含原始 EEG 和缓存。

## 原文参照与本地结果

原文参照取自出版方的 [Source Data（xlsx）](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-61064-x/MediaObjects/41467_2025_61064_MOESM9_ESM.xlsx)，
`Fig. 6` 工作表 E/F 面板，包含明确的被试编号。图注确认 EEGNET_2 / EEGNET_3 / EEGNET_4 分别对应 1-4、1-2-4、1-2-3-4。
提取的 630 条 EEGNet 数值及单元格引用存于 `reports/results/paper_reproduction_20261024/paper_fig6_reference.json`。

原文全 21 被试、4–40 Hz 均值如下；部分复现被试只能与原文相同被试子集比较：

| 任务 | 1-4 | 1-2-4 | 1-2-3-4 |
|---|---:|---:|---:|
| ME | 75.65% | 51.35% | 42.17% |
| MI | 77.58% | 55.55% | 43.61% |

当前已下载的核验快照、对比表及图表见 `reports/results/paper_reproduction_20261024/`。
其中 `snapshot.json` 记录采集时刻与运行状态，不能将它视为实时服务器状态。
完整权重和预测快照在本地忽略目录 `experiment_results/paper_reproduction_fixed_20261024/snapshots/`，
`download_verification.json` 记录下载包及其 SHA256。
