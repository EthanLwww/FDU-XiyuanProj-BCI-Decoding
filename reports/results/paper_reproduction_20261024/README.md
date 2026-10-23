# 论文离线 EEGNet 修正版复现：阶段结果

本地快照时间：2026-10-24 02:41:57（北京时间）。这是阶段快照，不是全部实验完成记录。

已完成 S01、S02 的 ME 拇指/小指、4–40 Hz、trial 分组五折训练。
完整队列为 21 名被试 × ME/MI × 二/三/四分类，共 126 条件、630 个 fold；
采集快照时 S03、S04、S05 正在训练，无失败条件。
队列每完成一个条件，自动重算并核验远端对比表与图表。

## 修复内容

- 在 1024 Hz 原始信号上切 1024 点窗口、步长 128 点，精确对应 1 秒/125 ms。
- 先排除含 NaN 的原始窗口，再逐窗口重采样至 100 点，保留短 trial 中的有效部分。
- 对齐作者公开实现：CAR、四阶 Butterworth、零填充后 lfilter、裁剪、逐通道 z-score。
- 评估前加载最高验证窗口准确率 checkpoint，并核对预测准确率与保存轮次的日志。
- 按 trial 分组五折，重叠窗口不跨训练/验证折；按文件加载、磁盘缓存、batch=16。
- 图表生成前核验 checkpoint SHA256、折划分、OOF 覆盖、投票、汇总均值和日志。

12 项回归测试通过，包含窗口数量/数值、短 trial、分组隔离与异常结果检出。
S01 中 27 个短 trial 的有效窗口得到保留，未将其整体丢弃。

## 与原文相同被试对照

原文参照来自出版方 Source Data 的 `Fig. 6` 工作表，按被试 ID、任务、手指组合和频带匹配。
表内复现数值为五折准确率均值。两种复现指标必须分开理解。

| 被试 | 原文 Accuracy | 复现 trial 投票 | 复现窗口 | trial 差值 | 窗口差值 | 原文单元格 |
|---|---:|---:|---:|---:|---:|---|
| S01 | 84.13% | 88.81% | 77.19% | +4.68 pp | −6.94 pp | K34 |
| S02 | 86.25% | 90.00% | 78.33% | +3.75 pp | −7.92 pp | K35 |
| 同被试均值，n=2 | 85.19% | 89.40% | 77.76% | +4.21 pp | −7.43 pp | — |

不能据此前两名被试宣称总体超过原文。原文未完整公开离线折索引、checkpoint 选择和窗口/trial 汇总脚本；
这里同时保留两种复现口径。held-out fold 同时参与回调和模型选择，因此这些是 CV 验证成绩，**没有独立测试集**。
不得与原文 80.56% 的在线 MI 成绩混为同一实验。

## 图表与证据

- [原文与同被试复现对照](figures/paper_vs_reproduction_matched.png)
- [五折 trial 箱线图与窗口/trial 对照](figures/corrected_pilot_accuracy.png)
- [五折验证 acc/loss 曲线](figures/corrected_pilot_learning_curves.png)
- [Fig. 6 风格 trial 图](figures/fig6_style_trial_accuracy.png)
- [Fig. 6 风格窗口图](figures/fig6_style_window_accuracy.png)
- [逐条件精确数值与差值](paper_comparison.csv)、[核验结果](verification.json)、[快照进程状态](snapshot.json)

箱线图的 pilot 点为同一被试的五个 fold，不是五名独立被试。
Fig. 6 风格图每点为一名被试的五折均值，未完成的 MI 面板明确标注 Pending。
本阶段仅运行宽频带 EEGNet 的三个代表手指组合；单频带扫描、其余手指组合、FBCSP、ERD 和在线闭环尚未复现。

`conditions/` 内保留数据审计、划分、逐轮历史、逐 trial 投票和指标。
完整预测与 10 个已完成 checkpoint 在本地：
`experiment_results/paper_reproduction_fixed_20261024/snapshots/20261023T184157Z/`。
下载包 SHA256：`c2d9d10b520d1b553dbb18d9762e3a29f69e5ee51bda80ce10ba5c5b4dcf97d1`，已校验 93 个文件。
原始 EEG 与缓存未下载、未纳入版本库。

## 继续运行的位置

远端实验：`/home/kxy/LYC/experiments/paper_reproduction_fixed_20261024`

远端隔离源码：上述目录下 `code/`。训练和自动图表任务已经运行，检查状态时不要重复启动。

```bash
ssh zhanglihua 'cat /home/kxy/LYC/experiments/paper_reproduction_fixed_20261024/suite.status.json'
```

训练环境：`/home/kxy/anaconda3/envs/bci_wjd_20261022/bin/python`。
绘图环境：`/home/kxy/anaconda3/bin/python`。
进一步的参数和评估限制见 `docs/PAPER_REPRODUCTION.md`。
