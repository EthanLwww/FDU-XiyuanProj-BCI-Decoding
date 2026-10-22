# FDU-XiyuanProj-BCI-Decoding

基于 [bfinl/Finger-BCI-Decoding](https://github.com/bfinl/Finger-BCI-Decoding) 的手指 EEG 解码研究项目，包含 EEGConformer、可插拔频带前端、宽带残差，以及跨被试预训练和微调实验。

## 当前实现

- 四子带：4–8、8–13、13–30、30–40 Hz，各自独立提取特征，支持可学习加权、均值和拼接融合。
- 可选 4–40 Hz 宽带残差，支持固定相加或 sigmoid 门控，初始门控系数为 0.1。
- 先划分 trial 再切分窗口，避免重叠窗口跨训练与验证集。
- 大规模实验使用磁盘缓存、内存映射和分批加载，保存逐轮指标、配置和模型验证记录。

实际训练入口通过 `bci/EEGModels_tf.py` 与 `bci/FilterBankModels_tf.py` 构建模型。`reference_models/EEGConformer_tf.py` 和 `EEG-Conformer-main/` 保留既有参考实现，不能直接当作本轮实验的模型定义。

## 环境与使用

已在 Linux、Python 3.11、TensorFlow 2.17.0 环境验证。数据集和模型权重单独存储，不包含在 Git 仓库中。

```bash
python -m pip install -r requirements-filterbank.txt
python train.py --help
python run_experiments.py --help
```

多被试预训练示例，将数据根目录指向包含 `S01` 等目录的位置：

```bash
python train.py 1,2,5 2 2 ME Orig EEGConformer \
  --filter-bank weighted --broadband-residual gated \
  --depth 1 --windowlen 3 --seed 42 \
  --data-dir /Dataset2/dataset_bci_decode_kxy/data \
  --save-dir ./Save
```

| 配置 | `--filter-bank` | `--broadband-residual` |
|---|---|---|
| 普通模型 | `off` | `off` |
| 频带 | `weighted` | `off` |
| 频带＋残差 | `weighted` | `gated` |

微调使用 `Finetune` 和 `--init-model` 指定兼容的预训练权重。详细说明见 [频带前端](docs/FILTER_BANK.md) 和 [宽带残差](docs/BROADBAND_RESIDUAL.md)。

根目录只保留三个启动脚本：

| 入口 | 用途 |
|---|---|
| `train.py` | 单被试、多被试预训练与微调；subjects 为单个 ID 或逗号分隔的 ID |
| `run_experiments.py` | 消融、重复实验、数据量对照及结果比较 |
| `main_online_processing.py` | BCPy2000 在线 EEGNet 解码入口，由 BCPy2000 加载 |

```bash
python run_experiments.py ablation --help
python run_experiments.py repeats --help
python run_experiments.py scaling --help
python run_experiments.py scaling-residual --help
python run_experiments.py compare --help
```

实验实现位于 `experiment_runners/` 和 `scaling_experiments/`，共用 `bci/` 内的模型、预处理和训练工具。默认路径、GPU 编号和参考 checkpoint 对应实验服务器，迁移环境时需要调整。模型结构、数据划分、超参数和已有结果不因本次目录整理而改变。目录职责与旧命令迁移见 [目录说明](docs/REPOSITORY_LAYOUT.md)。

## 已完成实验

统一任务为 ME 拇指/小指二分类，EEGConformer depth=1，3 秒窗口。

| 实验 | 预训练 | 微调 | 重复 |
|---|---|---|---|
| 三配置消融 | S01/S02/S05 | S20 Session 2 Base | 每种 10 次，seed 42–51 |
| 普通模型数据量对照 | 3 / 5 / 10 / 20 名被试 | S21 Session 2 Base | 每组 1 次，seed 42 |
| 频带残差数据量对照 | 相同四组 | 相同 S21 划分 | 每组 1 次，seed 42 |

3 人参考组为 S01/S02/S05，其余为 S01–S05、S01–S10 和 S01–S20。

- 十次消融的 S20 平均验证准确率：普通 62.92%，频带 63.11%，频带＋残差 64.80%。
- S21 普通模型：55.88%、58.82%、64.71%、67.65%。
- S21 频带残差：66.18%、66.18%、69.12%、79.41%。

均为最佳验证准确率 checkpoint 的结果，没有独立测试集。十个种子不等于十个独立被试，单次数据量对照不能推断统计显著性。完整插件同时改变空间卷积、归一化、参数量和微调参数，收益不能全部归因于频带划分。

正式报告、精简 PPT 和 draw.io 架构图见 [reports/](reports/)。可用于重新绘图的指标、逐轮历史和图表见 [reports/results/](reports/results/)。早期图表保留在 `Pictures/`，与上述受控实验分开归档。完整实验目录、权重、原始数据、缓存、日志及临时探测文件不上传。

## 回归验证

`tests/` 统一存放模型、训练入口、流式加载和源码快照的正式回归测试。使用合成数据与临时目录，不启动完整数据集实验。

```bash
CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 MPLBACKEND=Agg \
  python -m unittest discover -s tests -v

```

## 上游来源与许可

本仓库新增实验不属于原论文结果。原始项目说明单独保留在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。保留根目录 [MIT LICENSE](LICENSE) 与 `EEG-Conformer-main/` 的原许可证。

以下为上游来源说明：

Optional four-band EEGNet/EEGConformer frontend: see [FILTER_BANK.md](docs/FILTER_BANK.md) for switches, reproducible comparisons, checkpoints and verification.

Optional broadband residual branch: see [BROADBAND_RESIDUAL.md](docs/BROADBAND_RESIDUAL.md) for fixed-addition and learnable-gate ablations.

The upstream repository provides scripts used for the study: “EEG-based Brain-Computer Interface Enables Real-time Robotic Hand Control at Individual Finger Level”.

Ding, Y., Udompanyawit, C., Zhang, Y., & He, B. (2025). EEG-based brain-computer interface enables real-time robotic hand control at individual finger level. Nature communications, 16(1), 5401. https://doi.org/10.1038/s41467-025-61064-x

The scripts include the online processing and decoding as well as offline deep learning model training. EEGNet is used in this study and the tensorflow implementation of EEGNet is downloaded from the Army Research Laboratory (ARL) EEGModels project [1].  

This work was supported by the National Institutes of Health via grants NS124564, NS131069, NS127849, and NS096761 to Dr. Bin He.

[1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of neural engineering, 15, 056013. (2018).

Army Research Laboratory (ARL) EEGModels project repository: https://github.com/vlawhern/arl-eegmodels
