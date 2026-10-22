# 目录与启动方式

根目录保留三个启动脚本，单被试与多被试训练合并为 `train.py`。原参数顺序和开关保持一致，只需更换命令中的脚本名称。

```text
train.py                     单/多被试预训练和微调
run_experiments.py           实验调度与结果汇总
main_online_processing.py   BCPy2000 在线解码
bci/                        共用模型、预处理、训练及记录工具
experiment_runners/         三配置消融、重复实验
scaling_experiments/        数据量实验、流式加载及分析
tests/                      统一回归测试
reference_models/           未接入正式训练入口的参考实现
EEG-Conformer-main/         上游参考代码及其许可证
docs/                       插件说明、目录说明、历史设计记录
reports/                    PPT、draw.io 和正式实验结果
Pictures/                   早期图表
```

`bci/EEGModels_tf.py`、`bci/FilterBankModels_tf.py` 是正式模型定义；`bci/Functions.py` 提供数据读取与训练；`bci/filter_bank.py` 和 `bci/experiment_window_cache.py` 提供频带预处理与窗口缓存。回调、checkpoint 元数据、CLI 与源码快照也集中在 `bci/`。

原先根目录与 `scaling_experiments/` 中完全相同的六份模块已合并。不同实验协议的调度实现仍各自保留，避免混淆普通模型复用 checkpoint 和残差模型的额外核验。

## 命令迁移

| 原命令 | 新命令 |
|---|---|
| `python main_model_training.py ...` | `python train.py ...` |
| `python main_multi_subject_training.py ...` | `python train.py ...` |
| `python run_conformer_ablation.py ...` | `python run_experiments.py ablation ...` |
| `python repeated_experiments.py ...` | `python run_experiments.py repeats ...` |
| `python scaling_experiments/run_scaling.py ...` | `python run_experiments.py scaling ...` |
| `python scaling_experiments/run_scaling_residual.py ...` | `python run_experiments.py scaling-residual ...` |
| `python scaling_experiments/summarize_scaling.py ...` | `python run_experiments.py summarize ...` |
| `python scaling_experiments/summarize_residual.py ...` | `python run_experiments.py summarize-residual ...` |
| `python scaling_experiments/compare_scaling_models.py ...` | `python run_experiments.py compare ...` |

`train.py` 接受一个被试（例如 `20`）或多个被试（例如 `1,2,5`）。默认 `Data/`、`Save/` 仍相对于仓库根目录。具体参数用 `--help` 查看；批量实验保留原服务器路径与协议，运行前检查数据路径和 GPU。

Python 导入改为显式包名，例如 `from bci.FilterBankModels_tf import build_model`。加载已训练的自定义层模型时先执行 `from bci import FilterBankModels_tf`，HDF5 checkpoint 及其伴随配置无需搬动。

本地一次性计数/上传脚本及旧主机记录归档到 `.local-tools/`，由 Git 忽略。数据、权重、完整实验目录、缓存和日志继续留在原路径，不上传。

## 验证

```bash
CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 MPLBACKEND=Agg \
  python -m unittest discover -s tests -v
```

测试覆盖合成数据训练/微调、模型保存重载、流式加载、从其他工作目录启动 CLI，以及实验源码快照脱离仓库后的独立启动。BCPy2000 在线入口需在原硬件环境中运行。
