# 实验结果归档

本目录从完整实验目录中提取可用于审阅和重新绘图的正式结果。

- `repeats10/`：三种模型各 10 次的原始指标、统计、逐次历史与箱形图。
- `scaling_baseline/`：普通 EEGConformer 四种预训练规模，S21 微调。
- `scaling_residual/`：频带残差四种规模，以及与普通模型的配对结果。

保留原始报告中的实验路径作为来源记录，它们不是当前仓库内的相对路径。完整模型、划分数组、原始数据清单和资源日志保留在服务器 `/home/kxy/LYC/experiments/` 及本地被 Git 忽略的 `experiment_results/`。Git 归档不包含 EEG 数据和训练权重。

均使用验证集选择 checkpoint，没有独立测试集。准确率与对应 loss 来自同一 checkpoint，不能用最低 loss 替换对应值。
