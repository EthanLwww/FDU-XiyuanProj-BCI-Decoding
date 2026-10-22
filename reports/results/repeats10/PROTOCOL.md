# EEGConformer 十次重复实验

每种配置运行 10 个随机种子：42–51。
seed=42–46 复用已经完成并核验的前一批实验；新增 seed=47–51，共 15 组预训练＋微调。
三种配置是 broadband baseline、weighted filter bank、
weighted filter bank + gated broadband residual。

沿用原实验：ME 拇指/小指二分类，S01/S02/S05 多被试预训练，
S20 Session 2 Base 数据微调，depth=1，windowlen=3 s，128 通道，
降采样 100 Hz，batch=32。每个种子同时改变初始化和 80/20 trial 划分；
同一种子内三种配置共用完全相同的 trial 与窗口划分。

滤波、标准化均逐窗口进行，不依赖数据划分。共享预处理缓存已通过小数据
与真实数据的逐元素等价性检查，保留原流程的窗口顺序和数值精度。
训练与验证 trial 无交集。其他学习率、早停、微调冻结策略和 checkpoint
选择规则保持第一次实验的设置。

每个箱形图包含三个模型的箱体，分别展示预训练和微调两个阶段。
每个点代表一次完整实验，不把 epoch 当作独立样本。
Accuracy 指标为各次最高验证准确率；loss 使用同一个 accuracy 最佳
checkpoint 的验证 loss。另行保存每次最低 val loss，避免混淆两种指标。
叠加原始散点和种子标签，配对连线表示同一种子。
输出均值、样本标准差（ddof=1）、中位数、四分位数和每次原始结果。

这些结果反映初始化与随机划分共同造成的波动，属于验证集结果，
没有独立测试集。每个 seed 在统计中只计一次。

远端项目：`/home/kxy/LYC/Finger-BCI-Decoding-main`。
远端结果：`/home/kxy/LYC/experiments/conformer_depth1_win3_ME2_repeats10_20261020`。
原始数据：`/Dataset2/dataset_bci_decode_lyc`。

结果目录按 `seed_XX/variant/` 组织。每个目录包含两个阶段的 checkpoint、
config、逐轮 history、trial/window split，以及完整验证集复算记录。
`source/` 为本轮实际运行代码；`source_sha256.json` 用于验证源码。
seed=42 的源码与记录保留在第一次实验目录。

恢复未完成任务或重新绘图（在远端结果目录中）：

```bash
/home/kxy/anaconda3/bin/python source/repeated_experiments.py suite
/home/kxy/anaconda3/bin/python source/repeated_experiments.py plot
```
