# 可开关的滤波器组前端

整频带残差插件现已可用：见 [BROADBAND_RESIDUAL.md](BROADBAND_RESIDUAL.md)。通过 `--broadband-residual off|fixed|gated` 开关，默认关闭；开启后新增第 5 路整频带输入。

远端项目：`/home/kxy/LYC/Finger-BCI-Decoding-main`。
独立上游 baseline：`/home/kxy/LYC/Finger-BCI-Decoding-baseline`，未修改。

## 实验开关

统一的 `train.py` 入口接受 `--filter-bank off|mean|weighted|concat`，默认 `off`。

| 模式 | 预处理与前端 | 融合 |
|---|---|---|
| off | 原 [4,40] Hz 单带预处理、原模型 | 无额外融合 |
| mean | 四频带、独立时空分支 | 固定等权平均 |
| weighted | 与 mean 相同 | 4 个可训练 logits 经 Softmax 加权求和 |
| concat | 与 mean 相同 | 特征拼接；Conformer 投影回 40 维 |

固定频带顺序：4–8、8–13、13–30、30–40 Hz。输入由 `(N,C,300,1)` 变为 `(N,C,300,4)`，C 从数据读取，不固定为 64。窗口仍为 3 秒，重采样至 100 Hz。

每个频带单独执行时间卷积、沿电极维的 DepthwiseConv2D、BN/ELU/池化，融合前不混合频带。EEGNet 保留每个分支的可分离卷积；Conformer 每带产生 14×40 token，在融合后使用一套共享的 3 层、10 头 Transformer 和原尺寸分类头。融合前使用无可训练缩放/偏置的 LayerNorm。

这是受 filter-bank 思路启发的深度模型，不等同于经典 FBCSP 的 CSP + 对数方差管线。Conformer 开启插件时的空间卷积由普通卷积改为 depthwise，因此 off 对 weighted 比较的是整个插件；mean 对 weighted 更适合单独检验学习频带权重的收益。

## 运行

以下是在远端 shell 中执行的命令。已有可用解释器为 `/home/kxy/anaconda3/bin/python`（本次测试 TensorFlow 2.17.0）；不使用项目内从 Windows 复制来的虚拟环境。

```bash
cd /home/kxy/LYC/Finger-BCI-Decoding-main

# 当前模型对照
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank off --seed 42

# 开启学习频带权重
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank weighted --seed 42

# 等权对照：检验学习权重是否优于固定均值
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank mean --seed 42

# 拼接对照
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank concat --seed 42

# 多被试；将架构名换为 EEGNet 即可使用 EEGNet
/home/kxy/anaconda3/bin/python train.py "1,2,3" 2 2 ME Orig EEGNet --filter-bank weighted --seed 42
```

默认数据目录为项目 `Data`（存在时），否则 `/Dataset4/bci`；使用 `--data-dir` 覆盖。
输出默认在 `Save`，使用 `--save-dir` 覆盖。
调试时可加 `--epochs 1 --batch-size 8`，正式实验默认预训练 300 轮、微调 100 轮，保留早停和学习率衰减。

默认不指定种子时保留原随机行为。比较不同模式时使用相同 `--seed`、相同数据文件、被试、session、训练轮数及其他参数；种子固定试次划分和初始化随机数，跨硬件/框架版本不保证位级相同。多被试仍是合并后按试次划分，并非留一被试评估。

输出名例如：
`S20_Sess02_ME_2class_EEGConformer_FB-weighted_Seed42_Orig.h5`。
off 不带 FB 后缀；不传新参数时单被试保留旧文件名。多被试文件名包含具体被试 ID，避免同人数不同被试组混淆。相同配置再次运行仍会覆盖同名文件，请用独立 `--save-dir` 保留重复实验。

## 微调

```bash
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Finetune EEGConformer --filter-bank weighted --seed 42

# 从指定多被试模型微调
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Finetune EEGConformer --filter-bank weighted --seed 42 --init-model /absolute/path/to/multi_subject_Orig.h5
```

默认寻找同一配置名称的 Orig 模型，`--init-model` 可显式指定来源。
滤波器组模型必须同时携带其 `.config.json`；模式、频带、采样率、窗口和结构必须匹配。不能直接把单带权重加载到四带模型。参数冻结发生在 compile 前：保留末 12 层可训练，并额外允许 weighted 融合 logits 更新。

## 结果文件和权重解释

与 checkpoint 同名前缀：
- `.config.json`：实际配置、数据源文件清单（路径/大小/修改时间）、框架版本、参数量、最佳验证准确率、checkpoint SHA256。
- `.split.npz`：原始试次训练/验证索引、每个窗口对应的试次索引和试次标签，便于核对无窗口泄漏。
- `.history.json`、`_history.png`：训练曲线。
- `.band_weights.json`：weighted 的最佳 checkpoint 全局权重，或 mean 的固定等权值；off/concat 无此文件。

权重按固定频带顺序排列，和为 1。它表示模型偏好，不直接证明某个手指、被试或脑区的因果重要性。每窗口/通道/频带的 z-score 移除了绝对频带功率；若研究 ERD/ERS 的绝对功率，应另设功率保留实验。按手指或被试的结论需独立测试集上的分组消融或其他验证。

加载时先导入自定义层注册：
```python
import tensorflow as tf
from bci import FilterBankModels_tf
from bci.filter_bank import preprocess_windows

model = tf.keras.models.load_model("Save/your_model.h5", compile=False)
# windows 必须已按训练方式完成 CAR、窗口提取和重采样；形状 (N,C,300)
x = preprocess_windows(windows, sampling_rate=100, mode="weighted")
probabilities = model(x, training=False).numpy()
```

新的前端用四阶 Butterworth SOS 因果滤波，保留每侧 100 点零填充；off 路径保留原 lfilter。窗口独立过滤，推理时应采用相同方式，不能随意改成零相位或流式带状态滤波。

## 资源与支持边界

四频带不增加滑窗数；输入/分支激活相对相同 float32 单带约四倍，整体显存取决于模型和 batch。预处理按最多 128 个窗口一批，结果为 float32，但仍会完整保留最终训练/验证数组。内存不足时减小 batch 只能缓解模型显存，不能消除全量数组内存。

本次接入离线单/多被试训练、微调和独立模型推理；现有 BCPy2000 在线入口仍为单带 EEGNet，不能直接加载四带模型。独立 `reference_models/EEGConformer_tf.py` 不参与本训练入口。

## 验证

```bash
cd /home/kxy/LYC/Finger-BCI-Decoding-main
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 MPLBACKEND=Agg /home/kxy/anaconda3/bin/python -m unittest discover -s tests -v
```

测试使用合成 EEG，覆盖频带选择性、分支独立性、权重梯度、关闭插件的数值回归、序列化、命令行和训练/微调。Keras 对 HDF5 发出的 legacy-format 提示是预期提示，保留该格式以兼容现有项目。通过这些测试不代表真实数据准确率提升。
