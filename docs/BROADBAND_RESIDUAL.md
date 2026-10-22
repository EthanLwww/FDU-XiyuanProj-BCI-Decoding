# 可插拔整频带残差支路

该插件位于远端 `/home/kxy/LYC/Finger-BCI-Decoding-main`，适用于 EEGNet / EEGConformer 的离线单被试、多被试训练和微调。

## 开关与数据流

新增 `--broadband-residual off|fixed|gated`，默认 off：

| 模式 | 残差系数 | 输入形状（3 秒、100 Hz） |
|---|---|---|
| off | 无额外支路 | (N,C,300,4)，关闭 filter-bank 时仍为单带 |
| fixed | α = 1，不学习 | (N,C,300,5) |
| gated | α = sigmoid(a)，初始化 0.1，可学习 | (N,C,300,5) |

启用残差要求 `--filter-bank mean/weighted/concat`；与 `--filter-bank off` 同时启用会报错，避免产生无意义的重复整频带支路。

输入前四路仍是独立标准化的四子带，第五路从同一个 CAR 后、重采样后的窗口独立生成：4–40 Hz 因果带通、全频带一次 Z-score。第五路不会从四个已归一化子带重建。预处理仍分批完成。

融合为：

```text
Z = Z_filterbank + alpha * LayerNorm(P(Z_broadband))
```

整频带使用与原模型相同结构的卷积前端：
- Conformer：时间 Conv2D → 普通空间 Conv2D → BN/ELU/池化 → 40 维投影，得到 14×40 token。
- EEGNet：时间卷积 → 深度空间卷积 → 可分离卷积，得到 1×9×16 特征。

P 是无偏置线性投影，输出宽度与滤波器组融合结果匹配。LayerNorm 无可训练缩放和偏置。Conformer 在共享 Transformer 前相加；EEGNet 在 Flatten/分类头前相加。concat 模式也支持：EEGNet 将整频带投影到 64 维，Conformer 仍为 40 维。

原始卷积前端指网络结构一致，训练时重新初始化；不会自动加载或复用旧整频带 checkpoint 的参数。

## 消融命令

在远端 shell 中执行。以下例子使用被试 20、Session 2、ME 二分类；请保持数据、种子、轮数、batch 和其他配置一致。

```bash
cd /home/kxy/LYC/Finger-BCI-Decoding-main

# 原整频带模型
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank off --seed 42

# 四子带，不加整频带残差
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank weighted --broadband-residual off --seed 42

# 四子带 + 固定相加
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank weighted --broadband-residual fixed --seed 42

# 四子带 + 可学习残差（推荐尝试）
/home/kxy/anaconda3/bin/python train.py 20 2 2 ME Orig EEGConformer --filter-bank weighted --broadband-residual gated --seed 42

# 多被试、EEGNet 也采用相同开关
/home/kxy/anaconda3/bin/python train.py "1,2,3" 2 2 ME Orig EEGNet --filter-bank weighted --broadband-residual gated --seed 42
```

fixed 与 gated 的区别同时包含是否学习系数和初始残差强度（1 与 0.1）。若要严格隔离系数学习本身的贡献，还需固定 0.1 的额外对照；不要把当前两个模式的差异全部归因于“学到了更好的权重”。

## 保存、加载与微调

新输出名带有 `_BB-fixed` / `_BB-gated`，例如：
`S20_Sess02_ME_2class_EEGConformer_FB-weighted_BB-gated_Seed42_Orig.h5`。

关闭残差的文件名、输入与模型结构保持兼容。旧配置缺少 `broadband_residual` 时按 off 处理。固定/门控/无残差 checkpoint 不允许互相直接微调，必须使用匹配模式；目前没有单带或四带权重到五路模型的自动迁移。

新增 `.residual.json` 导出最佳验证准确率 checkpoint 的实际 α；`.band_weights.json` 仍只记录四个子带内的归一化权重。α 不是五个频带中的百分比，与四个子带权重相加也没有“总和为 1”的含义。

微调仍按原规则冻结前面各层、训练末 12 层，并显式保留频带权重（weighted）与残差门控（gated）可训练。整频带卷积前端不会整体解冻。

```python
import tensorflow as tf
from bci import FilterBankModels_tf
from bci.FilterBankModels_tf import get_residual_info
from bci.filter_bank import preprocess_windows

model = tf.keras.models.load_model("your_model.h5", compile=False)
# windows 已完成 CAR、3 秒窗口提取和 100 Hz 重采样，形状 (N,C,300)
x = preprocess_windows(windows, mode="weighted", broadband_residual="gated")
probabilities = model(x, training=False).numpy()
print(get_residual_info(model))
```

## 验证与边界

测试覆盖新增第五路的数值、原子带不变、固定/门控残差公式、支路独立性、非零梯度、各种融合模式、HDF5 重载、旧配置兼容、训练/微调、单/多被试命令行。

新增输入相对四带增加 25%，还会增加一条 CNN 支路的参数与激活；共享 Transformer 的层数和宽度保持不变。参数量增长与泛化收益需分开评价。

本次仅验证实现正确性；没有运行真实数据完整训练，也没有证明准确率提升。在线 BCPy2000 入口尚未接入该插件。
