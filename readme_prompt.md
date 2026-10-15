# 时间序列区间预测项目：V0–V5 分阶段开发提示词

> 目标：从零构建一个统一、可复现、便于消融实验的时间序列区间预测项目。  
> 输入统一为：
>
> \[
> X\in\mathbb R^{B\times L\times C}
> \]
>
> 输出统一为：
>
> \[
> \hat Y\in\mathbb R^{B\times P\times C\times 3}
> \]
>
> 最后一维固定表示：
>
> \[
> [\text{Lower},\text{Point},\text{Upper}]
> \]
>
> 且必须由网络结构天然保证：
>
> \[
> Lower < Point < Upper
> \]
>
> 禁止通过 `sort()`、交换位置或后处理强制排序。

---

# 0. 全项目统一约束

整个项目按以下版本逐步演进：

\[
\boxed{
\text{V0: Transformer + Quantile Head}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V1: Pretrain + Frozen Backbone + Adapter}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V2: + Horizon Attention}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V3: + Operator Attention}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V4: + Diagonal / Low-rank Operator}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{V5: + Dynamics-aware Interval Width}
}
\]

核心开发原则：

1. 每一个版本开始前，必须先生成对应 `specs/*.md`。
2. Spec 中必须先明确：
   - 任务定义；
   - 数学结构；
   - 张量维度；
   - 文件改动；
   - 执行计划；
   - 测试计划；
   - 验收标准。
3. 不允许 AI 一上来直接修改代码。
4. V1–V5 必须尽量复用前一版本的：
   - DataLoader；
   - 训练循环；
   - 指标；
   - 日志；
   - checkpoint；
   - 实验框架。
5. 后续版本原则上只改变 `models/` 中的网络与必要配置，避免重复实现整套训练流程。
6. 所有模型必须统一输入输出接口，保证可以直接通过 `epx_main.py` 对比。
7. 所有模型必须输出：
   ```python
   pred.shape == [B, P, C, 3]
   ```
8. 最后一维固定：
   ```python
   pred[..., 0]  # Lower
   pred[..., 1]  # Point / Median
   pred[..., 2]  # Upper
   ```
9. 必须结构性保证：
   ```python
   lower < point < upper
   ```

---

# 1. V0 Prompt：从空项目构建统一实验框架 + Transformer Interval Baseline

你现在面对的是一个**全新的空项目**。

你的任务不是只实现一个模型，而是先搭建一个后续 V0–V5 都可以复用的统一实验框架，然后实现第一个基线：

\[
\boxed{
\text{V0: Transformer + Structurally Non-crossing Quantile Head}
}
\]

---

## 1.1 第一原则：不要直接写代码

第一步必须先生成：

```text
specs/V0_project_and_baseline_spec.md
```

在 Spec 中先完成：

1. 项目目录设计；
2. 数据格式约定；
3. DataLoader 设计；
4. 模型统一接口；
5. 训练接口；
6. 验证接口；
7. 测试 / 推理接口；
8. 指标系统；
9. checkpoint；
10. logging；
11. config；
12. V0 网络结构；
13. tensor shape flow；
14. 单元测试；
15. 执行计划；
16. 验收标准。

只有 Spec 清楚以后，再开始创建文件。

---

## 1.2 项目目录必须保持清晰简洁

项目初始化时建议使用如下结构：

```text
project_root/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── dataload/
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataloader.py
│   └── scaler.py
│
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   └── v0_transformer.py
│
├── losses/
│   ├── __init__.py
│   ├── quantile_loss.py
│   └── interval_loss.py
│
├── metrics/
│   ├── __init__.py
│   ├── point_metrics.py
│   └── interval_metrics.py
│
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── config.py
│
├── configs/
│   └── v0.yaml
│
├── specs/
│   └── V0_project_and_baseline_spec.md
│
├── reports/
│
├── checkpoints/
│
├── logs/
│
├── results/
│
├── epx_main.py
├── run.py
├── requirements.txt
└── README.md
```

注意：

- `datasets/`：只负责存放数据；
- `dataload/`：只负责数据读取、切窗、划分、归一化；
- `models/`：只负责网络定义；
- `losses/`：统一 loss；
- `metrics/`：统一评估指标；
- `epx_main.py`：统一实验框架；
- `run.py`：统一执行入口；
- `configs/`：不同模型的超参数；
- `results/`：实验结果；
- `reports/`：每版实验总结；
- `checkpoints/`：模型参数；
- `logs/`：训练日志。

后续 V1–V5 不重新搭建训练框架，尽量只新增：

```text
models/v1_*.py
models/v2_*.py
...
configs/v1.yaml
configs/v2.yaml
...
```

---

## 1.3 `epx_main.py` 的职责

`epx_main.py` 必须作为统一实验核心。

要求至少包含类似接口：

```python
def train_one_epoch(...):
    ...

def validate(...):
    ...

def test(...):
    ...

def predict(...):
    ...

def run_experiment(
    model,
    train_loader,
    val_loader,
    test_loader,
    config
):
    ...
```

所有 V0–V5 都必须调用同一套：

- train；
- validation；
- test；
- metric；
- checkpoint；
- logger；

逻辑。

目的：

> 后续只修改模型类和配置文件，就可以公平执行不同版本。

---

## 1.4 `run.py` 的职责

`run.py` 是用户唯一主要执行入口。

例如：

```bash
python run.py --model v0 --config configs/v0.yaml
```

未来应支持：

```bash
python run.py --model v1 --config configs/v1.yaml
python run.py --model v2 --config configs/v2.yaml
...
```

建议使用模型注册机制：

```python
MODEL_REGISTRY = {
    "v0": V0Transformer,
    "v1": V1PretrainAdapter,
    "v2": V2HorizonAttention,
    ...
}
```

禁止在 `run.py` 中写大量：

```python
if model == "v0":
    ...
elif model == "v1":
    ...
```

尽量保持结构可扩展。

---

## 1.5 DataLoader 统一规范

输入：

\[
X\in\mathbb R^{B\times L\times C}
\]

target：

\[
Y\in\mathbb R^{B\times P\times C}
\]

因此一个 batch 必须输出：

```python
x.shape == [B, L, C]
y.shape == [B, P, C]
```

Dataset 负责：

1. 加载时间序列；
2. 按时间顺序切分 train / val / test；
3. 只使用 train 数据拟合 scaler；
4. 滑动窗口构造：
   - history length = `seq_len`
   - prediction length = `pred_len`
5. 禁止未来信息泄漏。

必须将：

- `seq_len`
- `pred_len`
- `stride`
- `batch_size`
- `feature_cols`
- `target_cols`

写入 config。

---

## 1.6 数据目录

默认：

```text
datasets/
├── raw/
│   └── dataset.csv
│
├── processed/
│   └── ...
│
└── README.md
```

不要把数据路径写死在 Python 文件里。

统一使用：

```yaml
data:
  path: datasets/raw/dataset.csv
```

---

## 1.7 BaseModel 统一接口

在：

```text
models/base_model.py
```

定义统一规范。

所有模型：

```python
class BaseForecastModel(nn.Module):

    def forward(self, x):
        """
        x:
            [B, L, C]

        return:
            pred: [B, P, C, 3]
        """
```

后续 V1–V5 都必须遵循。

---

# 2. V0 模型本身

V0 实现：

```text
Input
[B,L,C]
   ↓
Input Projection / Patch Embedding
   ↓
Transformer Encoder
   ↓
Forecast Representation
   ↓
Center Head
Lower Width Head
Upper Width Head
   ↓
Non-crossing Prediction Interval
   ↓
[B,P,C,3]
```

V0 暂时禁止加入：

- masked pretraining；
- LoRA；
- adapter；
- horizon attention；
- operator attention；
- Koopman；
- conformal；
- dynamics uncertainty。

V0 只是清晰、可靠的 Transformer baseline。

---

## 2.1 Transformer

使用标准 Transformer Encoder：

```text
Embedding
↓
Multi-Head Self-Attention
↓
Residual + LayerNorm
↓
FFN
↓
Residual + LayerNorm
```

至少支持：

```yaml
model:
  d_model: 128
  n_heads: 4
  n_layers: 3
  d_ff: 256
  dropout: 0.1
```

如果使用 patch：

\[
[B,L,C]
\rightarrow
[B,N,S\cdot C]
\rightarrow
[B,N,D].
\]

如果第一版希望保持简单，也可以先使用 timestep embedding：

\[
[B,L,C]
\rightarrow
[B,L,D].
\]

但必须在 Spec 中解释选择。

---

## 2.2 Non-crossing Interval Head

禁止直接输出三个自由预测值。

预测：

\[
\mu\in\mathbb R^{B\times P\times C}
\]

\[
a^-\in\mathbb R^{B\times P\times C}
\]

\[
a^+\in\mathbb R^{B\times P\times C}.
\]

定义：

\[
w^-=\operatorname{Softplus}(a^-)+\epsilon
\]

\[
w^+=\operatorname{Softplus}(a^+)+\epsilon.
\]

然后：

\[
Lower=\mu-w^-
\]

\[
Point=\mu
\]

\[
Upper=\mu+w^+.
\]

最终：

```python
pred = torch.stack(
    [lower, point, upper],
    dim=-1
)
```

输出：

\[
[B,P,C,3].
\]

必须测试：

```python
assert torch.all(lower < point)
assert torch.all(point < upper)
```

不能使用 `sort()`。

---

## 2.3 概率定义

第一版默认构建 90% 区间：

\[
Lower\approx Q_{0.05}
\]

\[
Point\approx Q_{0.50}
\]

\[
Upper\approx Q_{0.95}.
\]

因此中间点解释为 median forecast。

---

## 2.4 Loss

实现 Pinball / Quantile Loss：

\[
L_Q
=
\rho_{0.05}
+
\rho_{0.5}
+
\rho_{0.95}.
\]

可以额外实现 Interval Score，但第一轮 baseline 默认保持简单。

---

## 2.5 Metrics

点预测：

- MAE
- RMSE

区间预测：

- PICP
- MPIW
- PINAW
- Interval Score / Winkler Score
- Coverage Error

要求同时输出：

- overall；
- different horizons。

---

## 2.6 V0 必须完成的测试

至少包含：

1. Dataset shape test；
2. DataLoader shape test；
3. model forward shape；
4. backward test；
5. non-crossing test；
6. one-batch overfit test；
7. train / val / test no leakage；
8. scaler inverse transform；
9. checkpoint save/load；
10. metric unit tests。

---

## 2.7 V0 最终验收

最终必须可以直接：

```bash
python run.py --model v0 --config configs/v0.yaml
```

完成：

```text
Load Data
→ Train
→ Validation
→ Save Best Checkpoint
→ Test
→ Prediction
→ Metrics
→ Save Results
```

输出：

```text
reports/V0_result.md
```

记录：

- 模型参数量；
- trainable params；
- peak GPU memory；
- train time；
- inference time；
- MAE；
- RMSE；
- PICP；
- PINAW；
- Interval Score。

---

# 3. V1 Prompt：Masked Pretraining + Frozen Backbone + Adapter

当前 V0 已完成。

现在实现：

\[
\boxed{
\text{V1: Pretrain + Frozen Backbone + Low-rank Adapter}
}
\]

第一步必须生成：

```text
specs/V1_pretrain_adapter_spec.md
```

禁止直接修改代码。

---

## 3.1 开发原则

V1 必须复用：

- `datasets/`
- `dataload/`
- `epx_main.py`
- metrics
- logging
- checkpoint
- V0 output protocol

新增：

```text
models/v1_pretrain_adapter.py
configs/v1.yaml
```

必要时新增：

```text
pretrain.py
```

但不要重新复制整套训练框架。

---

## 3.2 Stage I：Masked Temporal Pretraining

输入：

\[
[B,L,C]
\]

如果采用 patch：

\[
[B,N,S\cdot C]
\]

经过 embedding：

\[
[B,N,D].
\]

使用：

1. Random Mask
2. Tail Mask

Random Mask 学习：

- local temporal dependency；
- periodic structure；
- contextual representation。

Tail Mask 缩小 reconstruction 和 forecasting 的任务差距。

---

## 3.3 Pretraining 结构

```text
Original Sequence
    ↓
Patchify
    ↓
Mask
    ↓
Patch Embedding
    ↓
Transformer Encoder
    ↓
H [B,N,D]
    ↓
Lightweight Reconstruction Head
    ↓
Reconstructed Patches
```

Reconstruction head：

```python
Linear(D, patch_dim)
```

或者最多一个很小的 MLP。

不要使用复杂 Transformer decoder。

---

## 3.4 预训练参数

训练结束得到：

\[
\theta_{\mathrm{pre}}^*.
\]

它包括：

- patch embedding；
- positional embedding；
- Transformer Q/K/V/O；
- FFN；
- LayerNorm。

Reconstruction head 只是预训练临时组件。

Forecast 阶段丢弃 reconstruction head。

---

## 3.5 Stage II：Frozen Backbone

冻结：

\[
\theta_{\mathrm{pre}}^*
\]

然后对 Q/K 加低秩 adaptation：

\[
W_Q=W_Q^*+B_QA_Q
\]

\[
W_K=W_K^*+B_KA_K.
\]

其中：

\[
rank=r\ll D.
\]

第一版默认：

\[
W_V=W_V^*
\]

保持冻结。

解释：

- Q/K：适配 forecasting-specific relation；
- V：保留 pretrained information representation。

---

## 3.6 Forecast Head

继续使用 V0：

\[
Lower=\mu-\operatorname{Softplus}(a^-)-\epsilon
\]

\[
Point=\mu
\]

\[
Upper=\mu+\operatorname{Softplus}(a^+)+\epsilon.
\]

输出仍然：

\[
[B,P,C,3].
\]

---

## 3.7 V1 实验

比较：

```text
V0:
Transformer from scratch

V1-A:
Pretrain + Full Fine-tuning

V1-B:
Pretrain + Frozen Backbone + Q/K Adapter

V1-C:
Pretrain + Frozen Backbone + Q/K/V Adapter
```

记录：

- total params；
- trainable params；
- trainable ratio；
- GPU memory；
- convergence；
- MAE；
- RMSE；
- PICP；
- PINAW；
- Interval Score。

---

## 3.8 V1 验收

必须确认：

```python
frozen_parameter.grad is None
adapter_parameter.grad is not None
```

并生成：

```text
reports/V1_result.md
```

---

# 4. V2 Prompt：Horizon-Aware Temporal Attention

在 V1 基础上实现：

\[
\boxed{
\text{V2: + Horizon Attention}
}
\]

第一步生成：

```text
specs/V2_horizon_attention_spec.md
```

---

## 4.1 核心问题

不同预测步：

\[
t+1,\dots,t+P
\]

不应完全依赖相同历史表示。

因此每个 horizon 创建独立 query。

---

## 4.2 Tensor Flow

Transformer：

\[
H\in\mathbb R^{B\times N\times D}.
\]

Historical Key：

\[
K_{hist}=HW_K
\]

采用低秩关系空间：

\[
D\rightarrow R
\]

得到：

\[
[B,N,R].
\]

Historical Value：

\[
V_{hist}=HW_V
\]

得到：

\[
[B,N,D].
\]

---

## 4.3 Horizon Query

建立 horizon embedding：

\[
E_H\in\mathbb R^{P\times D}.
\]

Global context：

\[
g=Pool(H)\in[B,D].
\]

构建：

\[
Q_h=(g+E_H)W_Q
\]

得到：

\[
[B,P,R].
\]

---

## 4.4 Temporal Attention

\[
A_h=
softmax
\left(
\frac{Q_hK_{hist}^{\top}}{\sqrt R}
\right)
\]

得到：

\[
[B,P,N].
\]

再：

\[
C_h=A_hV_{hist}
\]

得到：

\[
[B,P,D].
\]

---

## 4.5 输出

V2 暂时不要 Operator。

```text
Horizon Context
[B,P,D]
   ↓
Forecast Head
   ↓
[B,P,C,3]
```

---

## 4.6 KV Cache

Inference 时支持：

```python
K_cache = K_hist
V_cache = V_hist
```

不同 horizon 重用。

注意：

KV Cache 是 inference optimization，不作为主要算法创新。

---

## 4.7 V2 Ablation

比较：

```text
V1
V2
V2 without horizon embedding
V2 full-rank QK
V2 low-rank QK
```

生成：

```text
reports/V2_result.md
```

---

# 5. V3 Prompt：Multi-Head Operator Attention

在 V2 基础上实现：

\[
\boxed{
\text{V3: + Operator Attention}
}
\]

先生成：

```text
specs/V3_operator_attention_spec.md
```

---

## 5.1 核心逻辑

Horizon Attention 回答：

> 第 p 个未来时间点应该关注哪些历史？

Operator Attention 回答：

> 当前应该使用哪一种 latent dynamics？

---

## 5.2 Latent Dynamics Space

V2：

\[
C_h\in[B,P,D].
\]

映射：

\[
D\rightarrow D_K.
\]

得到：

\[
Z_{context}\in[B,P,D_K].
\]

当前状态：

\[
z_0=W_{init}Pool(H)
\]

得到：

\[
[B,D_K].
\]

---

## 5.3 Multi-Head

设：

\[
H_o
\]

个 operator heads。

例如：

\[
D_K=64,\quad H_o=4
\]

则：

\[
d_h=16.
\]

---

## 5.4 Operator Bank

V3 暂时使用完整矩阵：

\[
K_{h,m}\in\mathbb R^{d_h\times d_h}.
\]

每个 head 有：

\[
M
\]

个 operators。

Tensor：

```text
[Ho,M,dh,dh]
```

同时每个 operator 有一个 key：

```text
[Ho,M,dh]
```

---

## 5.5 Operator Attention

Horizon context reshape：

\[
[B,P,H_o,d_h].
\]

计算：

\[
s_{b,p,h,m}
=
\frac{
q_{b,p,h}^{\top}e_{h,m}
}{
\sqrt{d_h}
}.
\]

Softmax：

\[
\alpha_{b,p,h,m}.
\]

shape：

\[
[B,P,H_o,M].
\]

有效 operator：

\[
K_{b,p,h}^{eff}
=
\sum_m
\alpha_{b,p,h,m}K_{h,m}.
\]

---

## 5.6 Latent Rollout

\[
z^{(0)}=z_0
\]

\[
z^{(p)}
=
K_p^{eff}z^{(p-1)}.
\]

收集：

\[
Z_{future}\in[B,P,D_K].
\]

再进入 V0 interval head。

---

## 5.7 V3 Ablation

比较：

```text
V2
V3-single-operator
V3-single-head-mixture
V3-multi-head-mixture
```

测试：

```text
M = 2,4,8,16
Ho = 1,2,4,8
```

生成：

```text
reports/V3_result.md
```

---

# 6. V4 Prompt：Diagonal + Low-Rank Operator

在 V3 基础上实现：

\[
\boxed{
\text{V4: + Diagonal / Low-rank Operator}
}
\]

第一步生成：

```text
specs/V4_lowrank_operator_spec.md
```

---

## 6.1 核心目标

V3：

\[
K_{h,m}\in\mathbb R^{d_h\times d_h}
\]

参数复杂度：

\[
O(d_h^2).
\]

V4 改为：

\[
\boxed{
K_{h,m}
=
Diag(d_{h,m})
+
U_{h,m}V_{h,m}^{\top}
}
\]

其中：

\[
U,V\in\mathbb R^{d_h\times r_k}
\]

且：

\[
r_k\ll d_h.
\]

---

## 6.2 不显式构造矩阵

禁止默认：

```python
K = diag(D) + U @ V.T
out = K @ z
```

应该：

\[
Kz
=
D\odot z
+
U(V^{\top}z).
\]

vectorize 实现：

```python
diag_part = D * z
inner = einsum(...)
low_rank_part = einsum(...)
out = diag_part + low_rank_part
```

Operator mixture：

\[
z'
=
\sum_m
\alpha_m
\left[
D_m\odot z
+
U_m(V_m^\top z)
\right].
\]

---

## 6.3 V4 Ablation

比较：

```text
V3 Full Operator
V4 Diagonal Only
V4 Low-rank Only
V4 Diagonal + Low-rank
```

测试：

```text
rk = 1,2,4,8,16
```

必须报告：

- 参数量；
- compression ratio；
- GPU memory；
- latency；
- accuracy；
- interval quality。

---

## 6.4 数学等价测试

随机：

\[
D,U,V,z.
\]

验证：

```python
K = torch.diag_embed(D) + U @ V.transpose(-1, -2)
out1 = K @ z
out2 = D * z + U @ (V.T @ z)
```

必须：

```python
torch.allclose(out1, out2)
```

生成：

```text
reports/V4_result.md
```

---

# 7. V5 Prompt：Dynamics-Aware Interval Width

在 V4 基础上实现最终版：

\[
\boxed{
\text{V5: + Dynamics-aware Interval Width}
}
\]

第一步生成：

```text
specs/V5_dynamics_uncertainty_spec.md
```

---

## 7.1 核心思想

V4 已经得到：

\[
\alpha_{b,p,h,m}
\]

表示不同 operator 的选择概率。

V5 不直接把：

\[
Attention Entropy
\]

定义成 uncertainty。

它只能作为：

\[
\boxed{
\text{Dynamics ambiguity feature}
}
\]

由实验验证其是否与预测误差相关。

---

## 7.2 Operator Entropy

\[
H_{op}
=
-\sum_m
\alpha_m\log(\alpha_m+\epsilon).
\]

shape：

\[
[B,P,H_o].
\]

可：

- mean；
- max；
- learned fusion。

---

## 7.3 Operator Disagreement

对同一 latent state：

\[
z_m'=K_mz.
\]

计算 operator outputs 的 variance：

\[
D_{op}
=
Var_m(z'_m).
\]

它可以作为另一个 dynamics uncertainty signal。

---

## 7.4 Local Volatility

从历史序列提取简单统计量：

- rolling std；
- first-difference magnitude；
- local residual variance。

不要复杂化。

---

## 7.5 Interval Width Head

Point：

\[
\mu=f_\mu(Z_{future}).
\]

Width features：

\[
U_{feat}
=
[
Z_{future},
H_{op},
D_{op},
Volatility,
HorizonEmbedding
].
\]

然后：

\[
a^-=f_-(U_{feat})
\]

\[
a^+=f_+(U_{feat})
\]

\[
w^-=\operatorname{Softplus}(a^-)+\epsilon
\]

\[
w^+=\operatorname{Softplus}(a^+)+\epsilon.
\]

最终：

\[
Lower=\mu-w^-
\]

\[
Point=\mu
\]

\[
Upper=\mu+w^+.
\]

严格 non-crossing。

---

## 7.6 Optional：Reconstruction Familiarity

使用 V1 pretrained reconstruction model 对当前输入计算：

\[
R_{rec}.
\]

它只能解释为：

\[
\text{representation familiarity}
\]

或：

\[
\text{reconstruction difficulty}.
\]

禁止直接称其为 predictive uncertainty。

---

## 7.7 V5 Ablation

比较：

```text
V4
V5-A: + Operator Entropy
V5-B: + Operator Disagreement
V5-C: + Entropy + Disagreement
V5-D: + Local Volatility
V5-E: + Reconstruction Familiarity
```

重点指标：

- PICP
- PINAW
- MPIW
- Interval Score
- MAE
- RMSE
- Coverage by Horizon
- Width by Horizon

---

# 8. V0–V5 最终统一实验表

最终至少自动生成：

| Model | Pretrain | Adapter | Horizon Attn | Operator Attn | Low-rank Operator | Dynamics Width |
|---|---|---|---|---|---|---|
| V0 | × | × | × | × | × | × |
| V1 | ✓ | ✓ | × | × | × | × |
| V2 | ✓ | ✓ | ✓ | × | × | × |
| V3 | ✓ | ✓ | ✓ | ✓ | × | × |
| V4 | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| V5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

同时报告：

```text
MAE
RMSE
PICP
MPIW
PINAW
Interval Score
Total Params
Trainable Params
Peak GPU Memory
Train Time
Inference Latency
```

---

# 9. 推荐统一 Config 设计

例如：

```yaml
experiment:
  name: v0_baseline
  seed: 42
  device: cuda

data:
  path: datasets/raw/dataset.csv
  seq_len: 96
  pred_len: 24
  stride: 1
  batch_size: 32
  train_ratio: 0.7
  val_ratio: 0.1
  test_ratio: 0.2

model:
  name: v0
  d_model: 128
  n_heads: 4
  n_layers: 3
  d_ff: 256
  dropout: 0.1

interval:
  lower_quantile: 0.05
  median_quantile: 0.50
  upper_quantile: 0.95
  epsilon: 1.0e-6

train:
  epochs: 100
  lr: 1.0e-3
  weight_decay: 1.0e-4
  early_stopping: 10

output:
  checkpoint_dir: checkpoints/
  log_dir: logs/
  result_dir: results/
  report_dir: reports/
```

后续：

```text
configs/v1.yaml
configs/v2.yaml
...
```

只覆盖模型特有参数。

---

# 10. 推荐 `run.py` 最终使用方式

```bash
python run.py --model v0 --config configs/v0.yaml
```

```bash
python run.py --model v1 --config configs/v1.yaml
```

```bash
python run.py --model v2 --config configs/v2.yaml
```

最终支持：

```bash
python run.py --model v5 --config configs/v5.yaml
```

如果要做统一对比，可进一步支持：

```bash
python run.py --compare v0 v1 v2 v3 v4 v5
```

但这属于后续优化，第一版不必过度设计。

---

# 11. 最重要的工程要求

任何 Coding Agent 在实现时都必须遵守：

1. **先 Spec，后编码。**
2. 不改变统一输入：
   \[
   [B,L,C]
   \]
3. 不改变统一输出：
   \[
   [B,P,C,3]
   \]
4. 不允许通过 `sort()` 保证区间顺序。
5. 不允许不同模型拥有各自独立的训练循环。
6. 所有 V0–V5 共用 `epx_main.py`。
7. `run.py` 只负责：
   - 读取 config；
   - 构建 dataset；
   - 构建 model；
   - 调用实验框架。
8. DataLoader 不得嵌入模型逻辑。
9. 模型文件不得嵌入训练循环。
10. 每个版本都必须可独立复现。
11. 每个版本必须保存自己的：
    - config；
    - checkpoint；
    - log；
    - result；
    - report。
12. 所有新增模块必须有 shape test。
13. V3 之后所有复杂 `einsum` 必须在 Spec 中提前写明输入输出维度。
14. 每次改动必须保证旧版本仍然可以运行。
15. 不允许为了新版本破坏旧版本实验结果的可复现性。

---

# 12. 最终研究逻辑

整个项目最终应形成清晰的递进关系：

```text
V0
标准 Transformer 区间预测基线
        ↓
V1
学习通用时序结构，并低秩适配 forecasting
        ↓
V2
不同 horizon 学习不同历史依赖
        ↓
V3
不同 horizon 动态选择不同 latent dynamics
        ↓
V4
用 Diagonal + Low-rank 表达高效动力学
        ↓
V5
利用 dynamics ambiguity 改善预测区间宽度
```

最终论文中需要验证的不是：

> “V5 比 V0 好。”

而是：

> 每一次结构演进是否解决了一个明确问题，并通过消融实验证明该模块的独立价值。

