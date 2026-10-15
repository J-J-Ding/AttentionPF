# V1 Masked Pretraining + Frozen Backbone + Low-rank Adapter 规范

## 1. 任务定义

V1 在已完成的 V0 统一实验框架上实现两阶段时间序列区间预测：

\[
\boxed{\text{Stage I: Masked Temporal Pretraining}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{Stage II: Frozen Pretrained Backbone + Low-rank Adapter}}
\]

V1 的目标不是重新实现数据、训练和评估框架，而是验证 masked temporal pretraining 与参数高效适配是否改善预测性能和训练效率。

统一监督预测接口保持不变：

\[
X\in\mathbb{R}^{B\times L\times C_{in}}
\rightarrow
\hat{Y}\in\mathbb{R}^{B\times P\times C_{out}\times 3}.
\]

最后一维固定为 `[Lower, Point, Upper]`，并由网络结构保证 `Lower < Point < Upper`。

## 2. 复用边界

V1 必须直接复用：

- `datasets/` 中的数据布局；
- `dataload/` 中的时间切分、窗口构造、M/MS 模式和 scaler；
- `epx_main.py` 中的监督 train/validation/test/predict 生命周期；
- V0 的 loss、metrics、logging、checkpoint 与 visualization；
- V0 的模型输出协议和结构性非交叉区间头；
- `run.py` 的模型注册式入口。

V1 新增：

```text
models/v1_pretrain_adapter.py
configs/v1.yaml
specs/V1_pretrain_adapter_spec.md
reports/V1_result.md
```

Stage I 目标与监督预测不同，因此允许新增：

```text
pretrain.py
```

`pretrain.py` 只负责 masked reconstruction 的训练生命周期，不复制监督预测的整套实验框架。公共 seed、logger、checkpoint、配置解析和数据读取工具必须复用现有实现或提取最小公共函数。

## 3. 非目标

V1 不包含：

- horizon attention；
- operator attention；
- diagonal/low-rank forecasting operator；
- Koopman dynamics；
- conformal calibration；
- dynamics-aware interval width；
- 复杂 Transformer reconstruction decoder；
- 自回归 decoder；
- 预训练时使用 validation/test split 数据；
- 在模型输出后使用排序修复区间。

## 4. Stage I 数据边界

### 4.1 数据来源

预训练只允许使用正式 train split 中的历史窗口 `x:[B,L,C_in]`。不得读取正式 validation 或 test split 的输入或目标，以避免 representation leakage。

scaler 仍然只由正式 train rows 拟合。Stage I 和 Stage II 必须共享同一特征顺序与 scaler 统计量。

### 4.2 预训练内部验证集

为了选择最佳 reconstruction checkpoint，在正式 train windows 内按时间顺序再次划分：

```text
pretrain train windows: 前 90%
pretrain val windows:   后 10%
```

默认 `pretrain.val_ratio=0.1`。该划分只用于 Stage I early stopping，不改变正式 train/validation/test 边界。

### 4.3 输入通道

M 和 MS 模式的预训练输入均为全部 `feature_cols`，因此 Stage I 的 patch reconstruction 总是重建全部 `C_in` 个输入通道。MS 只在 Stage II forecast target 中输出最后一个通道。

## 5. Patchify

V1 使用 patch token，而不是 V0 的 timestep token。设：

- patch length：`S`；
- patch stride：`R`；
- patch dimension：`S*C_in`；
- patch count：`N`。

默认：

```yaml
patch:
  length: 16
  stride: 8
```

不使用 padding。必须满足：

\[
L\ge S,
\qquad
(L-S)\bmod R=0.
\]

patch 数量为：

\[
N=1+\frac{L-S}{R}.
\]

张量变化为：

\[
[B,L,C_{in}]
\rightarrow
[B,N,S,C_{in}]
\rightarrow
[B,N,S\cdot C_{in}].
\]

Patchify 必须保持时间顺序，不允许跨 batch 或窗口边界拼接。

## 6. Patch Embedding 与位置编码

每个展平 patch 经过：

\[
E_{patch}=\operatorname{Linear}(S\cdot C_{in},D).
\]

输出：

\[
E_{patch}\in\mathbb{R}^{B\times N\times D}.
\]

V1 使用 learnable patch positional embedding：

\[
E=E_{patch}+E_{pos},
\qquad
E_{pos}\in\mathbb{R}^{1\times N\times D}.
\]

Patch embedding 和 positional embedding 都属于预训练 backbone，并写入 `theta_pre*`。位置长度必须与 checkpoint 的 `seq_len/patch.length/patch.stride` 契约一致。

## 7. Mask 策略

### 7.1 通用规则

Mask 在 patch 索引上采样。实现顺序固定为：

1. 从原序列提取 target patches；
2. 计算 patch embeddings；
3. 将被 mask 位置的 embedding 替换为 learnable mask token；
4. 添加 positional embedding；
5. 输入 Transformer Encoder。

这样 reconstruction target 永远保留原始 patch，而 encoder 看不到被 mask patch 的原始数值。

每个样本必须至少 mask 1 个 patch，并至少保留 1 个可见 patch。

### 7.2 Random Mask

Random Mask 在每个样本中无放回均匀抽取 patch，默认：

```yaml
mask:
  random_ratio: 0.4
```

mask 数量为：

\[
M_{random}=\operatorname{clamp}(\operatorname{round}(N\cdot r_{random}),1,N-1).
\]

Random Mask 用于学习局部时间依赖、周期结构和上下文表示。

### 7.3 Tail Mask

Tail Mask 连续遮盖末尾 patches，默认：

```yaml
mask:
  tail_ratio: 0.25
```

mask 数量为：

\[
M_{tail}=\operatorname{clamp}(\operatorname{ceil}(N\cdot r_{tail}),1,N-1).
\]

mask 索引固定为：

\[
N-M_{tail},\ldots,N-1.
\]

Tail Mask 使 reconstruction 更接近“由过去推断未来尾部”的 forecasting 场景。

### 7.4 混合策略

每个 batch 只选择一种 mask 类型，默认等概率：

```yaml
mask:
  random_probability: 0.5
```

选择过程使用受统一 seed 控制的 PyTorch RNG。validation mask 必须使用固定 seed 和固定索引，保证不同 epoch 的 reconstruction loss 可比较。

## 8. Stage I Backbone

Masked embeddings 输入标准 Transformer Encoder：

```text
Masked patch embeddings [B,N,D]
→ Multi-Head Self-Attention
→ Residual + LayerNorm
→ FFN
→ Residual + LayerNorm
→ H [B,N,D]
```

默认 backbone 与 V0 宽度/深度配置一致：

```yaml
model:
  d_model: 128
  n_heads: 4
  n_layers: 3
  d_ff: 256
  dropout: 0.1
  activation: gelu
```

Encoder 不使用 causal mask；被遮盖的原始数值已由 mask token 隔离，可见 patches 可以双向建模上下文。

## 9. Reconstruction Head 与 Loss

Stage I 使用单层 lightweight reconstruction head：

```python
nn.Linear(d_model, patch_length * input_dim)
```

输出：

\[
\hat{X}_{patch}\in\mathbb{R}^{B\times N\times(S\cdot C_{in})}.
\]

默认 reconstruction loss 为 masked-only MSE：

\[
L_{recon}=\frac{1}{|\mathcal M|\cdot S\cdot C_{in}}
\sum_{i\in\mathcal M}
\lVert \hat{x}_i-x_i\rVert_2^2.
\]

未 mask patches 不进入 loss，防止模型通过复制可见输入主导目标。V1 不使用 Transformer decoder，也不叠加复杂辅助损失。

## 10. 预训练参数与 checkpoint

Stage I 训练得到：

\[
\theta_{pre}^{*}.
\]

必须保存的 backbone 参数包括：

- patch embedding；
- learnable positional embedding；
- Transformer Q/K/V/O projection；
- FFN；
- LayerNorm。

mask token 和 reconstruction head 属于 Stage I 临时组件；Stage II 不加载它们进入 forecast forward。

预训练 checkpoint 至少包含：

```text
format_version
stage = pretrain
model_name = v1
epoch / global_step / best_metric
backbone_state_dict
mask_token_state_dict
reconstruction_head_state_dict
optimizer_state_dict
config
scaler_state_dict
rng_state
pretrain_train_generator_state
```

加载时必须校验：数据路径与特征顺序、`seq_len`、`input_dim`、patch length/stride、`d_model`、head 数、层数和 FFN 维度。契约不一致时不得部分加载。

## 11. Stage II Backbone 初始化

Stage II 必须从显式配置的 Stage I best checkpoint 初始化，禁止静默回退为随机初始化。

加载顺序：

1. 构建与 checkpoint 契约一致的 patch backbone；
2. 严格加载 `backbone_state_dict`；
3. 丢弃 reconstruction head 和 mask token；
4. 根据实验变体设置 backbone 的 `requires_grad`；
5. 创建 adapter；
6. 创建新的 V0-compatible forecast head。

Stage II forecast representation 固定使用最后一个 patch token：

\[
h=H[:,-1,:]\in\mathbb{R}^{B\times D}.
\]

随后一次性映射全部 horizon、channel 和三个区间参数。

## 12. Low-rank Adapter

### 12.1 数学定义

对冻结的预训练 projection：

\[
W_Q=W_Q^{*}+sB_QA_Q,
\]

\[
W_K=W_K^{*}+sB_KA_K.
\]

其中：

\[
A\in\mathbb{R}^{r\times D},
\qquad
B\in\mathbb{R}^{D\times r},
\qquad
r\ll D,
\]

\[
s=\frac{\alpha}{r}.
\]

默认：

```yaml
adapter:
  rank: 8
  alpha: 16
  dropout: 0.0
```

`A` 使用 Kaiming 初始化，`B` 初始化为零。因此 adapter 刚插入时：

\[
BA=0,
\]

模型行为与 pretrained backbone 完全一致。

### 12.2 Q/K Adapter

默认 V1-B 只对每层 self-attention 的 Q 和 K projection 添加 adapter：

- `W_Q*`、`W_K*` 冻结；
- `A_Q/B_Q`、`A_K/B_K` 可训练；
- V、O、FFN、LayerNorm、patch embedding 和 positional embedding 全部冻结；
- forecast head 可训练。

Q/K 用于适配 forecasting-specific relation；冻结 V 保留预训练的信息表示。

### 12.3 Q/K/V Adapter

V1-C 在 V1-B 基础上为 V projection 增加同结构 adapter：

\[
W_V=W_V^{*}+sB_VA_V.
\]

O projection 仍保持冻结。该变体用于检验调整 value representation 是否带来额外收益。

### 12.4 工程约束

Adapter 必须作用于每个 Encoder layer 的真实 Q/K/V projection，不能只在 Encoder 输入或输出外侧增加普通 MLP 并称为 Q/K adapter。

若 PyTorch `MultiheadAttention` 使用融合 `in_proj_weight`，实现必须显式、可测试地分离 Q/K/V 计算或提供等价 parametrization。checkpoint 映射必须逐层校验，禁止按 shape 猜测加载。

## 13. Forecast Head

Stage II 继续使用 V0 的结构性非交叉参数化。Forecast representation 经线性映射得到：

\[
\mu,a^{-},a^{+}\in\mathbb{R}^{B\times P\times C_{out}}.
\]

定义：

\[
Lower=\mu-\operatorname{Softplus}(a^{-})-\epsilon,
\]

\[
Point=\mu,
\]

\[
Upper=\mu+\operatorname{Softplus}(a^{+})+\epsilon.
\]

最终：

```python
pred = torch.stack((lower, point, upper), dim=-1)
```

输出严格为：

\[
[B,P,C_{out},3].
\]

禁止使用 `sort()`、交换位置或后处理修复顺序。Stage II 默认继续使用 V0 quantile loss 和相同 evaluation coverage。

## 14. V1 实验矩阵

必须完成：

| 实验 | Backbone 初始化 | Backbone 更新 | Adapter | Forecast head |
|---|---|---|---|---|
| V0 | 随机 | 全量训练 | 无 | 可训练 |
| V1-A | `theta_pre*` | 全量 fine-tuning | 无 | 可训练 |
| V1-B | `theta_pre*` | 冻结 | Q/K | 可训练 |
| V1-C | `theta_pre*` | 冻结 | Q/K/V | 可训练 |

V1-A/B/C 必须使用完全相同的 patch backbone、预训练 checkpoint、监督数据 split、forecast head 结构、optimizer 预算、early stopping、seed 和 metrics。只有参数更新策略不同。

V0 保留当前 timestep baseline，不为匹配 V1 而改写。论文解释中必须说明 V0→V1 同时引入 patch tokenization 与 pretraining；V1-A/B/C 之间才是严格控制变量的 adaptation 消融。

每个实验至少记录：

- total parameters；
- trainable parameters；
- trainable ratio；
- peak GPU memory；
- train time 与 inference time；
- convergence curves；
- MAE、RMSE；
- PICP、MPIW、PINAW；
- Coverage Error；
- Interval Score；
- overall 与 by-horizon 指标。

## 15. 配置设计

`configs/v1.yaml` 在 V0 公共配置基础上新增：

```yaml
model:
  name: v1
  d_model: 128
  n_heads: 4
  n_layers: 3
  d_ff: 256
  dropout: 0.1
  activation: gelu

patch:
  length: 16
  stride: 8

mask:
  random_ratio: 0.4
  tail_ratio: 0.25
  random_probability: 0.5

pretrain:
  checkpoint: null
  epochs: 50
  val_ratio: 0.1
  early_stopping_patience: 10
  learning_rate: 0.001
  weight_decay: 0.0001

adaptation:
  mode: qk       # full | qk | qkv
  rank: 8
  alpha: 16
  dropout: 0.0
```

Stage I 命令要求 `pretrain.checkpoint: null`；Stage II 命令要求其指向存在的 Stage I best checkpoint。`adaptation.mode=full` 表示 V1-A，不创建 adapter；`qk` 和 `qkv` 分别表示 V1-B/C。

## 16. 执行入口与产物

### 16.1 Stage I

建议入口：

```bash
python pretrain.py --config configs/v1.yaml
```

每次预训练使用唯一 experiment ID，保存：

```text
checkpoints/<pretrain_experiment_id>/best.pt
checkpoints/<pretrain_experiment_id>/last.pt
logs/<pretrain_experiment_id>/train.log
logs/<pretrain_experiment_id>/metrics.jsonl
results/<pretrain_experiment_id>/config.resolved.yaml
results/<pretrain_experiment_id>/metrics.json
results/<pretrain_experiment_id>/figures/training_curves.png
```

### 16.2 Stage II

监督预测仍通过统一入口：

```bash
python run.py --model v1 --config configs/v1.yaml
```

`epx_main.py` 不增加 V1 专属训练循环，只消费符合 `BaseForecastModel` 协议的 V1 模型。

每个 V1-A/B/C 运行结果保存在各自唯一目录，并最终汇总生成：

```text
reports/V1_result.md
```

报告必须注明 Stage I checkpoint ID、adaptation mode、rank、alpha、参数统计和全部规定指标。

## 17. 文件改动

计划新增：

```text
models/v1_pretrain_adapter.py
configs/v1.yaml
pretrain.py
tests/test_v1_patch_mask.py
tests/test_v1_pretrain.py
tests/test_v1_adapter.py
tests/test_v1_integration.py
reports/V1_result.md
```

计划最小修改：

```text
models/__init__.py          注册 v1
utils/config.py             支持版本化 V0/V1 schema
utils/checkpoint.py         支持 pretrain stage 契约
utils/visualization.py      复用/扩展预训练收敛图
run.py                      复用注册表，不增加 if/elif 链
README.md                   增加 Stage I/II 命令
requirements.txt            原则上不新增依赖
```

`dataload/` 原则上只复用；若预训练 train 内部切分需要辅助函数，只允许增加通用 subset 构造，不得复制 CSV 读取、scaler 或窗口逻辑。

## 18. 测试计划

### 18.1 Patch 与 Mask

1. `[B,L,C]→[B,N,S*C]` shape 与手算 patch 内容一致；
2. 非法 `L/S/R` 明确失败，不静默 padding；
3. Random Mask 数量正确、无重复、每个样本至少一 mask/一 visible；
4. Tail Mask 只覆盖连续末尾 patches；
5. 固定 seed 得到相同 mask；
6. validation mask 跨 epoch 保持相同；
7. masked embeddings 不包含原始 patch 数值。

### 18.2 Pretraining

1. reconstruction 输出为 `[B,N,S*C]`；
2. MSE 只计算 masked positions；
3. unmasked prediction 的改变不影响 loss；
4. patch embedding、positional embedding、Q/K/V/O、FFN、LayerNorm 获得有限梯度；
5. reconstruction head 获得有限梯度；
6. 一个 ETTh1 小 batch 上 reconstruction loss 可下降；
7. pretraining 数据索引不超出正式 train split；
8. checkpoint round-trip 后 reconstruction 一致；
9. 不兼容 patch/backbone 契约明确拒绝。

### 18.3 Frozen Backbone 与 Adapter

1. 加载 checkpoint 后 backbone 权重逐项一致；
2. 零初始化 adapter 时输出与无 adapter backbone 一致；
3. V1-B 仅 Q/K adapter 与 forecast head `requires_grad=True`；
4. V1-C 仅 Q/K/V adapter 与 forecast head `requires_grad=True`；
5. V1-A 全 backbone 与 forecast head 可训练且没有 adapter；
6. backward 后所有 frozen parameters 的 `grad is None`；
7. backward 后所有 adapter parameters 的 `grad is not None` 且有限；
8. Q/K 模式的 V adapter 不存在，Q/K/V 模式的 V adapter 存在；
9. trainable parameter count 与显式枚举一致；
10. reconstruction head 不出现在 Stage II state dict/forward 中。

### 18.4 Forecast 输出与框架复用

1. M 输出 `[B,P,C_in,3]`；
2. MS 输出 `[B,P,1,3]`；
3. 所有有限输出满足 `Lower<Point<Upper`；
4. 禁止 `sort/argsort/交换 lower upper`；
5. quantile loss backward 只更新允许训练的参数；
6. V1 通过现有 `epx_main.py` 完成 train/validation/test/predict；
7. best checkpoint、metrics、predictions 和 figures 写入同一实验 ID 对应产物目录；
8. CPU smoke run 固定 seed 可复现。

### 18.5 消融与报告

1. V1-A/B/C 都能从同一 pretrain best checkpoint 启动；
2. 三组实验的非 adaptation 配置完全一致；
3. 报告包含 total/trainable/trainable ratio；
4. 报告包含 GPU memory、convergence、点指标和区间指标；
5. 在相同 backbone 与 forecast head 下，V1-B 的 trainable parameters 和 trainable ratio 必须严格低于 V1-A；报告不预设任意百分比阈值。

## 19. 执行计划

Spec 获得确认后，按以下顺序实施：

1. 建立 V1 版本化配置 schema 和失败测试；
2. 实现并测试 patchify、Random Mask、Tail Mask；
3. 实现 patch backbone、mask token 与 lightweight reconstruction head；
4. 实现 masked-only reconstruction loss；
5. 实现 `pretrain.py`，复用公共 seed/logger/checkpoint/path 工具；
6. 在 ETTh1 train split 上完成 Stage I 小规模 smoke 与 checkpoint round-trip；
7. 实现显式 Q/K/V projection 和 low-rank adapter；
8. 实现 pretrain checkpoint 到 Stage II backbone 的严格加载；
9. 实现 V1-A/B/C 参数冻结策略和梯度测试；
10. 接入 V0 non-crossing forecast head 与模型注册表；
11. 使用现有 `epx_main.py` 完成 V1 监督 smoke；
12. 运行完整单元、集成、泄漏、可复现性和禁用排序检查；
13. 依次执行 V1-A/B/C 正式实验；
14. 汇总 `reports/V1_result.md` 与论文图像。

所有代码实现遵循 TDD：先写失败测试并确认失败原因，再做最小实现，最后运行完整回归测试。不得为了 V1 修改 V0 的数学输出协议。

## 20. 验收标准

V1 完成必须同时满足：

1. `specs/V1_pretrain_adapter_spec.md`、`models/v1_pretrain_adapter.py` 和 `configs/v1.yaml` 存在；
2. Stage I 仅使用正式 train split，Random/Tail Mask 均可运行；
3. Patchify shape、mask 索引和 masked-only loss 正确；
4. Reconstruction head 仅为单层 `Linear(D, patch_dim)`，无复杂 decoder；
5. pretrain best checkpoint 包含完整 `theta_pre*` 与可复现状态；
6. Stage II 严格加载 checkpoint 并丢弃 reconstruction head；
7. V1-A 完成 full fine-tuning；
8. V1-B 完成 frozen backbone + Q/K adapter；
9. V1-C 完成 frozen backbone + Q/K/V adapter；
10. V1-B 中 `frozen_parameter.grad is None`；
11. V1-B/C 中 `adapter_parameter.grad is not None` 且有限；
12. V1-A/B/C 输出均为 `[B,P,C_out,3]`；
13. 所有有限输出结构性满足 `Lower<Point<Upper`，没有排序后处理；
14. Stage II 全部复用 `epx_main.py`、metrics、logging、checkpoint 和 visualization；
15. V0 全部回归测试继续通过；
16. V1 patch/mask/pretrain/adapter/integration 测试全部通过；
17. `python pretrain.py --config configs/v1.yaml` 可生成 Stage I best checkpoint；
18. `python run.py --model v1 --config configs/v1.yaml` 可完成监督训练与测试；
19. V1-A/B/C 记录 total params、trainable params、trainable ratio、GPU memory、convergence、MAE、RMSE、PICP、PINAW 和 Interval Score；
20. 自动生成 `reports/V1_result.md`，并能追溯所用 pretrain checkpoint 和完整 resolved config。
