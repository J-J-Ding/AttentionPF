# V2 Horizon-Aware Temporal Attention 规范

## 1. 任务定义

V2 在 V1 的 masked temporal pretraining、patch Transformer backbone 与参数高效适配基础上，引入 horizon-aware temporal attention：

\[
\boxed{\text{V2: V1 + Horizon Attention}}
\]

V1 使用最后一个 patch token 的单一表示，一次性生成全部未来步。V2 改为让每个预测 horizon 使用自己的 query，从同一组历史 token 中读取与该未来步最相关的信息。

统一监督预测接口保持不变：

\[
X\in\mathbb{R}^{B\times L\times C_{in}}
\rightarrow
\hat{Y}\in\mathbb{R}^{B\times P\times C_{out}\times 3}.
\]

最后一维固定为 `[Lower, Point, Upper]`，并继续由结构保证 `Lower < Point < Upper`。

## 2. 设计目标与成功标准

V2 必须满足：

1. 每个 horizon 都有独立、可学习的 horizon embedding；
2. 所有 horizon 共享同一历史表示，但可学习不同的历史注意力分布；
3. 主模型使用 `D -> R` 的低秩 Q/K 关系空间，Value 保持 `D` 维；
4. horizon context 直接进入共享 forecast head，不引入 Operator；
5. 推理阶段可以缓存历史 K/V，且缓存路径与非缓存路径数值等价；
6. 继续复用 V1 的预训练 checkpoint、backbone、适配策略和统一实验框架；
7. 完成 V1、标准 V2、无 horizon embedding、full-rank QK、low-rank QK 的受控消融；
8. 自动生成可追溯的 `reports/V2_result.md`。

## 3. 方案选择

### 3.1 采用方案：独立 Horizon Attention 读出层

数据流固定为：

```text
Input [B,L,C_in]
  -> V1 patchify + pretrained/adapted Transformer backbone
  -> Historical hidden states H [B,N,D]
  -> HorizonAttention
  -> Horizon contexts C_h [B,P,D]
  -> shared structural non-crossing forecast head
  -> Prediction [B,P,C_out,3]
```

Horizon Attention 是 backbone 后的独立模块。它不修改 V1 encoder 内部 self-attention，也不使用自回归 decoder。

### 3.2 未采用方案

- **复用最后一层 encoder 的 Q/K/V projection**：会把 horizon readout 与 encoder 内部注意力强耦合，难以独立消融和测试；
- **新增 Transformer decoder/cross-attention stack**：增加深度、残差与 FFN 等额外变量，超出 V2 的最小创新范围；
- **为每个 horizon 建立完全独立预测网络**：参数量随 `P` 大幅增长，无法把收益归因于 temporal attention。

## 4. 复用边界

V2 必须直接复用：

- V1 的 `patchify`、patch embedding、learned positional embedding 和 Transformer backbone；
- V1 Stage I best checkpoint 及其严格契约校验；
- V1 的 `full | qk | qkv` backbone adaptation 策略；
- V0/V1 的数据切分、M/MS 模式、scaler、loss、metrics、logging、checkpoint、visualization；
- `epx_main.py` 的 train/validation/test/predict 生命周期；
- `run.py` 的模型注册式入口；
- 结构性非交叉区间参数化。

V2 新增版本化文件：

```text
models/v2_horizon_attention.py
configs/v2.yaml
specs/V2_horizon_attention_spec.md
tests/test_v2_horizon_attention.py
tests/test_v2_config.py
tests/test_v2_integration.py
reports/V2_result.md
```

允许对以下公共文件做最小扩展：

```text
models/__init__.py
utils/config.py
utils/checkpoint.py
epx_main.py
README.md
tests/test_checkpoint.py
tests/test_cli.py
tests/test_project_structure.py
tests/test_smoke_experiment.py
```

V2 不复制 V1 backbone、训练循环、指标或绘图实现。

## 5. 明确非目标

V2 不包含：

- forecasting Operator、diagonal operator 或 low-rank operator；
- Koopman dynamics；
- 自回归 decoder；
- 多层 cross-attention decoder；
- conformal calibration；
- dynamics-aware interval width；
- 重新训练或改变 Stage I masked pretraining 目标；
- 对预测区间使用 `sort()`、索引交换或后处理修复；
- 将 KV cache 宣称为主要算法创新；
- 修改 V0 或 V1 的数学行为以适配 V2。

## 6. 符号与 Tensor 契约

| 符号 | Shape | 含义 |
|---|---|---|
| `B` | scalar | batch size |
| `L` | scalar | 输入时间长度 |
| `N` | scalar | patch token 数量 |
| `P` | scalar | 预测 horizon 数量 |
| `C_in` | scalar | 输入通道数 |
| `C_out` | scalar | 输出通道数 |
| `D` | scalar | backbone hidden dimension |
| `R` | scalar | horizon Q/K relation dimension |
| `H` | `[B,N,D]` | V1 backbone 历史表示 |
| `K_hist` | `[B,N,R]` | 历史 Key |
| `V_hist` | `[B,N,D]` | 历史 Value |
| `E_H` | `[P,D]` | horizon embedding |
| `g` | `[B,D]` | 全局历史上下文 |
| `Q_h` | `[B,P,R]` | horizon Query |
| `A_h` | `[B,P,N]` | horizon-to-history 注意力 |
| `C_h` | `[B,P,D]` | horizon context |
| `pred` | `[B,P,C_out,3]` | 区间预测 |

所有模块必须显式验证 rank、固定维度和有限值。禁止依赖意外广播掩盖 shape 错误。

## 7. V1 Backbone 输出

输入首先沿用 V1 数据流：

\[
X:[B,L,C_{in}]
\rightarrow
X_{patch}:[B,N,S\cdot C_{in}]
\rightarrow
H:[B,N,D].
\]

V2 必须加载与 V1 相同契约的 Stage I checkpoint。V2 不再使用 V1 的：

\[
H[:,-1,:]\rightarrow\operatorname{Linear}(D,P\cdot C_{out}\cdot3)
\]

作为 forecast readout；它由 Horizon Attention 和逐 horizon 的共享 forecast head 取代。

V1 backbone 的参数冻结与 adapter 规则保持不变。Horizon Attention 与新的 forecast head 始终属于 Stage II 新增可训练参数。

## 8. Historical Key 与 Value

给定：

\[
H\in\mathbb{R}^{B\times N\times D},
\]

Historical Key 使用无偏置线性投影进入关系空间：

\[
K_{hist}=HW_K,
\qquad
W_K\in\mathbb{R}^{D\times R},
\]

得到：

\[
K_{hist}\in\mathbb{R}^{B\times N\times R}.
\]

Historical Value 使用无偏置线性投影并保持 hidden dimension：

\[
V_{hist}=HW_V,
\qquad
W_V\in\mathbb{R}^{D\times D},
\]

得到：

\[
V_{hist}\in\mathbb{R}^{B\times N\times D}.
\]

`W_Q`、`W_K`、`W_V` 是 V2 新增的 Stage II 参数，不从 V1 encoder projection 复制或共享权重。这样能保持 V1 checkpoint 契约不变，并使 V2 消融只作用于新增 readout。

## 9. Horizon Query

### 9.1 Horizon Embedding

建立可学习参数：

\[
E_H\in\mathbb{R}^{P\times D}.
\]

第 `p` 行只对应预测步 `t+p+1`。初始化采用 `trunc_normal_(std=0.02)`，与 V1 learned positional embedding 的初始化风格一致。

`P` 是模型 checkpoint 契约的一部分。加载不同 `pred_len` 的监督 checkpoint 必须明确失败，不允许截断、插值或部分加载 horizon embedding。

### 9.2 Global Context

`Pool` 固定为沿历史 token 维的无参数 mean pooling：

\[
g=\frac{1}{N}\sum_{n=1}^{N}H[:,n,:]
\in\mathbb{R}^{B\times D}.
\]

不使用 last-token pooling、max pooling 或额外 attention pooling，避免引入新的可训练混杂因素。

### 9.3 Query Projection

先广播相加：

\[
Z_h=g[:,None,:]+E_H[None,:,:]
\in\mathbb{R}^{B\times P\times D},
\]

再使用无偏置线性投影：

\[
Q_h=Z_hW_Q,
\qquad
W_Q\in\mathbb{R}^{D\times R},
\]

得到：

\[
Q_h\in\mathbb{R}^{B\times P\times R}.
\]

## 10. Temporal Attention

注意力 logits 为：

\[
S_h=\frac{Q_hK_{hist}^{\top}}{\sqrt{R}}
\in\mathbb{R}^{B\times P\times N}.
\]

softmax 只沿历史 token 维 `N` 计算：

\[
A_h=\operatorname{softmax}(S_h,\operatorname{dim}=-1)
\in\mathbb{R}^{B\times P\times N}.
\]

每个有限输入必须满足：

\[
A_h\ge 0,
\qquad
\sum_{n=1}^{N}A_h[:,:,n]=1.
\]

Horizon context 为：

\[
C_h=A_hV_{hist}
\in\mathbb{R}^{B\times P\times D}.
\]

V2 不使用 causal mask：Key/Value 全部来自已观测历史窗口，不包含未来 target。实现采用 batched matrix multiplication，不允许按 horizon 编写 Python 循环。

## 11. Low-rank 与 Full-rank 定义

V2 的 “low-rank QK” 指 Q/K 被直接投影到较小的关系维度 `R`，不是 V1 encoder 中的 LoRA-style adapter rank。

- **low-rank QK**：`1 <= R < D`；正式默认值为 `R=32`；
- **full-rank QK**：`R=D`；在当前默认 `D=128` 时即 `R=128`；
- 两种模式的 Value 均为 `D -> D`；
- 两种模式使用完全相同的 pooling、horizon embedding、forecast head、训练预算和数据；
- `sqrt(R)` scaling 必须使用当前实际 relation dimension。

配置中的 `adaptation.rank` 与 `horizon_attention.rank` 是两个不同概念，必须分别记录和校验。

## 12. Forecast Head

Horizon context 通过一个跨 horizon 共享的线性层：

\[
\operatorname{Linear}(D,C_{out}\cdot3).
\]

它逐 horizon 作用于 `[B,P,D]`，输出 raw tensor：

\[
R_{out}\in\mathbb{R}^{B\times P\times C_{out}\times3}.
\]

共享 head 不含 horizon-specific 参数；horizon 差异必须来自 `Q_h -> A_h -> C_h`。这保证 “without horizon embedding” 消融真正移除 horizon identity，而不是从预测头旁路泄漏 horizon 信息。

结构性区间参数化保持不变：

\[
Point=R_{out}[:,:,:,0],
\]

\[
Lower=Point-\operatorname{Softplus}(R_{out}[:,:,:,1])-\epsilon,
\]

\[
Upper=Point+\operatorname{Softplus}(R_{out}[:,:,:,2])+\epsilon.
\]

最终：

```python
pred = torch.stack((lower, point, upper), dim=-1)
```

输出严格为 `[B,P,C_out,3]`。禁止排序或后处理修复。

## 13. Without Horizon Embedding 消融语义

当 `use_horizon_embedding: false` 时：

\[
Z_h=g[:,None,:]
\]

被显式扩展到 `P` 个 horizon，且不创建或不使用可训练的 `E_H`。由于 Q/K/V 和 forecast head 共享，该变体对同一样本的所有 horizon 产生相同 context 和预测。

这种退化是受控消融的预期行为：它验证 horizon identity 是否是产生 horizon-specific historical dependency 的必要来源。实现不得通过 horizon-specific bias、独立 head 或位置索引旁路补回 horizon 信息。

## 14. KV Cache

### 14.1 定位

KV cache 仅是 inference optimization，不改变训练目标、注意力公式或模型表达能力，也不作为 V2 的主要算法创新。

### 14.2 接口与行为

Horizon Attention 模块应提供可独立测试的内部接口：

```python
project_history(H: Tensor) -> tuple[Tensor, Tensor]
build_queries(global_context: Tensor, horizon_indices: Tensor | None = None) -> Tensor
attend(Q_h: Tensor, K_cache: Tensor, V_cache: Tensor) -> tuple[Tensor, Tensor]
```

其中：

```python
K_cache = K_hist  # [B,N,R]
V_cache = V_hist  # [B,N,D]
```

在一次 vectorized forward 中，所有 horizon 必须天然共享同一份 `K_hist/V_hist`。显式 cache 接口用于推理时分块计算或重复请求不同 horizon，而不重复执行历史 K/V projection。

### 14.3 Cache 约束

- cache 只允许在 `model.eval()` 与 inference/no-grad 路径使用；
- cache 不写入 checkpoint，不跨输入窗口、模型权重、device 或 dtype 复用；
- `K_cache/V_cache` 的 batch、token、rank、device、dtype 必须严格匹配 query；
- 非缓存、一次性缓存和分块 horizon cache 三条路径必须在容差内数值一致；
- cache 不得改变 attention weights 的 softmax 维度或精度；
- 训练 forward 不接受外部 cache，避免陈旧图或错误梯度。

## 15. 参数冻结与训练边界

V2 继续支持 V1 的 backbone adaptation：

| `adaptation.mode` | V1 backbone | Encoder adapters | Horizon Attention | Forecast head |
|---|---|---|---|---|
| `full` | 全量可训练 | 无 | 可训练 | 可训练 |
| `qk` | 冻结 | 每层 Q/K 可训练 | 可训练 | 可训练 |
| `qkv` | 冻结 | 每层 Q/K/V 可训练 | 可训练 | 可训练 |

标准 V2 及 V2 内部消融固定使用同一个 `adaptation.mode`；正式比较默认采用 V1 已选定的 `qkv` 配置。不得在比较 low-rank/full-rank 或有/无 horizon embedding 时同时改变 encoder adapter 策略。

V2 新增参数包括：

- `horizon_embedding`（启用时）；
- `query_projection`；
- `key_projection`；
- `value_projection`；
- shared forecast head。

所有新增参数均参与 Stage II 优化。冻结参数在 backward 后必须保持 `grad is None`。

## 16. 配置设计

新增 `configs/v2.yaml`，继承 V1 的全部公共、patch、mask、pretrain 与 adaptation 字段，并增加严格的 `horizon_attention` section：

```yaml
experiment:
  name: v2_etth1_m

model:
  name: v2
  d_model: 128
  n_heads: 4
  n_layers: 3
  d_ff: 256
  dropout: 0.1
  activation: gelu
  positional_encoding: learned

horizon_attention:
  mode: low_rank              # low_rank | full_rank
  rank: 32                    # low_rank: 1 <= rank < D; full_rank: rank == D
  use_horizon_embedding: true
  cache_kv: true
```

严格校验规则：

1. `model.name` 必须为 `v2`；
2. `mode` 只允许 `low_rank | full_rank`；
3. `rank` 必须是正整数；
4. `low_rank` 要求 `rank < model.d_model`；
5. `full_rank` 要求 `rank == model.d_model`；
6. `use_horizon_embedding` 与 `cache_kv` 必须是 boolean；
7. 未知键、缺失键和错误类型必须尽早失败；
8. `data.pred_len`、`model.d_model`、`horizon_attention.mode/rank/use_horizon_embedding` 必须写入监督 checkpoint 契约。

V2 仍通过 `pretrain.checkpoint` 加载 V1 Stage I best checkpoint；该 checkpoint 的 stage/model 契约仍为 V1 pretraining，不改写为 V2。

## 17. 模型注册与 Checkpoint

`MODEL_REGISTRY` 新增 `v2` builder。构建顺序固定为：

1. 从配置构建与 V1 相同的 patch backbone；
2. 严格加载 V1 Stage I `backbone_state_dict`；
3. 应用 `adaptation.mode`；
4. 创建 Horizon Attention；
5. 创建 shared forecast head；
6. 记录 pretrain provenance、adapter 设置和 horizon attention 设置。

V2 监督 checkpoint 的 `model_name` 为 `v2`，且至少能够拒绝以下不兼容恢复：

- `pred_len` 不同；
- `d_model` 不同；
- relation `mode/rank` 不同；
- `use_horizon_embedding` 不同；
- patch/backbone/pretrain checkpoint 契约不同；
- M/MS 导致的 `output_dim` 不同。

不得按 shape 猜测或部分加载 V2 专属参数。

## 18. V2 消融矩阵

所有实验使用相同 ETTh1 split、输入长度、预测长度、seed、Stage I checkpoint、V1 encoder adaptation、optimizer、epoch budget、early stopping、loss、metrics 与输出通道模式。

| 实验 | Readout | Horizon embedding | Q/K relation | Value | 目的 |
|---|---|---|---|---|---|
| V1 | last patch token + V1 flattened head | 无 | 无新增 Q/K | 无新增 V | 基线 |
| V2 | Horizon Attention + shared head | 有 | low-rank, `R=32` | `D->D` | 标准模型 |
| V2 w/o horizon embedding | Horizon Attention + shared head | 无 | low-rank, `R=32` | `D->D` | 验证 horizon identity |
| V2 full-rank QK | Horizon Attention + shared head | 有 | `R=D=128` | `D->D` | 比较完整关系空间 |
| V2 low-rank QK | Horizon Attention + shared head | 有 | `R=32` | `D->D` | 验证低秩效率/效果 |

标准 V2 与 “V2 low-rank QK” 在数学配置上相同；前者是主结果行，后者在消融表中作为 low-rank 对照行引用同一次正式 run，不重复训练或伪造第二组独立结果。若论文表格要求五行，必须共享 experiment ID 并明确标注 alias。

为避免信息泄漏，任何 rank 选择只能使用 validation 指标；test split 只在确定配置并恢复 best validation checkpoint 后评估一次。

## 19. 指标、效率与报告

每个独立实验至少记录：

- experiment ID 与 resolved config；
- Stage I checkpoint ID/path；
- adaptation mode/rank/alpha；
- horizon attention mode/rank 与 horizon embedding 开关；
- total parameters、trainable parameters、trainable ratio；
- peak GPU memory；
- train time、inference time、单样本 latency；
- KV cache 开关以及缓存/非缓存延迟（优化对比，不作为主精度结果）；
- best epoch、best validation loss 与 convergence curves；
- MAE、RMSE；
- PICP、MPIW、PINAW、Coverage Error、Interval Score；
- overall 与每个 horizon 的完整指标；
- attention diagnostics：至少保存平均 horizon-to-history attention map `[P,N]`，仅用于解释，不参与 loss。

汇总文件固定为：

```text
reports/V2_result.md
```

报告必须区分算法消融与推理优化。不得把 KV cache 的速度收益解释为预测精度创新，也不得只报告最优变体而省略失败或退化变体。

## 20. 错误处理与数值安全

- 非法输入 shape、rank、horizon index 或 cache shape 必须抛出清晰 `ValueError`；
- `x/H/Q/K/V/logits/attention/context/prediction` 出现 NaN/Inf 时必须尽早失败，不静默修补；
- softmax 必须沿历史 token 维执行；
- relation scaling 必须使用 `sqrt(R)`，不能误用 `sqrt(D)` 或 encoder `head_dim`；
- 空 horizon 选择、重复或越界 horizon index 必须拒绝；
- 训练模式传入 cache 必须拒绝；
- `use_horizon_embedding=false` 时不得保留未使用的可训练 horizon 参数；
- 不允许对 cache 做隐式 device/dtype 转换，以免隐藏调用错误；
- 不允许通过循环逐 horizon 计算标准 forward。

## 21. 测试计划

### 21.1 Horizon Attention 单元测试

1. `H:[B,N,D]` 产生 `K:[B,N,R]`、`V:[B,N,D]`；
2. mean pooling 与手算结果一致；
3. `E_H` shape 为 `[P,D]`，固定初始化 seed 可复现；
4. `Q:[B,P,R]`、logits/weights `[B,P,N]`、context `[B,P,D]`；
5. attention 非负且沿 `N` 求和为 1；
6. 小型确定性张量的 scaled dot-product attention 与手算一致；
7. scaling 使用 `sqrt(R)`；
8. 修改单个 horizon embedding 只直接改变对应 horizon query；
9. 启用 horizon embedding 时，不同 embedding 可产生不同 attention；
10. 禁用 horizon embedding 时，全部 horizon query、attention、context 和输出相同；
11. low-rank 与 full-rank 的 projection shape 和参数量符合定义；
12. forward 不按 horizon 使用 Python 循环。

### 21.2 KV Cache 测试

1. 非缓存 forward 与缓存 forward 输出、attention 在容差内一致；
2. 全 horizon 一次计算与多个 horizon chunk 拼接结果一致；
3. mock/计数验证 K/V projection 在 cache 构建时各执行一次；
4. cache 的 batch、N、R、D、device、dtype 不匹配时明确失败；
5. training mode 使用外部 cache 明确失败；
6. cache 不出现在 `state_dict` 或 checkpoint 中；
7. 切换输入或参数后不静默复用旧 cache。

### 21.3 Forecast 与梯度测试

1. M 输出严格为 `[B,P,C_in,3]`；
2. MS 输出严格为 `[B,P,1,3]`；
3. 所有有限输出满足 `Lower<Point<Upper`；
4. 源码和执行路径不使用 `sort/argsort/交换边界`；
5. loss backward 后 Horizon Attention 与 forecast head 梯度存在且有限；
6. frozen V1 backbone 参数 `grad is None`；
7. 所选 encoder adapter 参数梯度存在且有限；
8. V2 不实例化 V1 的 flattened `Linear(D,P*C*3)` head；
9. `P=1`、`B=1` 和不同合法 `N/R` 均可运行；
10. 错误输入维度和非有限输入明确失败。

### 21.4 配置、注册与 Checkpoint 测试

1. `configs/v2.yaml` 严格加载，registry 包含 `v2`；
2. 未知/缺失 `horizon_attention` 字段失败；
3. low-rank 的 `R>=D` 失败，full-rank 的 `R!=D` 失败；
4. 非 boolean 开关失败；
5. V1 Stage I checkpoint 能严格初始化 V2 backbone；
6. 不兼容 V2 checkpoint 契约逐项拒绝；
7. V0/V1 配置与 checkpoint 回归测试保持通过。

### 21.5 集成与实验测试

1. V2 通过现有 `epx_main.py` 完成 train/validation/checkpoint reload/test/predict；
2. CPU smoke run 生成 metrics、predictions、figures 与 V2 report；
3. 固定 seed 的重复 smoke run 关键输出一致；
4. 五个报告行都能追溯到配置和 experiment ID；
5. 标准 V2 与 low-rank alias 不重复训练且引用同一结果；
6. V2 各变体除被消融字段外配置一致；
7. 完整 V0/V1/V2 测试套件通过。

## 22. 实施顺序

Spec 获得确认后，按 TDD 顺序实施：

1. 新增 V2 配置 schema、示例配置与失败测试；
2. 新增 Horizon Attention 的 shape、手算注意力和数值失败测试；
3. 实现 mean pooling、Q/K/V projection、horizon embedding 与 vectorized attention；
4. 新增 shared structural non-crossing forecast head 并验证 M/MS 输出；
5. 组合 V1 backbone 与 V2 readout，接入冻结/adapter 规则；
6. 注册 V2 model builder，严格复用 V1 Stage I checkpoint；
7. 扩展 V2 supervised checkpoint 契约与 provenance；
8. 实现并测试 inference-only KV cache；
9. 接入统一实验、报告和 attention diagnostics；
10. 完成 CPU smoke、V0/V1 回归与完整测试；
11. 固定正式配置后依次运行 V1/V2 消融；
12. 汇总并核验 `reports/V2_result.md`。

每一步先编写能表达行为且因缺失实现而失败的测试，再做最小实现，最后运行相关回归测试。不得为了 V2 修改 V1 的已有预测行为。

## 23. 验收标准

V2 完成必须同时满足：

1. `specs/V2_horizon_attention_spec.md`、`models/v2_horizon_attention.py`、`configs/v2.yaml` 存在；
2. V2 严格复用 V1 Stage I backbone checkpoint 与 Stage II adapter 规则；
3. backbone 输出为 `[B,N,D]`，Horizon Attention 输出为 `[B,P,D]`；
4. 每个 horizon 使用 `g+E_H[p]` 构建 query；
5. low-rank Q/K 为 `D->R` 且 `R<D`，full-rank 为 `R=D`；
6. attention logits 使用 `sqrt(R)` scaling，softmax 沿 `N`；
7. Value 为 `D->D`，context 为 `[B,P,D]`；
8. forecast head 跨 horizon 共享，输出为 `[B,P,C_out,3]`；
9. 所有有限预测结构性满足 `Lower<Point<Upper`；
10. V2 不包含 Operator、自回归 decoder 或排序后处理；
11. 无 horizon embedding 变体不存在 horizon identity 旁路；
12. KV cache 与非缓存结果数值一致，且只用于 inference；
13. frozen backbone 与 adapter 的梯度边界符合配置；
14. V2 继续复用统一数据、训练、评估、指标、绘图和产物目录；
15. V0/V1 行为及其回归测试不受影响；
16. V2 单元、集成、checkpoint、cache 和 smoke 测试全部通过；
17. 五个消融报告行除指定变量外配置一致；
18. test split 只在配置确定并恢复 best validation checkpoint 后评估一次；
19. `reports/V2_result.md` 包含精度、区间、效率、参数量、逐 horizon 与 provenance；
20. 标准 V2 和 low-rank alias 的结果引用关系清晰、无重复运行伪装。
