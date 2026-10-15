# V0 统一实验框架与 Transformer 区间预测基线规范

## 1. 文档目的与版本边界

本文档定义从空项目构建统一时间序列区间预测实验框架，以及首个模型版本：

\[
\boxed{\text{V0: Transformer Encoder + Structurally Non-crossing Quantile Head}}
\]

V0 的目标有两部分：

1. 建立后续 V1–V5 共用的数据、训练、验证、测试、推理、指标、日志、配置和 checkpoint 基础设施；
2. 实现一个足够简单、可复现、可公平消融的 Transformer 区间预测基线。

V0 不包含预训练、Adapter、Horizon Attention、Operator Attention、低秩算子或动态区间宽度机制。这些能力属于后续版本。后续版本原则上只新增模型文件和配置，并复用 V0 实验框架。

## 2. 任务定义

### 2.1 输入与目标

历史输入：

\[
X\in\mathbb{R}^{B\times L\times C_{in}}
\]

监督目标：

\[
Y\in\mathbb{R}^{B\times P\times C_{out}}
\]

模型输出：

\[
\hat{Y}\in\mathbb{R}^{B\times P\times C_{out}\times 3}
\]

其中：

- `B`：batch size；
- `L`：历史窗口长度 `seq_len`；
- `P`：预测长度 `pred_len`；
- `C_in`：输入变量数；
- `C_out`：目标变量数；
- 输出最后一维固定为 `[Lower, Point, Upper]`。

对应代码语义：

```python
pred[..., 0]  # Lower
pred[..., 1]  # Point / Median
pred[..., 2]  # Upper
```

### 2.2 支持的预测模式

V0 支持 ETT 常用的两种模式：

- `M`：多变量输入、多变量输出，`C_out = C_in`；
- `MS`：多变量输入、单变量输出，`C_out = 1`。

数据进入 Dataset 前已经完成列排序，目标变量固定为数值输入的最后一列。模型和 Dataset 不负责识别或重排目标列。

MS 模式下：

```python
x = history[:, :]       # [L, C_in]
y = future[:, -1:]      # [P, 1]，必须保留通道维
```

禁止使用 `future[:, -1]`，因为该写法会丢失目标通道维。

### 2.3 时间列约定

ETT 风格 CSV 包含一个 `date` 列和若干数值列。V0 中 `date` 仅用于：

- 按时间升序排序；
- 检查时间戳重复和顺序错误；
- 执行按时间顺序的数据划分。

V0 不将 `date`、日历编码或其他时间特征输入模型。

## 3. 项目目录设计

```text
project_root/
├── datasets/
│   ├── raw/
│   │   └── dataset.csv
│   ├── processed/
│   └── README.md
├── dataload/
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataloader.py
│   └── scaler.py
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   └── v0_transformer.py
├── losses/
│   ├── __init__.py
│   ├── quantile_loss.py
│   └── interval_loss.py
├── metrics/
│   ├── __init__.py
│   ├── point_metrics.py
│   └── interval_metrics.py
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── config.py
├── configs/
│   └── v0.yaml
├── specs/
│   └── V0_project_and_baseline_spec.md
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_checkpoint.py
│   └── test_smoke_experiment.py
├── reports/
├── checkpoints/
├── logs/
├── results/
├── epx_main.py
├── run.py
├── requirements.txt
└── README.md
```

职责边界：

- `datasets/`：仅保存数据和数据说明，不包含读取逻辑；
- `dataload/`：读取、校验、时间划分、scaler 和滑动窗口；
- `models/`：统一模型抽象与各版本网络；
- `losses/`：训练目标函数；
- `metrics/`：无状态、可聚合的评价指标；
- `utils/`：随机种子、配置、日志与 checkpoint 等横向能力；
- `epx_main.py`：版本无关的实验生命周期；
- `run.py`：用户入口、对象组装与模型注册；
- `configs/`：所有可变实验参数；
- `results/`：机器可读的最终指标和预测结果；
- `reports/`：人工阅读的版本实验总结；
- `logs/`：逐轮训练日志；
- `checkpoints/`：模型和恢复训练所需状态。

运行时产物目录保留在仓库中，但产物文件应由 `.gitignore` 排除，仅提交必要的占位文件。

## 4. 数据格式约定

### 4.1 CSV 格式

默认数据路径由配置指定：

```yaml
data:
  path: datasets/raw/dataset.csv
```

ETT 风格示例：

```text
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,...
```

约束如下：

- `date_col` 默认是 `date`；
- 数值特征不得包含 NaN、正负无穷；
- 时间戳必须可解析、无重复，排序后严格递增；
- `feature_cols` 明确给出输入列及其顺序；
- 目标变量在输入文件准备阶段已经放在 `feature_cols` 的最后；
- `target_cols` 仍写入配置用于记录实验语义和校验，但 V0 的模式行为由 `features` 决定；
- `M` 要求 `target_cols == feature_cols`；
- `MS` 要求 `target_cols == [feature_cols[-1]]`。

不满足约束时应在构建数据集阶段快速失败，并给出包含文件路径和违规字段的错误信息。V0 不自动填补缺失值，也不隐式删除异常行。

### 4.2 配置中的数据字段

```yaml
data:
  path: datasets/raw/dataset.csv
  date_col: date
  feature_cols: [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]
  target_cols: [OT]
  features: MS
  seq_len: 96
  pred_len: 24
  stride: 1
  batch_size: 32
  num_workers: 0
  pin_memory: false
  split:
    type: ratio
    train: 0.7
    val: 0.1
    test: 0.2
```

V0 使用通用的按比例时间划分，使相同框架也可处理符合 ETT 表结构的其他数据。划分点在原始时间轴上确定，禁止随机划分。若未来需要严格复现 ETT 官方边界，应新增显式的固定边界配置，而不是在 Python 中按数据集名称写死。

## 5. DataLoader 设计

### 5.1 时间划分与防泄漏

处理顺序固定为：

1. 加载并校验 CSV；
2. 按 `date` 升序排列；
3. 根据完整时间轴计算 train、validation、test 的边界；
4. 仅使用 train 时间段的数值行拟合 scaler；
5. 使用同一个 scaler 变换三个时间段；
6. 分别建立滑动窗口索引。

validation 和 test 可以携带各自边界之前最多 `seq_len` 行作为历史上下文，但其监督预测区间必须完全落在对应 split 内。上下文不会成为该 split 的预测目标，因此既保留边界处样本，也不产生未来信息泄漏。

对于窗口起点 `s`：

\[
X_s = Z[s:s+L],\qquad Y_s = Z[s+L:s+L+P]
\]

相邻窗口起点间隔为 `stride`。任何窗口都必须满足完整的历史段和预测段存在。

### 5.2 Dataset 返回值

每个样本返回：

```python
x: torch.FloatTensor  # [L, C_in]
y: torch.FloatTensor  # M: [P, C_in]；MS: [P, 1]
```

每个 batch 返回：

```python
x.shape == [B, L, C_in]
y.shape == [B, P, C_out]
```

Dataset 不返回时间特征。训练 DataLoader 可开启 shuffle；validation 和 test DataLoader 必须关闭 shuffle。为保证复现性，DataLoader 的 generator 和 worker 初始化函数使用统一 seed。

### 5.3 Scaler

`dataload/scaler.py` 提供可序列化的逐通道标准化器：

\[
z_{t,c}=\frac{x_{t,c}-\mu_c}{\max(\sigma_c,\epsilon_s)}
\]

- 均值和标准差只从 train 行计算；
- 零方差通道通过 `epsilon_s` 防止除零；
- 支持 `fit`、`transform`、`inverse_transform`、`state_dict` 和 `load_state_dict`；
- inverse transform 必须同时支持 `[B,P,C_out]` 与 `[B,P,C_out,3]`；
- MS 模式只对最后一个目标通道使用相应的 scaler 统计量；
- scaler 状态写入 checkpoint，确保独立推理可复现。

## 6. 配置系统

`configs/v0.yaml` 是单次实验的完整声明。配置加载后应执行 schema 级校验，禁止静默忽略未知键。

建议结构：

```yaml
experiment:
  name: v0_etth_ms
  seed: 2026
  device: auto
  output_root: .

data:
  path: datasets/raw/dataset.csv
  date_col: date
  feature_cols: [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]
  target_cols: [OT]
  features: MS
  seq_len: 96
  pred_len: 24
  stride: 1
  batch_size: 32
  num_workers: 0
  pin_memory: false
  split:
    type: ratio
    train: 0.7
    val: 0.1
    test: 0.2

model:
  name: v0
  d_model: 128
  n_heads: 4
  n_layers: 3
  d_ff: 256
  dropout: 0.1
  activation: gelu
  positional_encoding: sinusoidal

interval:
  lower_quantile: 0.05
  median_quantile: 0.50
  upper_quantile: 0.95
  epsilon: 1.0e-6

loss:
  interval_weight: 0.0

optimizer:
  name: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4

scheduler:
  name: none

training:
  epochs: 50
  early_stopping_patience: 10
  monitor: val_loss
  monitor_mode: min
  grad_clip_norm: 1.0
  resume_from: null

evaluation:
  coverage: 0.9
  save_predictions: true

logging:
  level: INFO
```

关键校验：

- `seq_len > 0`、`pred_len > 0`、`stride > 0`；
- `d_model % n_heads == 0`；
- `0 < q_lower < 0.5 < q_upper < 1`；
- V0 固定要求中间 quantile 为 `0.5`；
- `evaluation.coverage == q_upper - q_lower`；
- split 比例均为正且总和为 1；
- `features` 只能为 `M` 或 `MS`；
- M/MS 的 `feature_cols` 与 `target_cols` 满足第 4.1 节约束；
- `interval.epsilon > 0`。

命令行只覆盖少量运行参数，例如模型名、配置路径、设备和恢复路径；核心超参数来自 YAML，并将解析后的完整配置快照保存到实验目录。

## 7. 模型统一接口

`models/base_model.py` 定义抽象基类：

```python
class BaseForecastModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C_in]
        Returns:
            pred: [B, P, C_out, 3]
        """
        raise NotImplementedError
```

构造模型时显式传入至少以下信息：

- `input_dim = C_in`；
- `output_dim = C_out`；
- `pred_len = P`；
- 网络超参数。

统一约束：

- forward 只接收 `x`，不依赖 `y`；
- 不在模型中读取全局配置或数据路径；
- 不在模型中判断 M/MS；
- 不在模型中执行 inverse transform；
- 不使用 `sort()`、交换位置或任何输出后排序；
- 所有版本都返回相同的四维结构，只有 `C_out` 随任务模式改变。

## 8. V0 网络结构

### 8.1 总体结构

V0 使用 encoder-only Transformer：

1. 输入线性投影：`C_in -> d_model`；
2. 加入固定正弦位置编码；
3. 通过 `n_layers` 层 Transformer Encoder；
4. 取 Encoder 最后一个时间位置的表示；
5. 在 Encoder 之后使用独立线性预测头，一次性产生所有 horizon、目标通道和三个原始参数；
6. 使用中心—正宽度参数化构造不交叉区间。

V0 Encoder 默认不使用 causal mask。每个输入 token 只能访问给定的历史窗口，而整个历史窗口在预测时均已知，因此历史窗口内部的双向注意力不构成未来泄漏。

V0 明确选择 timestep embedding，而不使用 patch embedding：每个历史时间步对应一个 token，shape 从 `[B,L,C_in]` 直接变为 `[B,L,d_model]`。这样可减少 V0 特有的数据变换与超参数，保持基线清晰；patch 化留待后续版本在独立 Spec 中评估。

### 8.2 位置编码

固定正弦位置编码不引入与 `seq_len` 绑定的可学习参数：

\[
E_0=\operatorname{Linear}(X)+PE
\]

实现需要在构造时验证输入窗口不超过位置编码支持长度，或按实际 `seq_len` 构造足够长度的 buffer。

### 8.3 Encoder 与预测头

\[
H=\operatorname{TransformerEncoder}(E_0)
\]

其中：

\[
H\in\mathbb{R}^{B\times L\times d_{model}}
\]

读出向量：

\[
h=H[:, -1, :]\in\mathbb{R}^{B\times d_{model}}
\]

线性预测头位于 Transformer Encoder 之后：

\[
Z=\operatorname{Linear}(h)
\]

并 reshape 为：

\[
Z\in\mathbb{R}^{B\times P\times C_{out}\times 3}
\]

该预测头一次性预测全部 horizon，但不包含 horizon query 或 horizon attention，从而为 V2 留下清晰的单变量改动点。

### 8.4 结构性不交叉区间头

将 `Z` 最后一维解释为三个未约束原始量：

```python
raw_point = Z[..., 0]
raw_lower_gap = Z[..., 1]
raw_upper_gap = Z[..., 2]
```

定义：

\[
Point=raw\_point
\]

\[
Lower=Point-\operatorname{softplus}(raw\_lower\_gap)-\epsilon
\]

\[
Upper=Point+\operatorname{softplus}(raw\_upper\_gap)+\epsilon
\]

其中 `epsilon = interval.epsilon > 0`。最终按固定顺序 stack：

```python
pred = torch.stack([lower, point, upper], dim=-1)
```

因为 `softplus(z) > 0` 且 `epsilon > 0`，网络结构天然满足：

\[
Lower < Point < Upper
\]

该过程不是排序或后处理纠正，而是模型输出参数化本身。实现和测试使用有限值输入；若网络产生 NaN 或无穷，训练循环应立即报错，而不是尝试修复。

## 9. Tensor Shape Flow

以 M/MS 共用符号表示：

| 阶段 | 张量形状 |
|---|---|
| DataLoader 输入 | `[B, L, C_in]` |
| Input projection | `[B, L, d_model]` |
| Positional encoding | `[B, L, d_model]` |
| Transformer Encoder | `[B, L, d_model]` |
| Last-state readout | `[B, d_model]` |
| Linear prediction head | `[B, P*C_out*3]` |
| Reshape raw parameters | `[B, P, C_out, 3]` |
| Non-crossing parameterization | `[B, P, C_out, 3]` |

模式差异只体现在 `C_out`：

- M：`C_out = C_in`；
- MS：`C_out = 1`。

## 10. Loss 系统

### 10.1 Quantile loss

对误差 `e = y - y_hat`，pinball loss 为：

\[
\rho_q(e)=\max(qe,(q-1)e)
\]

V0 默认：

```text
q_lower = 0.05
q_point = 0.5
q_upper = 0.95
```

分别作用于 `pred[..., 0]`、`pred[..., 1]`、`pred[..., 2]`，随后对 batch、horizon、channel 和 quantile 求均值。实现必须检查预测与目标 shape 可广播关系，禁止因意外广播掩盖维度错误。

### 10.2 Interval loss 扩展点

`losses/interval_loss.py` 提供可选的 interval score 或 coverage-width 正则实现，但 V0 默认 `interval_weight = 0.0`，总训练目标仅为 quantile loss。保留该模块是为了后续消融，不允许默认改变 V0 的优化目标。

## 11. 指标系统

指标统一在原始数据尺度上计算。validation 用于 early stopping 的 `val_loss` 保持在标准化尺度，以与训练目标一致；同时记录 inverse transform 后的业务指标。

### 11.1 点预测指标

使用 `Point = pred[..., 1]` 与 `Y` 计算：

- MAE；
- MSE；
- RMSE。

V0 不默认使用 MAPE，避免真实值接近零时产生不稳定结果。

### 11.2 区间指标

使用 `Lower = pred[..., 0]`、`Upper = pred[..., 2]`：

- PICP（Prediction Interval Coverage Probability）；
- MPIW（Mean Prediction Interval Width）；
- PINAW（使用测试目标整体 range 归一化的平均区间宽度，range 为零时明确返回不可计算标记）；
- Winkler / Interval Score，名义覆盖率由 `evaluation.coverage` 指定。
- Coverage Error：`abs(PICP - nominal_coverage)`。

覆盖判断固定使用闭区间：

\[
\mathbb{1}[Lower\leq Y\leq Upper]
\]

所有点预测与区间指标必须同时输出 overall 和逐 horizon 结果；逐 channel 结果可选。指标聚合必须按元素数加权，不能简单平均大小不同 batch 的 batch metric。

## 12. 统一训练、验证、测试与推理接口

`epx_main.py` 至少暴露：

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device, config):
    ...

def validate(model, loader, loss_fn, scaler, device, config):
    ...

def test(model, loader, loss_fn, scaler, device, config):
    ...

def predict(model, loader, scaler, device, config):
    ...

def run_experiment(model, train_loader, val_loader, test_loader, config):
    ...
```

### 12.1 训练接口

`train_one_epoch`：

- 设置 `model.train()`；
- 将 batch 移至目标 device；
- forward 并验证输出 shape；
- 检查 loss 和输出为有限值；
- backward；
- 按配置执行 gradient clipping；
- optimizer step；
- 按样本元素数聚合 epoch loss。

### 12.2 验证接口

`validate`：

- 设置 `model.eval()`；
- 使用 `torch.inference_mode()`；
- 返回标准化尺度 loss；
- inverse transform 后计算点预测与区间指标；
- 不更新模型、optimizer、scheduler 或 scaler。

### 12.3 测试接口

`test` 只在训练结束并恢复最佳 checkpoint 后调用一次。它返回完整指标字典；根据配置保存预测、真实值和必要元数据。

### 12.4 推理接口

`predict` 不要求 target 参与模型 forward，但当前 DataLoader 可以同时提供 `x, y` 以便离线评估。返回 inverse transform 后：

```python
predictions: [N, P, C_out, 3]
```

保存为 `.npz`，至少包含 `predictions`；测试模式同时保存 `targets`。文件旁保存配置快照或实验标识，以支持追溯。

### 12.5 实验生命周期

`run_experiment` 固定执行：

1. 设置随机种子和确定性选项；
2. 创建唯一实验目录；
3. 保存解析后的配置；
4. 逐 epoch 训练与验证；
5. 记录日志和指标；
6. 根据 monitor 保存最佳 checkpoint；
7. 执行 early stopping；
8. 加载最佳 checkpoint；
9. 在 test split 上评估一次；
10. 保存结果与可选预测；
11. 返回最终结果字典。

## 13. `run.py` 与模型注册

用户主要入口：

```bash
python run.py --model v0 --config configs/v0.yaml
```

模型通过注册表构建：

```python
MODEL_REGISTRY = {
    "v0": V0Transformer,
}
```

后续版本只向注册表增加条目。禁止在 `run.py` 中堆叠长链式 `if/elif`。

`run.py` 负责：

- 解析命令行；
- 加载和校验配置；
- 设置 seed；
- 构建数据集、DataLoader 和 scaler；
- 从注册表构建模型；
- 检查 `--model` 与配置 `model.name` 一致；
- 调用 `run_experiment`。

`run.py` 不实现训练细节、指标公式或模型 forward。

## 14. Checkpoint 设计

最佳 checkpoint 至少包含：

```python
{
    "format_version": 1,
    "model_name": "v0",
    "epoch": int,
    "global_step": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict | None,
    "scaler_state_dict": dict,
    "config": dict,
    "best_metric": float,
    "rng_state": dict,
}
```

保存策略：

- 每个实验至少维护 `best.pt`；
- 可选维护 `last.pt` 以支持中断恢复；
- 先写临时文件，再以原子替换方式完成 checkpoint 保存，避免部分写入；
- 加载时校验 `format_version`、模型名以及关键维度；
- 推理至少恢复 model、scaler 和 config；
- 恢复训练还需恢复 optimizer、scheduler、epoch、global step 与 RNG 状态。

默认 monitor 为 `val_loss`，方向为 `min`。测试必须使用 `best.pt`，不能直接使用最后一个 epoch 的内存权重。

## 15. Logging 与实验结果

每次运行创建唯一实验 ID，例如：

```text
v0_etth_ms_seed2026_YYYYMMDD-HHMMSS
```

对应输出：

```text
logs/<experiment_id>/train.log
logs/<experiment_id>/metrics.jsonl
checkpoints/<experiment_id>/best.pt
checkpoints/<experiment_id>/last.pt
results/<experiment_id>/config.resolved.yaml
results/<experiment_id>/metrics.json
results/<experiment_id>/predictions.npz
reports/V0_result.md
```

日志至少记录：

- 实验 ID、模型版本、device、seed；
- 数据路径、split 样本数、窗口数；
- 输入输出维度和模型参数量；
- 每个 epoch 的 train loss、val loss、学习率和耗时；
- validation 主要点/区间指标；
- 最佳 epoch 与 early stopping 原因；
- 最终 test 指标和产物路径。

`reports/V0_result.md` 是 V0 的固定人类可读总结，至少记录：

- total parameters 与 trainable parameters；
- peak GPU memory（CPU 运行时标记为不适用）；
- 总训练时间；
- 推理总时间、样本数与单样本 latency；
- MAE、RMSE、PICP、MPIW、PINAW、Coverage Error、Interval Score；
- overall 与逐 horizon 指标；
- 最佳 epoch、实验 ID、配置快照和 checkpoint 路径。

报告由实验结果自动生成或更新，不依赖人工复制控制台数字。不同实验不得相互覆盖；`reports/V0_result.md` 可作为最近一次指定 V0 实验的摘要，同时在 `results/<experiment_id>/` 中保留不可覆盖的完整结果。

控制台和文件日志使用同一 logger，避免重复 handler。结构化指标使用 JSON Lines，最终摘要使用 JSON。不得把完整 batch 张量写入日志。

## 16. 可复现性要求

统一 seed 工具设置：

- Python `random`；
- NumPy；
- PyTorch CPU/CUDA；
- DataLoader generator 和 worker seed。

配置允许开启 PyTorch deterministic algorithms。若某设备上的算子无法确定性执行，应明确报错或记录降级，不得静默声称完全可复现。

实验记录还应包含 Python、PyTorch、CUDA、操作系统和设备信息。相同环境、数据、配置和 seed 下，CPU 测试应产生一致结果；GPU 复现遵循 PyTorch 对相应算子的确定性保证。

## 17. 错误处理原则

- 配置、数据 schema、shape 或有限值检查失败时尽早抛出明确异常；
- 不自动排序模型输出；
- 不自动修补 NaN/Inf；
- 不静默跳过损坏样本；
- 不在训练失败后自动更换模型或超参数；
- 输出目录冲突时创建新的唯一实验 ID，不覆盖既有实验；
- 用户指定恢复 checkpoint 不兼容时停止并说明差异。

## 18. 单元测试与集成测试计划

### 18.1 数据测试

`tests/test_dataset.py`：

- 合成 ETT 风格 CSV 能正确读取；
- date 仅用于排序，不出现在 `x`；
- M batch shape 为 `[B,L,C_in]` 和 `[B,P,C_in]`；
- MS batch shape 为 `[B,L,C_in]` 和 `[B,P,1]`；
- MS 的 target 等于未来数组最后一列且保留通道维；
- scaler 只拟合 train 数据；
- validation/test 目标不越过各自边界；
- 边界历史上下文不会进入监督目标；
- stride 和窗口数量正确；
- 重复日期、缺失列、NaN/Inf、非法 split 明确报错。

### 18.2 模型测试

`tests/test_model.py`：

- M 模式输出严格为 `[B,P,C_in,3]`；
- MS 模式输出严格为 `[B,P,1,3]`；
- 多组 batch size、seq_len、pred_len 可 forward；
- 对全部有限输出，逐元素断言 `lower < point` 和 `point < upper`；
- backward 后模型参数获得有限梯度；
- 模型源码和执行路径不调用 `sort` 或交换 lower/upper；
- 错误输入通道数时给出清晰异常。

### 18.3 Loss 测试

`tests/test_losses.py`：

- pinball loss 与手算小样本一致；
- 完美预测时 loss 为零；
- 三个 quantile 与输出索引对应正确；
- 错误 shape 不发生隐式广播而是报错；
- loss 能反向传播且梯度有限。

### 18.4 指标测试

`tests/test_metrics.py`：

- MAE/MSE/RMSE 与手算值一致；
- PICP 使用闭区间边界；
- MPIW、PINAW、Interval Score 与手算值一致；
- 不同大小 batch 的聚合按元素数加权；
- 零 range 的 PINAW 行为符合定义。

### 18.5 Checkpoint 测试

`tests/test_checkpoint.py`：

- 保存后加载得到相同模型输出；
- scaler 和配置能够恢复；
- 不兼容 `format_version` 或维度明确失败；
- `best.pt` 的选择符合 monitor 方向。

### 18.6 Smoke test

`tests/test_smoke_experiment.py` 使用小型合成 CSV：

- 运行 1–2 个 epoch；
- 完成 train、validate、best checkpoint reload、test、predict；
- 生成预期日志和结果文件；
- 最终预测 shape 正确且严格不交叉；
- 固定 seed 的两次 CPU smoke run 关键输出一致。

### 18.7 One-batch overfit test

使用无 dropout 的极小模型和固定单 batch 训练若干步，验证最终 quantile loss 显著低于初始 loss。该测试用于确认 forward、loss、backward 与 optimizer 的连接正确，不要求三个 quantile 在有限步内达到零损失。为控制常规测试耗时，可将其标记为 slow，但在 V0 验收时必须执行。

## 19. 执行计划

V0 在本 Spec 获批后按以下顺序实现：

1. 初始化目录、包文件、`.gitignore`、`requirements.txt` 和 README 骨架；
2. 先编写数据层失败测试，再实现 CSV 校验、时间划分、scaler、窗口 Dataset 和 DataLoader；
3. 编写 BaseModel 与 V0 shape/不交叉失败测试；
4. 实现输入投影、位置编码、Transformer Encoder、末状态读出和结构性非交叉预测头；
5. 编写并实现 quantile loss 与可选 interval loss；
6. 编写并实现点预测和区间指标；
7. 实现配置加载、严格校验、随机种子与模型注册；
8. 编写并实现 checkpoint 与 logging；
9. 实现 `epx_main.py` 的 train/validate/test/predict/run_experiment；
10. 实现 `run.py` 参数解析和对象组装；
11. 完成 smoke test、完整测试和最小合成数据运行；
12. 补全 README 使用方式，并实现 `reports/V0_result.md` 的结果生成。

每一步遵循测试先行：先写能表达行为且会失败的测试，再写最小实现使其通过，最后在测试保护下重构。

## 20. 验收标准

同时满足以下条件才认为 V0 完成：

1. 目录职责符合第 3 节，训练逻辑没有散落到模型或入口脚本；
2. `python run.py --model v0 --config configs/v0.yaml` 是主要执行方式；
3. 数据路径和所有关键超参数均来自配置，不硬编码数据路径；
4. date 不进入模型输入；
5. 只用 train 数据拟合 scaler，validation/test 不泄漏未来信息；
6. M 模式 batch 与预测 shape 分别为 `[B,L,C_in]`、`[B,P,C_in]`、`[B,P,C_in,3]`；
7. MS 模式 batch 与预测 shape 分别为 `[B,L,C_in]`、`[B,P,1]`、`[B,P,1,3]`；
8. 所有模型输出最后一维严格为 `[Lower,Point,Upper]`；
9. 对全部有限输出逐元素满足 `Lower < Point < Upper`；
10. 实现中没有通过 `sort()`、索引交换或后处理修复区间顺序；
11. V0 预测头位于 Transformer Encoder 之后，使用最后位置表示一次性产生所有 horizon 输出；
12. 同一套 `epx_main.py` 完成 train、validation、test 和 predict；
13. 测试只在恢复最佳 validation checkpoint 后执行一次；
14. checkpoint 足以恢复推理，并包含 scaler、配置和格式版本；
15. 点指标和区间指标在原始尺度正确计算并保存；
16. 日志、checkpoint、配置快照、最终指标和可选预测均写入独立实验目录；
17. 全部单元测试和 smoke test 通过；
18. 固定环境、数据、配置和 seed 的 CPU 重复 smoke run 结果一致；
19. 新增后续模型时不需要复制或改写训练循环；
20. README 能让新用户准备 ETT 风格 CSV、运行 V0 并定位输出产物。
21. 指标同时提供 overall 与逐 horizon 结果，并包含 Coverage Error；
22. one-batch overfit 测试通过；
23. `reports/V0_result.md` 包含参数量、可训练参数量、峰值 GPU 显存、训练时间、推理延迟及规定的点/区间指标。

## 21. V0 明确不做的事项

- 不输入 `date` 或日历时间特征；
- 不支持输入列和目标列任意映射；MS 目标固定为最后一个输入通道；
- 不自动下载 ETT 数据；
- 不在 V0 中实现 S（单变量到单变量）模式；
- 不实现概率分布参数化或采样；
- 不实现 V1–V5 的网络能力；
- 不在模型外对预测区间排序或交换；
- 不把每个未来步建模为独立 horizon query。

