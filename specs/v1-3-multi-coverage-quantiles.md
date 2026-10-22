# V1.3 多覆盖率分位区间预测改造规范

## 0. 文档状态

- 目标分支：`v1-3-10quantile`
- 状态：**已确认并完成代码实施；正式 Stage-II 重训练待执行**
- 本文定义修改目标、接口、指标口径、兼容性和验收标准。
- 实现已通过项目完整测试，并已验证现有 Stage-I checkpoint 可加载到新的监督预测头。

## 1. 背景与目标

当前 V1 的监督预测头只输出三个分位点：

```text
[Q0.05, Q0.50, Q0.95]
```

它只能直接形成一个 90% 中心预测区间。为了与同时报告多个覆盖率的论文 baseline 公平比较，本次改造要求 V1 直接学习 19 个分位点，并由它们构造 10% 至 90% 的九个中心预测区间。

本次核心目标为：

1. V1 Stage-II 直接输出 5% 至 95%、步长为 5% 的 19 个条件分位点；
2. 从这些分位点直接构造 10%、20%、……、90% 九种中心预测区间；
3. 以 Q0.50 作为唯一点预测；
4. 在不插值内部预测分位点的前提下计算各覆盖率的区间指标；
5. 使用全部 19 个预测分位点计算更可靠的 CRPS；
6. 将 QICE 定义为九种中心区间的平均覆盖校准误差，并以百分点为单位输出；
7. 保持 V1 Stage-I 预训练流程和 V0 三分位行为不变。

## 2. 术语与固定集合

### 2.1 预测分位点

分位点集合固定为：

\[
\mathcal Q=\{0.05,0.10,0.15,\ldots,0.90,0.95\}.
\]

共 19 个严格递增的分位水平。Q0.50 位于索引 9（从 0 开始），作为点预测。

### 2.2 中心覆盖区间

覆盖率集合固定为：

\[
\mathcal C=\{0.10,0.20,0.30,\ldots,0.80,0.90\}.
\]

对覆盖率 \(c\)，中心预测区间定义为：

\[
I_c=
\left[
Q_{(1-c)/2},
Q_{(1+c)/2}
\right].
\]

固定映射如下：

| 覆盖率 | 下界 | 上界 |
|---:|---:|---:|
| 10% | Q0.45 | Q0.55 |
| 20% | Q0.40 | Q0.60 |
| 30% | Q0.35 | Q0.65 |
| 40% | Q0.30 | Q0.70 |
| 50% | Q0.25 | Q0.75 |
| 60% | Q0.20 | Q0.80 |
| 70% | Q0.15 | Q0.85 |
| 80% | Q0.10 | Q0.90 |
| 90% | Q0.05 | Q0.95 |

不允许通过现有三个分位点插值生成这些内部边界；19 个分位点必须由模型直接预测并参与训练。

## 3. 范围与非目标

### 3.1 本次修改范围

- V1 Stage-II 监督预测头；
- V1 配置契约；
- 通用 Quantile Loss 对任意分位点数量的支持；
- 监督训练、验证、测试与预测中的形状校验；
- scaler 对任意长度分位轴的支持；
- CRPS、QICE 和多覆盖率区间指标；
- `metrics.json`、`predictions.npz`、Markdown 报告与图表；
- checkpoint 兼容性校验；
- 对应单元测试、集成测试和说明文档。

### 3.2 非目标

- 不修改 V1 Stage-I masked pretraining 目标和网络；
- 不修改 V0 的三分位输出协议；
- 不修改数据集划分、scaler 拟合范围或滑动窗口逻辑；
- 不引入分布参数化假设（如 Gaussian、Student-t 或混合分布）；
- 不使用输出后排序修复 quantile crossing；
- 不把插值产生的分位点作为模型预测结果；
- 不在本次改造中加入 conformal calibration；
- 不改变 adapter 的 `full`、`qk`、`qkv` 三种模式。

## 4. 配置契约

### 4.1 V1 新配置

V1 使用显式列表配置预测分位点和评估覆盖率：

```yaml
interval:
  quantiles:
    - 0.05
    - 0.10
    - 0.15
    - 0.20
    - 0.25
    - 0.30
    - 0.35
    - 0.40
    - 0.45
    - 0.50
    - 0.55
    - 0.60
    - 0.65
    - 0.70
    - 0.75
    - 0.80
    - 0.85
    - 0.90
    - 0.95
  epsilon: 1.0e-6

evaluation:
  coverage_levels:
    - 0.10
    - 0.20
    - 0.30
    - 0.40
    - 0.50
    - 0.60
    - 0.70
    - 0.80
    - 0.90
  primary_coverage: 0.90
  save_predictions: true
```

V1 不再同时保存 `lower_quantile`、`median_quantile`、`upper_quantile`，避免两套配置产生不一致。V0 继续使用现有三分位配置，不强制迁移。

### 4.2 严格校验

V1 配置必须满足：

- `interval.quantiles` 恰好等于 0.05 至 0.95、步长 0.05 的 19 个值；
- 分位水平有限、唯一、严格递增且全部位于 `(0, 1)`；
- 必须且只能包含一个 0.50；
- `coverage_levels` 恰好等于 0.10 至 0.90、步长 0.10 的九个值；
- 每个覆盖率对应的上下分位点必须存在于 `interval.quantiles`；
- `primary_coverage` 必须属于 `coverage_levels`，本版本固定为 0.90；
- `epsilon > 0`；
- 未知键、缺失键和重复值必须立即报错。

配置 schema 应按模型版本区分：V0 保留旧 `interval/evaluation` schema，V1 使用本节的新 schema。

## 5. 模型输出协议

### 5.1 输出形状

V1 Stage-II 输出由：

```text
[B, P, C_out, 3]
```

改为：

```text
[B, P, C_out, 19]
```

最后一维与 `interval.quantiles` 一一对应，禁止依赖隐含或未保存的顺序。

### 5.2 结构性非交叉输出头

输出投影产生 19 个 raw 值。以 Q0.50 为锚点：

\[
\hat Q_{0.50}=r_9.
\]

中位数左侧从内向外递减：

\[
\hat Q_i=\hat Q_{i+1}-\operatorname{softplus}(r_i)-\epsilon,
\quad i=8,7,\ldots,0.
\]

中位数右侧从内向外递增：

\[
\hat Q_i=\hat Q_{i-1}+\operatorname{softplus}(r_i)+\epsilon,
\quad i=10,11,\ldots,18.
\]

该结构必须保证：

\[
\hat Q_{0.05}<\hat Q_{0.10}<\cdots<\hat Q_{0.95}.
\]

不得使用 `sort`、裁剪、交换上下界或推理后修复。

### 5.3 参数初始化

- 输出头维度变更为 `pred_len * output_dim * 19`；
- 保持当前 PyTorch 默认权重初始化，除非测试证明累计 gap 导致明显数值异常；
- 如需专门初始化 gap bias，必须单独记录设计依据和消融结果，不在未确认情况下引入。

## 6. 训练目标

### 6.1 多分位 Pinball Loss

对所有 19 个分位点计算等权 pinball loss：

\[
L_{quantile}
=\frac{1}{19N}
\sum_{q\in\mathcal Q}
\sum_{n=1}^{N}
\rho_q(y_n-\hat Q_{q,n}),
\]

其中：

\[
\rho_q(e)=\max(qe,(q-1)e).
\]

`QuantileLoss` 必须支持任意长度的严格递增分位列表，不再写死为 3。

### 6.2 可选 Interval Score 辅助项

为保持原有超参数语义，`loss.interval_weight > 0` 时只对 `primary_coverage=0.90` 的 Q0.05–Q0.95 计算辅助 Interval Score：

\[
L=L_{quantile}+\lambda L_{IS,0.90}.
\]

本次不对九个覆盖率的 Interval Score 求和，避免重复加权造成训练目标尺度发生不可控变化。默认 `interval_weight=0` 时训练目标仅为多分位 pinball loss。

## 7. 评估指标口径

所有指标均在 inverse transform 后的原始数据尺度计算。`overall` 对全部测试样本、forecast horizon 和输出通道的标量元素等权聚合；`by_horizon` 在每个 horizon 内对样本和通道等权聚合。

### 7.1 点预测指标

只使用 Q0.50 计算：

- MAE；
- MSE；
- RMSE。

### 7.2 九个覆盖率的区间指标

对每个 \(c\in\mathcal C\) 分别计算：

- PICP；
- MPIW；
- PINAW；
- Coverage Error：`abs(PICP - c)`；
- Interval Score，使用该覆盖率自己的 `alpha = 1 - c`。

现有顶层 `picp/mpiw/pinaw/coverage_error/interval_score` 字段继续保留，但明确代表 `primary_coverage=0.90`，以降低下游报告的破坏性变化。

### 7.3 QICE

根据本次需求，QICE 使用九个**中心覆盖区间**的经验覆盖误差：

\[
\operatorname{QICE}
=\frac{100}{|\mathcal C|}
\sum_{c\in\mathcal C}
\left|\widehat{PICP}_c-c\right|.
\]

输出单位为百分点，字段名为 `qice`。例如 `qice = 2.5` 表示九个覆盖率的平均绝对覆盖偏差为 2.5 个百分点，而不是比例值 0.025。

本定义不是 CARD 风格的“十个等概率相邻 quantile bins”QICE。两种口径不得混称或直接比较。论文对比前必须确认 baseline 使用的公式；如目标论文采用 equal-bin QICE，应另行扩展指标并使用不同字段名，不能静默替换本定义。

### 7.4 CRPS

CRPS 使用 19 个直接预测分位点构造分段线性 quantile function，并计算：

\[
\operatorname{CRPS}(F,y)
=2\int_0^1\rho_q(y-Q(q))\,dq.
\]

要求：

- 0.05 至 0.95 内只使用模型直接预测的分位点做分段线性连接；
- 0 至 0.05、0.95 至 1 的尾部沿相邻分位段线性外推；
- 积分采用现有解析分段实现，不创建高密度网格张量；
- 输出 `crps` 为原始目标尺度上的平均值，越低越好；
- 报告必须注明这是 quantile-integral CRPS，不得与归一化 CRPS 或 `CRPS_sum` 混为同一指标。

## 8. 结果与产物协议

### 8.1 `predictions.npz`

必须保存：

```text
predictions      [N, P, C_out, 19]
targets          [N, P, C_out]
quantile_levels  [19]
coverage_levels  [9]
```

`quantile_levels` 和 `coverage_levels` 必须与实际计算使用的配置完全一致，禁止只依赖外部 YAML 推断最后一维含义。

### 8.2 `metrics.json`

`overall` 和每个 `by_horizon` 记录均包含：

```json
{
  "mae": 0.0,
  "mse": 0.0,
  "rmse": 0.0,
  "crps": 0.0,
  "qice": 0.0,
  "primary_coverage": 0.9,
  "picp": 0.0,
  "mpiw": 0.0,
  "pinaw": 0.0,
  "coverage_error": 0.0,
  "interval_score": 0.0,
  "coverage_intervals": [
    {
      "coverage": 0.1,
      "lower_quantile": 0.45,
      "upper_quantile": 0.55,
      "picp": 0.0,
      "mpiw": 0.0,
      "pinaw": 0.0,
      "coverage_error": 0.0,
      "interval_score": 0.0
    }
  ]
}
```

`coverage_intervals` 按覆盖率升序保存九条记录。顶层区间指标必须与其中 `coverage=0.9` 的记录完全相等。

### 8.3 Markdown 报告

报告包含：

1. Overall 点预测与分布指标表：MAE、RMSE、CRPS、QICE；
2. Overall 多覆盖率表：九行覆盖率及其 PICP、MPIW、PINAW、Coverage Error、Interval Score；
3. By-horizon 主表：MAE、RMSE、CRPS、QICE 和 primary 90% 区间指标；
4. 对 CRPS 与 QICE 计算口径及单位的简短说明。

不在 Markdown 中展开 `96 horizons × 9 coverages` 的完整大表；完整数据保留在 `metrics.json`。

### 8.4 可视化

- Forecast interval 图继续默认绘制 primary 90% 区间与 Q0.50；
- 新增 overall coverage calibration 图：横轴 nominal coverage 10%–90%，纵轴 empirical PICP；
- Horizon metrics 图展示 CRPS、QICE 和 primary 90% 区间指标；
- 如展示嵌套区间，最多选择 50%、80%、90% 三层，避免九层填充遮挡；
- PNG/PDF 双格式及现有 DPI 配置保持不变。

## 9. 通用代码与版本边界

### 9.1 V1 专属行为

- `V1PretrainAdapter` 使用 19 分位非交叉头；
- `configs/v1.yaml` 使用新分位和覆盖率配置；
- V1 的监督预测、指标和产物采用本文协议。

### 9.2 V0 向后兼容

- `V0Transformer` 继续输出 `[B,P,C_out,3]`；
- `configs/v0.yaml` 保持原有 lower/median/upper 和单一 coverage；
- 通用 loss、metrics、scaler 和 experiment engine 可以支持任意 Q，但不得改变 V0 的既有数值语义；
- V0 的既有 checkpoint 和测试必须继续有效。

### 9.3 Scaler

当前 scaler 通过 `last_dim == 3` 判断分位轴，该逻辑必须移除。新协议按张量语义处理：

- 目标 `[N,P,C_out]` 的通道轴为最后一维；
- 分位预测 `[N,P,C_out,Q]` 的通道轴为倒数第二维；
- Q 可以是 3、19 或其他经配置验证的正整数；
- inverse transform 不得把分位轴误当通道轴。

## 10. Checkpoint 兼容性

### 10.1 Stage-I 预训练 checkpoint

Stage-I checkpoint 契约不包含监督分位输出头和 interval 配置。只要 backbone、patch、mask、数据列及 scaler 契约一致，现有 V1 Stage-I checkpoint 可以继续复用，不要求重新预训练。

### 10.2 Stage-II 监督 checkpoint

旧监督 checkpoint 的输出头形状为 3，新模型为 19，且 `interval` 配置契约发生变化，因此：

- 必须明确拒绝把旧三分位监督 checkpoint 加载到新模型；
- 错误信息应指出 output head 或 interval contract 不兼容；
- 不允许部分加载旧输出头；
- 新版本需要重新进行 Stage-II 监督训练；
- 新 checkpoint 必须保存完整 quantile/coverage 配置以支持确定性恢复。

## 11. 预计修改文件

实现阶段预计涉及：

```text
configs/v1.yaml
models/base_model.py
models/v1_pretrain_adapter.py
models/__init__.py
losses/quantile_loss.py
losses/interval_loss.py
metrics/__init__.py
metrics/interval_metrics.py
metrics/probabilistic_metrics.py
dataload/scaler.py
utils/config.py
utils/checkpoint.py
utils/visualization.py
epx_main.py
run.py
README.md
tests/test_config.py
tests/test_model.py
tests/test_losses.py
tests/test_metrics.py
tests/test_scaler.py
tests/test_checkpoint.py
tests/test_experiment.py
tests/test_visualization.py
tests/test_v1_adapter.py
tests/test_v1_integration.py
tests/test_smoke_experiment.py
```

实施时可以根据职责拆分新增小型辅助模块，但不得复制一套 V1 专用训练引擎。

## 12. 实施顺序

1. 增加 V1 版本化配置 schema 和 quantile/coverage 解析辅助函数；
2. 将 Quantile Loss 和 Interval Loss 从固定三分位推广到配置驱动；
3. 实现 V1 的 19 分位结构性非交叉输出头；
4. 泛化 experiment engine 的形状和单调性校验；
5. 修正 scaler 的任意分位轴处理；
6. 实现九覆盖率区间指标、本文定义的 QICE 和 19 分位 CRPS；
7. 扩展 NPZ、JSON、Markdown 和图表输出；
8. 更新 checkpoint 契约错误和文档；
9. 完成单元、集成、smoke 和回归测试；
10. 使用新 Stage-II 配置重新训练并记录实验结果。

## 13. 测试计划

### 13.1 配置测试

- 接受精确的 19 分位和九覆盖率配置；
- 拒绝缺失、重复、乱序、越界或不对称配置；
- 拒绝不存在对应上下分位点的 coverage；
- 验证 V0 旧配置仍然有效。

### 13.2 模型测试

- V1 输出形状严格为 `[B,P,C_out,19]`；
- 所有相邻分位点严格递增；
- 极端负 gap logits 下仍保持最小 `epsilon` 间距；
- 测试禁止调用 `sort`；
- `full/qk/qkv` 下输出头参数均可训练并得到有限梯度。

### 13.3 Loss 测试

- 用手算小例子验证 19 分位 pinball loss；
- 验证 prediction/target/quantile 数量不匹配时拒绝广播；
- 验证非零 `interval_weight` 只加入 90% Interval Score；
- 验证 loss 和梯度均有限。

### 13.4 指标测试

- 完美校准构造下九个 coverage error 和 QICE 均为 0；
- 已知偏差构造下验证 QICE 的百分点数值；
- 验证每个 coverage 使用正确的上下分位索引；
- 验证顶层区间指标等于 90% 记录；
- 验证 CRPS 的解析积分与高精度数值积分一致；
- 验证 overall 与 by-horizon 均按标量元素而不是 batch 等权聚合。

### 13.5 Scaler 与产物测试

- 三分位和 19 分位预测均能正确 inverse transform；
- MS 与 M 模式的通道轴均正确；
- NPZ 包含 predictions、targets、quantile_levels、coverage_levels；
- JSON/Markdown 中 QICE 单位明确为 `%`/百分点；
- 图表能在 headless 环境生成 PNG/PDF。

### 13.6 回归测试

- V0 原有三分位行为和测试不变；
- V1 Stage-I checkpoint 加载与预训练测试不变；
- 旧 V1 Stage-II checkpoint 被清晰拒绝；
- V1 端到端 smoke experiment 能完成训练、保存、重载和测试；
- 全量测试不得保留已知失败项。

## 14. 验收标准

只有同时满足以下条件，本次改造才算完成：

- [x] V1 直接输出 19 个分位点；
- [x] 所有样本、horizon、通道上的 19 个分位点严格不交叉；
- [x] 可直接构造 10%–90% 九个中心覆盖区间；
- [x] Q0.50 是 MAE/MSE/RMSE 的唯一点预测；
- [x] 每个覆盖率均输出 PICP、MPIW、PINAW、Coverage Error 和 Interval Score；
- [x] QICE 按本文九覆盖率公式计算并以百分点输出；
- [x] CRPS 使用全部 19 个直接预测分位点；
- [x] NPZ 保存分位水平和覆盖率元数据；
- [x] JSON、报告和图表明确指标口径；
- [x] V0 三分位行为不变；
- [x] Stage-I checkpoint 可复用，旧 Stage-II checkpoint 被明确拒绝；
- [x] 相关测试和项目全量测试全部通过（165 passed）；
- [ ] 完成一次新的 V1 Stage-II 实验并生成可复现实验产物。

## 15. 待用户确认的关键决策

实施前请确认以下五点：

1. 分位点固定为 0.05–0.95、步长 0.05，共 19 个；
2. 覆盖率固定为 10%–90%、步长 10%，共九个中心区间；
3. `qice` 使用九个中心区间 Coverage Error 的平均值并乘 100，而不是 equal-bin QICE；
4. 非零 `interval_weight` 仍只作用于 primary 90% 区间；
5. V0 保持三分位兼容，现有 V1 Stage-I checkpoint 复用，V1 Stage-II 重新训练。

