# V2 严格结构审计、Attention 诊断与公平消融规范

## 1. 范围及顺序

在 `feature/v2` 执行：先保存本规范；随后只读审计现有 V1/V2，记录原始 attention 并完成诊断；最后新增隔离的公平消融模型并运行四组实验。当前 V1/V2 forward、原配置、原 checkpoint 和原报告保留。禁止推进 V3/V4，禁止先通过修改原网络改善诊断结果。

## 2. 审计对象与证据

- 当前 `configs/v1.yaml`、`configs/v2.yaml`，ETTh1 M，L=P=96、C=7、D=128、3 encoder layers、QKV adapters rank=8。
- 现有正式 V1/V2 best checkpoints，分别来自 `v1_horizon_baseline_seed2026_20261016-144820-008596_e622d6` 和 `v2_horizon_low_rank_seed2026_20261016-144949-667162_06ae41`。
- 从配置重建真实模型并严格加载 checkpoint，以 `named_parameters` 的参数身份去重，按 `requires_grad` 统计；不能只根据 state dict 推断训练状态。
- 每个参数保存 name、shape、numel、trainable、所属分类；打印每个分类及每层子模块的 total/trainable/frozen。必须满足 total=trainable+frozen，分类和等于总数，无遗漏、无重复。

分类固定为 Patch Embedding、Transformer Q/K/V/O（分别列出）、FFN、Adapter、Horizon Attention、Forecast Head，并额外单列 Position Embedding、LayerNorm、Mask Token，避免把不属于上述类别的参数隐藏在总数中。Horizon Attention 再拆分 Q、K、V、Horizon Embedding；Adapter 按层和 Q/K/V 拆分。

## 3. 已从代码确定、待运行核验的结构差异

V1 forecast head 为 `Linear(D,P*C*3)`，V2 为跨 horizon 共享的 `Linear(D,C*3)`；现有代码没有额外多层 Forecast Decoder。

在正式配置下，预期精确分解为：

| 项目 | V1 | 当前 V2 rank32 |
|---|---:|---:|
| Adapter 可训练参数 | 18,432 | 18,432 |
| Forecast Head 可训练参数 | 260,064 | 2,709 |
| Horizon Attention 可训练参数 | 0 | 36,864 |
| 全部可训练参数 | 278,496 | 58,005 |
| 冻结参数 | 413,440 | 413,440 |
| 全部参数 | 691,936 | 471,445 |

`278496 - 260064 + 2709 + 36864 = 58005`。即原头减少 257,355 参数，新增 attention 36,864 参数，净减少 220,491 参数。运行审计必须核验 Adapter 数量、尺寸、冻结策略和全部非 head/non-attention 模块是否一致，不以此预期表代替实际统计。

结论必须区分：结构事实可证实；原 V1/V2 精度差异同时混入 readout/head 变化，不能作为仅增加 Horizon Attention 的因果证据。

## 4. 当前 V2 Attention 诊断

### 4.1 数据与保存

使用现有 V2 best checkpoint，eval + inference mode，禁用随机采样。主诊断集为完整 validation split（预计 1,647 个窗口）；不使用 test 选择 rank 或阈值。逐 batch 记录原始 `A:[B,P,N]`，合并为 `[S,P,N]`，预计 `[1647,96,11]`。

保存原始 attention、稳定样本顺序/窗口索引、horizon 索引、patch 覆盖区间、split、checkpoint/config/source commit/hash。不能仅对已平均的 `[P,N]` 图计算诊断。所有 attention 必须有限、非负、沿 N 归一化。

### 4.2 指标定义

对每个样本独立计算，再在样本维汇总，保留分布和分位数：

1. **Pairwise cosine**：`cos(a_p,a_q)`，排除对角线，输出平均 `[P,P]` 矩阵、逐样本 off-diagonal mean、整体 p10/p50/p90，以及相邻 horizon cosine。
2. **Entropy**：`H_p=-sum_n a_pn log(a_pn)`，0log0=0；保存 nats 和 `H_p/log(N)`，N=1 时规范化 entropy 定义为0并标记退化。
3. **Effective tokens**：`exp(H_p)`；以及 top1 mass、`KL(a_p || Uniform)=log(N)-H_p`。
4. **Pairwise total variation**：`0.5*sum_n |a_pn-a_qn|`。cosine 对概率向量天然偏高，必须用 TV 与 entropy 联合判断，不能仅凭高 cosine 宣告 collapse。
5. **预注册诊断旗标（启发式，不是统计证明）**：逐样本 horizon-collapse 候选为 off-diagonal mean cosine≥0.99 且 mean TV≤0.01；uniform 候选为 normalized entropy≥0.95；尖锐候选为 normalized entropy≤0.20 或 top1 mass≥0.90。报告旗标比例，不调整阈值去匹配结果。
6. 三段 horizon：short=1–32、middle=33–64、long=65–96。分别汇总 entropy/top1、段内与跨段 cosine/TV。一般 P 用尽可能等长的三个非空连续分组；P<3 不计算三段比较，显式标注。

### 4.3 图像

生成 300 DPI PNG 和矢量 PDF：完整 `[P,N]` 平均 heatmap、短/中/长三面板 heatmap、平均 pairwise cosine heatmap、entropy 随 horizon 的均值及 p10/p90 区间图。额外保存 validation 第0、中间、最后样本的三段 heatmap，检查均值是否掩盖样本异质性。

x 轴表示历史 patch，标注其在历史窗口中的 `[n*stride,n*stride+length-1]` 区间；y 轴为真实预测步。三段 heatmap 使用相同色标，图中明确是否为样本平均。不得把 heatmap 外观当成精度提高的证据。

## 5. 公平消融的严格定义

### 5.1 实验矩阵

| 变体 | Backbone / adapters | Forecast head | Horizon Attention |
|---|---|---|---|
| V1 | 原 V1 QKV adapter | 原 V1 完整头 | 无，使用最后历史 token |
| V1 + full-rank HA | 相同 | 同一结构/相同初始权重 | R=D=128 |
| V1 + rank64 HA | 相同 | 同一结构/相同初始权重 | R=64 |
| V1 + rank32 HA | 相同 | 同一结构/相同初始权重 | R=32 |

这些是四个独立训练实验，当前 V2 原结果只作为结构审计/诊断对象，不混入公平对照。

### 5.2 保留原 Forecast Head

原 V1 head 权重为 `[P*C*3,D]`、bias 为 `[P*C*3]`。将同一参数张量视为 `[P,C*3,D]` 和 `[P,C*3]`：

`raw[b,p,j] = sum_d context[b,p,d] * W[p,j,d] + bias[p,j]`。

V1 的 `context[b,p,:]=H[b,-1,:]`，所以该分块计算与原 V1 flatten linear 输出等价。HA 组则使用 `context=C_h`。不缩小 head，不新增 head 参数，不做 mean-pooling 后再丢弃 horizon 差异，不增加额外残差/FFN/Operator。

新的实验模型类型为 `v2_fair`，与当前 V2 分开注册。它继承 V1 初始化路径，先创建完全相同的 backbone/adapters/head，最后才创建 HA，避免 attention rank 改变非 HA 初始化 RNG。

### 5.3 参数与初始化门禁

- 每组在训练前枚举所有非 HA 参数的 name/shape/requires_grad/初值 hash，必须与 V1 完全一致；训练后权重允许不同，这是注意力介入训练后的结果。
- 所有非 HA 可训练参数固定 278,496，冻结参数固定 413,440。
- 预期新增 HA 参数：R128=61,440；R64=45,056；R32=36,864。总可训练参数分别339,936、323,552、315,360。
- 每组保持相同 Stage I checkpoint/scaler、QKV adapter rank/alpha/dropout、head、noncrossing 参数化、loss/quantiles、optimizer及其参数组规则、lr/weight_decay、epoch上限50、patience10、batch size32、seed2026、数据路径/特征顺序/时间切分、L=P=96。
- epoch预算/early-stopping规则相同，实际停止 epoch 可以不同，不能因此强行更换 best checkpoint。
- 运行前保存受控配置 diff 和非 HA 参数 hash；发现差异直接停止训练。
- 所有测试使用各自 validation best checkpoint。固定配置后每组只做一次 test 评估；新实验报告不得覆盖原 `reports/V1_result.md` 或 `reports/V2_result.md`。

## 6. 预期结论及证据强度

1. 可证实/否证 head 缩小是否解释原参数下降，以及 Adapter 是否真的保持不变。
2. 可证实当前 V2 在 validation 上是否具有高相似、近 uniform 或尖锐 attention 的候选特征；候选不等于已证明其造成性能下降。
3. 公平矩阵用于区分 head 缩小混杂因素与 HA/rank 的影响，不预设 HA 优于 V1，也不预设 low-rank 优于 full-rank。
4. 比较 overall 和完整 by-horizon 的 MAE/RMSE/PICP/MPIW/PINAW/Coverage Error/Interval Score，以及参数量、训练耗时、early-stopping与显存。一个 seed 不能支持显著性或普适性宣称。
5. 若公平 HA 仍逊于 V1，必须如实报告；不在本任务中调参、修改原网络或推进 V3/V4。

## 7. 文件改动

新增：

```text
specs/V2_diagnosis_spec.md
diagnose_v2.py                          只读结构审计、validation attention 采集/指标/图像
models/v2_fair_horizon.py               隔离的公平 HA 实验模型
v2_fair_ablation.py                     受控配置/参数哈希门禁、四组实验、聚合报告
configs/diagnosis/v1.yaml
configs/diagnosis/v1_horizon_full.yaml
configs/diagnosis/v1_horizon_rank64.yaml
configs/diagnosis/v1_horizon_rank32.yaml
tests/test_v2_diagnosis.py
tests/test_v2_fair.py
reports/V2_diagnosis_result.md
reports/V2_fair_ablation_result.md
```

最小扩展：`models/__init__.py`、`utils/config.py` 注册 `v2_fair`；`run.py` 仅扩展其报告标签/provenance，`README.md` 增加命令。原网络公式和训练循环保持不变。

诊断产物使用唯一目录 `results/v2_diagnosis_<id>/`，保存参数明细 JSON、结构审计 txt/JSON、原始 attention NPZ、诊断 JSON 和 figures。公平实验仍使用现有唯一 experiment ID 目录，新增不可覆盖的控制配置/初值 hash manifest。控制台必须打印结构分项表，不只提供文件路径。

## 8. 执行与验收

1. 规范落盘后执行原 V1/V2 只读结构审计，首先打印逐模块统计及参数差额解释。
2. 对现有 V2 validation 采集 raw attention 并计算/绘制上述指标。
3. 诊断完成后新增公平实验模型，先测试 V1 原头与分块头计算等价、非 HA 参数初值一致、各 rank 参数差额、冻结梯度和输出 shape。
4. 证明不公平配置/参数门禁会失败，再运行四组正式消融。
5. 运行相关测试及完整回归，核验每组配置、初值 hash、checkpoint与结果来源；原 V1/V2 文件不发生网络行为变更。
6. 输出两份独立报告；所有结构结论有参数枚举证据，attention结论基于逐样本指标，消融结论基于相同非 HA 模型容量与训练预算。
