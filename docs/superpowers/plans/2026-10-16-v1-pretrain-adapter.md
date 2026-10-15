# V1 Masked Pretraining and Low-rank Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 V0 框架上实现只使用 train split 的 masked patch pretraining，以及 V1-A full fine-tuning、V1-B Q/K adapter、V1-C Q/K/V adapter 三种监督预测模式。

**Architecture:** Stage I 将 ETTh1 历史窗口 patchify，以 Random/Tail Mask 和 masked-only MSE 训练共享 patch Transformer backbone，并保存严格契约化的 pretrain checkpoint。Stage II 严格加载该 backbone，按 `full/qk/qkv` 设置可训练参数，继续通过现有 `epx_main.py`、V0 非交叉区间头、指标、checkpoint 和图像系统完成预测。

**Tech Stack:** Python 3.12、PyTorch、NumPy、Pandas、PyYAML、Matplotlib、pytest。

**Spec:** `specs/V1_pretrain_adapter_spec.md`

## Global Constraints

- 输入保持 `[B,L,C_in]`，监督输出保持 `[B,P,C_out,3]`，最后一维严格为 `[Lower,Point,Upper]`。
- 非交叉必须由 `Softplus + epsilon` 参数化保证；禁止 `sort/argsort`、索引交换和后处理修复。
- Stage I 只能消费正式 train split；正式 validation/test 的输入和目标都不得参与预训练。
- 默认 `patch.length=16`、`patch.stride=8`，不 padding，必须完整覆盖 `seq_len`。
- 默认 Random Mask ratio 0.4、Tail Mask ratio 0.25、batch 级 Random 概率 0.5；每个样本至少一 masked 和一 visible patch。
- Reconstruction head 固定为 `Linear(D, patch_dim)`，loss 只计算 masked patches。
- V1-B 只训练 Q/K adapters 与 forecast head；V1-C 只训练 Q/K/V adapters 与 forecast head；其余 backbone 参数冻结。
- Stage II 必须显式指定兼容的 Stage I best checkpoint，不得静默随机初始化或部分加载。
- Stage II 必须复用 `epx_main.py`；不得复制监督训练循环。
- 用户当前未提交的 `configs/v0.yaml` 和 `reports/V0_result.md` 修改属于用户，实施时不得覆盖或误提交。
- 所有 Python 测试使用 `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest ... -p no:cacheprovider`。

## Review Focus

- 重叠 patch 不得让 masked token 保留原始 patch 数值；Task 2 用输入扰动验证 masked embeddings 隔离。
- Stage I resume 必须同时恢复全局 RNG、mask RNG 和 DataLoader generator；Task 4 比较连续与中断续训的下一 epoch mask/顺序。
- 融合或分离 Q/K/V projection 的 checkpoint 映射不得按 shape 猜测；Task 5 逐层逐参数验证加载和值相等。
- `full/qk/qkv` 模式的 `requires_grad` 与实际 backward gradients 可能不一致；Task 5 同时检查两者。
- 相同 shape 但不同 feature order、patch contract 或 backbone 配置必须拒绝；Task 4/5 覆盖语义契约冲突。

---

### Task 1: V0/V1 版本化配置契约

**Files:**
- Create: `configs/v1.yaml`
- Modify: `utils/config.py`
- Modify: `tests/test_config.py`
- Test: `tests/test_v1_config.py`

**Interfaces:**
- Consumes: `load_config(path) -> dict` 与 V0 现有严格 schema。
- Produces: `validate_config(config)` 对 `model.name=v0/v1` 分派严格 schema；V1 配置包含 `patch/mask/pretrain/adaptation`。

- [ ] **Step 1: 写 V1 配置失败测试**

覆盖合法默认配置，以及未知字段、patch 不完整覆盖、非法 ratio、`rank>d_model`、`alpha<=0`、非法 adaptation mode、Stage II checkpoint 类型错误。保留所有 V0 配置测试。

- [ ] **Step 2: 运行 RED**

Run: `...python.exe -m pytest tests/test_config.py tests/test_v1_config.py -q -p no:cacheprovider`

Expected: FAIL，因为 V1 顶层 section 和字段尚未被 schema 接受。

- [ ] **Step 3: 实现版本化 schema**

在 `utils/config.py` 提供：

```python
def validate_config(config: dict[str, Any]) -> None
def validate_v0_config(config: dict[str, Any]) -> None
def validate_v1_config(config: dict[str, Any]) -> None
```

共享 data/interval/optimizer/evaluation/visualization 验证，V1 单独验证 patch、mask、pretrain、adaptation。创建与 Spec 完全一致的 `configs/v1.yaml`，默认 `adaptation.mode: qk`、`pretrain.checkpoint: null`。

- [ ] **Step 4: 运行 GREEN 与 V0 回归**

Run: `...python.exe -m pytest tests/test_config.py tests/test_v1_config.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add configs/v1.yaml utils/config.py tests/test_config.py tests/test_v1_config.py
git commit -m "feat: add strict V1 configuration"
```

---

### Task 2: Patchify、Random Mask 与 Tail Mask

**Files:**
- Create: `dataload/pretrain.py`
- Modify: `dataload/__init__.py`
- Create: `tests/test_v1_patch_mask.py`

**Interfaces:**
- Consumes: train dataset 的 `x:[B,L,C]` 与统一 seed。
- Produces:

```python
def patchify(x: Tensor, patch_length: int, patch_stride: int) -> Tensor
def sample_random_mask(batch_size: int, n_patches: int, ratio: float, *, generator: Generator, device: device) -> BoolTensor
def sample_tail_mask(batch_size: int, n_patches: int, ratio: float, *, device: device) -> BoolTensor
def choose_mask(..., random_probability: float, generator: Generator) -> tuple[BoolTensor, str]
```

- [ ] **Step 1: 写 patch/mask 失败测试**

手算一个 `[1,6,2]` 序列在 `S=4,R=2` 下的三个 patches；测试非法覆盖拒绝、Random 无重复与数量、Tail 连续末尾、固定 seed、概率 0/1 分支、至少一 masked/visible。

- [ ] **Step 2: 运行 RED**

Run: `...python.exe -m pytest tests/test_v1_patch_mask.py -q -p no:cacheprovider`

Expected: FAIL，模块或 API 不存在。

- [ ] **Step 3: 实现纯函数**

使用 `Tensor.unfold` 保持时间顺序；mask 返回 `[B,N]` bool。所有 ratio 与 shape 错误给出实际值，不做 padding。

- [ ] **Step 4: 运行 GREEN**

Run: `...python.exe -m pytest tests/test_v1_patch_mask.py -q -p no:cacheprovider`

Expected: patch/mask 纯函数测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add dataload/pretrain.py dataload/__init__.py tests/test_v1_patch_mask.py
git commit -m "feat: add temporal patch masking"
```

---

### Task 3: V1 Patch Backbone 与 Masked Reconstruction

**Files:**
- Create: `models/v1_pretrain_adapter.py`
- Create: `losses/reconstruction_loss.py`
- Modify: `losses/__init__.py`
- Modify: `tests/test_v1_patch_mask.py`
- Create: `tests/test_v1_pretrain.py`

**Interfaces:**
- Consumes: `patchify`、`mask:[B,N]`、V1 model config。
- Produces:

```python
class V1PatchBackbone(nn.Module):
    def embed_patches(self, patches: Tensor, mask: BoolTensor | None = None) -> Tensor
    def forward(self, patches: Tensor, mask: BoolTensor | None = None) -> Tensor

class V1MaskedPretrainer(nn.Module):
    def forward(self, x: Tensor, mask: BoolTensor) -> tuple[Tensor, Tensor]
    def backbone_state_dict(self) -> dict[str, Tensor]

def masked_reconstruction_loss(reconstructed: Tensor, target: Tensor, mask: BoolTensor) -> Tensor
```

- [ ] **Step 1: 写 backbone/reconstruction 失败测试**

断言 embedding `[B,N,D]`、encoder `[B,N,D]`、reconstruction `[B,N,S*C]`；learnable position shape 正确；Q/K/V/O、FFN、LayerNorm、patch embedding 和 head 获得有限梯度。改变 masked 原始 patches 后，传给 Encoder 的 masked token 位置必须不变；visible 位置必须变化。

- [ ] **Step 2: 写 masked-only loss 手算测试**

构造 2 个 patches，只 mask 第 2 个；改变第 1 个 reconstruction 不影响 loss，改变第 2 个按手算 MSE 攐变。空 mask、shape 不匹配和 NaN 明确失败。

- [ ] **Step 3: 运行 RED**

Run: `...python.exe -m pytest tests/test_v1_patch_mask.py tests/test_v1_pretrain.py -q -p no:cacheprovider`

Expected: FAIL，因为 V1 classes/loss 不存在。

- [ ] **Step 4: 实现显式 Q/K/V/O attention 与 post-norm block**

在同一 V1 module 中实现可供 Stage I/II 共用的 projection-separated self-attention、FFN 和 LayerNorm；Stage I adapters disabled。`embed_patches` 先线性投影，再用 mask token 替换被 mask 位置，最后加 learnable position。

- [ ] **Step 5: 实现单层 reconstruction head 与 masked loss**

Reconstruction head 只能是 `nn.Linear(d_model, patch_length*input_dim)`。`backbone_state_dict()` 不包含 head 或 mask token。

- [ ] **Step 6: 运行 GREEN 与 backward 检查**

Run: `...python.exe -m pytest tests/test_v1_patch_mask.py tests/test_v1_pretrain.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 7: 提交**

```bash
git add models/v1_pretrain_adapter.py losses/reconstruction_loss.py losses/__init__.py tests/test_v1_patch_mask.py tests/test_v1_pretrain.py
git commit -m "feat: add V1 masked patch backbone"
```

---

### Task 4: Stage I 数据内部切分、训练引擎与 Pretrain Checkpoint

**Files:**
- Modify: `dataload/pretrain.py`
- Create: `pretrain.py`
- Modify: `utils/checkpoint.py`
- Modify: `utils/logger.py`
- Modify: `utils/visualization.py`
- Modify: `tests/test_v1_pretrain.py`
- Create: `tests/test_v1_pretrain_engine.py`

**Interfaces:**
- Consumes: V0 `DataBundle.train_loader.dataset`、`V1MaskedPretrainer`、masked loss、公共 seed/logger/save_checkpoint。
- Produces:

```python
def create_pretrain_loaders(bundle: DataBundle, config: dict) -> tuple[DataLoader, DataLoader]
def train_pretrain_epoch(...) -> dict[str, float | int]
def validate_pretrain(...) -> dict[str, float | int]
def run_pretraining(config: dict) -> dict[str, Any]
def load_pretrain_checkpoint(path, *, expected_config, map_location="cpu") -> dict[str, Any]
```

- [ ] **Step 1: 写数据边界失败测试**

验证 train dataset 按窗口顺序 90/10 切分、两子集无交集、最大原始 target/history 索引不超过正式 `train_end`；val loader 不 shuffle。

- [ ] **Step 2: 写 Stage I 生命周期失败测试**

用 ETTh1 的少量真实 train windows 运行 1 epoch，断言 best/last checkpoint、resolved config、metrics JSON/JSONL、训练曲线存在；checkpoint 包含 Spec 规定字段。

- [ ] **Step 3: 写 checkpoint 契约与 resume 失败测试**

相同 tensor shape 但 feature order、patch stride 或层数不同必须拒绝；连续两 epoch与 1 epoch+resume 的下一轮 loader 顺序、mask 类型/索引、模型权重一致。

- [ ] **Step 4: 运行 RED**

Run: `...python.exe -m pytest tests/test_v1_pretrain.py tests/test_v1_pretrain_engine.py -q -p no:cacheprovider`

Expected: FAIL，Stage I engine/checkpoint API 不存在。

- [ ] **Step 5: 实现 loaders 与训练/验证函数**

训练 mask 使用可 checkpoint 的独立 `torch.Generator`；validation 按 sample index 与固定 seed 生成稳定 mask。聚合 loss 必须按 masked elements 加权。

- [ ] **Step 6: 实现 pretrain checkpoint 和 CLI**

复用 atomic `save_checkpoint`，保存 backbone、mask token、head、optimizer、scaler、配置、三类 RNG 与 early-stopping 状态。`pretrain.py --config configs/v1.yaml` 为唯一 Stage I 主入口。

- [ ] **Step 7: 扩展通用路径/绘图，不复制基础设施**

为 pretrain experiment 创建唯一目录，训练曲线使用现有 Matplotlib 论文风格；不要创建监督 predictions 或 interval metrics。

- [ ] **Step 8: 运行 GREEN、ETTh1 小批 overfit 与 V0 回归**

Run: `...python.exe -m pytest tests/test_v1_pretrain.py tests/test_v1_pretrain_engine.py tests/test_utils.py tests/test_checkpoint.py -q -p no:cacheprovider`

Expected: PASS；小批 reconstruction loss 下降。

- [ ] **Step 9: 提交**

```bash
git add dataload/pretrain.py pretrain.py utils/checkpoint.py utils/logger.py utils/visualization.py tests/test_v1_pretrain.py tests/test_v1_pretrain_engine.py
git commit -m "feat: add reproducible V1 pretraining"
```

---

### Task 5: Low-rank Q/K/QKV Adapter 与冻结策略

**Files:**
- Modify: `models/v1_pretrain_adapter.py`
- Create: `tests/test_v1_adapter.py`

**Interfaces:**
- Consumes: Stage I `backbone_state_dict` 与 config `adaptation.mode/rank/alpha/dropout`。
- Produces:

```python
class LowRankProjectionAdapter(nn.Module):
    def forward(self, x: Tensor) -> Tensor

class V1PretrainAdapter(BaseForecastModel):
    def load_pretrained_backbone(self, state: dict[str, Tensor]) -> None
    def configure_adaptation(self, mode: Literal["full", "qk", "qkv"]) -> None
    def forward(self, x: Tensor) -> Tensor
    def parameter_summary(self) -> dict[str, int | float]
```

- [ ] **Step 1: 写零初始化与 projection 失败测试**

Adapter 的 `B` 全零、`A` 非零；插入 adapter 前后相同输入输出逐元素一致。直接修改 Q adapter 后只影响 Q 路径；K/V 同理。

- [ ] **Step 2: 写 checkpoint 严格加载失败测试**

逐层比较 Stage I/II patch embed、position、Q/K/V/O、FFN、LayerNorm；丢失键、额外键和语义契约不一致都必须失败，不能按相同 shape 误载。

- [ ] **Step 3: 写 full/qk/qkv 冻结与梯度失败测试**

对真实 quantile loss backward：

```python
assert all(parameter.grad is None for frozen parameters)
assert all(parameter.grad is not None and finite for adapter parameters)
assert all(parameter.grad is not None for forecast-head parameters)
```

并验证 V1-A 无 adapter且整个 backbone 可训练；V1-B 无 V adapter；V1-C 有 V adapter；B/C trainable ratio 均小于 A。

- [ ] **Step 4: 运行 RED**

Run: `...python.exe -m pytest tests/test_v1_adapter.py -q -p no:cacheprovider`

Expected: FAIL，adapter/forecast model 尚未实现。

- [ ] **Step 5: 实现 adapter 与模式配置**

计算 `s*(dropout(x) @ A.T @ B.T)` 并加到冻结 base projection；`A:[r,D]` Kaiming、`B:[D,r]` zeros。冻结通过 `requires_grad_(False)` 显式完成，不依赖 optimizer 过滤掩盖错误。

- [ ] **Step 6: 实现 V0-compatible forecast head**

最后 patch token `[B,D]` 一次线性映射到 `[B,P,C_out,3]` raw slots，使用 V0 `Softplus+epsilon` 组装 `[Lower,Point,Upper]`。

- [ ] **Step 7: 运行 GREEN 与禁用排序扫描**

Run: `...python.exe -m pytest tests/test_v1_adapter.py tests/test_model.py -q -p no:cacheprovider`

Run: `rg -n "torch\.(sort|argsort)|\bsort\(" models`

Expected: tests PASS；扫描无预测排序调用。

- [ ] **Step 8: 提交**

```bash
git add models/v1_pretrain_adapter.py tests/test_v1_adapter.py
git commit -m "feat: add frozen V1 projection adapters"
```

---

### Task 6: 模型注册、统一监督实验与 V1 报告

**Files:**
- Modify: `models/__init__.py`
- Modify: `run.py`
- Modify: `epx_main.py`
- Modify: `utils/logger.py`
- Modify: `utils/checkpoint.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_experiment.py`
- Create: `tests/test_v1_integration.py`

**Interfaces:**
- Consumes: `V1PretrainAdapter`、Stage I checkpoint、现有 `run_experiment`。
- Produces: registry-based `build_model("v1",...)`、通用 V0/V1 report renderer、trainable ratio 与 pretrain provenance。

- [ ] **Step 1: 写 registry/CLI 失败测试**

`MODEL_REGISTRY` 同时包含 v0/v1；`build_model("v1")` 必须要求 pretrain checkpoint；`run.py` 不增加模型 if/elif 链。

- [ ] **Step 2: 写统一引擎失败测试**

用微型 Stage I checkpoint 和 ETTh1 子集分别运行 full/qk/qkv 1 epoch；断言现有 `epx_main.py` 生成 best/last、metrics、predictions、figures，输出严格不交叉。

- [ ] **Step 3: 写报告失败测试**

V1 report 必须包含 pretrain checkpoint ID、mode/rank/alpha、total/trainable/ratio、GPU memory、timing、MAE/RMSE/PICP/PINAW/Interval Score 与 by-horizon。

- [ ] **Step 4: 运行 RED**

Run: `...python.exe -m pytest tests/test_cli.py tests/test_experiment.py tests/test_v1_integration.py -q -p no:cacheprovider`

Expected: FAIL，v1 未注册且报告硬编码为 V0。

- [ ] **Step 5: 将 registry 改为 builder callables**

V0/V1 各自 builder 消费同一 `build_model(name, config, input_dim, output_dim)` 接口；注册表选择 builder，不在 `run.py` 写模型分支。

- [ ] **Step 6: 扩展统一结果字段**

`epx_main.py` 统一记录 `trainable_ratio`；若模型提供 provenance/parameter summary，则无分支地并入结果。监督训练循环本身不变。

- [ ] **Step 7: 通用化报告路径与 renderer**

根据 `model.name` 输出 `reports/V0_result.md` 或 `reports/V1_result.md`；保留向后兼容的 V0 内容并增加 V1 字段。

- [ ] **Step 8: 运行 GREEN 与 V0 回归**

Run: `...python.exe -m pytest tests/test_cli.py tests/test_experiment.py tests/test_v1_integration.py tests/test_smoke_experiment.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 9: 提交**

```bash
git add models/__init__.py run.py epx_main.py utils/logger.py utils/checkpoint.py tests/test_cli.py tests/test_experiment.py tests/test_v1_integration.py
git commit -m "feat: integrate V1 with shared experiments"
```

---

### Task 7: 文档、端到端验收与完整回归

**Files:**
- Modify: `README.md`
- Create: `reports/V1_result.md`
- Modify: `tests/test_project_structure.py`
- Modify: `tests/test_smoke_experiment.py`

**Interfaces:**
- Consumes: Stage I/II CLI 与全部公共 artifacts。
- Produces: 可复现的 V1 用户流程与结构验收。

- [ ] **Step 1: 写项目结构和两阶段 smoke 失败测试**

断言 Spec、V1 model/config、pretrain entry、V1 report 存在；用 ETTh1 稀疏窗口执行 Stage I 1 epoch，再将其 best checkpoint 注入 Stage II qk 1 epoch。

- [ ] **Step 2: 运行 RED**

Run: `...python.exe -m pytest tests/test_project_structure.py tests/test_smoke_experiment.py -q -p no:cacheprovider`

Expected: 新 V1 smoke 在文档/产物尚未齐全处 FAIL。

- [ ] **Step 3: 更新 README 与初始报告模板**

记录：环境、Stage I 命令、如何把 best path 写入 V1 config、V1-A/B/C 命令与产物位置。报告由实验结果自动覆写，不手工填造指标。

- [ ] **Step 4: 运行完整测试**

Run: `...python.exe -m pytest tests -q -p no:cacheprovider`

Expected: 全部 PASS，包括 V0 回归、ETTh1 overfit、V1 pretrain/adapter/two-stage smoke。

- [ ] **Step 5: 静态与产物验证**

Run: `...python.exe -m compileall -q dataload models losses metrics utils epx_main.py run.py pretrain.py`

Run: `git diff --check`

Run: `rg -n "torch\.(sort|argsort)|\bsort\(" models`

Expected: compile/diff PASS；无预测排序。

- [ ] **Step 6: 清理测试生成的 ignored artifacts**

仅删除已核验位于当前 workspace 的 `results/` 测试目录，保留 `.gitkeep`；不删除用户正式结果或未核验路径。日志/checkpoint 按同样的明确测试 experiment ID 清理。

- [ ] **Step 7: 提交**

```bash
git add README.md reports/V1_result.md tests/test_project_structure.py tests/test_smoke_experiment.py
git commit -m "docs: complete V1 two-stage workflow"
```

---

### Task 8: ETTh1 正式 V1-A/B/C 消融与最终报告

**Files:**
- Modify: `reports/V1_result.md`
- Generated/ignored: `checkpoints/<experiment_id>/`, `logs/<experiment_id>/`, `results/<experiment_id>/`

**Interfaces:**
- Consumes: 固定 `configs/v1.yaml`、同一个 Stage I best checkpoint、同一 seed/data/supervised budget。
- Produces: V1-A/B/C 可追溯结果、图像和汇总报告。

- [ ] **Step 1: 运行正式 Stage I**

Run: `...python.exe pretrain.py --config configs/v1.yaml`

Expected: best/last checkpoint 与 reconstruction convergence artifacts 完整；记录 best checkpoint ID。

- [ ] **Step 2: 锁定三份 resolved experiment config**

仅改变 `adaptation.mode` 为 `full/qk/qkv`，三组指向同一个 Stage I best checkpoint。比较其余字段，必须完全一致。

- [ ] **Step 3: 依次运行 V1-A/B/C**

Run: `...python.exe run.py --model v1 --config <v1-a-resolved.yaml>`，并对 B/C 重复。

Expected: 三组均完成 best checkpoint reload、test、metrics、predictions 和 figures。

- [ ] **Step 4: 验证正式 artifacts**

每组 NPZ 必须有限、shape 为 `[N,P,C_out,3]` 且严格不交叉；metrics JSON 必须包含参数、显存、时间和规定指标；B/C trainable ratio 小于 A。

- [ ] **Step 5: 自动汇总 `reports/V1_result.md`**

报告使用表格比较 V0/V1-A/V1-B/V1-C，并链接/记录各 experiment ID、pretrain checkpoint、resolved config、best checkpoint 和 figures。

- [ ] **Step 6: 最终完整验证**

Run: `...python.exe -m pytest tests -q -p no:cacheprovider`

Expected: 全部 PASS。随后检查 `git status --short`，只提交报告和应跟踪源码，不提交 ignored 大体积 artifacts。

- [ ] **Step 7: 提交**

```bash
git add reports/V1_result.md
git commit -m "docs: report V1 adapter ablations"
```
