# V2 Horizon-Aware Temporal Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在 V1 预训练 patch backbone 与 adapter 基础上，实现每个预测步独立 query 的低秩 Horizon-Aware Temporal Attention、推理 KV cache 和可复现消融报告。

**Architecture:** V2 继承 `V1PretrainAdapter` 以复用 Stage I checkpoint 严格加载和 `full/qk/qkv` 冻结规则，但用独立 `HorizonTemporalAttention` 与跨 horizon 共享的非交叉 forecast head 替换 V1 flattened head。Horizon 模块以 mean-pooled history 加 learned horizon embedding 生成 Q，将历史 H 投影成低秩 K 和全维 V；显式 cache 仅在 eval/inference 中复用 K/V。统一训练、评估、checkpoint、metrics 和 visualization 继续由现有框架负责，独立消融运行器只编排配置和汇总结果。

**Tech Stack:** Python 3.12、PyTorch、NumPy、PyYAML、pytest；不新增运行时依赖。

**Spec:** `specs/V2_horizon_attention_spec.md`

## Global Constraints

- 监督输入保持 `[B,L,C_in]`，输出保持 `[B,P,C_out,3]`，最后一维严格为 `[Lower,Point,Upper]`。
- 非交叉继续由 `Softplus + epsilon` 结构保证；禁止 `sort/argsort`、索引交换或后处理修复。
- V2 必须严格复用 V1 Stage I checkpoint、patch backbone 和 `full/qk/qkv` adaptation；不得复制预训练或监督训练循环。
- `Pool(H)` 固定为沿 token 维的无参数 mean pooling。
- Q/K 使用无 bias 的 `D->R` projection；V 使用无 bias 的 `D->D` projection；attention scaling 固定为 `sqrt(R)`，softmax 固定沿 `N`。
- 默认 low-rank relation `R=32` 且 `R<D`；full-rank 固定 `R=D`。`horizon_attention.rank` 与 `adaptation.rank` 不得混用。
- Forecast head 固定为跨 horizon 共享的 `Linear(D,C_out*3)`；horizon identity 只能来自 `E_H`。
- 禁用 horizon embedding 时不得创建未使用的可训练 `E_H`，也不得通过独立 head/bias 补回 horizon identity。
- KV cache 只允许在 `eval()` + inference/no-grad 中使用；不得进入 checkpoint，不得跨输入、权重、device 或 dtype 静默复用。
- V0/V1 数学行为保持不变；对 V1 的重构只允许为复用接口且必须由原有回归测试保护。
- 当前已提交的 `configs/v1.yaml` 和 `reports/V1_result.md` 是正式 V1 实验状态，不得改回模板值；依赖 `checkpoint:null/mode:qk` 的测试改用独立测试 fixture。
- 正式 V2 及内部消融固定相同 Stage I checkpoint、`adaptation.mode=qkv`、数据 split、seed、训练预算、loss 和 metrics。
- 所有 pytest 命令使用 `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest ... -p no:cacheprovider`；测试产物使用 pytest `tmp_path`，不得写入仓库 `results/`。

## Review Focus

- `P=1`、`B=1` 或单 token `N=1` 时 squeeze/广播不得丢失维度；Task 2 的 shape 参数化测试固定覆盖。
- 禁用 horizon embedding 后不得残留 horizon-specific 参数或预测差异；Task 2/3 同时检查 state dict、query、attention、context 和最终预测。
- cache 在 optimizer step 或 `load_state_dict` 后不得继续被接受；Task 5 用参数 version stamp 验证陈旧 cache 拒绝。
- horizon index 必须是一维、唯一、非空、升序可选但输出保持请求顺序，越界/重复/错误 dtype 明确失败；Task 2/5 覆盖全部输入类。
- V2 从 V1 Stage I checkpoint 初始化时，V2 专属 `model.name/horizon_attention` 不得误参与 pretrain contract，而 supervised resume 必须校验它们；Task 4 同时覆盖两种 checkpoint 边界。

---

### Task 1: V2 严格配置契约与文件骨架

**Files:**
- Create: `configs/v2.yaml`
- Create: `tests/test_v2_config.py`
- Modify: `utils/config.py:11-183`
- Modify: `tests/test_v1_config.py:40-44`
- Modify: `tests/test_cli.py:31-37`
- Modify: `tests/test_project_structure.py:25-33`
- Create: `reports/V2_result.md`

**Interfaces:**
- Consumes: `load_config(path) -> dict[str, Any]`、`validate_config(config) -> None`、V1 的 `patch/mask/pretrain/adaptation` schema。
- Produces: `validate_v2_config(config: dict[str, Any]) -> None`；`model.name=v2` 配置新增严格 `horizon_attention={mode,rank,use_horizon_embedding,cache_kv}`。

- [x] **Step 1: 写 V2 配置 RED 测试**

在 `tests/test_v2_config.py` 定义 `valid_v2_config()`，从 `valid_v1_config()` 深拷贝后设置 `model.name="v2"` 与：

```python
"horizon_attention": {
    "mode": "low_rank",
    "rank": 32,
    "use_horizon_embedding": True,
    "cache_kv": True,
}
```

测试合法 low-rank/full-rank；参数化拒绝未知/缺失字段、非整数/非正 rank、low-rank `R>=D`、full-rank `R!=D`、非 boolean 开关和未知 mode。`tests/test_project_structure.py` 此任务只新增 V2 spec/config/report 路径断言；model 路径在 Task 3 模型实际存在时再加入。

- [x] **Step 2: 运行配置 RED**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_config.py tests/test_project_structure.py -q -p no:cacheprovider`

Expected: FAIL，因为 `v2` 分派、schema、配置和报告文件尚不存在。

- [x] **Step 3: 实现 schema 与 `configs/v2.yaml`**

在 `utils/config.py` 增加 `_V2_SCHEMA` 和：

```python
def validate_v2_config(config: dict[str, Any]) -> None
```

共享 `_validate_shared_config` 和 V1 patch/mask/pretrain/adaptation 校验，避免复制逻辑：将 V1 专属通用校验抽为 `_validate_patch_pretrain_adaptation(config)`，由 V1/V2 调用。创建 `configs/v2.yaml`，以当前正式 M、`L=P=96`、V1 Stage I checkpoint、`adaptation.mode=qkv` 为基线，新增 spec 固定的 low-rank `R=32` 设置。创建只有标题、实验矩阵和“结果待正式运行填充”说明的 `reports/V2_result.md`，不得填造指标。

- [x] **Step 4: 修正 V1 测试与正式配置的耦合**

`tests/test_v1_config.py::test_v1_yaml_loads_with_strict_sections` 只断言 schema/类型和当前已提交正式值，不再假设模板 `checkpoint:null/mode:qk`。`tests/test_cli.py` 用 `valid_v1_config()` 构造 null-checkpoint case，继续验证 builder 必须显式 checkpoint；不得修改 V1 生产代码。

- [x] **Step 5: 运行配置 GREEN 与 V0/V1 回归**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_config.py tests/test_v1_config.py tests/test_v2_config.py tests/test_cli.py tests/test_project_structure.py -q -p no:cacheprovider`

Expected: PASS；V0/V1/V2 配置均严格验证。

- [x] **Step 6: 提交**

```bash
git add configs/v2.yaml reports/V2_result.md utils/config.py tests/test_v1_config.py tests/test_v2_config.py tests/test_cli.py tests/test_project_structure.py
git commit -m "feat: add strict V2 configuration"
```

---

### Task 2: Vectorized Horizon Temporal Attention 核心

**Files:**
- Create: `models/v2_horizon_attention.py`
- Create: `tests/test_v2_horizon_attention.py`

**Interfaces:**
- Consumes: `history: Tensor[B,N,D]`、可选 `horizon_indices: LongTensor[P_selected]`。
- Produces:

```python
class HorizonTemporalAttention(nn.Module):
    def __init__(self, *, d_model: int, pred_len: int, relation_rank: int,
                 use_horizon_embedding: bool) -> None
    def pool_history(self, history: Tensor) -> Tensor
    def project_history(self, history: Tensor) -> tuple[Tensor, Tensor]
    def build_queries(self, global_context: Tensor,
                      horizon_indices: Tensor | None = None) -> Tensor
    def attend(self, queries: Tensor, keys: Tensor,
               values: Tensor) -> tuple[Tensor, Tensor]
    def forward(self, history: Tensor,
                horizon_indices: Tensor | None = None) -> tuple[Tensor, Tensor]
```

`forward` 返回 `(context[B,P_selected,D], attention[B,P_selected,N])`。

- [x] **Step 1: 写 shape、参数与 pooling RED 测试**

参数化 `(B,N,D,R,P)=(2,5,8,3,4),(1,1,4,2,1)`，断言 K/V/Q/attention/context shape；Q/K/V projection `bias is None`；mean pooling 与 `history.mean(dim=1)` 相等；`horizon_embedding` shape/初始化可复现。禁用 embedding 时断言 `horizon_embedding is None` 且 state dict 无该参数。

- [x] **Step 2: 写数学与 horizon identity RED 测试**

用固定小张量和手工设置 projection weights，断言 logits 除以 `sqrt(R)`、softmax 沿 `N`、attention 非负且和为 1、context 与手算一致。修改一个 `E_H[p]` 只改变该 query；禁用 embedding 时所有 horizon 的 query/attention/context 相同。

- [x] **Step 3: 写错误输入与向量化 RED 测试**

拒绝错误 rank/shape、空 batch/token、NaN/Inf、horizon indices 的空值/重复/越界/非整数/非一维。通过 monkeypatch `attend` 调用次数或源码检查证明标准 forward 只执行一次 batched attention，不按 horizon 循环。

- [x] **Step 4: 运行核心 RED**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_horizon_attention.py -q -p no:cacheprovider`

Expected: FAIL，因为 V2 module/API 尚不存在。

- [x] **Step 5: 实现最小 HorizonTemporalAttention**

使用 `nn.Linear(D,R,bias=False)` 构建 Q/K、`nn.Linear(D,D,bias=False)` 构建 V，`trunc_normal_(std=0.02)` 初始化可选 `E_H`。集中实现 `_validate_history`、`_normalize_horizon_indices` 和 finite checks；索引输出保持请求顺序。`attend` 只使用 batched matmul 与 `torch.softmax(..., dim=-1)`。

- [x] **Step 6: 运行 GREEN**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_horizon_attention.py -q -p no:cacheprovider`

Expected: 全部 Horizon Attention unit tests PASS。

- [x] **Step 7: 提交**

```bash
git add models/v2_horizon_attention.py tests/test_v2_horizon_attention.py
git commit -m "feat: add horizon temporal attention"
```

---

### Task 3: V2 Forecaster、共享非交叉 Head 与梯度边界

**Files:**
- Modify: `models/v2_horizon_attention.py`
- Create: `tests/test_v2_model.py`
- Modify: `tests/test_v1_adapter.py`
- Modify: `tests/test_project_structure.py:25-33`

**Interfaces:**
- Consumes: Task 2 `HorizonTemporalAttention`；V1 `V1PretrainAdapter` 的 backbone、`configure_adaptation`、`load_pretrained_backbone` 和 `parameter_summary`。
- Produces:

```python
class V2HorizonAttentionForecaster(V1PretrainAdapter):
    def __init__(..., relation_rank: int,
                 use_horizon_embedding: bool, cache_kv: bool) -> None
    def encode_history(self, x: Tensor) -> Tensor
    def decode_context(self, context: Tensor) -> Tensor
    def forward_with_attention(self, x: Tensor,
                               horizon_indices: Tensor | None = None
                               ) -> tuple[Tensor, Tensor]
    def forward(self, x: Tensor) -> Tensor
```

`decode_context` 返回 `[B,P_selected,C_out,3]`；`forward` 始终返回全部 `P`。

- [x] **Step 1: 写输出 shape 与 head 结构 RED 测试**

为 M/MS、`B=1`、`P=1/4` 构建 tiny model，断言输出 shape、有限值与逐元素 `Lower<Point<Upper`。断言最终 `output_projection` 为 `Linear(D,C_out*3)` 而不是 V1 的 `Linear(D,P*C_out*3)`，并且同一个 head 逐 horizon 共享。

同时在 `tests/test_project_structure.py` 加入 `models/v2_horizon_attention.py`，此时文件已由 Task 2 创建，结构测试应保持 GREEN。

- [x] **Step 2: 写 horizon embedding 消融 RED 测试**

在 dropout=0 的确定性模型中，`use_horizon_embedding=False` 时全部 horizon 的 context 和 prediction 完全相同；启用后手工设置不同 embedding，至少两个 horizon 的 attention/context/prediction 不同。源码/模块枚举确认不存在 horizon-specific head 或 Operator。

- [x] **Step 3: 写 adaptation 梯度边界 RED 测试**

参数化 `full/qk/qkv`：Horizon Attention 与 shared head 总是 `requires_grad=True` 且 backward 梯度有限；frozen backbone 参数 `grad is None`；对应 encoder adapter 参数存在有限梯度；`full` 无 adapter 且 backbone 可训练。保留并运行 V1 原有 adapter 行为测试。

- [x] **Step 4: 运行模型 RED**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_model.py tests/test_v1_adapter.py -q -p no:cacheprovider`

Expected: FAIL，因为 V2 forecaster 尚不存在。

- [x] **Step 5: 实现 V2 forecaster**

调用 `super().__init__` 复用 V1 backbone/adaptation，然后立即将 `output_projection` 替换为 shared `nn.Linear(d_model, output_dim*3)` 并创建 `HorizonTemporalAttention`。`encode_history` 复用 `patchify + self.backbone`；`decode_context` 使用 V1 相同 Softplus/epsilon 公式；`forward` 仅调用 `forward_with_attention(x)[0]`。不得修改 V1 forward。

- [x] **Step 6: 运行 GREEN 与 V1 回归**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_horizon_attention.py tests/test_v2_model.py tests/test_v1_adapter.py tests/test_model.py -q -p no:cacheprovider`

Expected: PASS；V1 输出与 adapter 测试继续通过。

- [x] **Step 7: 提交**

```bash
git add models/v2_horizon_attention.py tests/test_v2_model.py tests/test_v1_adapter.py tests/test_project_structure.py
git commit -m "feat: compose V2 horizon forecaster"
```

---

### Task 4: Registry、V1 Pretrain 初始化与 V2 Supervised Checkpoint 契约

**Files:**
- Modify: `models/__init__.py:8-74`
- Modify: `utils/checkpoint.py:17-55`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_checkpoint.py`
- Create: `tests/test_v2_integration.py`

**Interfaces:**
- Consumes: `load_pretrain_checkpoint(path, expected_config=...)`、Task 3 `V2HorizonAttentionForecaster`。
- Produces: `_build_v2(config,input_dim,output_dim) -> BaseForecastModel`；`MODEL_REGISTRY["v2"]`；supervised checkpoint contract 包含完整 `horizon_attention`。

- [x] **Step 1: 写 registry/pretrain RED 测试**

在 `tests/test_v2_integration.py` 使用 tiny V1 Stage I checkpoint fixture，调用 `build_model("v2", config, ...)`，断言 backbone 权重逐项等于 checkpoint、`model_name` 为 V2、pretrain scaler/provenance 保留，且 provenance 新增 `horizon_attention_mode/rank/use_horizon_embedding/cache_kv`。null/不存在/incompatible pretrain checkpoint 明确失败。

- [x] **Step 2: 写 checkpoint contract RED 测试**

在 `tests/test_checkpoint.py` 以同 shape 配置逐项改变 `data.pred_len`、`model.d_model`、`horizon_attention.mode/rank/use_horizon_embedding`、patch、adaptation、pretrain checkpoint、M/MS output contract，断言 `validate_checkpoint_contract` 或 `load_checkpoint` 指出具体不兼容路径。另测 V2 config 能加载 `stage=pretrain, model_name=v1` 的 Stage I checkpoint，因为 pretrain contract 不比较 `model.name/horizon_attention`。

- [x] **Step 3: 运行 registry/checkpoint RED**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_integration.py tests/test_checkpoint.py tests/test_cli.py -q -p no:cacheprovider`

Expected: FAIL，因为 registry 和 supervised `horizon_attention` contract 尚未实现。

- [x] **Step 4: 实现 `_build_v2` 与 provenance**

在 `models/__init__.py` 用与 `_build_v1` 相同顺序加载 Stage I checkpoint，构建 `V2HorizonAttentionForecaster`，严格载入 `backbone_state_dict`，附加 scaler/provenance，并导出 V2 class。不得把 `model.name=v2` 加入 `PRETRAIN_CONTRACT_PATHS`。

- [x] **Step 5: 扩展 supervised checkpoint contract**

在 `utils/checkpoint.py::_CONTRACT_PATHS` 增加 `horizon_attention`；保留现有 `model`、`patch`、`adaptation`、`pretrain.checkpoint` 检查。依赖 strict `model.load_state_dict` 拒绝 output_dim 不同的权重，不做部分加载。

- [x] **Step 6: 运行 GREEN 与 V1 checkpoint 回归**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_integration.py tests/test_checkpoint.py tests/test_cli.py tests/test_v1_integration.py tests/test_v1_pretrain_engine.py -q -p no:cacheprovider`

Expected: PASS；V1 Stage I/II checkpoint round-trip 不受影响。

- [x] **Step 7: 提交**

```bash
git add models/__init__.py utils/checkpoint.py tests/test_cli.py tests/test_checkpoint.py tests/test_v2_integration.py
git commit -m "feat: register V2 checkpoint contracts"
```

---

### Task 5: Inference-only KV Cache 与分块 Horizon 预测

**Files:**
- Modify: `models/v2_horizon_attention.py`
- Modify: `tests/test_v2_horizon_attention.py`
- Modify: `tests/test_v2_model.py`

**Interfaces:**
- Consumes: Task 2 `pool_history/project_history/build_queries/attend`；Task 3 `encode_history/decode_context`。
- Produces:

```python
@dataclass(frozen=True)
class HorizonKVCache:
    global_context: Tensor
    keys: Tensor
    values: Tensor
    producer_versions: tuple[int, ...]

class HorizonTemporalAttention(nn.Module):
    def build_cache(self, history: Tensor) -> HorizonKVCache
    def forward_from_cache(self, cache: HorizonKVCache,
                           horizon_indices: Tensor | None = None
                           ) -> tuple[Tensor, Tensor]

class V2HorizonAttentionForecaster(...):
    def build_inference_cache(self, x: Tensor) -> HorizonKVCache
    def predict_from_cache(self, cache: HorizonKVCache,
                           horizon_indices: Tensor | None = None
                           ) -> tuple[Tensor, Tensor]
```

- [x] **Step 1: 写 cache 等价与 projection 次数 RED 测试**

在 `model.eval()` + `torch.inference_mode()` 下，断言非缓存全 horizon、一次缓存全 horizon、按 `[0,1]`/`[2,3]` 分块后按请求顺序拼接的 prediction/attention 数值一致。通过 mock `key_projection.forward/value_projection.forward` 验证 cache 只构建一次，后续多个 horizon 请求不重复 K/V projection。

- [x] **Step 2: 写 cache 生命周期 RED 测试**

拒绝 training mode、grad-enabled context、batch/N/R/D/device/dtype 不匹配、错误 horizon index。cache 不注册为 parameter/buffer，不出现在 `state_dict`。构建 cache 后对 backbone、Q/K/V、horizon embedding 或 forecast head 任一参数执行 in-place optimizer update 或 `load_state_dict`，任一 `parameter._version` 变化时必须以 stale-cache 错误拒绝。

- [x] **Step 3: 运行 cache RED**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_horizon_attention.py tests/test_v2_model.py -q -p no:cacheprovider`

Expected: FAIL，因为 cache dataclass/API 尚不存在。

- [x] **Step 4: 实现 cache**

`build_cache` 只保存 mean-pooled global context、K/V tensor 和 producer parameter `_version` stamps；不持有 history 或 model 引用。Horizon module 直接使用时 stamp 其全部参数；forecaster 对外 cache 再覆盖为 backbone + Horizon Attention + forecast head 的全部参数版本。`forward_from_cache` 先验证 eval/no-grad、shape/device/dtype 和当前 versions，再构建指定 horizon queries 并调用 `attend`。普通训练 `forward` 永不接受 cache。

- [x] **Step 5: 运行 GREEN 与 checkpoint 排除检查**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_horizon_attention.py tests/test_v2_model.py tests/test_checkpoint.py -q -p no:cacheprovider`

Expected: PASS；cache 和非 cache 数值一致，state dict 无 cache。

- [x] **Step 6: 提交**

```bash
git add models/v2_horizon_attention.py tests/test_v2_horizon_attention.py tests/test_v2_model.py
git commit -m "feat: add V2 inference KV cache"
```

---

### Task 6: 统一实验接入、Attention Diagnostics 与 V2 报告字段

**Files:**
- Modify: `epx_main.py:79-152,300-450`
- Modify: `run.py:34-102`
- Modify: `models/v2_horizon_attention.py`
- Modify: `tests/test_experiment.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_v2_model.py`
- Modify: `tests/test_smoke_experiment.py`

**Interfaces:**
- Consumes: model 可选 `forward_with_attention(x) -> (prediction,attention)` 与 cache APIs；现有 `_evaluate/run_experiment/render_report`。
- Produces: `_evaluate(..., collect_attention: bool = False)`；`execute(..., render_single_report: bool = True)`；`V2HorizonAttentionForecaster.benchmark_kv_cache(x, horizon_chunks, repeats) -> dict[str, float | int]`；V2 result/provenance、`results/<id>/attention_map.npy`、cache benchmark metadata 和 V2 单次报告。

- [x] **Step 1: 写可选 attention 收集 RED 测试**

在 `tests/test_experiment.py` 用 tiny dummy V2-style model，断言 `_evaluate(..., collect_attention=True)` 对支持 `forward_with_attention` 的模型按 batch/sample 加权汇总 `[P,N]` attention mean；默认 `False` 和普通 V0/V1 model 保持原调用路径。attention shape/finite/归一化错误明确失败。`validate` 不收集，`test` 收集。

- [x] **Step 2: 写产物与 report RED 测试**

`run_experiment` 在 attention 可用时保存非空 `attention_map.npy` 并在 `paths` 返回；V0/V1 不创建该文件。`render_report` 对 `model_name=V2` 输出 pretrain provenance、adaptation、horizon attention mode/rank、embedding/cache 开关、standard inference latency 和 cache benchmark 字段。`execute(..., render_single_report=False)` 必须跳过单次报告写入，供 Task 7 消融使用；默认值保持现有 CLI 行为。

- [x] **Step 3: 写 cache benchmark 契约 RED 测试**

为 V2 model 增加测试专用 tiny batch，基准结果固定包含 `cache_build_seconds`、`uncached_chunk_seconds`、`cached_chunk_seconds`、`chunks`，所有值非负；只验证计时字段和 cached/uncached 数值一致，不写“cached 必须更快”的脆弱断言。GPU 路径在计时前后 synchronize，CPU 直接 perf counter。`tests/test_cli.py` 的 report 测试和所有 smoke tests 改用 `tmp_path`，不再向仓库 `results/test_report.md` 或 `results/test-runs` 写测试产物。

- [x] **Step 4: 运行实验 RED**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_experiment.py tests/test_cli.py tests/test_smoke_experiment.py -m "not slow" -q -p no:cacheprovider`

Expected: FAIL，因为 attention artifact 和 V2 report 字段尚不存在。

- [x] **Step 5: 实现通用可选 hook**

在 `_evaluate` 中只在 `collect_attention=True` 时通过 `getattr(model,"forward_with_attention",None)` 探测能力；V2 聚合 attention，V0/V1 行为不变。在 `run_experiment` 保存 diagnostics，检测 `benchmark_kv_cache` 时仅对第一个 test batch 做固定 chunk/repeat 的推理 benchmark，并把结果写入 metrics JSON。所有 numpy/tensor 转换发生在 inference/no-grad 后。在 V2 model 中实现 benchmark method，cached/uncached 使用同一 hidden history 和 horizon chunks，避免把 backbone 时间误算成 KV projection 收益。

- [x] **Step 6: 扩展 V2 单次报告**

`run.render_report` 将 V1/V2 共同 pretrain/adaptation 字段复用，V2 追加 horizon/cache 字段；保持 V0 输出不变。不要在这里实现五行消融汇总。

- [x] **Step 7: 运行 GREEN 与 slow V2 smoke**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_experiment.py tests/test_cli.py -q -p no:cacheprovider`

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_smoke_experiment.py::test_real_etth1_v2_horizon_attention_smoke -v -p no:cacheprovider`

Expected: PASS；smoke 生成 V2 checkpoint、metrics、predictions、figures、attention map 与 `V2_result.md`，输出严格不交叉。

- [x] **Step 8: 提交**

```bash
git add epx_main.py run.py models/v2_horizon_attention.py tests/test_experiment.py tests/test_cli.py tests/test_v2_model.py tests/test_smoke_experiment.py
git commit -m "feat: integrate V2 experiment artifacts"
```

---

### Task 7: 受控消融编排与聚合 `reports/V2_result.md`

**Files:**
- Create: `v2_ablation.py`
- Create: `tests/test_v2_ablation.py`
- Modify: `reports/V2_result.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `run.execute(model_name, config, render_single_report=False) -> result`、`configs/v1.yaml`、`configs/v2.yaml`。
- Produces:

```python
def build_ablation_configs(v1_config: dict[str, Any],
                           v2_config: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]
def run_v2_ablation(v1_config: dict[str, Any],
                    v2_config: dict[str, Any]) -> dict[str, dict[str, Any]]
def render_v2_ablation_report(results: dict[str, dict[str, Any]],
                              path: str | Path) -> None
```

- [x] **Step 1: 写配置隔离 RED 测试**

`build_ablation_configs` 产生四次独立运行：V1、标准 V2 low-rank、V2 w/o horizon embedding、V2 full-rank；断言除 `model.name`/readout 以及指定消融字段外，Stage I checkpoint、data、seed、adaptation、optimizer、training、loss、metrics 完全相同。标准 V2 与第五行 “V2 low-rank QK” 不重复运行。

- [x] **Step 2: 写 mock 编排与报告 RED 测试**

mock `execute`，断言只调用四次且均传入 `render_single_report=False`、每次 test result 有唯一 experiment ID；报告包含五行，其中标准 V2 和 low-rank alias 共享同一 experiment ID，并包含参数量、训练/推理效率、cache metadata、overall 与 by-horizon 指标、checkpoint/config provenance。缺失任何 run/metric 或 alias ID 不一致时报告生成失败，不静默填 N/A。

- [x] **Step 3: 运行消融 RED**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_ablation.py -q -p no:cacheprovider`

Expected: FAIL，因为消融编排器不存在。

- [x] **Step 4: 实现 CLI 与聚合报告**

CLI 固定支持：

```text
python v2_ablation.py --v1-config configs/v1.yaml --v2-config configs/v2.yaml
```

在调用任何训练前验证两个配置共享受控字段；依次以 `render_single_report=False` 运行四个独立实验，避免覆盖现有 `reports/V1_result.md` 或中途覆盖 V2 汇总；最后一次性原子写入 `reports/V2_result.md`。执行失败时保留已有实验目录并终止，不用部分结果覆盖正式报告。标准 V2/low-rank alias 在报告中明确标记同一次 run。

- [x] **Step 5: 更新 README 与占位报告**

README 说明 V2 单次运行、KV cache 定位、消融命令和产物位置。正式实验未执行前，`reports/V2_result.md` 只保留表头、配置和待运行状态；测试不得硬编码虚构指标。

- [x] **Step 6: 运行 GREEN**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/test_v2_ablation.py tests/test_cli.py tests/test_v2_config.py -q -p no:cacheprovider`

Expected: PASS；mock 证明四次运行、五行报告和 alias provenance 正确。

- [x] **Step 7: 提交**

```bash
git add v2_ablation.py tests/test_v2_ablation.py reports/V2_result.md README.md
git commit -m "feat: add controlled V2 ablations"
```

---

### Task 8: 全量验证、正式实验与最终报告

**Files:**
- Modify: `reports/V2_result.md`
- Modify: `configs/v2.yaml` only if validation-selected formal settings differ; never use test metrics to choose them

**Interfaces:**
- Consumes: Tasks 1-7 全部代码与测试；正式 ETTh1 数据和已提交 V1 Stage I checkpoint。
- Produces: 通过完整回归的 V2 分支与可追溯正式消融报告。

- [ ] **Step 1: 运行静态差异检查**

Run: `git diff --check`

Expected: 无 whitespace errors。确认 `rg -n "sort\(|argsort|Operator" models/v2_horizon_attention.py` 不包含预测排序或 Operator 实现；若字符串只出现在错误消息/注释，人工核对。

- [ ] **Step 2: 运行非 slow 完整测试**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests -m "not slow" -q -p no:cacheprovider`

Expected: PASS，0 failures/errors。

- [ ] **Step 3: 运行 slow/smoke 完整测试**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests -m slow -v -p no:cacheprovider`

Expected: PASS，V0/V1/V2 smoke 均成功且测试产物仅位于 pytest tmp directory。

- [ ] **Step 4: 运行完整套件**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests -v -p no:cacheprovider`

Expected: PASS，0 failures/errors；记录通过数量和耗时。

- [ ] **Step 5: 冻结正式配置并运行消融**

确认 rank 仅依据 validation 或预先固定值选择，随后运行：

```text
D:\Application\ApplicationData\anaconda3\envs\py312\python.exe v2_ablation.py --v1-config configs/v1.yaml --v2-config configs/v2.yaml
```

Expected: 四个独立 experiment ID 完成；每个只在 best validation checkpoint 恢复后评估 test 一次；最终报告有五行且 low-rank alias 指向标准 V2 run。

- [ ] **Step 6: 审计正式产物**

逐行核对 `reports/V2_result.md` 与各 `metrics.json/config.resolved.yaml/best.pt`：experiment ID、Stage I checkpoint、adaptation、relation rank、embedding/cache 开关、参数量、时间、MAE/RMSE、PICP/MPIW/PINAW/Coverage Error/Interval Score 和 by-horizon 完整一致。确认 attention map `[P,N]` 有限且每行归一化。

- [ ] **Step 7: 提交正式结果**

只提交源代码、配置、规范、测试、README 和汇总报告；遵守现有 `.gitignore`，不提交大体积 checkpoints/results/logs：

```bash
git add configs/v2.yaml reports/V2_result.md README.md
git commit -m "exp: record formal V2 horizon ablations"
```

- [ ] **Step 8: 最终分支验证**

Run: `git status --short --branch`

Run: `git log --oneline --decorate -10`

Expected: `feature/v2` 工作区干净，提交历史按 Task 1-8 可审计；此时再进入 code review/推送流程。
