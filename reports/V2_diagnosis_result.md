# V2 structure and attention diagnosis

## Actual parameter inventories

```text
V1: parameter inventory
Category / module                                                       Total    Trainable       Frozen
Patch Embedding                                                        14,464            0       14,464
Transformer Q                                                          49,536            0       49,536
Transformer K                                                          49,536            0       49,536
Transformer V                                                          49,536            0       49,536
Transformer O                                                          49,536            0       49,536
FFN                                                                   197,760            0      197,760
Adapter                                                                18,432       18,432            0
Horizon Attention                                                           0            0            0
Forecast Head                                                         260,064      260,064            0
Position Embedding                                                      1,408            0        1,408
LayerNorm                                                               1,536            0        1,536
Mask Token                                                                128            0          128
TOTAL                                                                 691,936      278,496      413,440
backbone.layers.0.linear1                                              33,024            0       33,024
backbone.layers.0.linear2                                              32,896            0       32,896
backbone.layers.0.norm1                                                   256            0          256
backbone.layers.0.norm2                                                   256            0          256
backbone.layers.0.self_attn.adapters.k                                  2,048        2,048            0
backbone.layers.0.self_attn.adapters.q                                  2,048        2,048            0
backbone.layers.0.self_attn.adapters.v                                  2,048        2,048            0
backbone.layers.0.self_attn.k_proj                                     16,512            0       16,512
backbone.layers.0.self_attn.out_proj                                   16,512            0       16,512
backbone.layers.0.self_attn.q_proj                                     16,512            0       16,512
backbone.layers.0.self_attn.v_proj                                     16,512            0       16,512
backbone.layers.1.linear1                                              33,024            0       33,024
backbone.layers.1.linear2                                              32,896            0       32,896
backbone.layers.1.norm1                                                   256            0          256
backbone.layers.1.norm2                                                   256            0          256
backbone.layers.1.self_attn.adapters.k                                  2,048        2,048            0
backbone.layers.1.self_attn.adapters.q                                  2,048        2,048            0
backbone.layers.1.self_attn.adapters.v                                  2,048        2,048            0
backbone.layers.1.self_attn.k_proj                                     16,512            0       16,512
backbone.layers.1.self_attn.out_proj                                   16,512            0       16,512
backbone.layers.1.self_attn.q_proj                                     16,512            0       16,512
backbone.layers.1.self_attn.v_proj                                     16,512            0       16,512
backbone.layers.2.linear1                                              33,024            0       33,024
backbone.layers.2.linear2                                              32,896            0       32,896
backbone.layers.2.norm1                                                   256            0          256
backbone.layers.2.norm2                                                   256            0          256
backbone.layers.2.self_attn.adapters.k                                  2,048        2,048            0
backbone.layers.2.self_attn.adapters.q                                  2,048        2,048            0
backbone.layers.2.self_attn.adapters.v                                  2,048        2,048            0
backbone.layers.2.self_attn.k_proj                                     16,512            0       16,512
backbone.layers.2.self_attn.out_proj                                   16,512            0       16,512
backbone.layers.2.self_attn.q_proj                                     16,512            0       16,512
backbone.layers.2.self_attn.v_proj                                     16,512            0       16,512
backbone.mask_token                                                       128            0          128
backbone.patch_embedding                                               14,464            0       14,464
backbone.position_embedding                                             1,408            0        1,408
output_projection                                                     260,064      260,064            0

V2: parameter inventory
Category / module                                                       Total    Trainable       Frozen
Patch Embedding                                                        14,464            0       14,464
Transformer Q                                                          49,536            0       49,536
Transformer K                                                          49,536            0       49,536
Transformer V                                                          49,536            0       49,536
Transformer O                                                          49,536            0       49,536
FFN                                                                   197,760            0      197,760
Adapter                                                                18,432       18,432            0
Horizon Attention                                                      36,864       36,864            0
Forecast Head                                                           2,709        2,709            0
Position Embedding                                                      1,408            0        1,408
LayerNorm                                                               1,536            0        1,536
Mask Token                                                                128            0          128
TOTAL                                                                 471,445       58,005      413,440
backbone.layers.0.linear1                                              33,024            0       33,024
backbone.layers.0.linear2                                              32,896            0       32,896
backbone.layers.0.norm1                                                   256            0          256
backbone.layers.0.norm2                                                   256            0          256
backbone.layers.0.self_attn.adapters.k                                  2,048        2,048            0
backbone.layers.0.self_attn.adapters.q                                  2,048        2,048            0
backbone.layers.0.self_attn.adapters.v                                  2,048        2,048            0
backbone.layers.0.self_attn.k_proj                                     16,512            0       16,512
backbone.layers.0.self_attn.out_proj                                   16,512            0       16,512
backbone.layers.0.self_attn.q_proj                                     16,512            0       16,512
backbone.layers.0.self_attn.v_proj                                     16,512            0       16,512
backbone.layers.1.linear1                                              33,024            0       33,024
backbone.layers.1.linear2                                              32,896            0       32,896
backbone.layers.1.norm1                                                   256            0          256
backbone.layers.1.norm2                                                   256            0          256
backbone.layers.1.self_attn.adapters.k                                  2,048        2,048            0
backbone.layers.1.self_attn.adapters.q                                  2,048        2,048            0
backbone.layers.1.self_attn.adapters.v                                  2,048        2,048            0
backbone.layers.1.self_attn.k_proj                                     16,512            0       16,512
backbone.layers.1.self_attn.out_proj                                   16,512            0       16,512
backbone.layers.1.self_attn.q_proj                                     16,512            0       16,512
backbone.layers.1.self_attn.v_proj                                     16,512            0       16,512
backbone.layers.2.linear1                                              33,024            0       33,024
backbone.layers.2.linear2                                              32,896            0       32,896
backbone.layers.2.norm1                                                   256            0          256
backbone.layers.2.norm2                                                   256            0          256
backbone.layers.2.self_attn.adapters.k                                  2,048        2,048            0
backbone.layers.2.self_attn.adapters.q                                  2,048        2,048            0
backbone.layers.2.self_attn.adapters.v                                  2,048        2,048            0
backbone.layers.2.self_attn.k_proj                                     16,512            0       16,512
backbone.layers.2.self_attn.out_proj                                   16,512            0       16,512
backbone.layers.2.self_attn.q_proj                                     16,512            0       16,512
backbone.layers.2.self_attn.v_proj                                     16,512            0       16,512
backbone.mask_token                                                       128            0          128
backbone.patch_embedding                                               14,464            0       14,464
backbone.position_embedding                                             1,408            0        1,408
horizon_attention.horizon_embedding                                    12,288       12,288            0
horizon_attention.key_projection                                        4,096        4,096            0
horizon_attention.query_projection                                      4,096        4,096            0
horizon_attention.value_projection                                     16,384       16,384            0
output_projection                                                       2,709        2,709            0
```

## Structural finding

V1 trainable 278,496 = Adapter 18,432 + Head 260,064.
V2 trainable 58,005 = Adapter 18,432 + Head 2,709 + HA 36,864.
Head delta -257,355; HA delta +36,864; net delta -220,491.
No separate multilayer forecast decoder exists. Backbone/Adapter names, shapes and trainability match; all frozen tensors match.

V1 to current V2 is not an HA-only ablation: the forecast head also shrinks. Adapter capacity is unchanged.

## Validation attention diagnostics

Raw tensor: `(1647, 96, 11)`; computed per sample before aggregation.
- Mean off-diagonal cosine: 0.638012
- Mean pairwise TV: 0.427840
- Normalized entropy: {'mean': 0.8255614037270255, 'p10': 0.6935917287559016, 'p50': 0.8447219658724477, 'p90': 0.9255071873320618, 'min': 0.48218976577457684, 'max': 0.9948214830580456}
- Effective token count: {'mean': 7.400334006312861, 'p10': 5.27595828168744, 'p50': 7.580315403844579, 'p90': 9.200605613536828, 'min': 3.177963427240104, 'max': 10.864251620615365}
- Top1 mass: {'mean': 0.2587272834328324, 'p10': 0.18064436912536622, 'p50': 0.24992141127586365, 'p90': 0.3497013211250305, 'min': 0.10864009708166122, 'max': 0.6100167632102966}
- Candidate rates: {'collapse_sample_fraction': 0.0, 'uniform_fraction': 0.038706739526411654, 'sharp_fraction': 0.0}

High cosine alone does not prove collapse. Joint cosine/TV flags and entropy/top1 are heuristic diagnostics, not proof of the cause of forecast error.
The validation mean heatmap must be read with the per-sample examples and quantiles. No test data was used to choose ranks or thresholds.

## Short / middle / long horizons

| Group | Horizons | Mean normalized entropy | Mean top1 |
|---|---|---:|---:|
| short | 1-32 | 0.820802 | 0.263087 |
| middle | 33-64 | 0.824530 | 0.258645 |
| long | 65-96 | 0.831352 | 0.254450 |

## Sources and figures

- Immutable diagnosis directory: `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\results\v2_diagnosis_20261016-191632_cfc026`
- V1 checkpoint: `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\checkpoints\v1_horizon_baseline_seed2026_20261016-144820-008596_e622d6\best.pt`
- V2 checkpoint: `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\checkpoints\v2_horizon_low_rank_seed2026_20261016-144949-667162_06ae41\best.pt`
- Raw per-sample tensor: `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\results\v2_diagnosis_20261016-191632_cfc026\attention_raw.npz`
- Full statistics / pair matrices / group comparisons: `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\results\v2_diagnosis_20261016-191632_cfc026\attention_statistics.json`
- Parameter names, shapes, flags: `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\results\v2_diagnosis_20261016-191632_cfc026\v1_parameters.json`, `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\results\v2_diagnosis_20261016-191632_cfc026\v2_parameters.json`
- Provenance: `D:\DingJunjie\CodeFiles\PythonCode\Project2\AttentionPF\.worktrees\feature-v1\results\v2_diagnosis_20261016-191632_cfc026\provenance.json`

### attention_full

![attention_full](D:/DingJunjie/CodeFiles/PythonCode/Project2/AttentionPF/.worktrees/feature-v1/results/v2_diagnosis_20261016-191632_cfc026/figures/attention_full.png)

### attention_groups_sample_0

![attention_groups_sample_0](D:/DingJunjie/CodeFiles/PythonCode/Project2/AttentionPF/.worktrees/feature-v1/results/v2_diagnosis_20261016-191632_cfc026/figures/attention_groups_sample_0.png)

### attention_groups_sample_1646

![attention_groups_sample_1646](D:/DingJunjie/CodeFiles/PythonCode/Project2/AttentionPF/.worktrees/feature-v1/results/v2_diagnosis_20261016-191632_cfc026/figures/attention_groups_sample_1646.png)

### attention_groups_sample_823

![attention_groups_sample_823](D:/DingJunjie/CodeFiles/PythonCode/Project2/AttentionPF/.worktrees/feature-v1/results/v2_diagnosis_20261016-191632_cfc026/figures/attention_groups_sample_823.png)

### attention_groups_validation_mean

![attention_groups_validation_mean](D:/DingJunjie/CodeFiles/PythonCode/Project2/AttentionPF/.worktrees/feature-v1/results/v2_diagnosis_20261016-191632_cfc026/figures/attention_groups_validation_mean.png)

### entropy_by_horizon

![entropy_by_horizon](D:/DingJunjie/CodeFiles/PythonCode/Project2/AttentionPF/.worktrees/feature-v1/results/v2_diagnosis_20261016-191632_cfc026/figures/entropy_by_horizon.png)

### horizon_cosine

![horizon_cosine](D:/DingJunjie/CodeFiles/PythonCode/Project2/AttentionPF/.worktrees/feature-v1/results/v2_diagnosis_20261016-191632_cfc026/figures/horizon_cosine.png)
