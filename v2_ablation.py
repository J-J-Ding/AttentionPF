"""Run four controlled experiments and report five rows (one low-rank alias)."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from run import execute, render_report, format_report
from utils.config import load_config, validate_config


RUN_LABELS = ('V1', 'V2', 'V2 without horizon embedding', 'V2 full-rank QK')
LOW_RANK_ALIAS = 'V2 low-rank QK'
METRICS = ('mae', 'rmse', 'picp', 'mpiw', 'pinaw', 'coverage_error', 'interval_score')


def build_ablation_configs(v1_config: dict[str, Any], v2_config: dict[str, Any]
                           ) -> list[tuple[str, str, dict[str, Any]]]:
    validate_config(v1_config)
    validate_config(v2_config)
    if v1_config['model']['name'] != 'v1' or v2_config['model']['name'] != 'v2':
        raise ValueError('ablation requires v1 and v2 model configurations')
    # Compare every shared section, permitting only model identity and run name.
    for section, left in v1_config.items():
        right = v2_config[section]
        if section == 'model':
            left = {k: v for k, v in left.items() if k != 'name'}
            right = {k: v for k, v in right.items() if k != 'name'}
        elif section == 'experiment':
            left = {k: v for k, v in left.items() if k != 'name'}
            right = {k: v for k, v in right.items() if k != 'name'}
        if left != right:
            raise ValueError(f'controlled ablation configurations differ in {section}')
    if v2_config['training']['resume_from'] is not None:
        raise ValueError('ablation training.resume_from must be null')
    options = v2_config['horizon_attention']
    if options['mode'] != 'low_rank' or not options['use_horizon_embedding']:
        raise ValueError('standard V2 requires low_rank and horizon embedding')
    if not v2_config['pretrain']['checkpoint']:
        raise ValueError('ablation requires pretrain.checkpoint')
    runs = []
    for index, label in enumerate(RUN_LABELS):
        config = deepcopy(v1_config if index == 0 else v2_config)
        config['experiment']['name'] = (
            'v1_horizon_baseline', 'v2_horizon_low_rank', 'v2_horizon_no_embedding',
            'v2_horizon_full_rank')[index]
        if index == 2:
            config['horizon_attention']['use_horizon_embedding'] = False
        if index == 3:
            config['horizon_attention'].update(mode='full_rank', rank=config['model']['d_model'])
        validate_config(config)
        runs.append((label, config['model']['name'], config))
    return runs


def _validate_results(results: dict[str, dict[str, Any]]) -> None:
    if set(results) != set(RUN_LABELS) | {LOW_RANK_ALIAS}:
        raise ValueError('ablation report requires all five result rows')
    main, alias = results['V2'], results[LOW_RANK_ALIAS]
    if (main['experiment_id'] != alias['experiment_id']
            or main['test_metrics'] != alias['test_metrics']):
        raise ValueError('low-rank alias must reference the standard V2 run')
    ids = [results[label]['experiment_id'] for label in RUN_LABELS]
    if len(set(ids)) != 4:
        raise ValueError('independent ablations require unique experiment IDs')
    try:
        for label in RUN_LABELS:
            result = results[label]
            for field in ('total_params', 'trainable_params', 'trainable_ratio', 'train_seconds',
                          'inference_seconds', 'inference_latency_seconds', 'best_val_loss'):
                if not math.isfinite(result[field]):
                    raise ValueError(f'non-finite {field} in {label}')
            horizons = result['test_metrics']['by_horizon']
            if not horizons or [h['horizon'] for h in horizons] != list(range(1, len(horizons)+1)):
                raise ValueError('by-horizon results must be complete and ordered')
            for row in [result['test_metrics']['overall'], *horizons]:
                for key in METRICS:
                    if row[key] is None and key == 'pinaw':
                        continue  # PINAW is undefined for zero target range.
                    if not math.isfinite(row[key]):
                        raise ValueError(f'non-finite {key} in {label}')
            for key in ('best_checkpoint', 'resolved_config', 'metrics_json'):
                if not result['paths'][key]:
                    raise ValueError(f'missing {key}')
    except (KeyError, TypeError) as error:
        raise ValueError(f'incomplete ablation result: {error}') from error


def render_v2_ablation_report(results: dict[str, dict[str, Any]], path: str | Path) -> None:
    _validate_results(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# V2 Horizon-Aware Temporal Attention Results', '',
             'Four independent runs; V2 low-rank QK is an alias of standard V2.', '',
             '## Overall comparison', '',
             '| Model | MAE | RMSE | PICP | MPIW | PINAW | Coverage Error | Interval Score |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in (*RUN_LABELS, LOW_RANK_ALIAS):
        row = results[label]['test_metrics']['overall']
        values = ['N/A' if row[k] is None else f'{row[k]:.6f}' for k in METRICS]
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    lines += ['', '## Parameters and efficiency', '',
              '| Model | Total | Trainable | Ratio | Peak GPU MiB | Train s | Inference s | s/sample | Experiment ID |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for label in (*RUN_LABELS, LOW_RANK_ALIAS):
        r = results[label]
        memory = r['peak_gpu_memory_bytes']
        memory_text = 'N/A (CPU)' if memory is None else f'{memory/1024**2:.2f}'
        lines.append(f"| {label} | {r['total_params']} | {r['trainable_params']} | "
                     f"{r['trainable_ratio']:.6f} | {memory_text} | {r['train_seconds']:.3f} | "
                     f"{r['inference_seconds']:.3f} | {r['inference_latency_seconds']:.6f} | `{r['experiment_id']}` |")
    lines += ['', '## Interpretation and scope', '',
              'Configuration is fixed before test evaluation. One seed is reported; results do not establish multi-seed significance.',
              'V1 to V2 changes both temporal readout and forecast head parameterization. V2 internal ablations isolate embedding and Q/K dimension.',
              'Without horizon embedding all horizons share the same query, context and forecast. This is the expected ablation behavior.',
              'KV cache is an inference optimization. Timings below exclude backbone encoding and compare repeated horizon chunks over one history.',
              'Inference time in the main table includes the shared evaluation pipeline and is not a model-only kernel benchmark.', '']
    # Reuse the existing version-aware renderer for complete per-run provenance and horizons.
    for label in RUN_LABELS:
        content = format_report(results[label]).splitlines()[1:]
        lines.extend([f'## {label}: run details', '', *content, ''])
        attention = results[label]['paths'].get('attention_map')
        if attention:
            lines.extend([f'- Attention map `[P,N]`: `{attention}`', ''])
        lines.extend([f"- Metrics JSON: `{results[label]['paths']['metrics_json']}`", ''])
    # A normally created sibling inherits destination ACLs on Windows. Secure
    # tempfile directories would otherwise carry restrictive ACLs through replace.
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write('\n'.join(lines).rstrip()+'\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_v2_ablation(v1_config: dict[str, Any], v2_config: dict[str, Any]
                    ) -> dict[str, dict[str, Any]]:
    configs = build_ablation_configs(v1_config, v2_config)
    results = {}
    for label, model_name, config in configs:
        results[label] = execute(model_name, config, render_single_report=False)
        # Keep a single-run summary with immutable experiment artifacts.
        render_report(results[label], Path(results[label]['paths']['metrics_json']).parent / 'report.md')
    results[LOW_RANK_ALIAS] = results['V2']
    report = Path(v2_config['experiment']['output_root']) / 'reports' / 'V2_result.md'
    render_v2_ablation_report(results, report)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-config', default='configs/v1.yaml')
    parser.add_argument('--v2-config', default='configs/v2.yaml')
    args = parser.parse_args(argv)
    run_v2_ablation(load_config(args.v1_config), load_config(args.v2_config))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
