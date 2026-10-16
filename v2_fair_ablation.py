"""Four HA-only experiments with identical V1 non-attention capacity and initialization."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import torch

from dataload import create_dataloaders
from diagnose_v2 import parameter_inventory
from epx_main import run_experiment
from models import build_model
from run import format_report, render_report
from utils.checkpoint import validate_scaler_state
from utils.config import load_config, validate_config
from utils.seed import set_seed


LABELS=('V1','V1 + full-rank HA','V1 + rank64 HA','V1 + rank32 HA')
CONFIG_FILES=('v1.yaml','v1_horizon_full.yaml','v1_horizon_rank64.yaml','v1_horizon_rank32.yaml')


def non_attention_manifest(model: torch.nn.Module) -> dict:
    manifest={}
    for name,parameter in model.named_parameters():
        if name.startswith('horizon_attention.'):
            continue
        manifest[name]=dict(shape=list(parameter.shape),trainable=parameter.requires_grad,
            numel=parameter.numel(),dtype=str(parameter.dtype),
            sha256=hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    return manifest


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode('utf-8')).hexdigest()


def assert_fair_initialization(reference: torch.nn.Module, candidate: torch.nn.Module) -> None:
    left,right=non_attention_manifest(reference),non_attention_manifest(candidate)
    if left!=right:
        changed=sorted(name for name in left.keys()|right.keys() if left.get(name)!=right.get(name))
        raise ValueError('unfair non-HA initialization: '+', '.join(changed))


def load_fair_configs(directory: Path) -> list[dict]:
    return [load_config(directory/name) for name in CONFIG_FILES]


def assert_fair_configs(configs: list[dict]) -> None:
    if len(configs)!=4 or configs[0]['model']['name']!='v1':
        raise ValueError('fair matrix requires V1 and three HA variants')
    for config in configs:
        validate_config(config)
    reference=deepcopy(configs[0])
    reference['experiment'].pop('name')
    reference['model'].pop('name')
    for config in configs[1:]:
        if config['model']['name']!='v2_fair':
            raise ValueError('HA variants must use v2_fair with the original V1 head')
        normalized=deepcopy(config)
        normalized.pop('horizon_attention')
        normalized['experiment'].pop('name')
        normalized['model'].pop('name')
        for section in reference.keys()|normalized.keys():
            if reference.get(section)!=normalized.get(section):
                raise ValueError(f'unfair configuration change in {section}')
    dimension=configs[0]['model']['d_model']
    for config,rank in zip(configs[1:],(dimension,dimension//2,dimension//4)):
        options=config['horizon_attention']
        if options!=dict(mode='full_rank' if rank==dimension else 'low_rank',rank=rank,
                         use_horizon_embedding=True,cache_kv=False):
            raise ValueError('fair horizon_attention matrix must be full, half and quarter rank with embedding')
    if configs[0]['training']['resume_from'] is not None:
        raise ValueError('fair training must start from the shared Stage-I checkpoint, not resume')


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def render_fair_report(results: list[dict], manifest_path: Path, path: Path) -> None:
    if len(results)!=4:
        raise ValueError('four completed fair experiments are required')
    hashes={r['fair_non_attention_sha256'] for r in results}
    if len(hashes)!=1:
        raise ValueError('non-HA initial parameter hashes differ')
    horizons={len(r['test_metrics']['by_horizon']) for r in results}
    if len(horizons)!=1 or next(iter(horizons))<=0:
        raise ValueError('fair result horizons must agree')
    lines=['# Fair V1 + Horizon Attention Ablation','',
        'Four independently trained experiments; all non-HA parameter names, shapes, trainability and initial values are identical.',
        'The full original V1 forecast head is retained. Only Horizon Attention parameters/readout are added.',
        '',f'- Control manifest: `{manifest_path.resolve()}`',
        f'- Common non-HA initial SHA256: `{next(iter(hashes))}`','',
        '| Model | Total | Trainable | Frozen | Non-HA trainable | Head trainable | HA trainable |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,r in zip(LABELS,results):
        inv=r['fair_initial_inventory']
        head=inv['categories']['Forecast Head']['trainable']
        ha=inv['categories']['Horizon Attention']['trainable']
        lines.append(f"| {label} | {r['total_params']} | {r['trainable_params']} | {r['total_params']-r['trainable_params']} | {r['trainable_params']-ha} | {head} | {ha} |")
    lines += ['', '| Model | MAE | RMSE | PICP | MPIW | PINAW | Coverage Error | Interval Score |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,r in zip(LABELS,results):
        overall=r['test_metrics']['overall']
        vals=[overall[key] for key in ('mae','rmse','picp','mpiw','pinaw','coverage_error','interval_score')]
        lines.append('| '+' | '.join([label,*['N/A' if v is None else f'{v:.6f}' for v in vals]])+' |')
    baseline=results[0]['test_metrics']['overall']
    lines += ['', '## Evidence and interpretation','',
        'Same Stage-I checkpoint and scaler; same seed, split, adapter settings, loss, optimizer, batch size, epoch budget and early-stopping rule.',
        'Each run selects its own best validation checkpoint; actual stopping epochs can differ. Test is evaluated once per run after selection.',
        'The old V1/current-V2 comparison also changed head capacity, whereas this matrix retains the 260,064-parameter V1 head.',
        'A single seed does not establish statistical significance or universal superiority. No ranks or thresholds were tuned from test results.','']
    for label,r in zip(LABELS[1:],results[1:]):
        metric=r['test_metrics']['overall']
        lines.append(f"- {label}: MAE delta vs V1 {metric['mae']-baseline['mae']:+.6f}; RMSE delta {metric['rmse']-baseline['rmse']:+.6f}; Interval Score delta {metric['interval_score']-baseline['interval_score']:+.6f} (negative is better).")
    for label,r in zip(LABELS,results):
        lines += ['',f'## {label}: complete result','',*format_report(r).splitlines()[1:],
            '',f"- Initial parameter manifest: `{r['paths']['initial_manifest']}`",
            f"- Metrics JSON: `{r['paths']['metrics_json']}`"]
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as handle:
            handle.write('\n'.join(lines).rstrip()+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_fair_ablation(configs: list[dict]) -> list[dict]:
    assert_fair_configs(configs)
    root=Path(configs[0]['experiment']['output_root'])
    control_dir=root/'results'/('v2_fair_control_'+datetime.now().strftime('%Y%m%d-%H%M%S')+'_'+uuid4().hex[:6])
    control_dir.mkdir(parents=True,exist_ok=False)
    set_seed(configs[0]['experiment']['seed'],configs[0]['experiment']['deterministic'])
    reference_bundle=create_dataloaders(configs[0])
    reference=build_model('v1',configs[0],reference_bundle.input_dim,reference_bundle.output_dim)
    reference_manifest=non_attention_manifest(reference)
    control=dict(configs=configs,labels=LABELS,non_ha_manifest=reference_manifest,
        common_sha256=manifest_sha256(reference_manifest),split=asdict(reference_bundle.split_metadata),
        optimizer_parameter_group_rule='one group containing all model parameters; frozen parameters have no gradient')
    _write_json(control_dir/'control.json',control)
    # Preflight every variant before any training can begin.
    for config in configs:
        set_seed(config['experiment']['seed'],config['experiment']['deterministic'])
        model=build_model(config['model']['name'],config,reference_bundle.input_dim,reference_bundle.output_dim)
        assert_fair_initialization(reference,model)
    results=[]
    for label,config in zip(LABELS,configs):
        set_seed(config['experiment']['seed'],config['experiment']['deterministic'])
        bundle=create_dataloaders(config)
        model=build_model(config['model']['name'],config,bundle.input_dim,bundle.output_dim)
        validate_scaler_state(model.pretrain_scaler_state,bundle.scaler.state_dict())
        assert_fair_initialization(reference,model)
        initial=non_attention_manifest(model)
        inventory=parameter_inventory(model)
        digest=manifest_sha256(initial)
        model.pretrain_provenance['fair_non_attention_sha256']=digest
        initial_path=control_dir/(config['experiment']['name']+'_initial.json')
        _write_json(initial_path,dict(non_attention=initial,inventory=inventory,config=config))
        print('FAIR GATE',label,'PASSED',inventory['summary'],'head',inventory['categories']['Forecast Head'],digest,flush=True)
        result=run_experiment(model,bundle.train_loader,bundle.val_loader,bundle.test_loader,config,bundle.scaler)
        result['fair_initial_inventory']=inventory
        result['paths']['initial_manifest']=initial_path.resolve()
        # Preserve immutable control metadata adjacent to each run's artifacts.
        _write_json(Path(result['paths']['metrics_json']).parent/'fair_control.json',dict(
            common_sha256=digest,initial_manifest=str(initial_path.resolve()),
            group_manifest=str((control_dir/'control.json').resolve()),config=config))
        render_report(result,Path(result['paths']['metrics_json']).parent/'report.md')
        results.append(result)
        del model,bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    render_fair_report(results,control_dir/'control.json',root/'reports/V2_fair_ablation_result.md')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir',type=Path,default=Path('configs/diagnosis'))
    args=parser.parse_args()
    run_fair_ablation(load_fair_configs(args.config_dir))


if __name__=='__main__':
    main()
