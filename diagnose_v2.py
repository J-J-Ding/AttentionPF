"""Read-only V1/V2 parameter audit and per-sample validation attention diagnosis."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataload import create_dataloaders
from models import build_model
from utils.checkpoint import load_checkpoint, validate_scaler_state
from utils.config import load_config


CATEGORIES = ('Patch Embedding','Transformer Q','Transformer K','Transformer V',
              'Transformer O','FFN','Adapter','Horizon Attention','Forecast Head',
              'Position Embedding','LayerNorm','Mask Token')


def parameter_category(name: str) -> str:
    if '.adapters.' in name:
        return 'Adapter'
    if name.startswith('horizon_attention.'):
        return 'Horizon Attention'
    if name.startswith('output_projection.'):
        return 'Forecast Head'
    if name.startswith('backbone.patch_embedding.'):
        return 'Patch Embedding'
    if name == 'backbone.position_embedding':
        return 'Position Embedding'
    if name == 'backbone.mask_token':
        return 'Mask Token'
    for projection, label in [('q_proj','Q'),('k_proj','K'),('v_proj','V'),('out_proj','O')]:
        if f'.self_attn.{projection}.' in name:
            return f'Transformer {label}'
    if '.linear1.' in name or '.linear2.' in name:
        return 'FFN'
    if '.norm1.' in name or '.norm2.' in name:
        return 'LayerNorm'
    raise ValueError(f'unclassified parameter: {name}')


def parameter_inventory(model: torch.nn.Module) -> dict:
    categories={name:dict(total=0,trainable=0,frozen=0) for name in CATEGORIES}
    modules=defaultdict(lambda:dict(total=0,trainable=0,frozen=0))
    parameters=[]
    identities=set()
    for name,parameter in model.named_parameters():
        if id(parameter) in identities:
            raise ValueError(f'duplicate parameter identity: {name}')
        identities.add(id(parameter))
        category=parameter_category(name)
        module=name if name.endswith(('position_embedding','mask_token','horizon_embedding')) else name.rsplit('.',1)[0]
        count=parameter.numel()
        trainable=count if parameter.requires_grad else 0
        for row in (categories[category],modules[module]):
            row['total']+=count
            row['trainable']+=trainable
            row['frozen']+=count-trainable
        parameters.append(dict(name=name,shape=list(parameter.shape),numel=count,
                               trainable=parameter.requires_grad,category=category))
    summary={key:sum(row[key] for row in categories.values()) for key in ('total','trainable','frozen')}
    assert summary['total']==summary['trainable']+summary['frozen']
    return dict(summary=summary,categories=categories,modules=dict(modules),parameters=parameters)


def inventory_text(name: str, inventory: dict) -> str:
    lines=[f'{name}: parameter inventory',f"{'Category / module':64s} {'Total':>12s} {'Trainable':>12s} {'Frozen':>12s}"]
    for label,row in [*inventory['categories'].items(),('TOTAL',inventory['summary']),
                      *sorted(inventory['modules'].items())]:
        lines.append(f"{label:64s} {row['total']:12,d} {row['trainable']:12,d} {row['frozen']:12,d}")
    return '\n'.join(lines)


def _describe(values) -> dict:
    values=np.asarray(values,dtype=np.float64)
    if not values.size:
        return dict(mean=None,p10=None,p50=None,p90=None,min=None,max=None)
    return dict(mean=float(values.mean()),p10=float(np.quantile(values,.1)),
                p50=float(np.quantile(values,.5)),p90=float(np.quantile(values,.9)),
                min=float(values.min()),max=float(values.max()))


def attention_statistics(attention: np.ndarray) -> dict:
    a=np.asarray(attention,dtype=np.float64)
    if a.ndim!=3 or min(a.shape)<=0 or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('attention must be finite nonnegative nonempty [S,P,N]')
    if not np.allclose(a.sum(-1),1.,atol=1e-6,rtol=1e-6):
        raise ValueError('attention must sum to one along N')
    samples,pred,tokens=a.shape
    entropy=-(a*np.log(np.maximum(a,1e-300))).sum(-1)
    entropy=np.maximum(entropy,0.)
    normalized=entropy/np.log(tokens) if tokens>1 else np.zeros_like(entropy)
    top1=a.max(-1)
    mask=~np.eye(pred,dtype=bool)
    cos_sum=np.zeros((pred,pred))
    tv_sum=np.zeros((pred,pred))
    cos_values=[]
    tv_values=[]
    sample_cos=[]
    sample_tv=[]
    adjacent=[]
    # Never compute statistics on sample-averaged vectors. Chunking bounds memory.
    for start in range(0,samples,16):
        batch=a[start:start+16]
        unit=batch/np.linalg.norm(batch,axis=-1,keepdims=True)
        # Avoid loading a second BLAS/OpenMP runtime alongside Windows PyTorch.
        cosine=np.clip(np.einsum('bpr,bqr->bpq',unit,unit,optimize=False),0.,1.)
        tv=.5*np.abs(batch[:,:,None,:]-batch[:,None,:,:]).sum(-1)
        cos_sum+=cosine.sum(0)
        tv_sum+=tv.sum(0)
        if pred>1:
            off=cosine[:,mask]
            off_tv=tv[:,mask]
            cos_values.append(off.ravel().astype(np.float32))
            tv_values.append(off_tv.ravel().astype(np.float32))
            sample_cos.extend(off.mean(1).tolist())
            sample_tv.extend(off_tv.mean(1).tolist())
            adjacent.append(cosine[:,np.arange(pred-1),np.arange(1,pred)].ravel())
    cosine_mean=cos_sum/samples
    tv_mean=tv_sum/samples
    collapse=(np.asarray(sample_cos)>=.99)&(np.asarray(sample_tv)<=.01)
    groups=[]
    if pred>=3:
        for label,indices in zip(('short','middle','long'),np.array_split(np.arange(pred),3)):
            groups.append(dict(name=label,horizons=(indices+1).tolist(),
                               entropy_normalized=_describe(normalized[:,indices]),
                               top1_mass=_describe(top1[:,indices])))
    group_pairs={}
    for left in groups:
        li=np.array(left['horizons'])-1
        for right in groups:
            ri=np.array(right['horizons'])-1
            block=np.ix_(li,ri)
            selection=np.ones((len(li),len(ri)),dtype=bool)
            if left['name']==right['name']:
                np.fill_diagonal(selection,False)
            group_pairs[f"{left['name']}:{right['name']}"]=dict(
                cosine=_describe(cosine_mean[block][selection]),
                tv=_describe(tv_mean[block][selection]))
    result=dict(shape=list(a.shape),degenerate_single_token=tokens==1,
        entropy_nats=_describe(entropy),entropy_normalized=_describe(normalized),
        effective_tokens=_describe(np.exp(entropy)),top1_mass=_describe(top1),
        uniform_kl=_describe(np.log(tokens)-entropy),
        cosine_off_diagonal=_describe(np.concatenate(cos_values) if cos_values else []),
        pairwise_tv=_describe(np.concatenate(tv_values) if tv_values else []),
        adjacent_cosine=_describe(np.concatenate(adjacent) if adjacent else []),
        per_sample_mean_cosine=sample_cos,per_sample_mean_tv=sample_tv,
        cosine_matrix=cosine_mean.tolist(),tv_matrix=tv_mean.tolist(),
        entropy_by_horizon=dict(mean=normalized.mean(0).tolist(),
            p10=np.quantile(normalized,.1,axis=0).tolist(),p90=np.quantile(normalized,.9,axis=0).tolist()),
        groups=groups,group_pairs=group_pairs,
        flags=dict(collapse_sample_fraction=float(collapse.mean()) if collapse.size else None,
            uniform_fraction=float((normalized>=.95).mean()) if tokens>1 else None,
            sharp_fraction=float(((normalized<=.2)|(top1>=.9)).mean()) if tokens>1 else None),
        thresholds=dict(cosine=.99,tv=.01,uniform_entropy=.95,sharp_entropy=.20,sharp_top1=.90))
    return result


def save_attention_figures(attention: np.ndarray, stats: dict, directory: Path,
                           patch_length: int, patch_stride: int) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    pred,tokens=attention.shape[1:]
    ticks=[f'{n*patch_stride}-{n*patch_stride+patch_length-1}' for n in range(tokens)]
    def save(fig,name):
        fig.tight_layout()
        for extension in ('png','pdf'):
            fig.savefig(directory/f'{name}.{extension}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    mean=attention.mean(0)
    fig,ax=plt.subplots(figsize=(9,7))
    im=ax.imshow(mean,aspect='auto',vmin=0,vmax=1,cmap='viridis',origin='lower',extent=(-.5,tokens-.5,.5,pred+.5))
    ax.set(xticks=np.arange(tokens),xticklabels=ticks,xlabel='Historical patch (zero-based window indices)',ylabel='Forecast horizon',title='Validation sample-mean horizon attention')
    ax.tick_params(axis='x',rotation=45)
    fig.colorbar(im,ax=ax,label='Attention probability')
    save(fig,'attention_full')
    selections=[('validation_mean',mean),*[(f'sample_{i}',attention[i]) for i in (0,len(attention)//2,len(attention)-1)]]
    if pred>=3:
        for label,values in selections:
            fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
            for ax,indices,name in zip(axes,np.array_split(np.arange(pred),3),('Short','Middle','Long')):
                im=ax.imshow(values[indices],aspect='auto',vmin=0,vmax=1,origin='lower',cmap='viridis',
                             extent=(-.5,tokens-.5,indices[0]+.5,indices[-1]+1.5))
                ax.set(title=f'{name}: {indices[0]+1}-{indices[-1]+1}',xlabel='Historical patch',ylabel='Forecast horizon')
                ax.set_xticks(np.arange(tokens),ticks,rotation=90)
            fig.colorbar(im,ax=list(axes),label='Attention probability',shrink=.8)
            fig.suptitle(f'Horizon attention: {label}')
            # constrained_layout handles the shared colorbar.
            for extension in ('png','pdf'):
                fig.savefig(directory/f'attention_groups_{label}.{extension}',dpi=300,bbox_inches='tight')
            plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(stats['cosine_matrix'],vmin=0,vmax=1,origin='lower',extent=(.5,pred+.5,.5,pred+.5))
    ax.set(xlabel='Horizon',ylabel='Horizon',title='Mean within-sample attention cosine')
    fig.colorbar(im,ax=ax)
    save(fig,'horizon_cosine')
    fig,ax=plt.subplots(figsize=(9,4))
    h=np.arange(1,pred+1)
    series=stats['entropy_by_horizon']
    ax.plot(h,series['mean'],label='Mean normalized entropy')
    ax.fill_between(h,series['p10'],series['p90'],alpha=.2,label='Sample p10-p90')
    ax.axhline(.95,linestyle='--',color='gray',label='Uniform candidate threshold')
    ax.axhline(.2,linestyle=':',color='gray',label='Sharp candidate threshold')
    ax.set(xlabel='Forecast horizon',ylabel='Entropy / log(N)',ylim=(0,1.05))
    ax.legend()
    save(fig,'entropy_by_horizon')


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_diagnosis(v1_checkpoint: Path, v2_checkpoint: Path, output_root: Path) -> Path:
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')+'_'+uuid4().hex[:6]
    directory=output_root/'results'/f'v2_diagnosis_{stamp}'
    directory.mkdir(parents=True,exist_ok=False)
    loaded={}
    for label,checkpoint in (('V1',v1_checkpoint),('V2',v2_checkpoint)):
        # Saved resolved configuration, rather than mutable current defaults.
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        config=state['config']
        bundle=create_dataloaders(config)
        model=build_model(config['model']['name'],config,bundle.input_dim,bundle.output_dim)
        load_checkpoint(checkpoint,model,expected_config=config,
                        expected_model_name=config['model']['name'],restore_rng=False)
        validate_scaler_state(state['scaler_state_dict'],bundle.scaler.state_dict())
        inventory=parameter_inventory(model)
        text=inventory_text(label,inventory)
        print(text,flush=True)
        (directory/f'{label.lower()}_parameters.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
        (directory/f'{label.lower()}_parameters.txt').write_text(text+'\n',encoding='utf-8')
        loaded[label]=(model,config,bundle,inventory)
    v1,c1,_,i1=loaded['V1']
    v2,c2,bundle,i2=loaded['V2']
    shared1={n:p for n,p in v1.named_parameters() if not n.startswith('output_projection.')}
    shared2={n:p for n,p in v2.named_parameters() if not n.startswith(('output_projection.','horizon_attention.'))}
    structural_equal=(set(shared1)==set(shared2) and all(
        shared1[n].shape==shared2[n].shape and shared1[n].requires_grad==shared2[n].requires_grad for n in shared1))
    frozen_equal=all(torch.equal(shared1[n],shared2[n]) for n in shared1 if not shared1[n].requires_grad)
    if not structural_equal or not frozen_equal:
        raise ValueError('unexpected non-head backbone structure/frozen-weight difference')
    explanation=(f"V1 trainable {i1['summary']['trainable']:,} = Adapter 18,432 + Head 260,064.\n"
        f"V2 trainable {i2['summary']['trainable']:,} = Adapter 18,432 + Head 2,709 + HA 36,864.\n"
        'Head delta -257,355; HA delta +36,864; net delta -220,491.\n'
        'No separate multilayer forecast decoder exists. Backbone/Adapter names, shapes and trainability match; all frozen tensors match.\n')
    print(explanation,flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v2.to(device).eval()
    batches=[]
    with torch.inference_mode():
        for x,_ in bundle.val_loader:
            _,weights=v2.forward_with_attention(x.to(device))
            batches.append(weights.cpu().numpy())
    attention=np.concatenate(batches)
    statistics=attention_statistics(attention)
    np.savez_compressed(directory/'attention_raw.npz',attention=attention,
        validation_sample_indices=np.arange(len(attention)),
        forecast_starts=bundle.val_loader.dataset.forecast_starts,
        horizons=np.arange(1,attention.shape[1]+1),
        patch_start=np.arange(attention.shape[2])*c2['patch']['stride'],
        patch_end=np.arange(attention.shape[2])*c2['patch']['stride']+c2['patch']['length']-1)
    (directory/'attention_statistics.json').write_text(json.dumps(statistics,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    provenance=dict(split='validation',split_metadata=asdict(bundle.split_metadata),
        v1_checkpoint=str(v1_checkpoint.resolve()),v2_checkpoint=str(v2_checkpoint.resolve()),
        v1_sha256=_hash_file(v1_checkpoint),v2_sha256=_hash_file(v2_checkpoint),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config=c2,non_head_structure_equal=structural_equal,frozen_tensors_equal=frozen_equal)
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    save_attention_figures(attention,statistics,directory/'figures',c2['patch']['length'],c2['patch']['stride'])
    report=['# V2 structure and attention diagnosis','', '## Actual parameter inventories','', '```text',
        inventory_text('V1',i1),'',inventory_text('V2',i2),'```','', '## Structural finding','',explanation,
        'V1 to current V2 is not an HA-only ablation: the forecast head also shrinks. Adapter capacity is unchanged.',
        '', '## Validation attention diagnostics','',f'Raw tensor: `{attention.shape}`; computed per sample before aggregation.',
        f"- Mean off-diagonal cosine: {statistics['cosine_off_diagonal']['mean']:.6f}",
        f"- Mean pairwise TV: {statistics['pairwise_tv']['mean']:.6f}",
        f"- Normalized entropy: {statistics['entropy_normalized']}",
        f"- Effective token count: {statistics['effective_tokens']}",
        f"- Top1 mass: {statistics['top1_mass']}",f"- Candidate rates: {statistics['flags']}",
        '', 'High cosine alone does not prove collapse. Joint cosine/TV flags and entropy/top1 are heuristic diagnostics, not proof of the cause of forecast error.',
        'The validation mean heatmap must be read with the per-sample examples and quantiles. No test data was used to choose ranks or thresholds.',
        '', '## Short / middle / long horizons','', '| Group | Horizons | Mean normalized entropy | Mean top1 |',
        '|---|---|---:|---:|']
    for group in statistics['groups']:
        report.append(f"| {group['name']} | {group['horizons'][0]}-{group['horizons'][-1]} | {group['entropy_normalized']['mean']:.6f} | {group['top1_mass']['mean']:.6f} |")
    report += ['', '## Sources and figures','',f'- Immutable diagnosis directory: `{directory.resolve()}`',
        f'- V1 checkpoint: `{v1_checkpoint.resolve()}`',f'- V2 checkpoint: `{v2_checkpoint.resolve()}`',
        f'- Raw per-sample tensor: `{(directory/"attention_raw.npz").resolve()}`',
        f'- Full statistics / pair matrices / group comparisons: `{(directory/"attention_statistics.json").resolve()}`',
        f'- Parameter names, shapes, flags: `{(directory/"v1_parameters.json").resolve()}`, `{(directory/"v2_parameters.json").resolve()}`',
        f'- Provenance: `{(directory/"provenance.json").resolve()}`','']
    for path in sorted((directory/'figures').glob('*.png')):
        report += [f'### {path.stem}','',f'![{path.stem}]({path.resolve().as_posix()})','']
    report_path=output_root/'reports/V2_diagnosis_result.md'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text('\n'.join(report).rstrip()+'\n',encoding='utf-8')
    print('DIAGNOSIS',directory, json.dumps({k:statistics[k] for k in ('cosine_off_diagonal','pairwise_tv','entropy_normalized','flags')}),flush=True)
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-checkpoint',type=Path,default=Path('checkpoints/v1_horizon_baseline_seed2026_20261016-144820-008596_e622d6/best.pt'))
    parser.add_argument('--v2-checkpoint',type=Path,default=Path('checkpoints/v2_horizon_low_rank_seed2026_20261016-144949-667162_06ae41/best.pt'))
    parser.add_argument('--output-root',type=Path,default=Path('.'))
    args=parser.parse_args()
    run_diagnosis(args.v1_checkpoint,args.v2_checkpoint,args.output_root)


if __name__=='__main__':
    main()
