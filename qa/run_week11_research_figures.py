"""Fresh fixed-checkpoint inference on six non-review research fields.

Reads a supplied ShockVortexML checkout without altering it. Keeps raw/model
outputs in scratch; derived figures and a portable provenance report can be
retained in the course. No training, threshold selection or human truth.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figure(fields, prob, masks, row, output, name):
    length=row['reference']['reference_length']
    x,y=fields['x']/length,fields['y']/length
    fluid=fields['observation_mask'] & ~fields['geometry']
    from scipy.ndimage import binary_erosion
    logrho=np.log(np.where(fluid,fields['rho'],row['reference']['rho_inf']))
    gy,gx=np.gradient(logrho,y,x,edge_order=2)
    # Fixed physical display transform, not a frame-dependent quantile.
    schlieren=1-np.exp(-.4*np.hypot(gx,gy))
    gradient_domain=binary_erosion(fluid,iterations=1,border_value=0)
    extent=[float(x[0]),float(x[-1]),float(y[0]),float(y[-1])]
    fig,axes=plt.subplots(1,3,figsize=(16,5.6),layout='constrained')
    for ax in axes:
        ax.imshow(np.ma.masked_where(~gradient_domain,schlieren),origin='lower',extent=extent,
                  cmap='Greys',vmin=0,vmax=1,interpolation='nearest')
        solid=np.zeros((*fluid.shape,4));solid[fields['geometry']]=[.13,.18,.23,1]
        ax.imshow(solid,origin='lower',extent=extent,interpolation='nearest')
        ax.set(xlabel='x / reference length',ylabel='y / reference length',aspect='equal')
    axes[0].set_title('Native-density schlieren',fontsize=13)
    for ax, k, title, color in [(axes[1],0,'Learned shock front','#e85d04'),
                               (axes[2],1,'Learned vortex cores','#0077b6')]:
        overlay=np.zeros((*fluid.shape,4))
        from matplotlib.colors import to_rgba
        overlay[masks[k]]=to_rgba(color,.90)
        ax.imshow(overlay,origin='lower',extent=extent,interpolation='nearest')
        ax.set_title(title,fontsize=13)
    fig.suptitle(f"{row['case']} | t = {row['time']:.3g} | fixed research model, ML-only",fontsize=16)
    fig.legend(handles=[Patch(color='#e85d04',label='shock (0.97)'),
                        Patch(color='#0077b6',label='vortex core (0.85)'),
                        Patch(color='#212e3b',label='solid')],loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(output/f'{name}.png',dpi=300)
    fig.savefig(output/f'{name}.pdf')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('tmp/week11_research_v1'))
    args=p.parse_args();root=args.research_root.resolve();out=args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new output directory; preserve completed results')
    kit=root/'output/replay_kits/shock_repair_v2_native86_v1'
    dataset=root/'data/processed/jcp_native_primitives_v1'
    manifest=json.loads((kit/'REPLAY_MANIFEST.json').read_text())
    for item in manifest['files']:
        if digest(kit/item['path'])!=item['sha256']:
            raise ValueError('Copied research engine hash mismatch: '+item['path'])
    index=json.loads((dataset/'index.json').read_text())
    review=set()
    for size in ('512x512','900x990'):
        m=json.loads((root/f'data/processed/jcp_v4/expert_review/{size}/annotation_manifest.json').read_text())
        review.update(r['dataset_id'] for r in m['frames'])
    selected=[]
    for case in ('re1e6_retained','grid_f180_cfl0p20'):
        candidates=sorted([r for r in index['records'] if r['case']==case and
                           r['split']=='test' and r['dataset_id'] not in review and r['time']>1],
                          key=lambda r:r['time'])
        if len(candidates)<3:
            raise ValueError('Insufficient non-review cases')
        selected.extend(candidates[i] for i in [0,len(candidates)//2,len(candidates)-1])
    out.mkdir(parents=True)
    plan={'selection':'first/middle/last eligible time after t>1 for two development-test trajectories; all review IDs excluded',
          'ids':[r['dataset_id'] for r in selected], 'checkpoint_sha256':manifest['checkpoint_sha256'],
          'new_solver_runs':False,'new_training':False,'human_accuracy':False}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2))
    sys.path.insert(0,str(kit/'src'));sys.path.insert(0,str(kit/'tools'))
    import torch
    from jcp2026.models import make_model
    from jcp2026.infer import predict_tiled,clean
    from jcp2026.diagnostics import build_inputs
    from ml.flow_aligned import FreestreamReference
    from replay_jcp_repair_native86 import primitive_fields
    from replay_jcp_shock_repair_cli import fingerprint
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    cp=torch.load(kit/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=make_model(cp['spec'],cp['base_channels']);model.load_state_dict(cp['model_state_dict']);model.eval()
    thresholds=json.loads((kit/'thresholds.json').read_text())['thresholds']
    if thresholds!={'shock':.97,'vortex_core':.85}:
        raise ValueError('Unexpected operating point')
    runtime=manifest['runtime_settings'];records=[];started=time.monotonic()
    for i,row in enumerate(selected):
        path=dataset/row['primitive_file']
        if digest(path)!=row['primitive_sha256']:
            raise ValueError('Primitive hash mismatch')
        fields=primitive_fields(path)
        ref=FreestreamReference(**row['reference'])
        inputs,_=build_inputs(fields,ref)
        if fingerprint(inputs)!=row['expected_inputs']:
            raise ValueError('Input contract mismatch')
        probability=predict_tiled(model,torch,inputs[:cp['input_channels']],
            tile_size=runtime['tile_size'],overlap=runtime['tile_overlap'],batch_size=runtime['tile_batch'])
        domain=fields['observation_mask']&~fields['geometry']
        masks=[clean((probability[h]>=thresholds[key])&domain,9)
               for h,key in enumerate(('shock','vortex_core'))]
        assert np.isfinite(probability).all() and all(not m[~domain].any() for m in masks)
        name=f"{'airfoil' if row['family']=='airfoil' else 'cylinder'}_{i%3+1}"
        np.savez_compressed(out/f'{name}_arrays.npz',**fields,probabilities=probability,
                            shock=masks[0],vortex_core=masks[1])
        figure(fields,probability,masks,row,out,name)
        records.append({k:row[k] for k in ('dataset_id','case','time','family','split','primitive_sha256')})
        records[-1].update(figure=name+'.png',shock_pixels=int(masks[0].sum()),
                          vortex_pixels=int(masks[1].sum()),input_rebuild_verified=True,
                          array_sha256=digest(out/f'{name}_arrays.npz'))
        print(f'Fresh inference {i+1}/6: {row["dataset_id"]}',flush=True)
    report={**plan,'runs':records,'thresholds':thresholds,'seconds':time.monotonic()-started,
            'scope':'New forward passes on previously inspected development-test CFD. No tuning, human labels, hybrid, independent accuracy or new CFD.',
            'figure_sha256':{f.name:digest(f) for f in out.glob('*.png')}}
    (out/'research_manifest.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE six research-model forward passes',flush=True)


if __name__=='__main__':
    main()
