"""Original Transolver on the Week 15 retrospective double-step family split (no new CFD).

Run from repository root: python qa/train_family_transolver.py --seed 17
Test fields are used only after validation selects a checkpoint.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import platform
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmllab.vendor.transolver.Transolver_Irregular_Mesh import Model
from train_week15_vanilla_deeponet import geometry_ids, case_metrics, sha256

DATA = ROOT / 'results/step_operator_audit/source/dataset.npz'
ARCHIVE = ROOT / 'results/step_operator_audit/wake_predictions.tgz'
OUT = ROOT / 'results/family_transolver'
MODEL_CONFIG = dict(space_dim=2, fun_dim=2, out_dim=3, n_layers=3,
                    n_hidden=64, n_head=4, slice_num=16, mlp_ratio=2,
                    dropout=0.0, Time_Input=False, unified_pos=False)


def load_data():
    assert sha256(DATA) == '28d4d4c440cdc4c1ac1d13749ce00b0690d99f29cf20fd56c65fc00b6a8058fd'
    with np.load(DATA, allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    split_path=ROOT/'results/step_geometry_generalization/source/family_frozen_split_recovered.json'
    split=json.loads(split_path.read_text())
    gids=geometry_ids(data['masks'])
    groups={'train':np.array([i for i in split['train'] if gids[i]!=5]),
            'validation':np.array(split['validation']), 'test':np.array(split['heldout'])}
    assert [len(groups[k]) for k in ('train','validation','test')]==[100,8,19]
    assert set(gids[groups['test']])=={12,45,46,47,48,49,50,51}
    assert set(range(len(gids)))-set(np.concatenate(list(groups.values())))==set(np.flatnonzero(gids==5))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(gids[groups[a]]) & set(gids[groups[b]])
    return data,gids,groups,{k:sorted(map(int,set(gids[v]))) for k,v in groups.items()}


def make_features(data):
    """Only coordinates, stored geometric SDF and Re; no flow target is an input."""
    q = data['queries'].astype('float32')
    re = np.broadcast_to(data['Re'][:,None,None], (*q.shape[:2],1))
    return np.concatenate([q,re], axis=-1).astype('float32')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--epochs',type=int,default=1200)
    p.add_argument('--steps-per-epoch',type=int,default=16)
    p.add_argument('--points',type=int,default=1024)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--benchmark',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(args.seed)
    data,gids,groups,group_ids=load_data()
    features=make_features(data)
    masks=data['masks'].astype(bool)
    train_mask=np.zeros_like(masks);train_mask[groups['train']]=masks[groups['train']]
    xm,xs=features[train_mask].mean(0),features[train_mask].std(0)
    ym,ys=data['raw'][train_mask].mean(0),data['raw'][train_mask].std(0)
    xs[xs==0]=1;ys[ys==0]=1
    x=torch.from_numpy((features-xm)/xs)
    y=torch.from_numpy((data['raw']-ym)/ys)
    indices=[np.flatnonzero(m) for m in masks]
    model=Model(**MODEL_CONFIG)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs*args.steps_per_epoch,eta_min=1e-5)
    start=time.perf_counter()
    dest=args.output/f'seed_{args.seed}'
    if not args.benchmark:
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'scaling.npz',input_mean=xm,input_std=xs,target_mean=ym,target_std=ys)
    history=[];best=float('inf');best_epoch=0

    def infer(ci):
        # One complete fluid cloud. Do not chunk the query set: attention is contextual.
        ids=indices[ci]
        with torch.no_grad():
            out=model(x[ci,ids,:2][None],x[ci,ids,2:][None])[0]
        return out

    for epoch in range(1,(1 if args.benchmark else args.epochs)+1):
        model.train();losses=[]
        for step in range(args.steps_per_epoch):
            ci=int(rng.choice(groups['train']))
            ids=rng.choice(indices[ci],min(args.points,len(indices[ci])),replace=False)
            opt.zero_grad(set_to_none=True)
            pred=model(x[ci,ids,:2][None],x[ci,ids,2:][None])[0]
            loss=(pred-y[ci,ids]).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();sched.step();losses.append(loss.item())
        if args.benchmark:
            print(json.dumps({'seconds_per_update':(time.perf_counter()-start)/args.steps_per_epoch,
                              'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
            return
        if epoch==1 or epoch%25==0 or epoch==args.epochs:
            model.eval()
            val=float(np.mean([float((infer(ci)-y[ci,indices[ci]]).square().mean()) for ci in groups['validation']]))
            if not np.isfinite(val): raise RuntimeError('Non-finite validation loss')
            record=dict(epoch=epoch,training_mse=float(np.mean(losses)),validation_mse=val,
                        elapsed_seconds=time.perf_counter()-start)
            history.append(record)
            if val<best:
                best=val;best_epoch=epoch
                torch.save(model.state_dict(),dest/'best.pt')
            (dest/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            print(json.dumps({'seed':args.seed,**record}),flush=True)
    model.load_state_dict(torch.load(dest/'best.pt',weights_only=True));model.eval()
    train_seconds=time.perf_counter()-start
    records=[];test_start=time.perf_counter()
    for ci in groups['test']:
        pred=np.zeros_like(data['raw'][ci]);pred[indices[ci]]=infer(ci).numpy()*ys+ym
        assert np.isfinite(pred).all()
        name=f'g{gids[ci]:03d}_Re{int(data["Re"][ci])}_medium'
        records.append(dict(case=name,seed=args.seed,checkpoint_epoch=best_epoch,
                            **case_metrics(data['raw'][ci],pred,masks[ci])))
        np.savez_compressed(dest/f'{name}_prediction.npz',prediction=pred,truth=data['raw'][ci],
                            coordinates=data['queries'][ci,:,:2]*[5,1],mask=masks[ci],shape=data['shape'],
                            row_index=ci,Re=data['Re'][ci],evaluation_group='retrospective_test')
    with (dest/'case_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    source_files=[Path(__file__),ROOT/'qa/train_week15_vanilla_deeponet.py',
                  *sorted((ROOT/'flowmllab/vendor/transolver').glob('*.py'))]
    manifest=dict(model='Original Transolver, compact irregular-mesh configuration',model_config=MODEL_CONFIG,
        parameters=sum(p.numel() for p in model.parameters()),seed=args.seed,
        epochs=args.epochs,steps_per_epoch=args.steps_per_epoch,points_per_update=args.points,
        input_features=['x/L','y/H','stored geometric SDF','Re'],targets=['u','v','Re*p_kinematic'],
        split={k:v.tolist() for k,v in groups.items()},geometry_ids=group_ids,
        dataset_sha256=sha256(DATA),split_source_sha256=sha256(ROOT/'results/step_geometry_generalization/source/family_frozen_split_recovered.json'),quarantined_geometry_ids=[5],
        source_sha256={str(f.relative_to(ROOT)):sha256(f) for f in source_files},
        checkpoint_sha256=sha256(dest/'best.pt'),checkpoint_epoch=best_epoch,
        selection='Minimum mean per-case scaled MSE over all fluid points of 8 validation cases; every 25 epochs plus epoch 1.',
        inference='Entire fluid point cloud per case in one call; no target-derived sampling; no query chunking.',
        training='Uniform training case and uniform fluid-point sample; unweighted normalized MSE; AdamW; cosine LR 1e-3 to 1e-5; clip 1.0.',
        training_seconds=train_seconds,test_inference_seconds=time.perf_counter()-test_start,
        software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        hardware=dict(device='cpu',threads=args.threads,processor=platform.processor()),
        limitation='Same 100/8/19 post-audit family split with g005 quarantined; 19,200 updates match the fresh suite ceiling, but losses, sampled-point exposure and architecture tuning differ. Not equal-FLOP superiority evidence. Historically inspected test.',
        history=history)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('STEP_TRANSOLVER_COMPLETE',args.seed,flush=True)

if __name__=='__main__':main()
