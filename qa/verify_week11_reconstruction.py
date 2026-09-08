"""Recompute every retained test score from saved weights (no retraining)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from run_week11_reconstruction import ROOT, UNet, degrade, load_cases


def verify(folder):
    torch.set_num_threads(4)
    protocol=json.loads((folder/'protocol.json').read_text())
    rows=json.loads((folder/'metrics.json').read_text())
    assert protocol['source_sha256']==hashlib.sha256((ROOT/'qa/run_week11_reconstruction.py').read_bytes()).hexdigest()
    cases=load_cases(ROOT/'data/modal_labs')
    truth=np.stack((cases[105]['u'],cases[105]['v']),1).astype('float32')
    train=np.concatenate([np.stack((cases[k]['u'],cases[k]['v']),1) for k in (90,110)]).astype('float32')
    t=torch.tensor(train)
    mean=t.mean((0,2,3),keepdim=True); std=t.std((0,2,3),keepdim=True).clamp_min(1e-8)
    np.testing.assert_allclose(mean.flatten(),protocol['mean'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(std.flatten(),protocol['std'],rtol=1e-6,atol=1e-6)
    dx=float(np.diff(cases[105]['x'])[0]);dy=float(np.diff(cases[105]['y'])[0])
    def diag(a):
        uy,ux=np.gradient(a[:,0],dy,dx,axis=(-2,-1),edge_order=2)
        vy,vx=np.gradient(a[:,1],dy,dx,axis=(-2,-1),edge_order=2)
        discriminant=(ux-vy)**2+4*uy*vx
        return np.sqrt(np.maximum(-discriminant,0))/2,vx-uy,ux+vy
    q=float(np.quantile(diag(train)[0][...,2:-2,2:-2],.9))
    np.testing.assert_allclose(q,protocol['swirl_threshold'],rtol=1e-6)
    label=diag(truth)[0]>q
    inputs=(degrade(torch.tensor(truth))-mean)/std
    for row in rows:
        if row['method']=='Interpolation': a=degrade(torch.tensor(truth)).numpy()
        else:
            task=row['method'];seed=row['seed']
            # PyTorch's version metadata is a string subclass, not a model callable.
            # Allow only that known type; never enable arbitrary pickle execution.
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint=torch.load(folder/f'{task}_{seed}.pt',map_location='cpu',weights_only=True)
            model=UNet(2 if task=='reconstruction' else 1)
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            history=np.load(folder/f'{task}_{seed}_loss.npy',allow_pickle=False)
            assert row['selected_epoch']==int(history[np.argmin(history[:,2]),0])
            with torch.no_grad(): pred=model(inputs)
            if task=='reconstruction': a=(pred*std+mean).numpy()
            else: mask=pred.sigmoid().numpy()[:,0]>row['probability_threshold']
        if row['method']!='segmentation':
            swirl,w,div=diag(a);mask=swirl>q
            sl=(slice(None),slice(None),slice(2,-2),slice(2,-2))
            vl2=np.linalg.norm((a-truth)[sl])/np.linalg.norm(truth[sl])
            wt=diag(truth)[1]
            wl2=np.linalg.norm((w-wt)[...,2:-2,2:-2])/np.linalg.norm(wt[...,2:-2,2:-2])
            np.testing.assert_allclose(vl2,row['velocity_relative_l2'],rtol=2e-5)
            np.testing.assert_allclose(wl2,row['vorticity_relative_l2'],rtol=2e-5)
        p=mask[...,2:-2,2:-2]; y=label[...,2:-2,2:-2]
        intersection=(p&y).sum((-2,-1));total=p.sum((-2,-1))+y.sum((-2,-1));union=(p|y).sum((-2,-1))
        dice=np.divide(2*intersection,total,out=np.ones_like(total,dtype=float),where=total!=0).mean()
        iou=np.divide(intersection,union,out=np.ones_like(union,dtype=float),where=union!=0).mean()
        np.testing.assert_allclose([dice,iou],[row['dice'],row['iou']],atol=1e-7)
    print('WEEK11_RECONSTRUCTION_PASS: all seven rows recomputed from saved models')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder)
