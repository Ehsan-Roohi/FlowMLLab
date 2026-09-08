"""Independent U-Net reconstruction/identification audit on retained LBM fields.

Run from the repository root. Requires optional CPU PyTorch. Never overwrites
an existing output directory. No Ricardo-course code or checkpoints are used.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmllab.modal_experiments import load_cases


class UNet(nn.Module):
    """Two-level compact U-Net; explicit interpolation handles odd grid widths."""
    def __init__(self, outputs):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),
                                 nn.Conv2d(b,b,3,padding=1),nn.ReLU())
        self.e1=block(2,8); self.e2=block(8,16); self.bridge=block(16,32)
        self.d2=block(48,16); self.d1=block(24,8)
        self.head=nn.Conv2d(8,outputs,1)

    def forward(self,x):
        a=self.e1(x); b=self.e2(F.max_pool2d(a,2))
        c=self.bridge(F.max_pool2d(b,2))
        c=self.d2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        c=self.d1(torch.cat((F.interpolate(c,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return self.head(c)


def degrade(a):
    # Area reduction 32x78 -> 8x19; width ratio is 78/19, not exactly four.
    return F.interpolate(F.interpolate(a,size=(8,19),mode='area'),
                         size=a.shape[-2:],mode='bilinear',align_corners=False)


def diagnostics(a,dx,dy):
    uy,ux=np.gradient(a[:,0],dy,dx,axis=(-2,-1),edge_order=2)
    vy,vx=np.gradient(a[:,1],dy,dx,axis=(-2,-1),edge_order=2)
    swirl=np.sqrt(np.maximum(-.25*(ux-vy)**2-uy*vx,0))
    return swirl,vx-uy,ux+vy


def mask_scores(p,y):
    p=np.asarray(p,dtype=bool)[...,2:-2,2:-2]
    y=np.asarray(y,dtype=bool)[...,2:-2,2:-2]
    tp=np.sum(p&y,axis=(-2,-1)); fp=np.sum(p&~y,axis=(-2,-1)); fn=np.sum(~p&y,axis=(-2,-1))
    def ratio(n,d): return np.divide(n,d,out=np.ones_like(n,dtype=float),where=d!=0)
    return {'dice':float(ratio(2*tp,2*tp+fp+fn).mean()),
            'iou':float(ratio(tp,tp+fp+fn).mean())}


def run(out,epochs,seeds):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    cases=load_cases(ROOT/'data/modal_labs')
    raw={k:torch.tensor(np.stack((d['u'],d['v']),axis=1),dtype=torch.float32) for k,d in cases.items()}
    train=torch.cat((raw[90],raw[110]))
    mean=train.mean((0,2,3),keepdim=True); std=train.std((0,2,3),keepdim=True).clamp_min(1e-8)
    dx=float(np.diff(cases[90]['x'])[0]); dy=float(np.diff(cases[90]['y'])[0])
    for d in cases.values():
        if not np.allclose(np.diff(d['x']),dx) or not np.allclose(np.diff(d['y']),dy):
            raise ValueError('All cases must share a uniform physical grid')
    # Frozen before training: top decile of training swirl, not test-dependent.
    threshold=float(np.quantile(diagnostics(train.numpy(),dx,dy)[0][...,2:-2,2:-2],.90))
    labels={k:diagnostics(v.numpy(),dx,dy)[0]>threshold for k,v in raw.items()}
    inputs={k:(degrade(v)-mean)/std for k,v in raw.items()}
    x=torch.cat((inputs[90],inputs[110]))
    protocol={'train':[90,110],'validation':100,'retained_test':105,
        'epochs':epochs,'seeds':seeds,'batch_size':32,'optimizer':'Adam','learning_rate':.001,
        'coarse_shape':[8,19],'fine_shape':[32,78],'swirl_threshold':threshold,
        'label':'ROI-velocity-derived swirling strength; weak reference, not human truth',
        'scope':'Previously inspected coarse incompressible LBM; no shock accuracy claim',
        'mean':mean.flatten().tolist(),'std':std.flatten().tolist(),
        'torch':torch.__version__,'numpy':np.__version__,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_manifest_sha256':hashlib.sha256((ROOT/'data/modal_labs/manifest.json').read_bytes()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]; predictions={'Native reference':raw[105].numpy(),'Interpolation':degrade(raw[105]).numpy()}
    def field_score(a):
        truth=raw[105].numpy(); sl=(slice(None),slice(None),slice(2,-2),slice(2,-2))
        swirl,w,div=diagnostics(a,dx,dy); _,wt,_=diagnostics(truth,dx,dy)
        return dict(velocity_relative_l2=float(np.linalg.norm((a-truth)[sl])/np.linalg.norm(truth[sl])),
                    vorticity_relative_l2=float(np.linalg.norm((w-wt)[...,2:-2,2:-2])/np.linalg.norm(wt[...,2:-2,2:-2])),
                    divergence_rms=float(np.sqrt(np.mean(div[...,2:-2,2:-2]**2))),
                    **mask_scores(swirl>threshold,labels[105]))
    rows.append(dict(method='Interpolation',seed=None,**field_score(predictions['Interpolation'])))
    for seed in seeds:
        for task in ('reconstruction','segmentation'):
            torch.manual_seed(seed)
            model=UNet(2 if task=='reconstruction' else 1)
            target=(train-mean)/std if task=='reconstruction' else torch.tensor(np.concatenate((labels[90],labels[110]))[:,None],dtype=torch.float32)
            valtarget=(raw[100]-mean)/std if task=='reconstruction' else torch.tensor(labels[100][:,None],dtype=torch.float32)
            lossfn=nn.MSELoss() if task=='reconstruction' else nn.BCEWithLogitsLoss()
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            best=float('inf'); history=[]; state=None
            for epoch in range(epochs):
                model.train(); total=0.
                for ix in torch.randperm(len(x)).split(32):
                    optimizer.zero_grad(); loss=lossfn(model(x[ix]),target[ix]); loss.backward(); optimizer.step()
                    total+=loss.item()*len(ix)
                model.eval()
                with torch.no_grad(): val=lossfn(model(inputs[100]),valtarget).item()
                history.append([epoch+1,total/len(x),val])
                if val<best: best=val; state=copy.deepcopy(model.state_dict()); selected=epoch+1
            model.load_state_dict(state)
            with torch.no_grad(): pred=model(inputs[105]); vp=model(inputs[100])
            row=dict(method=task,seed=seed,selected_epoch=selected,parameters=sum(p.numel() for p in model.parameters()))
            if task=='reconstruction':
                a=(pred*std+mean).numpy(); row.update(field_score(a))
                if seed==seeds[0]: predictions['U-Net reconstruction']=a
            else:
                vp=vp.sigmoid().numpy()[:,0]
                grid=np.linspace(.1,.9,17)
                cut=float(max(grid,key=lambda q:mask_scores(vp>q,labels[100])['dice']))
                mask=pred.sigmoid().numpy()[:,0]>cut
                row.update(probability_threshold=cut,**mask_scores(mask,labels[105]))
                if seed==seeds[0]: predictions['Direct mask']=mask
            rows.append(row)
            np.save(out/f'{task}_{seed}_loss.npy',np.asarray(history))
            torch.save({'state_dict':state,'protocol':protocol,'selected_epoch':selected},out/f'{task}_{seed}.pt')
            print(json.dumps(row),flush=True)
    (out/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':220})
    # Fixed midpoint frame and first seed; neither selected for visual success.
    fig,axes=plt.subplots(4,1,figsize=(11,9),layout='constrained')
    xx,yy=cases[105]['x'],cases[105]['y']; frame=140
    lim=float(np.max(np.abs(train[:,1].numpy())))
    for ax,name in zip(axes,list(predictions)[:3]+['Direct mask']):
        a=predictions[name]
        if name=='Direct mask':
            ax.pcolormesh(xx,yy,a[frame],vmin=0,vmax=1,cmap='Blues',shading='auto')
            ax.contour(xx,yy,labels[105][frame].astype(float),levels=[.5],colors='darkorange',linewidths=.8)
        else:
            im=ax.pcolormesh(xx,yy,a[frame,1],vmin=-lim,vmax=lim,cmap='RdBu_r',shading='auto')
            ax.contour(xx,yy,(diagnostics(a[frame:frame+1],dx,dy)[0][0]>threshold).astype(float),levels=[.5],colors='black',linewidths=.6)
            fig.colorbar(im,ax=ax,label='v/U',shrink=.8)
        ax.set(title=name,xlabel='x/D',ylabel='y/D',aspect='equal')
    fig.suptitle('Retained Re105, frame 140; first training seed (not best seed)')
    fig.savefig(out/'comparison.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    for ax,task in zip(axes,('reconstruction','segmentation')):
        for seed in seeds:
            h=np.load(out/f'{task}_{seed}_loss.npy')
            ax.semilogy(h[:,0],h[:,1],alpha=.65,label=f'{seed} train')
            ax.semilogy(h[:,0],h[:,2],'--',alpha=.65,label=f'{seed} validation')
        ax.set(title=task,xlabel='Epoch',ylabel='Normalized MSE' if task=='reconstruction' else 'Binary cross entropy'); ax.legend(fontsize=7,ncol=2)
    fig.savefig(out/'loss.png'); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    args=parser.parse_args()
    if args.epochs<1 or len(set(args.seeds))!=len(args.seeds): parser.error('Positive epochs and unique seeds required')
    run(args.output,args.epochs,args.seeds)
