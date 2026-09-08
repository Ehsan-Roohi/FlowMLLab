"""Publication layout from saved first-seed weights; no fitting or frame selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_week11_reconstruction import ROOT, UNet, degrade, diagnostics, load_cases


def plot(folder):
    torch.set_num_threads(4)
    p=json.loads((folder/'protocol.json').read_text())
    rows=json.loads((folder/'metrics.json').read_text())
    cases=load_cases(ROOT/'data/modal_labs'); seed=p['seeds'][0]; frame=140
    native=torch.tensor(np.stack((cases[105]['u'],cases[105]['v']),1),dtype=torch.float32)
    coarse=degrade(native)
    mean=torch.tensor(p['mean']).reshape(1,2,1,1);std=torch.tensor(p['std']).reshape(1,2,1,1)
    fields=[native.numpy(),coarse.numpy()]
    for task in ('reconstruction','segmentation'):
        model=UNet(2 if task=='reconstruction' else 1)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint=torch.load(folder/f'{task}_{seed}.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.no_grad(): result=model((coarse-mean)/std)
        if task=='reconstruction': fields.append((result*std+mean).numpy())
        else:
            cut=next(r['probability_threshold'] for r in rows if r['method']==task and r['seed']==seed)
            mask=result.sigmoid().numpy()[:,0]>cut
    x,y=cases[105]['x'],cases[105]['y'];dx=float(np.diff(x)[0]);dy=float(np.diff(y)[0]);q=p['swirl_threshold']
    reference=diagnostics(fields[0],dx,dy)[0]>q
    lim=max(float(np.max(np.abs(cases[k]['v']))) for k in p['train'])
    plt.rcParams.update({'font.size':12,'savefig.dpi':220})
    fig=plt.figure(figsize=(10,16))
    names=['Native ROI reference','Bilinear interpolation','U-Net reconstruction + diagnostic','Direct U-Net mask']
    for i,name in enumerate(names):
        bottom=.055+(3-i)*.23
        ax=fig.add_axes([.09,bottom,.78,.196])
        if i<3:
            im=ax.contourf(x,y,fields[i][frame,1],levels=np.linspace(-lim,lim,51),cmap='RdBu_r',vmin=-lim,vmax=lim)
            region=diagnostics(fields[i][frame:frame+1],dx,dy)[0][0]>q
            ax.contour(x,y,region.astype(float),levels=[.5],colors='black',linewidths=1)
            cax=fig.add_axes([.9,bottom+.012,.018,.172])
            fig.colorbar(im,cax=cax,ticks=[-.5,0,.5],label='v/U')
        else:
            ax.pcolormesh(x,y,mask[frame],cmap='Blues',vmin=0,vmax=1,shading='auto')
            ax.contour(x,y,reference[frame].astype(float),levels=[.5],colors='darkorange',linewidths=1.2)
        ax.set(title=name,xlabel='x/D' if i==3 else '',ylabel='y/D',aspect='equal',xlim=(x[0],x[-1]),ylim=(y[0],y[-1]))
    fig.suptitle(f'Retained Re105 | frame {frame} | seed {seed}',y=.986,fontsize=15)
    fig.savefig(folder/'comparison.png');plt.close(fig)
    (folder/'plot_provenance.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'frame':frame,'seed':seed,'display':'Native-grid filled-contour level crossings only; no change to metrics or fields',
        'layout':'Equal physical aspect ratio; one method per aligned row'},indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    plot(parser.parse_args().folder)
