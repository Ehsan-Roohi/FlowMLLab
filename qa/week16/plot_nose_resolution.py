"""Plot actual old/new exported meshes; this figure does not validate flow fields."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def read_mesh(folder):
    with (folder/'mesh.su2').open() as f:
        assert f.readline().strip()=='NDIME= 2'
        count=int(f.readline().split('=')[1])
        cells=np.array([[int(v) for v in f.readline().split()[1:5]] for _ in range(count)])
        count=int(f.readline().split('=')[1])
        xy=np.array([[float(v) for v in f.readline().split()[:2]] for _ in range(count)])
    return xy[cells]

def build(rejected,replacement,output):
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True)
    manifest={}
    for ax,folder,title in zip(axs,[rejected,replacement],['Rejected original mesh','Replacement mesh: flow validation pending']):
        with np.load(folder/'cad_meridian.npz') as a:x=a['x'];r=a['r']
        xn=float(x[0]);rn=float(r[0]);poly=read_mesh(folder)
        poly[:,:,0]=(poly[:,:,0]-xn)/rn;poly[:,:,1]/=rn
        mask=(poly[:,:,0].max(1)>=-3)&(poly[:,:,0].min(1)<=3)&(poly[:,:,1].min(1)<=3)
        rings=poly[mask];rings=np.concatenate([rings,rings[:,:1]],axis=1)
        ax.add_collection(LineCollection(rings,colors='#607985',linewidths=.5))
        ax.plot((x-xn)/rn,r/rn,color='black',lw=2)
        ax.plot([0,0],[0,1],color='#c35421',lw=2,label='Original flat cap')
        ax.set(xlim=(-3,3),ylim=(0,3),xlabel='(x - nose x) / nose radius',title=title)
        ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8)
        manifest[title]={'mesh_sha256':hashlib.sha256((folder/'mesh.su2').read_bytes()).hexdigest(),'metadata':json.loads((folder/'metadata.json').read_text())}
    axs[0].set_ylabel('r / nose radius')
    fig.suptitle('Same NASA nose geometry; actual Gmsh cell boundaries')
    fig.text(.5,.10,'Equal physical aspect ratio. A resolved mesh still requires independent flow validation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94));fig.savefig(output);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rejected',type=Path,required=True);p.add_argument('--replacement',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.rejected,a.replacement,a.output)
