"""Teaching figures from actual NASA mesh and convergence files; no solver run."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from analyze import load_csv


def build(root, runs):
    root=Path(root)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for row in runs:
        d=root/f"seeb_level_{row['level']:g}"
        h=load_csv(d/'history.csv');it=np.arange(len(h))
        label=f"{row['cells']:,} cells"
        axs[0].plot(it,h['rms[Rho]'],label=label)
        axs[1].plot(it,h['CD'],label=label)
    axs[0].axhline(-9,color='gray',ls='--',lw=.8,label='Residual criterion')
    axs[0].set(xlabel='Iteration',ylabel='log10 RMS density residual')
    axs[1].set(xlabel='Iteration',ylabel='SU2 pressure drag coefficient')
    for ax in axs:
        ax.axvline(2000,color='gray',ls=':',lw=.8)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('NASA case: residual convergence and force history (limiter freezes at 2000)')
    fig.tight_layout();fig.savefig(root/'seeb_convergence.png',dpi=180);plt.close(fig)
    # Read the real coarse exported quadrilateral mesh, rather than reconstructing a sketch.
    folder=root/f"seeb_level_{runs[0]['level']:g}"
    with (folder/'mesh.su2').open() as f:
        assert f.readline().strip()=='NDIME= 2'
        n=int(f.readline().split('=')[1]);cells=np.array([[int(v) for v in f.readline().split()[1:5]] for _ in range(n)])
        n=int(f.readline().split('=')[1]);xy=np.array([[float(v) for v in f.readline().split()[:2]] for _ in range(n)])
    polygons=xy[cells];centers=polygons.mean(axis=1)
    fig,axs=plt.subplots(3,1,figsize=(10,10))
    windows=[(-.5,5.5,0,2),(-.001,.018,0,.015),(1.35,1.65,1.17,1.23)]
    names=['Actual coarse Gmsh mesh: body wall, exposed axis and outer farfield',
           'Nose inset: as-built finite cap, two cells across the cap',
           'Observation-line inset: H/L = 21.2/17.667']
    with np.load(folder/'cad_meridian.npz') as a:gx=a['x'];gr=a['r']
    for ax,window,title in zip(axs,windows,names):
        xmin,xmax,ymin,ymax=window
        # Include any cell intersecting the view, without simplifying inset topology.
        mask=(polygons[:,:,0].max(1)>=xmin)&(polygons[:,:,0].min(1)<=xmax)&(polygons[:,:,1].max(1)>=ymin)&(polygons[:,:,1].min(1)<=ymax)
        rings=polygons[mask];rings=np.concatenate([rings,rings[:,:1]],axis=1)
        ax.add_collection(LineCollection(rings,colors='#567181',linewidths=.18))
        ax.plot(gx,gr,color='black',lw=1.2,label='Body/sting wall')
        ax.axhline(21.2/17.667,color='#c23b22',ls='--',lw=1,label='Pressure extraction')
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='x/L',ylabel='r/L',title=title)
        ax.set_aspect('equal',adjustable='box')
    axs[0].legend(fontsize=8,loc='upper left')
    fig.tight_layout();fig.savefig(root/'seeb_mesh.png',dpi=180);plt.close(fig)
