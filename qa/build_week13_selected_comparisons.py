"""Build the two selected Week 13 CFD/PINN teaching comparisons."""
from pathlib import Path
import hashlib, json, xml.etree.ElementTree as ET
import nbformat
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.interpolate import griddata

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/week13_deep_cavity'
OUT=ROOT/'results/week13_deep_cavity'

def read_vtu(path):
    tree=ET.parse(path)
    arrays={a.attrib.get('Name'):np.fromstring(a.text or '',sep=' ')
            for a in tree.iter('DataArray') if a.attrib.get('Name') in {'u','v','p','Points'}}
    points=arrays['Points'].reshape(-1,3)[:,:2]
    # Nektar element boundaries repeat physical points. Average duplicates.
    rounded=np.round(points,12)
    unique,inverse=np.unique(rounded,axis=0,return_inverse=True)
    fields={}
    for name in ('u','v','p'):
        total=np.bincount(inverse,weights=arrays[name])
        count=np.bincount(inverse)
        fields[name]=total/count
    return unique,fields

def main():
    with np.load(DATA/'field.npz',allow_pickle=False) as z:
        x,y=z['x'],z['y']; pinn={k:z[k].astype(float) for k in ('u','v','p')}
    points,cfd_scattered=read_vtu(DATA/'nektar_cavity_t120.vtu')
    X,Y=np.meshgrid(x,y); query=np.column_stack((X.ravel(),Y.ravel()))
    cfd={k:griddata(points,v,query,method='linear').reshape(Y.shape)
         for k,v in cfd_scattered.items()}
    if any(np.isnan(a).any() for a in cfd.values()):
        raise RuntimeError('Nektar interpolation left points outside the CFD convex hull')
    for fields in (cfd,pinn):
        fields['p']=fields['p']-np.trapezoid(np.trapezoid(fields['p'],x,axis=1),y)/2.2
        fields['speed']=np.hypot(fields['u'],fields['v'])
    np.savez_compressed(DATA/'nektar_cfd_on_pinn_grid.npz',x=x,y=y,**cfd)
    speed_max=max(cfd['speed'].max(),pinn['speed'].max())
    pmax=max(np.abs(cfd['p']).max(),np.abs(pinn['p']).max())
    fig,axs=plt.subplots(2,2,figsize=(9,11),layout='constrained',sharex=True,sharey=True)
    for row,(label,f) in enumerate((('Nektar++ CFD, t=120',cfd),('PINN, checkpoint 55118',pinn))):
        im=axs[row,0].pcolormesh(x,y,f['speed'],shading='auto',cmap='viridis',vmin=0,vmax=speed_max)
        axs[row,0].streamplot(x,y,f['u'],f['v'],color='white',density=.75,linewidth=.45,arrowsize=.55)
        fig.colorbar(im,ax=axs[row,0],shrink=.72,label='speed / lid speed')
        im=axs[row,1].pcolormesh(x,y,f['p'],shading='auto',cmap='RdBu_r',
                                 norm=SymLogNorm(.02,vmin=-pmax,vmax=pmax))
        fig.colorbar(im,ax=axs[row,1],shrink=.72,label='mean-zero pressure (symmetric log)')
        axs[row,0].set_ylabel(label+'\n\ny/W')
    axs[0,0].set_title('Speed and streamlines'); axs[0,1].set_title('Pressure, common gauge/full symmetric-log scale')
    for ax in axs.ravel(): ax.set(xlabel='x/W',aspect='equal',xlim=(0,1),ylim=(0,2.2))
    metrics=json.loads((DATA/'comparison.json').read_text())
    fig.suptitle('Re=1000, D/W=2.2: matched-grid CFD and PINN comparison\n'
                 f"velocity relative L2 = {100*metrics['velocity_relative_L2']:.2f}%")
    fig.savefig(OUT/'cfd_pinn_fields.png',dpi=190,bbox_inches='tight'); plt.close(fig)

    manifest=json.loads((DATA/'manifest.json').read_text())
    for name in ('nektar_cavity_t120.vtu','nektar_cfd_on_pinn_grid.npz','comparison.json'):
        manifest['files'][name]=hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    manifest['matched_comparison']={
        'cfd':'Nektar++ cavity.vtu at t=120, mapped to the 301x661 PINN grid',
        'pinn':'restart-55118.ckpt field export',
        'velocity_relative_L2':metrics['velocity_relative_L2'],
        'qualification':'Boundary/lid mismatch and lack of mesh/steady convergence remain explicit.'}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    path=ROOT/'notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb'
    n=nbformat.read(path,4)
    # Keep the primary cell as a numerical audit. The later selected comparison
    # is the only full-field rendering for D/W=2.2.
    primary=next(c for c in n.cells if c.cell_type=='code' and "deep_dir=ROOT/'data/week13_deep_cavity'" in c.source)
    cut=primary.source.index("fig,axs=plt.subplots(1,3")
    primary.source=primary.source[:cut]+"print({'speed_max':float(speed.max()),'pressure_range':[float(pressure.min()),float(pressure.max())]})\n"
    title='## Selected case A: D/W=2.2, CFD beside PINN'
    section=[nbformat.v4.new_markdown_cell(title+'''\n\nThe two rows below are not two PINN renderings. The first is the retained
Nektar++ CFD field at t=120; the second is PINN checkpoint 55118. Both are
mapped to one grid, use the same speed and pressure scales, and remove the same
area-weighted pressure gauge. The reported 3.59% velocity relative L2 remains
a near-matched comparison: the lid profiles differ, and the CFD archive does
not establish mesh or long-time convergence.'''),
        nbformat.v4.new_code_cell("""from IPython.display import Image, display
display(Image(filename=str(ROOT/'results/week13_deep_cavity/cfd_pinn_fields.png')))
comparison=json.loads((ROOT/'data/week13_deep_cavity/comparison.json').read_text())
print({'velocity_relative_L2_percent':100*comparison['velocity_relative_L2'],
       'CFD_last_interval_relative_L2':comparison['nektar_last_interval_relative_L2'],
       'lid_max_difference':comparison['lid_max_difference']})"""),
        nbformat.v4.new_markdown_cell('''## Selected case B: D/W=1, qualified near-matched CFD comparison\n\nThe square-cavity figure retains the full PINN velocity fields, the pointwise
PINN-minus-CFD velocity error and two CFD/PINN centreline comparisons. It is a
near-matched test because the PINN uses a smooth lid and the CFD reference uses
a classical lid. The frozen errors are 2.18% for u centreline, 4.42% for v
centreline and 3.10% for the interior velocity field.'''),
        nbformat.v4.new_code_cell("display(Image(filename=str(ROOT/'results/week04_2_pinn_cavity/qualified_validation.png')))" )]
    # Replace the old four-image gallery; keep its numerical audit table.
    start=next(i for i,c in enumerate(n.cells) if c.cell_type=='markdown' and
               (c.source.startswith('## 4. Geometry-faithful fields') or c.source.startswith(title)))
    end=next(i for i,c in enumerate(n.cells[start+1:],start+1) if c.cell_type=='markdown' and c.source.startswith('## 5.'))
    n.cells[start:end]=section
    nbformat.write(n,path)
    print(OUT/'cfd_pinn_fields.png')

if __name__=='__main__': main()
