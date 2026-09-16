"""Import the supplied checkpoint field as W13's primary retained example."""
from pathlib import Path
import hashlib,json
import nbformat as nb
from classroom_cells import md,code
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/week13_deep_cavity'
manifest={'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'field.npz',DATA/'audit.json']},
 'checkpoint':'restart-55118.ckpt','case':{'Re':1000,'depth_over_width':2.2},
 'source':'User-supplied Unity field export, received 2026-09-16',
 'qualification':'Retained field, not independently certified final convergence. No model weights or matching CFD fields supplied.'}
(DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
cells=[md(r'''
## Main case: a deep cavity, Re=1000 and D/W=2.2

This retained author-model field is evaluated from **restart-55118.ckpt**, on
a 301-by-661 physical grid. Follow the data checks and regenerate the figures
below before inspecting the shorter training exercise. The two examples use
different networks and parameter settings; the short CPU model does not
reproduce this checkpoint.

The supplied export includes u, v, p and derived streamfunction/vorticity, but
not model weights or matching CFD fields. A separately recovered continuation
history ends at checkpoint 65711, not this field checkpoint. The export therefore
supports field inspection, not a new claim of final optimizer convergence or
independent CFD validation. Streamfunction was integrated from u; extrema are
candidate vortex centres. Re-run residuals from the actual model before
interpreting finite-difference divergence as its autodifferentiated residual.
'''),code('''
from scipy.integrate import cumulative_trapezoid
from matplotlib.colors import SymLogNorm
deep_dir=ROOT/'data/week13_deep_cavity'
deep_manifest=json.loads((deep_dir/'manifest.json').read_text())
for filename,digest in deep_manifest['files'].items():
    assert hashlib.sha256((deep_dir/filename).read_bytes()).hexdigest()==digest
deep_audit=json.loads((deep_dir/'audit.json').read_text())
assert deep_audit['case']=={'Re':1000.,'depth_over_width':2.2,'tri':0.}
with np.load(deep_dir/'field.npz',allow_pickle=False) as archive:
    deep={key:archive[key].copy() for key in archive.files}
x,y=deep['x'],deep['y']; u,v,p=deep['u'],deep['v'],deep['p']
assert u.shape==v.shape==p.shape==(661,301)
assert all(np.isfinite(a).all() for a in deep.values())
assert np.all(np.diff(x)>0) and np.all(np.diff(y)>0)
np.testing.assert_allclose([x[0],x[-1],y[0],y[-1]],[0,1,0,2.2])
psi=cumulative_trapezoid(u,y,axis=0,initial=0.)
omega=np.gradient(v,x,axis=1,edge_order=2)-np.gradient(u,y,axis=0,edge_order=2)
div=np.gradient(u,x,axis=1,edge_order=2)+np.gradient(v,y,axis=0,edge_order=2)
np.testing.assert_allclose(psi,deep['psi'],atol=1e-12)
np.testing.assert_allclose(omega,deep['omega'],atol=1e-10)
np.testing.assert_allclose(abs(div).max(),deep_audit['max_abs_divergence_fd'],rtol=1e-10)
for edge in [np.hypot(u[0],v[0]),np.hypot(u[:,0],v[:,0]),np.hypot(u[:,-1],v[:,-1])]:
    assert edge.max()<1e-10
XX,YY=np.meshgrid(x,y)
away=~(((XX<.1)|(XX>.9))&(YY>2.1))
print('Checkpoint:',deep_audit['checkpoint'])
print({'FD_divergence_max_full':float(abs(div).max()),
       'FD_divergence_max_outside_top_corner_boxes':float(abs(div[away]).max()),
       'FD_divergence_RMS_full':float(np.sqrt(np.mean(div**2)))})
# Remove the physical-area-weighted pressure gauge, not an arbitrary point value.
integral=lambda a:np.trapezoid(np.trapezoid(a,x,axis=1),y)
pressure=p-integral(p)/2.2
speed=np.hypot(u,v)
fig,axs=plt.subplots(1,3,figsize=(12,8),layout='constrained')
im=axs[0].pcolormesh(x,y,speed,shading='auto',cmap='viridis',vmin=0,vmax=speed.max())
axs[0].streamplot(x,y,u,v,color='white',density=.8,linewidth=.5,arrowsize=.65)
fig.colorbar(im,ax=axs[0],shrink=.65,label='Speed / lid speed')
axs[0].set_title('Speed and streamlines')
for ax,field,title,threshold in [(axs[1],pressure,'Mean-zero pressure',.01),(axs[2],omega,'Vorticity',1.)]:
    lim=float(abs(field).max())
    im=ax.pcolormesh(x,y,field,shading='auto',cmap='RdBu_r',norm=SymLogNorm(threshold,vmin=-lim,vmax=lim))
    fig.colorbar(im,ax=ax,shrink=.65,label=title+' (symmetric-log scale)')
    ax.set_title(title)
for ax in axs: ax.set(xlabel='x/W',ylabel='y/W',aspect='equal',xlim=(0,1),ylim=(0,2.2))
fig.suptitle('Re=1000 | D/W=2.2 | retained checkpoint 55118; no range clipping')
out=ROOT/'results/week13_deep_cavity'; out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'fields.png',dpi=180); plt.show()
fig,axs=plt.subplots(1,2,figsize=(9,7),layout='constrained')
levels=np.unique(np.r_[-np.logspace(-7,-1,25),np.logspace(-7,-2,21)])
for ax in axs:
    ax.contour(x,y,psi,levels=levels,colors='black',linewidths=.6)
    ax.set(xlabel='x/W',ylabel='y/W',aspect='equal',xlim=(0,1))
axs[0].set(ylim=(0,2.2),title='Integrated streamfunction')
axs[1].set(ylim=(0,.5),title='Lower-cavity detail; same contour levels')
fig.savefig(out/'streamfunction.png',dpi=180); plt.show()
'''),md('''
### What these checks establish

The supplied field is finite, has the declared geometry and zero stationary-wall
velocity, and its derived fields reproduce the export. They do not certify
convergence. The full-grid FD divergence maximum is about 4.33 and must remain
visible alongside regional diagnostics. A hard streamfunction formulation can
be divergence-free under autodifferentiation while an exported grid has large
finite-difference error near sharp wall variations. Neither explanation alone
proves that this particular checkpoint is accurate; matching CFD and model
residuals are the next checks. Full color ranges are retained, with explicitly
labelled symmetric-log pressure/vorticity scales and no percentile clipping.
''')]
path=ROOT/'notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb'
n=nb.read(path,4)
if not n.metadata.get('deep_case_55118'):
    n.cells[0].source='# Week 13 — Learn PINNs through a deep-cavity case\n\nStart with the retained Re1000, D/W=2.2 field, then build and train a small CPU PINN. The older D=1 and D=2 cases remain as secondary research audits.'
    n.cells[0].source+='\n\n[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb)'
    n.cells[0].source+='\n\n<!-- MIE690A article-aligned validation v4 -->'
    n.cells[3:3]=cells
    n.metadata['deep_case_55118']=True
    for c in n.cells:
        if c.cell_type=='code': c.outputs=[]; c.execution_count=None
    nb.write(n,path)

if not n.metadata.get('deep_loss_65711'):
    plot_source=(ROOT/'qa/plot_week13_deep_loss.py').read_text()
    plot_source=plot_source[plot_source.index("d=np.loadtxt"):]
    plot_source += '\nplt.show()\n'
    position=next(i for i,c in enumerate(n.cells) if '### What these checks establish' in c.source)+1
    n.cells[position:position]=[md('''
## Observed continuation loss: a separate provenance record

The Unity `training/data/loss.dat` was recovered with `resume.json` identifying
`restart-65711.ckpt`. These are 74 distinct logged steps after removing identical
stage-boundary duplicates; the resumed-process clock is **not** total training
from scratch. Plot the two momentum training MSEs and their sum. The original
test columns repeated training values and are not treated as independent tests.

The supplied field above is checkpoint **55118**. Do not use this later history
as the field's matched residual audit. Small training loss alone establishes
neither CFD agreement nor whole-domain convergence.
'''),code(plot_source)]
    n.metadata['deep_loss_65711']=True
    nb.write(n,path)
