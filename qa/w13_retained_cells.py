"""Retained Week 13 code cells reused verbatim by build_week13_notebook.py (deep-case checks, continuation loss, four-case audit)."""

DEEP_CHECKS = r"""from scipy.integrate import cumulative_trapezoid
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
print('Checkpoint:', Path(deep_audit['checkpoint']).name, '(cluster path recorded in data/week13_deep_cavity/audit.json)')
print({'FD_divergence_max_full':float(abs(div).max()),
       'FD_divergence_max_outside_top_corner_boxes':float(abs(div[away]).max()),
       'FD_divergence_RMS_full':float(np.sqrt(np.mean(div**2)))})
# Remove the physical-area-weighted pressure gauge, not an arbitrary point value.
integral=lambda a:np.trapezoid(np.trapezoid(a,x,axis=1),y)
pressure=p-integral(p)/2.2
speed=np.hypot(u,v)
print({'speed_max':float(speed.max()),'pressure_range':[float(pressure.min()),float(pressure.max())]})
"""

LOSS_CONTINUATION = r"""d=np.loadtxt(ROOT/'data/week13_deep_cavity/loss_continuation.dat')
assert d.shape==(74,3) and np.isfinite(d).all()
assert np.all(np.diff(d[:,0])>0) and np.all(d[:,1:]>0)
fig,ax=plt.subplots(figsize=(9,5.5))
for y,label,color in [(d[:,1],'Momentum x','#1764ab'),(d[:,2],'Momentum y','#d95f02'),(d[:,1:].sum(1),'Total','#26384a')]:
    ax.semilogy(d[:,0],y,label=label,color=color,lw=2)
ax.set(xlabel='Logged step in the resumed process',ylabel='Training mean-square residual',
       title='Deep-cavity PINN: retained continuation history')
ax.grid(alpha=.2,which='both')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
fig.text(.5,.02,'History ends at checkpoint 65711; supplied field is checkpoint 55118.\nNot a from-scratch history, CFD error, or independent test loss.',ha='center',fontsize=10)
fig.subplots_adjust(bottom=.3)
out=ROOT/'results/week13_deep_cavity/loss_continuation.png'
fig.savefig(out,dpi=180,bbox_inches='tight')
print(out.relative_to(ROOT).as_posix())
print('rows',len(d),'last total',d[-1,1:].sum())

plt.show()
"""

LOAD_MATRIX = r"""records=[]
for Re,D in CASES:
    folder=RESULT/f're{Re}-d{D}'
    audit=json.loads((folder/'audit.json').read_text())
    history=[json.loads(line) for line in (folder/'optimizer-history.jsonl').read_text().splitlines() if line]
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    records.append(dict(Re=Re,D=D,folder=folder,audit=audit,history=history,hashes=files))
print('Loaded four immutable evidence packages.')
"""

LOSS_CURVES = r"""fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for ax,r in zip(axs.ravel(),records):
    rows=list({q['global_step']:q for q in r['history']}.values()); rows.sort(key=lambda q:q['global_step'])
    steps=[q['global_step'] for q in rows]
    ax.semilogy(steps,[np.sqrt(q['train_rx_mse']+q['train_ry_mse']) for q in rows],label='masked train')
    ax.semilogy(steps,[q['heldout_momentum_rms'] for q in rows],'k--',label='held-out full')
    ax.semilogy(steps,[np.sqrt(2)*q['top_corner_momentum_rms'] for q in rows],':',color='#009E73',label='top corners')
    ax.axvline(1000,color='#D55E00',lw=1); ax.set(title=f"Re={r['Re']}, D={r['D']}",xlabel='optimizer step',ylabel='residual RMS'); ax.grid(alpha=.2)
axs[0,0].legend(frameon=False); plt.show()
"""

EVIDENCE_TABLE = r"""rows=[]
for r in records:
    a=r['audit']; z=a['independent_residual']; c=a.get('cfd_comparison')
    rows.append((r['Re'],r['D'],np.hypot(z['momentum_x_rms'],z['momentum_y_rms']),
                 np.hypot(z['top_corner_momentum_x_rms'],z['top_corner_momentum_y_rms']),
                 max(max(w.values()) for w in a['wall_error'].values()),None if c is None else c['all_pass'],a['claim_status']))
print(f"{'Re':>5} {'D':>3} {'R_full':>11} {'R_corner':>11} {'wall max':>11} {'CFD gates':>10}  status")
for row in rows: print(f"{row[0]:5d} {row[1]:3d} {row[2]:11.3e} {row[3]:11.3e} {row[4]:11.3e} {str(row[5]):>10}  {row[6]}")
"""

CASE_A = r"""from IPython.display import Image, display
display(Image(filename=str(ROOT/'results/week13_deep_cavity/cfd_pinn_fields.png')))
comparison=json.loads((ROOT/'data/week13_deep_cavity/comparison.json').read_text())
print({'velocity_relative_L2_percent':100*comparison['velocity_relative_L2'],
       'CFD_last_interval_relative_L2':comparison['nektar_last_interval_relative_L2'],
       'lid_max_difference':comparison['lid_max_difference']})
"""

CASE_B = r"""display(Image(filename=str(ROOT/'results/week04_2_pinn_cavity/qualified_validation.png')))
"""
