"""Retain hash-checked formulation comparisons and English teaching figures."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flowmllab.pressure_velocity import centerlines,GHIA_X,GHIA_Y
from flowmllab.pressure_velocity_experiments import benchmark_errors,REFERENCE,ExperimentalConfig
from flowmllab.pressure_velocity import _residual,divergence
from flowmllab.streamfunction_experiments import sv
from flowmllab.cavity_visuals import RGB,flow_structure,field_normalization,field_scale_note
OUT=ROOT/'results/week01_2_formulations'
METHODS=['simple','piso','pimple','streamfunction-vorticity']
COLORS=['#1649b7','#d83723','#138044','#803cb2']

def load():
 rows=[]
 for p in sorted(OUT.glob('*-re*-n*.json')):
  d=json.loads(p.read_text())
  if d['status']=='failed':continue
  archive=p.with_suffix('.npz');assert hashlib.sha256(archive.read_bytes()).hexdigest()==d['field_sha256']
  with np.load(archive,allow_pickle=False) as f:r={**d,**{k:f[k] for k in f.files}}
  for k,v in benchmark_errors(r).items():assert np.isclose(v,r[k],rtol=1e-12,atol=1e-14)
  assert all(np.isfinite(a).all() for a in r.values() if isinstance(a,np.ndarray))
  assert abs(r['p'].mean())<1e-10
  if r['config']['method']=='streamfunction-vorticity':
   h=1/r['config']['cells'];dt=r['config']['dt']
   updated,_=sv.advance_vorticity(r['omega'].copy(),r['u'],r['v'],r['config']['reynolds'],dt,h,h)
   measured=float(np.max(np.abs((updated-r['omega'])[1:-1,1:-1]))/dt)
   assert np.isclose(measured,r['steady_vorticity_linf'],rtol=1e-10,atol=1e-12)
   assert r['u'].shape==(r['config']['cells']+1,)*2
   assert np.all(r['psi'][[0,-1],:]==0) and np.all(r['psi'][:,[0,-1]]==0)
   if r['converged']:assert r['steady_vorticity_linf']<1e-6 and r['continuity_linf']<1e-9
  else:
   measured=_residual(r['u_faces'],r['v_faces'],r['p'],ExperimentalConfig(**r['config']))
   assert np.isclose(measured,r['momentum_residual_linf'],rtol=1e-10,atol=1e-12)
   assert np.max(np.abs(divergence(r['u_faces'],r['v_faces'],1/r['config']['cells'])))<1e-9
   if r['converged']:assert r['momentum_residual_linf']<1e-6 and r['continuity_linf']<1e-9
  rows.append(r)
 return sorted(rows,key=lambda r:(r["config"]["reynolds"],r["config"]["cells"],METHODS.index(r["config"]["method"])))

def main():
 import matplotlib.pyplot as plt
 from matplotlib import font_manager
 for name in ['times.ttf','timesbd.ttf','timesi.ttf']:
  p=Path('C:/Windows/Fonts')/name
  if p.exists():font_manager.fontManager.addfont(str(p))
 plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False})
 rows=load();base=[r for r in rows if r['config']['cells']==32]
 def save(fig,name):fig.savefig(OUT/(name+'.png'),dpi=170,facecolor='white');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
 for i,re in enumerate([400,1000]):
  for method,color in zip(METHODS,COLORS):
   for r in rows:
    c=r['config']
    if c['reynolds']!=re or c['method']!=method:continue
    y,u,x,v=centerlines(r);style={32:'-',64:'--',128:':'}.get(c['cells'],'-.')
    axes[i,0].plot(u,y,style,color=color,label=f"{method}, h=1/{c['cells']}" + (" [NOT STEADY]" if not r['converged'] else ""))
    axes[i,1].plot(x,v,style,color=color)
  axes[i,0].plot(REFERENCE[re]['u'],GHIA_Y,'ko',ms=3,label='Ghia reference')
  axes[i,1].plot(GHIA_X,REFERENCE[re]['v'],'ko',ms=3)
  axes[i,0].set(xlabel='u / U',ylabel='y / L',title=f'Re={re}: vertical centerline')
  axes[i,1].set(xlabel='x / L',ylabel='v / U',title=f'Re={re}: horizontal centerline')
  axes[i,0].legend(fontsize=8)
 save(fig,'centerlines')
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for i,re in enumerate([400,1000]):
  for r in base:
   c=r['config'];method=c['method']
   if c['reynolds']!=re:continue
   color=COLORS[METHODS.index(method)]
   if method=='streamfunction-vorticity':
    axes[i,1].semilogy(r['residual_steps'],r['history_rate'],color=color,label='FD relative vorticity-update rate')
   else:
    axes[i,0].semilogy(r['history'][:,0],r['history'][:,3],color=color,label=method.upper())
  axes[i,0].axhline(1e-6,color='k',ls=':');axes[i,1].axhline(1e-9,color='k',ls=':',label='update tolerance / dt')
  axes[i,0].set(xlabel='FV outer iterations',ylabel='Steady momentum defect, Linf',title=f'Re={re}, h=1/32: FV equations')
  axes[i,1].set(xlabel='FD explicit time steps',ylabel='||delta omega||2 / (dt ||omega||2)',title=f'Re={re}, h=1/32: FD update history')
  axes[i,0].legend();axes[i,1].legend(fontsize=9)
 fig.suptitle('Different equations and stopping norms: do not compare ordinate magnitudes')
 save(fig,'convergence')
 fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
 for ax,re in zip(axes,[400,1000]):
  rr=[next(r for r in base if r['config']['reynolds']==re and r['config']['method']==m) for m in METHODS]
  ax.bar(np.arange(4),[r['end_to_end_seconds'] for r in rr],color=COLORS)
  ax.set_xticks(range(4),['SIMPLE','PISO','PIMPLE','psi-omega'],rotation=15)
  ax.set(ylabel='Measured end-to-end seconds',title=f'Re={re}, h=1/32; single sequential run')
  for j,r in enumerate(rr):ax.text(j,r['end_to_end_seconds'],f"{r['end_to_end_seconds']:.1f}s" + (" LIMIT" if not r['converged'] else ""),ha='center',va='bottom',fontsize=9)
  ax.margins(y=.15)
 save(fig,'runtime')
 for key,title in [('speed','Speed / U'),('p','Zero-mean pressure / rho U^2'),('omega','Vorticity L / U')]:
  fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
  vals=[np.hypot(r['u'],r['v']) if key=='speed' else r[key] for r in base]
  hi=max(float(np.abs(z).max()) for z in vals);lo=0 if key=='speed' else -hi
  norm=field_normalization(key,vals)
  for i,re in enumerate([400,1000]):
   for j,method in enumerate(METHODS):
    r=next(r for r in base if r['config']['reynolds']==re and r['config']['method']==method)
    z=np.hypot(r['u'],r['v']) if key=='speed' else r[key]
    ax=axes[i,j];im=ax.pcolormesh(r['x'],r['y'],z,cmap=RGB,norm=norm,shading='nearest',rasterized=True)
    flow_structure(ax,r['x'],r['y'],r['u'],r['v'])
    ax.set(aspect='equal',xlim=(0,1),ylim=(0,1),xlabel='x / L',ylabel='y / L',title=f"Re={re}: {method}" + (" [NOT STEADY]" if not r['converged'] else ""))
  fig.colorbar(im,ax=list(axes.flat),label=title,shrink=.8)
  fig.suptitle(f'{title}; matched h=1/32; FD nodes and FV cell centers differ')
  if key!='speed':fig.supxlabel(field_scale_note(key),fontsize=10)
  save(fig,key)
 fine=[r for r in rows if r['config']['cells']>=64]
 if fine:
  fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
  for i,re in enumerate([400,1000]):
   for j,(method,n) in enumerate([('simple',64),('streamfunction-vorticity',64),('streamfunction-vorticity',128)]):
    r=next(r for r in fine if r['config']['reynolds']==re and r['config']['method']==method and r['config']['cells']==n)
    ax=axes[i,j];im=ax.pcolormesh(r['x'],r['y'],np.hypot(r['u'],r['v']),cmap=RGB,vmin=0,vmax=1,shading='nearest')
    flow_structure(ax,r['x'],r['y'],r['u'],r['v'])
    ax.set(aspect='equal',xlim=(0,1),ylim=(0,1),xlabel='x / L',ylabel='y / L',title=f'Re={re}, h=1/{n}: {method}')
  fig.colorbar(im,ax=list(axes.flat),label='Speed / U')
  fig.suptitle('Mesh refinement: SIMPLE h=1/64; independent FD h=1/64 and 1/128')
  save(fig,'refinement')
 text=['# Week 1.2: streamfunction-vorticity versus pressure-velocity methods','',
 'Executed 20 October 2026. Fresh sequential CPU runs from rest, one BLAS/OMP thread. No external CFD solver. The original Week-1 and Re=100 sources are unchanged.','',
 '## What is matched, and what differs','',
 'Unit square, unit discontinuous lid, stationary no-slip walls, 2D laminar incompressible flow; Re=400 and 1000. All four methods use h=1/32 for the timing comparison. FD uses 33x33 boundary-inclusive nodes; FV uses 32x32 pressure cells and staggered face velocities. Matching spacing does not make their discrete equations identical. Central convection is used in both.','',
 'FV settings: velocity relaxation 0.5, pressure relaxation 0.2, steady momentum Linf <1e-6, continuity <1e-9; PISO 2 corrections, PIMPLE 3 outer x 2 inner, dt=0.05. SIMPLE is steady. FD: forward Euler dt=0.001, central differences, DST Poisson solve, wall-vorticity boundary condition; relative vorticity update <1e-12 at three consecutive 500-step checks after 5000 steps, plus independently recomputed steady vorticity RHS Linf <1e-6. Interior centered-divergence is checked separately. FD pressure is reconstructed after the solve by least squares, not solved by pressure correction.','',
 'FD update histories divided by dt describe a relative vorticity-change rate; they are not FV momentum residuals. A steady RHS check prevents treating a small time-step update as sufficient convergence. The pressure reconstruction discrepancy is retained, and pressure has no independent Ghia benchmark. No transient-order validation is claimed.','',
 '## Accuracy, convergence and timing','',
 '| Re | Method | h | Status | Steps/iterations | End-to-end seconds | u relative L2 | v relative L2, all | v sensitivity* | Continuity Linf | Steady native defect |',
 '|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  c=r['config'];defect=r.get('steady_vorticity_linf',r.get('momentum_residual_linf'))
  sensitivity=f"{r['ghia_v_relative_l2_without_x09063']:.5f}" if 'ghia_v_relative_l2_without_x09063' in r else 'n/a'
  text.append(f"| {c['reynolds']} | {c['method']} | 1/{c['cells']} | {r['status']} | {r.get('steps',r.get('iterations'))} | {r['end_to_end_seconds']:.3f} | {r['ghia_u_relative_l2']:.5f} | {r['ghia_v_relative_l2']:.5f} | {sensitivity} | {r['continuity_linf']:.2e} | {defect:.2e} |")
 text += ['', '*At Re=400 the published Ghia v entry at x=0.9063 is retained in the all-point metric. The separately labeled sensitivity omits that one entry; no value is replaced. Re=1000 uses all points. Ghia, Ghia & Shin (1982), tables I/II, DOI 10.1016/0021-9991(82)90058-4. Relative L2 here is the Euclidean norm at tabulated sample locations, not an area-integrated field norm.','',
 'Nonconverged cases are retained and marked NOT STEADY/LIMIT in plots. Their errors describe the last iterate, not an accepted steady solution. Timing bars for a limit case are consumed budget, not time to convergence.', '',
 'Initial rejected attempts are retained under attempts/: the Re=400 relative-update stop did not pass the independent steady gate, and Re=1000 reached its 200000-step limit. Accepted reruns use update tolerance 1e-12 and a 400000-step budget; the physical equations and steady-RHS gate were not relaxed.', '',
 'The final column is a vorticity-transport RHS for FD and a momentum defect for FV. These are different quantities and must not be ranked numerically against one another. The discrete divergence checks also live on different grids.','',
 'Timing includes each solver call and FD pressure recovery, excluding archive writing, plotting and notebook setup. It is one sequential sample per configuration on this host; other system activity was not controlled. Different dt, stopping norms, unknown counts and implementation costs mean these are measured costs of these implementations/settings, not a universal algorithm ranking. Time-to-equal-accuracy needs resolution studies and repeated timing.','']
 text += ['## Measured interpretation','']
 for re in [400,1000]:
  group=[r for r in base if r['config']['reynolds']==re]
  sf=next(r for r in group if r['config']['method']=='streamfunction-vorticity')
  simple=next(r for r in group if r['config']['method']=='simple')
  text.append(f"- Re={re}, h=1/32: FD errors are {100*sf['ghia_u_relative_l2']:.2f}% (u) and {100*sf['ghia_v_relative_l2']:.2f}% (all-point v); SIMPLE errors are {100*simple['ghia_u_relative_l2']:.2f}% and {100*simple['ghia_v_relative_l2']:.2f}%. FD/SIMPLE elapsed-cost ratio is {sf['end_to_end_seconds']/simple['end_to_end_seconds']:.2f} for these settings; FD convergence status is {sf['status']}. This is not a time-to-steady ratio when FD did not converge.")
  fine=[r for r in rows if r['config']['reynolds']==re and r['config']['method']=='streamfunction-vorticity' and r['config']['cells']==64]
  if fine:
   q=fine[0];sensitivity=(f"; Re=400 v sensitivity {100*q['ghia_v_relative_l2_without_x09063']:.2f}%" if re==400 else '')
   text.append(f"- Re={re}, FD refinement to h=1/64: u error {100*q['ghia_u_relative_l2']:.2f}%, all-point v error {100*q['ghia_v_relative_l2']:.2f}%{sensitivity}. This finer result is not used in the h=1/32 timing bars.")
  finest=[r for r in rows if r['config']['reynolds']==re and r['config']['method']=='streamfunction-vorticity' and r['config']['cells']==128]
  if finest:
   q=finest[0];text.append(f"- Re={re}, FD h=1/128: u error {100*q['ghia_u_relative_l2']:.2f}%, all-point v error {100*q['ghia_v_relative_l2']:.2f}%; convergence={q['status']}. This is a separate FD resolution study, not a four-method 128-cell timing comparison.")
  text.append(f"- Re={re}, FD pressure-reconstruction gradient discrepancy is {sf['pressure_grad_rel_residual']:.3f}. This full-grid diagnostic includes strong lid/corner effects and is not a validated pressure-error estimate; do not interpret pressure-color agreement as pressure accuracy.")
 text += ['', '### Solver work (different linear systems)', '', '| Re | Method | h | FV pressure corrections | FD streamfunction Poisson solves | FD pressure LSQR iterations |', '|---:|---|---|---:|---:|---:|']
 for r in rows:
  c=r['config'];text.append(f"| {c['reynolds']} | {c['method']} | 1/{c['cells']} | {r.get('pressure_solves','-')} | {r.get('poisson_solves','-')} | {r.get('pressure_lsqr_iterations','-')} |")
 text += ['', 'FV pressure corrections, FD DST solves and pressure-reconstruction LSQR iterations are distinct work units, not interchangeable operation counts.', '']
 text += ['', '### Primary vortex (sampled streamfunction minimum)', '', '| Re | Method | h | x/L | y/L | psi_min |', '|---:|---|---|---:|---:|---:|']
 for r in rows:
  c=r['config']
  if 'vortex_x' in r:text.append(f"| {c['reynolds']} | {c['method']} | 1/{c['cells']} | {r['vortex_x']:.4f} | {r['vortex_y']:.4f} | {r['psi_min']:.6f} |")
 text += ['', 'Vortex positions are sampled grid locations, not sub-grid fitted centers. The FV 32-cell calls used SIMPLE/PISO/PIMPLE order at Re=400, then Re=1000. Accepted FD 32-cell reruns followed the retained failed attempts; the 64-cell SIMPLE/FD calls followed afterward. Calls did not overlap. Factorization caches and unrelated host load can affect timing. The FD-only 128-cell calls were last. No repeated-run uncertainty interval is available; finer grids can sometimes require fewer transient steps and hence do not guarantee monotonically increasing wall time.', '']
 for name in ['centerlines','convergence','runtime','speed','p','omega','refinement']:text += [f'![{name}]({name}.png)','']
 text += ['## Student investigation','',
 'Start at 32 cells; finer grids and higher Re can require longer runs and more memory. Compare benchmark errors before comparing speed. Preserve all failures. At least two resolutions are needed to investigate mesh sensitivity, and two do not establish asymptotic order. A converged iteration is not necessarily an accurate flow. Report the primary vortex position, both velocity errors and the Re=400 reference sensitivity alongside any speed comparison. Small corner vortices may remain under-resolved.','',
 'Reproduce with `python qa/run_week01_2_formulations.py` then `python qa/run_week01_2_formulations.py --cells 64 --methods simple streamfunction-vorticity`, `python qa/run_week01_2_formulations.py --cells 128 --methods streamfunction-vorticity`, and `python qa/report_week01_2_formulations.py`. Fresh notebook runs are opt-in. Raw JSON/NPZ archives retain histories, settings, source hashes, software versions, and pressure-recovery diagnostics.']
 (OUT/'README.md').write_text('\n'.join(text)+'\n')
 agreement=[]
 for re in [400,1000]:
  group=[r for r in base if r['config']['reynolds']==re]
  ref=next(r for r in group if r['config']['method']=='simple')
  for r in group:
   if r['config']['method'] in ['piso','pimple']:
    diff={k:float(np.max(np.abs(r[k]-ref[k]))) for k in ['u','v','p']}
    assert max(diff.values())<2e-5
    agreement.append({'Re':re,'method':r['config']['method'],'linf_vs_simple':diff})
 validation={'date':'2026-10-20','retained_runs':len(rows),'same_grid_fv_agreement':agreement,'all_converged':all(r['converged'] for r in rows),'field_hashes_verified':True,'benchmark_metrics_recomputed':True,'native_steady_gates_checked':True,'source_unchanged':{p:all(r['source_sha256'][p]==hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for r in rows) for p in ['common/w4utils.py','flowmllab/pressure_velocity.py']}}
 (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
