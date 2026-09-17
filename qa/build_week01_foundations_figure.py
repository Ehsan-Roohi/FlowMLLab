"""Rebuild the Week-1 lecture figure using the existing notebook solver.

Only solver definitions and GHIA arrays are extracted; optional pressure runs,
Colab setup, and other notebook side effects are deliberately not executed.
Run from any working directory with NumPy and Matplotlib installed.
"""
import ast,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
(root/'lectures/source/week01_assets').mkdir(parents=True, exist_ok=True)
n=json.loads((root/'notebooks/week01/03_cavity_ghia.ipynb').read_text())
ns={'np':np}
keep={'build_grid','laplacian','apply_vorticity_boundary_conditions','solve_streamfunction_poisson','compute_velocity','advance_vorticity','run_cavity','centerline_profiles','ghia_errors'}
for c in n['cells']:
 if c['cell_type']!='code':continue
 tree=ast.parse(''.join(c['source']))
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in keep: exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('GHIA_') for t in node.targets):exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
r=ns['run_cavity'](report_every=2000)
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
a=axes[0];a.streamplot(r['x'],r['y'],r['u'],r['v'],density=1.1,color='#176578',linewidth=.7);a.set(xlabel='x/L',ylabel='y/L',title='Computed streamlines',aspect='equal')
x,y,u,v=ns['centerline_profiles'](r)
axes[1].plot(u,y,color='#176578',label='65 x 65 CFD');axes[1].scatter(ns['GHIA_U'],ns['GHIA_Y'],s=12,color='#b44f30',label='Ghia et al.');axes[1].set(xlabel='u/U at x/L = 0.5',ylabel='y/L');axes[1].legend(fontsize=8)
axes[2].plot(x,v,color='#176578');axes[2].scatter(ns['GHIA_X'],ns['GHIA_V'],s=12,color='#b44f30');axes[2].set(xlabel='x/L',ylabel='v/U at y/L = 0.5')
for a in axes:a.grid(alpha=.18)
fig.savefig(root/'lectures/source/week01_assets/cavity_validation.pdf')
e=ns['ghia_errors'](r)
record={'Re':100,'N':65,'dt':.001,'steps':10000,'poisson_iters':60,'relative_L2_u':e[0],'relative_L2_v':e[1],'last_reported_relative_vorticity_change':r['residual_values'][-1],'scope':'Fresh execution of existing Week-1 notebook solver functions; single-grid teaching comparison, not grid-independent certification.'}
(root/'lectures/source/week01_assets/validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(record)
