"""Generate additive PINN audit lessons; never launch training or change earlier labs."""
from pathlib import Path
import argparse
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    'ghia': ROOT/'notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb',
    'd5': ROOT/'notebooks/week13/W13_D5_Controlled_Optimizer_Study.ipynb',
}
BOOT = r'''
# FLOWMLLAB_COLAB_BOOTSTRAP_V1
# Run All reads retained evidence on CPU; it never starts training or HPC jobs.
# Outside a checkout, obtain the published repository once. Preserve existing work.
from pathlib import Path
import hashlib, json, subprocess, sys
from datetime import datetime
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents, Path('/content/FlowMLLab')]
             if (p/'flowmllab/ghia.py').is_file() and (p/'results').is_dir()), None)
if ROOT is None:
    candidate = Path('/content/FlowMLLab') if Path('/content').is_dir() else Path.cwd()/'FlowMLLab'
    if not candidate.exists():
        subprocess.run(['git','clone','--depth','1',
                        'https://github.com/Ehsan-Roohi/FlowMLLab.git',str(candidate)],check=True)
    if not (candidate/'flowmllab/ghia.py').is_file() or not (candidate/'results').is_dir():
        raise RuntimeError('Use an updated FlowMLLab checkout after saving your student work; '
                           'this cell never overwrites an existing directory or edited notebook.')
    ROOT = candidate
ROOT = ROOT.resolve()
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
print('Evidence checkout:', ROOT)

def verified_file(entry):
    path = (ROOT/entry['path']).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Evidence must stay inside the checkout.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('SHA256 mismatch: '+entry['path'])
    return path

def stamp(value):
    result = datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:
        raise ValueError('Use timezone-aware evidence timestamps.')
    return result

def load_campaign(folder, kind):
    path = ROOT/folder/'evaluation_manifest.json'
    if not path.exists():
        print('Fresh campaign results are not included in this checkout yet.')
        print('Study design and exercises are available; no numerical outcome is fabricated.')
        return {'records': [], 'status':'not_available'}
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assert manifest['schema_version']==1 and manifest['campaign_kind']==kind
    seen=set()
    for row in manifest['records']:
        if manifest.get('path_base')=='manifest_directory':
            for key in ['prediction','reference','comparison_prediction','history','training_manifest','training_contract']:
                if key in row:
                    evidence=(path.parent/row[key]['path']).resolve()
                    if not evidence.is_relative_to(ROOT.resolve()):
                        raise ValueError('Published evidence must remain inside the checkout.')
                    row[key]=dict(row[key],path=evidence.relative_to(ROOT.resolve()).as_posix())
        if 'history' in row: verified_file(row['history'])
        if row['run_id'] in seen:
            raise ValueError('Duplicate run_id; a restart is not another seed.')
        seen.add(row['run_id'])
        for provenance_key in ['training_manifest','training_contract']:
            if provenance_key in row: verified_file(row[provenance_key])
        if row['outcome'] not in ('scored','controlled_pilot_scored'):
            continue
        if not (row['prediction']['sha256']==row['freeze']['prediction_sha256']
                ==row['score']['prediction_sha256']):
            raise ValueError('Scored prediction differs from the frozen prediction.')
        if stamp(row['freeze']['at_utc'])>stamp(row['score']['reference_accessed_at_utc']):
            raise ValueError('Reference access predates recorded prediction freeze.')
        verified_file(row['prediction']); verified_file(row['reference'])
        if 'comparison_prediction' in row:
            if row['comparison_prediction']['source_prediction_sha256']!=row['prediction']['sha256']:
                raise ValueError('Derived comparison does not bind the original frozen prediction.')
            verified_file(row['comparison_prediction'])
        if 'history' in row: verified_file(row['history'])
        if 'ghia_profiles' in row: verified_file(row['ghia_profiles'])
    if manifest.get('path_base')=='manifest_directory':
        for item in manifest.get('figures',[]):
            resolved=(path.parent/item['path']).resolve()
            if not resolved.is_relative_to(ROOT.resolve()):raise ValueError('Figure must stay inside checkout.')
            item['path']=resolved.relative_to(ROOT.resolve()).as_posix()
            verified_file(item)
    print('Campaign status:',manifest['status'])
    return manifest

def show_history(rows):
    rows=[r for r in rows if 'history' in r]
    if not rows:
        print('No verified convergence history for this selection.'); return
    metric_names={r['history']['metric_name'] for r in rows}
    if len(metric_names)!=1: raise ValueError('Do not overlay unlike convergence metrics.')
    for r in rows:
        print(r['run_id'],r['history']['metric_label'],'; independent audit =',r['history']['independent_audit'])
        print('History location:',r['history'].get('loss_location','recorded in protocol'))
        print('Time scope:',r['history'].get('time_scope','recorded in protocol'))
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for r in rows:
        h=pd.read_csv(verified_file(r['history']))
        for ax,column in zip(axes,['objective_evaluations','elapsed_seconds']):
            label=f"{r['method']}, seed {r['seed']}"
            if 'collar' in r: label+=f", epsilon={r['collar']:g}"
            ax.plot(h[column],h['metric_value'],label=label)
            ax.set(xlabel=column.replace('_',' '),ylabel=r['history']['metric_label'])
            ax.set_yscale('symlog',linthresh=1e-12);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',
               bbox_to_anchor=(.5,-.16),ncol=2,frameon=False)
    fig.tight_layout();plt.show()

def show_native_profiles(row, heights):
    fields=[]
    for key in [('comparison_prediction' if 'comparison_prediction' in row else 'prediction'),'reference']:
        with np.load(verified_file(row[key]),allow_pickle=False) as f:
            fields.append({k:f[k].copy() for k in f.files})
    pred,ref=fields
    # D5 native exports name physical floor height h; no byte/file rewrite is needed.
    for field in fields:
        if 'y' not in field and 'h' in field: field['y']=field['h']
    for key in ['x','y']:
        if ref[key].ndim!=1 or not np.array_equal(pred[key],ref[key]):
            raise ValueError('Matched physical native coordinates are required.')
        if not np.all(np.diff(ref[key])>0): raise ValueError('Coordinates must increase.')
    variables=[v for v in ['u','v','p','omega'] if v in pred and v in ref]
    shape=(len(ref['y']),len(ref['x']))
    for field in fields:
        for v in variables:
            if field[v].shape!=shape or not np.isfinite(field[v]).all():
                raise ValueError('Invalid field: '+v)
    if 'p' in variables and 'comparison_prediction' not in row:
        def w(c):
            d=np.diff(c);return np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2]
        weights=np.outer(w(ref['y']),w(ref['x']))[1:-1,1:-1]
        pred['p']-=np.sum(weights*(pred['p']-ref['p'])[1:-1,1:-1])/weights.sum()
    for height in heights:
        j=int(np.argmin(abs(ref['y']-height)))
        print('Native physical row y/W =',ref['y'][j],'; requested =',height)
        fig,axes=plt.subplots(1,len(variables),figsize=(3.3*len(variables),3.1),squeeze=False)
        for ax,v in zip(axes[0],variables):
            ax.plot(ref['x'],ref[v][j],color='black',label='CFD reference')
            ax.plot(pred['x'],pred[v][j],color='tab:orange',ls='--',label='Frozen PINN')
            ax.set(xlabel='x/W',ylabel=v+' (nondimensional)');ax.grid(alpha=.2)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',
                   bbox_to_anchor=(.5,-.13),ncol=2,frameon=False)
        fig.tight_layout();plt.show()
'''

COMMON_PROTOCOL = r'''## Information, work and recovery

| Record | Required control |
| --- | --- |
| Initial parameters, samples, architecture, physics | Identical within each paired seed |
| Endpoint/audit rule | Ghia uses the last fixed-budget endpoint; final audits never select it |
| CFD/Ghia references | Read for scoring only after prediction freezing |
| Quasi-Newton objective | Freeze collocation and weights during every line search |
| Work | Count closures, line-search trials, candidate-pool residual evaluations and elapsed time |
| Seeds | Preserve every predeclared outcome and failure; a restart is not another seed |

Adam supplies moment-scaled first-order steps; L-BFGS uses secant information to
approximate inverse curvature. Staging can help ill-conditioning but cannot repair
wrong boundary conditions or an inadequate representation. Explain a measured gain
using residual-component gradient norms/cosines, accepted steps and physical audits;
the optimizer's name is not evidence. Compare evaluation and time budgets, because
one iteration can contain several expensive closures.

A complete restart contains model, optimizer/history, RNG, exact samples/weights,
stage, counters, frozen-endpoint identity, configuration and source identities. Any separately implemented best-state selection also needs its state saved. Verify
interrupted/uninterrupted agreement through the next accepted update in each stage.
A weight file alone cannot resume the same optimizer trajectory.
'''

MANIFEST = r'''## Post-freeze evidence interface

The parent scoring workflow writes `evaluation_manifest.json` separately from the
trainer. The notebook never launches training or silently downloads results.
The manifest has `schema_version: 1`, `campaign_kind`, `status`, and `records`.
Each record includes `run_id, Re, seed, method, outcome`; failed/unscored rows remain.
Ghia marks completed scores `scored`; D5 marks them `controlled_pilot_scored`.
The cost gate remains `cost_smoke_unscored` and cannot enter accuracy tables.
D5 exports may declare `path_base: manifest_directory`; those paths are resolved
inside the checkout. A cost-smoke row remains unscored and contains no reference
accuracy. A D5 `comparison_prediction` binds the original frozen prediction hash
and carries the declared common curl and already aligned global pressure gauge.
A scored row adds repository-relative `prediction`/`reference` file entries with
`path, sha256`, `freeze` with `prediction_sha256, at_utc`, and `score` with the same
`prediction_sha256` plus `reference_accessed_at_utc`. Optional `history` is a hashed
CSV containing `objective_evaluations, elapsed_seconds, metric_value` and `loss_location`.
Its file entry declares `metric_name`, `metric_label`, `independent_audit`, and cost scope.
The present Ghia history is `training_collar_momentum_rms`: the square root of the last
closure's training loss outside the collar, **not** an independent audit or a recomputed
accepted-endpoint residual. Only the fixed final model is scored. Its independent
outer/shell audits are separate final measurements, not a convergence history. Total residual-point
work includes adaptive candidate evaluation and audits; wall time must name its scope.

Native NPZs have increasing 1D `x,y` (or D5 `x,h`) and matching `(len(y),len(x))` arrays `u,v`;
optional `p,omega,psi` retain provenance. Here `y` is physical height above the floor
in units of width, not height normalized by cavity depth. Exact reference/prediction
coordinates are checked. Pressure comparisons use one whole-interior quadrature
mean shift; there is no separately fitted regional offset. A timestamp ledger records
ordering but cannot prove absence of unrecorded target access. Preserve source hashes,
trainer data-read records and independent review as well.

### Evidence and article-output contract

<!-- MIE690A article-aligned validation v3 -->
<!-- MIE690A additive course extension v1; prior lectures and notebooks remain unchanged -->

This is an additive teaching study, not a replacement for the manuscript's retained
figures or a claim that these fresh PINNs meet the paper's accuracy. Preserve the
separate evidence roles in [ARTICLE_FIGURE_MAP.md](../../ARTICLE_FIGURE_MAP.md).
Any later manuscript-facing figure must come from the frozen numerical outputs,
its documented generator and machine-readable metrics; do not redraw curves or
substitute a screenshot for a solver-to-reference comparison. A passed software
or provenance check is not passed scientific acceptance.

[Loss-landscape study](https://proceedings.mlr.press/v235/rathore24a.html) ·
[SOAP gradient-alignment study](https://arxiv.org/abs/2502.00604) ·
[Checked cavity literature](../../docs/CAVITY_ALL_RE_NEURAL_LITERATURE.md).
Published findings motivate experiments; they are not new local results.
'''


def make(kind):
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
    if kind=='ghia':
        md(r'''# Week 4, Lab 9 — Label-free PINNs across the Ghia cases

**Prerequisites:** complete Week 4 CFD/POD foundations and [Week 13 Part A: build a small streamfunction PINN](../week13/W13_Rectangular_Cavity_PINN_Research.ipynb) before this research audit. This is an advanced branch; its Week 4 location does not make it a prerequisite for introductory field learning.

This additive lab studies case-specific PINNs at
$Re=100,400,1000,3200,5000,7500,10000$. The admitted matrix has three seeds (7, 23, 61) and both declared collars per Re.
Each model uses four tanh layers of width 64, float64, 2048 fixed points, 2000 Adam
updates followed by 500 L-BFGS outer updates. Completed updates and actual closures
are different counts. Each model solves its own Re; this is not
one network predicting an unseen Re. [Lab 7](W4_Lab7_High_Re_CFD_Ghia.ipynb) supplies
CFD qualification and [Lab 8](W4_Lab8_High_Re_Field_DeepONet.ipynb) supplies a separate
supervised POD-DeepONet comparison. Earlier lessons remain intact.

**Learning goals:** audit exact boundary conditions, distinguish objective loss from
physical/field error, compare matched costs, and inspect weak vortices. Run All reads
retained evidence on CPU. Missing fresh results are reported explicitly; this lesson
alone does not demonstrate new accuracy.

[Campaign protocol](../../docs/CAVITY_PINN_CAMPAIGN_PROTOCOL.md) ·
[CFD evidence](../../results/cavity_highre_ghia/README.md)
''')
        code(BOOT+"\nCAMPAIGN=load_campaign('results/cavity_pinn_campaign','ghia_constant_lid')\ndisplay(pd.DataFrame(CAMPAIGN['records']).reindex(columns=['run_id','Re','seed','method','outcome','collar']))")
        md(r'''## 1. Equations and the exact lid

Let $x=X/W,y=Y/W\in[0,1]$, $u,v$ be velocities divided by lid speed $U$, and
$p=P/(\rho U^2)$. Width $W$, density $\rho$ and kinematic viscosity $\nu$ define
$Re=UW/\nu$. Subscripts denote derivatives in these physical nondimensional coordinates.

$$r_c=u_x+v_y=0,$$
$$r_x=uu_x+vu_y+p_x-Re^{-1}(u_{xx}+u_{yy})=0,$$
$$r_y=uv_x+vv_y+p_y-Re^{-1}(v_{xx}+v_{yy})=0.$$

The open top wall has $(u,v)=(1,0)$; the other open walls have $(0,0)$.
The incompatible top-corner limits are singular. A smoothed Gaussian/polynomial
lid changes the benchmark. A streamfunction $u=\psi_y,v=-\psi_x$ enforces continuity
by mixed-partial symmetry; signed vorticity is $\omega=v_x-u_y=-\nabla^2\psi$.
Manufactured residual, pressure-sign and random-network wall-derivative tests must
verify these identities before a production run.
''')
        md(r'''## 2. The corner treatment is an explicit approximation

The reviewed pilot uses leading Stokes corner velocity **and pressure** subtraction;
the precise formula and derivative tests are in the protocol. A generic rational
lifting imposed an uncorrectable leading singular defect and was rejected before
training. Even Stokes subtraction can leave $O(1/r)$ residuals after global blending
and nonlinear terms, where $r$ is distance to a top corner.

The declared collocation collar is $\varepsilon=0.02W$, with a paired $0.01W$
sensitivity experiment. The open-wall lid remains constant. Original unscaled
residuals must also be audited in excluded shells. A low loss outside the collar
does not establish accuracy inside it or full-domain strong-residual convergence.
Report collar norm, minimum sampled distance and sample counts; random avoidance
of a corner is not a documented regularization.
''')
        md(r'''## 3. Residual-guided sampling without changing the objective

The implemented Ghia baseline uses fixed uniform collocation; adaptive sampling
is a future controlled extension. The following estimator exercise does not imply
that an adaptive cavity model has been trained.

For $N$ candidate points, let $f_i=r_x(\mathbf x_i)^2+r_y(\mathbf x_i)^2$ and
$J_N=N^{-1}\sum_i f_i$. For $m$ draws with replacement, $P(I_j=i)=q_i>0$,

$$\widehat J_N=\frac1m\sum_{j=1}^m\frac{f_{I_j}}{Nq_{I_j}}.$$

A uniform proposal component keeps all probabilities positive. Clipping or
self-normalizing these weights changes the estimator. A plain adaptive sample mean
also changes the objective. This small finite-sum check is a mathematical exercise,
not a cavity output.
''')
        code('''x=np.linspace(0,1,101)
f=1+20*np.exp(-((x-.85)/.04)**2)
N=len(x);mixture=.2
q=mixture/N+(1-mixture)*f/f.sum()
values={'Uniform objective':f.mean(),'Expected weighted estimate':np.sum(q*f/(N*q)),
        'Expected unweighted adaptive estimate':np.sum(q*f)}
display(pd.Series(values,name='Finite-pool toy calculation'))
assert np.isclose(values['Uniform objective'],values['Expected weighted estimate'],rtol=1e-13)
''')
        md(r'''## A measured computation lesson: aggregate identical work

The GPU probe changes computational chunk size, while retaining the same initial
network and all 2048 collocation points in each full gradient. It is not a change
in stochastic minibatch size. Five consecutive Adam updates per chunk were timed
on one Quadro RTX 8000; the reported value is the median of the last three.
These are not independent timing replicates. The cost curve is not an accuracy
curve. Initial loss/gradient checks support equivalence at the tested state;
they do not prove every future trajectory or device is bitwise identical.
''')
        code('''cost_dir=ROOT/'results/cavity_pinn_campaign/cost_study'
if (cost_dir/'manifest.json').exists():
    cost_manifest=json.loads((cost_dir/'manifest.json').read_text())
    cost=json.loads(verified_file(cost_manifest['source_probe']).read_text())
    rows=cost['rows']
    for row in rows:
        assert np.isclose(np.median(row['times_seconds'][-3:]),row['median_last3_seconds'],rtol=1e-13)
    display(pd.DataFrame([{'chunk_points':r['chunk_size'],'median_update_seconds':r['median_last3_seconds'],
                           'peak_allocated_MB_decimal':r['peak_allocated_bytes']/1e6,
                           'initial_gradient_max_difference':r['gradient_max_abs_difference']} for r in rows]))
    from IPython.display import Image
    display(Image(filename=str(verified_file(cost_manifest['figure'])),width=700))
    print(cost_manifest['timing_scope'])
else:
    print('No verified chunk-cost probe is included.')
''')
        folder='ghia';default_re=100;heights='[.1,.5,.9]'
    else:
        md(r'''# Week 13 — Controlled optimizer experiments in a deep cavity

This additive lesson complements [the retained D5 regional study](W13_Deep_Cavity_D5_Regional_Accuracy.ipynb).
It examines fresh paired optimizer trials, separately from historical transferred
and refined workflows. Geometric depth $D/W=5$ is not five network layers.

**Learning goals:** preserve the exact lid/envelope physics, account for derivative
scaling and work, and assess weak lower vortices. Run All reads evidence on CPU and
does not launch training. Missing fresh results are labeled; historical outcomes
are never relabeled as fresh independent-seed experiments.

[Fresh protocol](../../docs/DEEP_CAVITY_ABLATION_PROTOCOL.md) ·
[Retained author data](../../data/week13_d5/README.md) ·
[Author paper v2](https://arxiv.org/abs/2610.38814v2)
''')
        code(BOOT+"\nCAMPAIGN=load_campaign('results/deep_cavity_ablation','d5_gaussian_lid')\ndisplay(pd.DataFrame(CAMPAIGN['records']).reindex(columns=['run_id','Re','seed','method','outcome']))")
        md(r'''## 1. Physical coordinates and the actual formulation

Use $x=X/W\in[0,1]$, $y=Y/W\in[0,5]$ measured upward from the floor, velocities
scaled by $U$, pressure by $\rho U^2$, and $Re=UW/\nu$. The residuals are

$$r_x=uu_x+vu_y+p_x-Re^{-1}(u_{xx}+u_{yy}),\qquad
r_y=uv_x+vv_y+p_y-Re^{-1}(v_{xx}+v_{yy}).$$

If the network uses $\eta=y/5$, then $\partial_y=\tfrac15\partial_\eta$ and
$\partial_{yy}=\tfrac1{25}\partial_{\eta\eta}$. Omitting these factors changes
the PDE. The author problem has a Gaussian-regularized lid and analytical wall
factor, rather than the square Ghia discontinuous lid. Preserve the exact widths,
coordinate convention and derivative factors in the frozen protocol.

The author representation is
$$\psi=\psi_{\rm lid}+B^2E(d)N_\psi,\qquad p=E(d)N_p,$$
$$E(d)=\exp\{-4.212[\operatorname{softplus}_{10}(d-0.25)
-\operatorname{softplus}_{10}(-0.25)]\},\quad
\operatorname{softplus}_{10}(z)=\tfrac1{10}\log(1+e^{10z}).$$

Here $d=5-y$ is distance below the lid in width units, $B$ the wall factor,
$\psi_{\rm lid}$ the lid lifting and $N_\psi,N_p$ network outputs. The protocol
records the exact source. With $\eta=y/5$,
$$B=16x(1-x)\eta(1-\eta),$$
$$\psi_{\rm lid}=(y-5)\eta^2G(x)e^{-(5-y)^2/\delta_y^2},$$
$$G(x)=(1-e^{-(x-1)^2/\delta_x^2})(1-e^{-x^2/\delta_x^2}),$$
where $\delta_x=\sqrt{0.001}$ and $\delta_y=0.1$. Original network inputs use
$y_{\rm source}=y-4$; the floor-height shift is a plotting convention, not an
unrecorded change of initialization/input scaling. Differentiate the complete products.
The verified source actually trains $Mr_x/E,Mr_y/E$, where
$$M=(1-e^{-r_L^2/a^2})(1-e^{-r_R^2/a^2}),\qquad a=0.01,$$
and $r_L,r_R$ are distances to the upper corners. Physical audits report the
original unmasked $r_x,r_y$ as well. Fresh trials use the authored balanced
sampler: 50% full domain, 30% upper region, 15% floor and 5% logarithmic
lower-corner samples. Only top corners are excluded; lower corners are sampled
down to radius $10^{-4}$. This differs from the older uniform sampler that
excluded all four corners. The smooth residual mask and sampling exclusion
are different mechanisms.
Removing the envelope or reusing the simpler older trainer is not an exact
reproduction. A smaller depth-scaled loss is not proof of a better lower vortex.
''')
        md(r'''## 2. Authentic optimizer variants and controlled attribution

The intended routes are Adam→L-BFGS and an authentic SOAP→self-scaled-Broyden
implementation. The final protocol states the actual admitted variants and source
identities. An unavailable compact implementation is a missing component; a dense
alternative cannot silently be called the exact author's optimizer.

SOAP preconditions gradients; self-scaled Broyden uses scaled secant information.
These can improve conditioning but also cost more per update. Log component gradient
norms/cosines, step lengths, closures, memory and audit errors to test the explanation.
Fresh paired seeds use the same initialization and samples, with no historical
transfer unless it is an explicitly controlled variable. If transfer is used, count
its prior cost. The retained 161×801 Re1000 workflow table has unequal histories
and cannot isolate optimizer causality.
''')
        folder='d5';default_re=1000;heights='[.25,2.5,4.75]'
    md(COMMON_PROTOCOL)
    if kind=='d5':
        md(r'''## Measured cost gate: line-search trials are not accepted endpoints

The short cost gate uses the same fresh initialization, 8192 balanced points and
one Quadro RTX 8000 allocation. Each arm has 50 first-order and only 10
quasi-Newton loss/gradient evaluations. It opens no CFD reference values and
has no accuracy ranking.

Every curve point is an evaluated closure. The orange quasi-Newton spikes are
trial values; the tiny SSB budget expires before its first accepted refinement
step, so the final model remains at the SOAP endpoint. This is not evidence
that adequately budgeted SSB is inferior. The figure's wall-time axis includes
audits and checkpoint I/O; pure optimizer-step time is reported separately.
The fresh three-seed scientific pilot uses a separately declared larger budget.
''')
        code(r'''
COST=load_campaign('results/deep_cavity_ablation/cost_study','d5_gaussian_lid')
if COST['records']:
    display(pd.DataFrame(COST['records'])[['method','outcome','training_seconds','active_seconds','function_evaluations','gradient_evaluations']])
    display(pd.DataFrame(COST.get('pair_checks',[])))
    from IPython.display import Image
    for item in COST.get('figures',[]):
        if item['path'].endswith('.png'):
            display(Image(filename=str(verified_file(item)),width=1000))
    print('Reference values accessed:',COST.get('reference_values_accessed'))
''')
    default_seed="7" if kind=='ghia' else "(CAMPAIGN.get('paired_seeds') or [12345])[0]"
    code(f"RE={default_re}\nSEED={default_seed}\nselected=[r for r in CAMPAIGN['records'] if r['Re']==RE and r['seed']==SEED]\nshow_history(selected)")
    md(r'''## Physical comparison and error meaning

For a declared region $R$ and physical quadrature weights $w_i$, report reference
velocity RMS $A_R$, absolute vector-error RMS $e_R$, and relative error $E_R=e_R/A_R$:

$$A_R^2=\frac{\sum_{i\in R}w_i(u_i^2+v_i^2)}{\sum_{i\in R}w_i},\qquad
 e_R^2=\frac{\sum_{i\in R}w_i[(\hat u_i-u_i)^2+(\hat v_i-v_i)^2]}{\sum_{i\in R}w_i}.$$

A weak lower vortex can have a very small $A_R$; preserve both absolute and relative
errors. Use one whole-interior pressure gauge and retain signed regional pressure
means/RMS; never refit a separate offset in each region. Signed vorticity and native
profiles reveal behavior that speed magnitude can hide. The cell below shows only
actual matched native rows and explicitly reports their heights. Missing pressure
or vorticity data are not silently reconstructed. Each Re/seed is inspected separately.
For the Ghia native profiles, omega compares PINN automatic derivatives with CFD
transported vorticity. The separate gallery computes the same centered curl from
both stored velocity fields on a common 509-by-509 grid; these are different
diagnostics and neither overwrites the frozen scores.

CFD grid sensitivity and PINN discrepancy are different quantities. The existing
square-cavity 256-to-512 velocity differences are 2.03%, 3.14%, 4.72%, 6.32% at
Re3200/5000/7500/10000; they are sensitivity measures, not exact-error bounds.
A steady root does not demonstrate asymptotic temporal stability.
''')
    code(f"scored=[r for r in selected if r['outcome'] in ('scored','controlled_pilot_scored')]\nfor row in scored:\n    print(row['run_id'],row['method'])\n    if 'scores' in row and 'regions' in row['scores']:\n        display(pd.DataFrame.from_dict(row['scores']['regions'],orient='index'))\n    for audit_key in ['physics_audit','physical_residual_audit']:\n        if audit_key in row: display(pd.json_normalize(row[audit_key]))\n    show_native_profiles(row,{heights})\nif not scored:\n    print('No scored frozen fields for this selection; no accuracy conclusion is available.')")
    if folder=='ghia':
        md(r'''## Required student work

Submit these numbered artifacts; preserve failed runs and the frozen campaign.

1. **Table A — problem identity:** for Re100 and one high-Re case, record lid,
   coordinate units, collar, network, seed, sample count and audit roles. Flag any
   mismatch before comparing optimizers.
2. **Table B — sampling:** change the toy mixture to 0.1 and 0.5. Check the exact
   expected weighted objective and explain the change in the unweighted estimate.
   Explain why weight clipping is not a harmless numerical convenience.
3. **Figure 1/Table C — work:** overlay paired collars' recorded training RMS
   against closures and time. State that the last closure need not be the accepted
   endpoint. Compare final independent outer/shell audits separately. There is no
   independent audit history in this baseline, so do not invent one or use it to
   select a checkpoint. Optimizer variants require a separate controlled study.
4. **Figure 2/Table D — collar sensitivity:** compare 0.02W and 0.01W with matched
   starts and budget. Overlay signed physical profiles and report original residuals
   above, between and below the collars. Do changes exceed the method difference?
5. **Figure 3/Table E — physics:** for all available predeclared seeds, report
   interior and lower-quarter $A_R,e_R,E_R$, signed vorticity, and secondary-vortex
   locations. Compare with CFD grid sensitivity and report all failures.
6. **Table F — restart:** in a separately labeled small development run, interrupt
   Adam and L-BFGS once each. Compare next accepted update, parameters, predictions,
   counters and optimizer/sample/RNG states with an uninterrupted control.

Finish with one paragraph naming observable-specific wins/losses and remaining
uncertainty. Case-specific all-seven neural results already exist in literature;
this exercise demonstrates controlled comparison and reproducibility, not priority
or held-out-Re generalization.
''')
    else:
        md(r'''## Required student work

Use a separate development directory; do not overwrite frozen evidence.

1. **Table A — formulation:** record Re, D/W, exact lid widths, wall factor,
   envelope, derivative scaling, network and raw/scaled residual definitions.
   Explain two code changes that would silently alter the intended experiment.
2. **Table B — matching:** verify initial-parameter/sample hashes, seeds, optimizer
   variants, budgets and reference-free checkpoint selection before ranking.
3. **Figure 1/Table C — cost:** plot the recorded metric with its actual label
   against closures and elapsed time. If only final raw audits exist, compare them
   separately; do not invent an audit trajectory. Split first-stage and curvature
   work, including setup and failed searches. Does a ranking change with the cost axis?
4. **Figures 2–3/Table D — weak flow:** inspect native profiles near y/W=0.25,2.5,
   4.75 and the protocol's exact floor masks. Report reference RMS, absolute and
   relative velocity error, signed pressure offsets and vorticity. Explain any
   disagreement between velocity and pressure rankings.
5. **Table E — seeds:** report each paired seed difference, failures, median and
   full range. Three seeds do not automatically yield a confidence interval or
   a universal optimizer ranking. Do not select a best seed using reference error.
6. **Table F — recovery:** interrupt each optimizer stage in a small demonstration.
   Compare the resumed next update and full state with an uninterrupted control.
   Explain which algorithmic information a weight-only checkpoint would lose.

Finish by separating observations, plausible mechanisms and untested proposals.
Discuss a deterioration as carefully as a gain. Historical transferred workflow
rankings need not match a fresh controlled optimizer study.
''')
    if kind=='ghia':
        code("display(pd.DataFrame(CAMPAIGN.get('collar_comparisons',[])))")
        md(r'''## The completed fixed-budget baseline: retain the failures

The numerical ledger below reports every frozen endpoint. Successful execution is
not successful solution. Compare the signed flow profiles with the recorded loss:
a small training objective on a sparse fixed set can coexist with a poor interior
field and failed independent residual tests. This baseline uses one architecture
and one optimizer schedule; it does not establish that PINNs in general fail.

A zero difference between collars is informative only if their actual sample
sets differ. With sparse fixed random points, the removed annulus may contain
no collocation points at all. Inspect sample hashes and annulus coverage before
claiming collar robustness. This is separate from genuine outer/shell audits.

The existing accurate supervised POD-DeepONet learns CFD fields and remains a
separate experiment. It must not be used to describe this label-free baseline.
''')
        code(r'''
figure_manifest_path=ROOT/'results/cavity_pinn_campaign/figures/figure_manifest.json'
if figure_manifest_path.exists():
    figures=json.loads(figure_manifest_path.read_text(encoding='utf-8'))
    verified_file(figures['source_evaluation'])
    display(pd.DataFrame([figures['summary']]))
    print('Physical acceptance:',figures['summary']['physical_acceptance'])
    from IPython.display import Image
    for item in figures['figures']:
        if item.get('kind')=='signed_ghia_centerlines':
            print('Re =',item['Re'],'; all three seeds and both collars')
            display(Image(filename=str(verified_file(item)),width=1000))
    print('Representative fields use predeclared seed 7, collar 0.02; no best-seed selection.')
    for item in figures['figures']:
        if item.get('kind') in ('native_fields_and_derived_centered_curl','matched_streamfunction_contours'):
            display(Image(filename=str(verified_file(item)),width=1000))
else:
    print('The verified result gallery has not been generated yet. No placeholder curves are shown.')
''')
    if kind=='ghia':
        md(r'''## A bounded sampling intervention after the failed baseline

The follow-up changes only the fixed point distribution and its global area
weights at $Re=100,7500$, using seeds 7, 23, 61 and both collars. Each run has
exactly its own original baseline closure and residual-point work cap; those
caps can differ between collars or seeds. This is a newly declared development
experiment after the original failures, not retroactive validation of them.

The comparison preserves every paired outcome. It does not add CFD labels,
choose a best seed, smooth the constant lid or establish full-domain corner
integrability. Inspect signed profiles, near-wall fields and independent raw
residuals together. The original 42-run gallery above is unchanged.

**Saved samples are part of the scientific state.** Cross-host regeneration of
annular coordinates can differ in the last bits. The repaired inference path
loads the exact saved samples and complete trained state; it does not resample,
refit or certify cross-platform optimizer continuation. The ledger below reports
any earlier partial target access instead of presenting the repaired pass as
an evaluation that had never opened references.

**Required follow-up work:**
1. Produce **Table G** with every paired closure/point-work cap, initial-state hash,
   stratum count and weight sum. Explain which collar comparisons have unequal work.
2. Produce **Figure 4/Table H** with signed profiles and errors for all seeds at both
   Re. Find a case where an independent residual decreases while field error grows.
3. Produce **Figure 5/Table I** using seed 7, collar 0.02 and the shared field scales.
   Describe missing or displaced circulation and distinguish native vorticity scores
   from the extra common finite-difference curl.
4. Produce **Table J**, a chronology of the earlier scoring reads and exact-state
   evaluation repair. Explain what was preserved and what was not certified.
''')
        code(r'''
comparison_path=ROOT/'results/cavity_pinn_stratified/comparison/comparison_manifest.json'
if comparison_path.exists():
    STRATIFIED=json.loads(comparison_path.read_text(encoding='utf-8'))
    assert STRATIFIED['schema_version']==1 and STRATIFIED['campaign_kind']=='ghia_uniform_vs_stratified_matched_work'
    expected={(re,seed,epsilon) for re in [100,7500] for seed in [7,23,61] for epsilon in [.01,.02]}
    assert len(STRATIFIED['records'])==12
    assert {(r['Re'],r['seed'],r['collar']) for r in STRATIFIED['records']}==expected
    for item in STRATIFIED['inputs'].values(): verified_file(item)
    for item in STRATIFIED.get('evaluation_provenance',{}).values(): verified_file(item)
    print(STRATIFIED.get('evaluation_chronology',{}).get('disclosure','See the completed-pass provenance ledger.'))
    paired=pd.read_csv(verified_file(STRATIFIED['outputs']['paired_metrics']))
    metrics=['field.interior.velocity_relative_l2','field.lower_quarter.velocity_relative_l2',
             'ghia.u.screened_relative_l2','ghia.v.screened_relative_l2',
             'physics.outer_uniform.momentum_rms']
    display(paired[paired['metric'].isin(metrics)][['Re','seed','collar','metric','baseline','stratified','outcome']])
    display(pd.DataFrame(STRATIFIED['records'])[['Re','seed','collar','closure_cap','matched_residual_point_evaluations']])
    display(pd.DataFrame(STRATIFIED['collar_comparisons']))
    figure_path=ROOT/'results/cavity_pinn_stratified/figures/figure_manifest.json'
    if figure_path.exists():
        gallery=json.loads(figure_path.read_text(encoding='utf-8'))
        assert gallery['source_comparison']['sha256']==hashlib.sha256(comparison_path.read_bytes()).hexdigest()
        from IPython.display import Image
        for item in gallery['figures']:
            if item['path'].endswith('.png'):
                print(item['kind'],'; Re =',item.get('Re'),'seed =',item.get('seed','all three'))
                display(Image(filename=str(verified_file(item)),width=1000))
    else:
        print('Scored pairs exist, but the verified physical gallery is not available yet.')
else:
    print('The completed 12-pair sampling comparison is not available yet; no gain is inferred.')
''')
    if kind=='d5':
        md(r'''## Completed controlled study: compare every region and seed

The two optimizer pipelines use fresh matched starts and frozen balanced samples.
A lower error for one observable is not a winner for every region. Compare
absolute velocity error with the local CFD amplitude before reading percentages;
retain signed pressure offsets after the one global interior gauge. The floor
pressure shape panels remove each plotted cut mean only as a separate diagnostic.

The tables below retain all three paired seeds and the exact reported regional
metrics. The optimizer timings exclude some work, while the history's active-wall
time includes audits and I/O; neither should be presented as a universal hardware
speed ranking. Three seeds give descriptive variability, not a significance test.
''')
        code(r'''
if CAMPAIGN.get('status')=='controlled_pilot_scored':
    assert CAMPAIGN['paired_seeds']==[12345,23456,34567]
    assert len(CAMPAIGN['records'])==6
    display(pd.DataFrame(CAMPAIGN['pair_checks']))
    display(pd.DataFrame(CAMPAIGN['records'])[['seed','method','function_evaluations','gradient_evaluations','training_seconds','active_seconds']])
    regional=pd.DataFrame(CAMPAIGN['metrics'])
    display(regional[['seed','method','region','CFD_velocity_RMS_U','velocity_RMS_U','velocity_NRMSE_percent',
                      'pressure_RMS_rhoU2','pressure_signed_mean_offset_rhoU2','vorticity_RMS_UoverW']])
    summary=pd.DataFrame(CAMPAIGN['descriptive_seed_summary'])
    display(summary[['region','metric','baseline_median','candidate_median',
                     'candidate_lower_error_seeds','candidate_higher_error_seeds']])
    print('Baseline route: Adam to L-BFGS. Candidate route: SOAP to self-scaled Broyden.')
    for row in CAMPAIGN['descriptive_seed_summary']:
        print(row['region'],row['metric'],': candidate lower error in',row['candidate_lower_error_seeds'],
              'of',row['paired_seed_count'],'paired seeds; higher in',row['candidate_higher_error_seeds'])
    print('No combined winner or full-domain convergence claim follows from these tables.')
else:
    print('Fresh six-run regional accuracy results are not available. The short cost gate remains separate.')
''')
    if kind=='d5':
        md(r'''## Fresh controlled-pilot figures, when frozen and scored

The main campaign gallery is separate from the cost gate. Every displayed image
is verified against its manifest. Physical tables retain signed pressure offsets
after one global gauge. A floor pressure panel explicitly labeled **cut-mean-centered
shape** is an additional shape diagnostic, not the original signed pressure error.
It must not replace the regional signed mean and RMS values.
''')
        code(r'''
if any(r['outcome'] in ('scored','controlled_pilot_scored') for r in CAMPAIGN['records']):
    from IPython.display import Image
    rgb_manifest_path=ROOT/'results/deep_cavity_ablation/lesson_figures/figure_manifest.json'
    if rgb_manifest_path.exists():
        rgb_gallery=json.loads(rgb_manifest_path.read_text(encoding='utf-8'))
        verified_file(rgb_gallery['source_evaluation'])
        images=rgb_gallery['figures']
    else:
        images=CAMPAIGN.get('figures',[])
    for item in images:
        if item['path'].endswith('.png'):
            display(Image(filename=str(verified_file(item)),width=1000))
else:
    print('No frozen, scored fresh D5 physical gallery is available yet. The cost gate is not an accuracy result.')
''')
    md(MANIFEST)
    relative=TARGETS[kind].relative_to(ROOT).as_posix()
    launcher=('https://colab.research.google.com/github/'
              'Ehsan-Roohi/FlowMLLab/blob/main/'+relative)
    heading,body=cells[0].source.split('\n',1)
    cells[0].source=(heading+'\n\n[Open in Colab]('+launcher+')\n'+body+
        '\n\n**CPU setup:** use the existing checkout, or let the first cell clone the published '
        'repository once when none is found. Existing folders and student edits are preserved. '
        'Colab already supplies NumPy, pandas, Matplotlib and IPython; a local environment needs '
        'those packages. This evidence-only route needs no Torch, TensorFlow or GPU. '
        'The initial repository download can take longer than the numerical audit. '
        'A missing campaign is reported explicitly; no external result archive is downloaded.')
    book=nb.v4.new_notebook(cells=cells)
    book.metadata.update({'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                          'language_info':{'name':'python'},
                          'flowmllab':{'additive_lesson':True,'launches_training':False,'campaign':kind}})
    return book


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    for kind,target in TARGETS.items():
        book=make(kind);nb.validate(book)
        for cell in book.cells:
            if cell.cell_type=='code':compile(cell.source,str(target),'exec')
        if args.check:
            old=nb.read(target,as_version=4)
            assert [(c.cell_type,c.source) for c in old.cells]==[(c.cell_type,c.source) for c in book.cells]
        else:
            if args.execute:
                from execute_cavity_lesson import execute
                book=execute(book,ROOT)
            target.write_text(nb.writes(book),encoding='utf-8')
        print(f'{target.name}: {len(book.cells)} cells; checked={args.check}; executed={args.execute and not args.check}')

if __name__=='__main__':main()
