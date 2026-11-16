"""Build only the additive Week 10.2 lesson; never rewrite earlier notebooks.

Scientific source extracts are owned by data/week10_2_kinetic_shock/manifest.json.
Default notebook execution reads those files, performs CPU moment calculations,
trains a bounded fresh BGK PINN, and runs its independent CPU DVM reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from flowmllab import kinetic_shock as ks

BOOK = ROOT / 'notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb'
RESULTS = ROOT / 'results/week10_2_kinetic_shock'
LECTURE_SOURCE = ROOT / 'lectures/source/week10_2_bgk_shakhov_shock_pinn_dvm.json'
LECTURE = ROOT / 'lectures/week10_2_bgk_shakhov_shock_pinn_dvm.pdf'
PAPER = 'https://arxiv.org/abs/2605.29211v2'
LABELS = {'rho': r'$\rho$', 'u': r'$u$', 'T': r'$T$', 'q': r'$q_x$',
          'sigma': r'$\sigma_{xx}$', 'm': r'$m_{xxx}$', 'R': r'$R_{xx}$',
          'Delta': r'$\Delta$', 'S': r'$S_{xx}$'}
COLORS = ['#2355b5', '#c52a32', '#16834b', '#8b4f9e']


def readable(method):
    names = {'reference': 'Matched DVM', 'parent': 'Parent equation-trained state',
             'final': 'Final equation-trained state', 'model': 'Retained PINN',
             'equation_trained': 'Equation-trained BGK', 'five_field': 'Five-field S0',
             'eight_moment': 'Final eight-moment BGK', 'stageB': 'Final eight-moment BGK',
             's0': 'Five-field S0', 'S0': 'Five-field S0',
             'five_field_S0': 'Five-field S0', 'data_assisted': 'Final eight-moment BGK',
             'original_envelope': 'Eight-moment model170',
             'common_tail_initial': 'Eight-moment model170T'}
    return names.get(method, method.replace('_', ' '))


def configure_plots(publication=False):
    import matplotlib as mpl
    if publication:
        from qa.lecture_typography import configure_matplotlib
        configure_matplotlib()
    mpl.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': 'black', 'text.color': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'})


def profile_figure(case, fields=('rho', 'T', 'q', 'sigma', 'R', 'Delta')):
    """Plot native comparison nodes, preserving signs and actual coverage."""
    import matplotlib.pyplot as plt
    fields = tuple(f for f in fields if f in case.columns)
    rows = (len(fields) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(9, 2.45 * rows + 0.6), squeeze=False)
    selected = ('five_field_S0', 'data_assisted', 'equation_trained') if 'five_field_S0' in case.methods else tuple(m for m in case.methods if m != 'reference')
    methods = ['reference'] + [m for m in selected if m in case.methods]
    for ax, field in zip(axes.flat, fields):
        for j, method in enumerate(methods):
            style = {'color': 'black', 'lw': 2.2} if method == 'reference' else {
                'color': COLORS[(j - 1) % len(COLORS)], 'lw': 1.6,
                'linestyle': ('--', '-.', ':', '-')[((j - 1) % 4)]}
            ax.plot(case.x, case.field(method, field), label=readable(method), **style)
        ax.set(xlabel=r'$x$', ylabel=LABELS[field], xlim=(case.x[0], case.x[-1]))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=True)
        ax.grid(alpha=.16)
    for ax in list(axes.flat)[len(fields):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1),
               ncol=min(2, len(methods)), frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .88 if len(methods) > 2 else .91), h_pad=1.1, w_pad=1.2)
    return fig


def score_case(case, case_id):
    rows = []
    for method in case.methods:
        if method == 'reference':
            continue
        for field in case.columns:
            result = ks.profile_metrics(case.x, case.field(method, field),
                                        case.field('reference', field))
            rows.append({'case': case_id, 'method': method, 'field': field,
                         'relative_l2_percent': float(result['relative_l2_percent'])})
    return rows


def m2_figure(seed=141001):
    import matplotlib.pyplot as plt
    stages = [('adam20000', 'Adam 20k'), ('ssb500', 'SSB 500'), ('fixed500', 'Split-half SSB 500')]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.9), sharex=True)
    first = ks.load_case(f'm2_seed{seed}_adam20000')
    for j, field in enumerate(('q', 'sigma')):
        axes[0, j].plot(first.x, first.field('reference', field), color='black', lw=2.2, label='Matched DVM')
        for i, (stage, label) in enumerate(stages):
            case = ks.load_case(f'm2_seed{seed}_{stage}')
            method = next(m for m in case.methods if m != 'reference')
            pred, ref = case.field(method, field), case.field('reference', field)
            axes[0, j].plot(case.x, pred, color=COLORS[i], ls=('--', '-.', ':')[i], lw=1.8, label=label)
            axes[1, j].plot(case.x, pred - ref, color=COLORS[i], ls=('--', '-.', ':')[i], lw=1.7)
        axes[0, j].set_ylabel(LABELS[field])
        axes[1, j].set(xlabel=r'$x$', ylabel='PINN - DVM')
        axes[1, j].axhline(0, color='0.45', lw=.6)
    for ax in axes.flat:
        ax.grid(alpha=.16)
        ax.set_xlim(first.x[0], first.x[-1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .88), h_pad=1.1)
    return fig


def schematic(kind):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis('off')
    def box(x, y, w, h, label):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                   fc='#f5f5f5', ec='black', lw=1))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=13)
    def arrow(a, b):
        ax.annotate('', b, a, arrowprops={'arrowstyle': '->', 'lw': 1.2, 'color': 'black'})
    if kind == 'model':
        box(.1, 1.25, 2.6, 1.15, 'One physical coordinate x\nThree velocity coordinates\nDistribution f(t, x, v)')
        box(3.5, 2.4, 2.5, 1.0, 'BGK\nMaxwellian target M')
        box(3.5, .35, 2.5, 1.15, 'Shakhov\nHeat-flux correction\nPr controls q relaxation')
        box(7.0, 1.25, 2.7, 1.15, 'Transport + collision\nKinetic shock structure\nSigned moments')
        arrow((2.8, 2.0), (3.4, 2.8)); arrow((2.8, 1.6), (3.4, 1.0))
        arrow((6.1, 2.8), (6.9, 2.0)); arrow((6.1, 1.0), (6.9, 1.6))
    else:
        box(.1, 1.35, 2.1, 1.0, 'Full distribution f\nIntegrate transverse v')
        box(3.1, 2.35, 2.3, .95, 'g: population\nh: transverse\nsecond moment')
        box(3.1, .4, 2.3, .95, 'k: transverse\nfourth moment')
        box(6.5, 2.35, 3.0, .95, 'Density, velocity, temperature\nStress, heat flux, m and S')
        box(6.5, .4, 3.0, .95, 'Separate R and Delta\nS = R + Delta / 3')
        arrow((2.3, 2.05), (3.0, 2.8)); arrow((2.3, 1.55), (3.0, .9))
        arrow((5.5, 2.8), (6.4, 2.8)); arrow((5.5, .9), (6.4, .9))
        arrow((5.5, 2.5), (6.5, 1.2))
    fig.tight_layout(pad=.5)
    return fig


def collision_figure():
    import matplotlib.pyplot as plt
    t = np.linspace(0, 3, 201)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(t, np.exp(-t), color=COLORS[0], lw=2, label='Both models: stress')
    axes[1].plot(t, np.exp(-t), color=COLORS[0], lw=2, label='BGK: Pr = 1')
    axes[1].plot(t, np.exp(-(2/3)*t), color=COLORS[1], ls='--', lw=2, label='Shakhov: Pr = 2/3')
    for ax, y in zip(axes, (r'$\sigma/\sigma_0$', r'$q/q_0$')):
        ax.set(xlabel=r'$t/\tau$', ylabel=y, xlim=(0, 3), ylim=(0, 1.04))
        ax.grid(alpha=.18)
    handles, labels = [], []
    for ax in axes:
        a, b = ax.get_legend_handles_labels(); handles.extend(a); labels.extend(b)
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .86))
    return fig


def build_figures():
    import matplotlib.pyplot as plt
    configure_plots(publication=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    figures = {'kinetic_model_schematic': schematic('model'),
               'moment_hierarchy': schematic('moments'), 'collision_rates': collision_figure(),
               'tube_bgk_profiles': profile_figure(ks.load_case('tube_bgk')),
               'tube_shakhov_profiles': profile_figure(ks.load_case('tube_shakhov')),
               'm2_optimizer_profiles': m2_figure()}
    supervised = [name for name in ks.available_cases() if name.startswith('supervised')]
    if len(supervised) != 1:
        raise ValueError('Require exactly one declared supervised comparison case')
    figures['supervised_bgk_profiles'] = profile_figure(ks.load_case(supervised[0]))
    for name, fig in figures.items():
        fig.savefig(RESULTS / f'{name}.png', dpi=200, facecolor='white')
        plt.close(fig)
    metrics = [r for name in ks.available_cases() for r in score_case(ks.load_case(name), name)]
    pd.DataFrame(metrics).to_csv(RESULTS / 'profile_metrics.csv', index=False, float_format='%.12g')
    print(f'Built {len(figures)} physical figures; rescored {len(metrics)} profile comparisons.', flush=True)


BOOT = '''# FLOWMLLAB_COLAB_BOOTSTRAP_V1
from pathlib import Path
import sys, subprocess, time, json
START = time.perf_counter()
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents, Path('/content/FlowMLLab')]
             if (p/'flowmllab/kinetic_shock.py').is_file()), None)
if ROOT is None:
    candidate = Path('/content/FlowMLLab') if Path('/content').is_dir() else Path.cwd()/'FlowMLLab'
    if candidate.exists():
        raise RuntimeError('Save your own work and update the existing checkout to a revision with Week 10.2. This cell never overwrites it.')
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse',
                    'https://github.com/Ehsan-Roohi/FlowMLLab.git',str(candidate)],check=True)
    subprocess.run(['git','-C',str(candidate),'sparse-checkout','set',
                    'flowmllab','data/week10_2_kinetic_shock'],check=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(candidate)],check=True)
    ROOT = candidate
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from IPython.display import display
from flowmllab import kinetic_shock as ks
print('Mode: fresh CPU collision/moment calculations + retained PINN/DVM rescoring.')
print('Data source commit:',ks.UPSTREAM_COMMIT)
print('Run All also trains a fresh CPU BGK PINN and solves independent DVM references in Sections 10-12.')
'''


def build_notebook(execute=False):
    import nbformat as nbf
    cells = []
    def md(identifier, text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip(), id=identifier))
    def code(identifier, text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip(), id=identifier))
    md('w10-2-title', r'''
        # Week 10.2 - One-dimensional kinetic shocks: BGK, Shakhov, PINNs and DVM

        [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb)
        · [Lecture](../../lectures/week10_2_bgk_shakhov_shock_pinn_dvm.pdf)
        · [Setup](README.md) · [Paper, revised 13 November 2026](https://arxiv.org/abs/2605.29211v2)

        **Question:** when density and temperature look right, can heat flux, stress
        or fourth-order moments still be wrong? Compute collision moments on CPU,
        then answer with the paper's actual matched PINN/DVM profiles.

        Prerequisites: Week 2 data roles, Week 3 Maxwellians and moments, Week 8 shock
        jumps, and Week 10 profile validation. The lecture introduces the PINN
        residual; prior completion of the full Week 13 research audit is optional.

        **Fresh work and retained research are kept distinct.** Tables B-D are
        fresh CPU calculations; E-G re-evaluate saved scientific predictions.
        Sections 10-14 train a new CPU BGK PINN, solve an independent DVM, and let
        you change depth, optimizer and loss weights (Tables H-J). Full historical
        paper training remains a separate research workflow.
        All variables are nondimensional, with gas constant $R_g=1$ and
        $\gamma=5/3$. One spatial dimension still has **three molecular velocity
        components**. Do not replace this model with a scalar Burgers equation.
    ''')
    code('w10-2-setup', BOOT)
    md('w10-2-case-map', r'''
        ## 1. Table A - Case and data-role registry

        Check the collision model, relaxation law, time/domain and reference role
        before comparing two curves. A stationary Mach-2 shock is a different
        problem from the short-time smoothed shock tube. The loader checks hashes,
        grid coordinates and case identities; it does not fit shifts or rescale a
        prediction to improve its score. Read the data-assisted comparison's
        metadata for its explicit common-grid interpolation.

        **Conditions behind the figures.** All cases use a monatomic gas,
        $R_g=1$, $\gamma=5/3$ and three molecular velocity components.

        | Figure / table | Physical problem | Mach and collision parameters | Reference |
        | --- | --- | --- | --- |
        | CPU PINN, H-J | Transient BGK tube, $t=0.02$, $x\in[-0.5,0.5]$ | No prescribed stationary upstream Mach; constant $\tau=\mathrm{Kn}=1$, $\Pr=1$ | Independent CPU DVM, 400/800 cells |
        | Saved tube profiles, E | Same transient initial states, BGK or Shakhov | No prescribed stationary upstream Mach; $\tau=\mathrm{Kn}=1$; $\Pr=1$ or $2/3$, respectively | Model-matched 3200-cell DVM |
        | Moment supervision, G | Same transient BGK tube and final time | Same $\tau=\mathrm{Kn}=1$, $\Pr=1$ | 6400-cell DVM, 399 common comparison nodes |
        | Stationary optimizer path, F | Steady normal shock; 81 retained core nodes, $x\in[-10,10]$ | Upstream $M_1=2$; Shakhov, $\Pr=2/3$, $\tau=1/\rho$ | Matched stationary DVM |

        The source's Kn convention is tied to its nondimensional relaxation law;
        it is not an imported DSMC mean-free-path estimate. In a transient tube,
        local $M(t,x)=|u|/\sqrt{\gamma R_gT}$ evolves from zero. A propagating
        shock's speed-based Mach would require a separate front-tracking measurement;
        neither quantity should be labelled as a prescribed stationary $M_1$.
    ''')
    code('w10-2-registry', '''
        manifest = ks.load_manifest()
        registry = []
        for case_id in ks.available_cases():
            case = ks.load_case(case_id)
            registry.append(dict(case=case_id, problem=case.metadata['problem'],
                model=case.metadata['collision_model'], tau=case.metadata.get('tau_law'),
                rows=len(case.x), xmin=case.x[0], xmax=case.x[-1],
                methods=', '.join(case.methods)))
        table_A = pd.DataFrame(registry)
        print('Table A: verified retained cases; these are not newly trained networks')
        display(table_A)
    ''')
    md('w10-2-jump-theory', r'''
        ## 2. Table B - Jump states and the correct steady fluxes

        Upstream $\rho_1=T_1=1$ and $u_1=M\sqrt{5/3}$. For a monatomic normal shock,
        $$\rho_2=\frac{4M^2}{M^2+3},\quad u_2=\frac{u_1}{\rho_2},\quad
        T_2=\frac{5M^2-1}{4\rho_2}.$$
        The kinetic stress is $\sigma=\int(c_x^2-c^2/3)f\,d\mathbf v$, and
        $q=\frac12\int c^2c_x f\,d\mathbf v$, where $\mathbf c=\mathbf v-u\mathbf e_x$.
        Preserve both signs. A steady solution conserves
        $$J_m=\rho u,\quad J_p=\rho u^2+\rho T+\sigma,\quad
        J_E=\frac12\rho u^3+\frac52\rho Tu+u\sigma+q.$$
        Equilibrium endpoints have $q=\sigma=0$. The same spatially constant-flux
        test does **not** apply to the transient shock tube.
    ''')
    code('w10-2-jumps', '''
        MACH = 2.0  # Exercise 1: also try 1.5 and 3.0, one case at a time.
        states = ks.jump_states(MACH)
        flux_rows = []
        for side in ('upstream','downstream'):
            s = states[side]
            flux = ks.stationary_fluxes(s['rho'],s['u'],s['T'],0.0,0.0)
            flux_rows.append({'side':side, **s, **{k:float(v) for k,v in flux.items()}})
        table_B = pd.DataFrame(flux_rows)
        print('Table B: fresh Rankine-Hugoniot states and fluxes')
        display(table_B)
        assert np.allclose(table_B.iloc[0][['mass','momentum','energy']].astype(float),
                           table_B.iloc[1][['mass','momentum','energy']].astype(float),rtol=1e-12)
    ''')
    md('w10-2-moment-theory', r'''
        ## 3. Table C - Which moments can the reduced distributions determine?

        $g=\int f\,dv_y dv_z$, $h=\int(v_y^2+v_z^2)f\,dv_y dv_z$ and
        $k=\int(v_y^2+v_z^2)^2f\,dv_y dv_z$. There is **no factor one-half in $h$**.
        For $c=v_x-u$, all following integrals are over $v_x$:
        $$3\rho T=\int(c^2g+h),\quad q=\frac12\int(c^3g+ch),\quad
        \sigma=\frac13\int(2c^2g-h).$$
        $$R=\frac13\int(2c^4g+c^2h-k)-7T\sigma,\qquad
        \Delta=\int(c^4g+2c^2h+k)-15\rho T^2.$$
        $S=R+\Delta/3$ is a derived ninth diagnostic. Eight-moment supervision
        means $\rho,u,T,q,\sigma,m,R,\Delta$, not eight independent entries plus $S$.

        The next calculation compares two **positive** full distributions with
        the same $g,h$. A has a transverse Gaussian temperature 1; B mixes
        transverse temperatures 0.5 and 1.5 equally. Both have $g=N(0,1)$ and
        $h=2g$, but $k_A=8g$ and $k_B=10g$. Predict $R,\Delta,S$ before running.
    ''')
    code('w10-2-counterexample', '''
        counter = ks.transverse_counterexample(order=64,cutoff=9)
        fields = ['rho','u','T','q','sigma','m','R','Delta','S']
        table_C = pd.DataFrame([{'case':name,**{f:counter[name][f] for f in fields}}
                                for name in ('A','B')])
        print('Table C: same g and h; different R and Delta')
        display(table_C)
        assert np.isclose(counter['B']['R'],-2/3,atol=1e-9)
        assert np.isclose(counter['B']['Delta'],2,atol=1e-9)
        assert np.isclose(counter['B']['S'],0,atol=1e-9)
        print('Quantities available without k:',list(counter['without_k']))
    ''')
    md('w10-2-collision-theory', r'''
        ## 4. Table D - A fresh CPU collision experiment

        Both models solve $\partial_t f+v_x\partial_x f=(\mathcal T-f)/\tau$.
        BGK uses the Maxwellian $\mathcal M$. Shakhov uses
        $$\mathcal T_S=\mathcal M\left[1+
        \frac{(1-\Pr)q c_x}{5\rho T^2}\left(\frac{c^2}{T}-5\right)\right].$$
        The fresh calculation uses the positive mixture
        $0.7\mathcal M(1,0.8,0.7)+0.3\mathcal M(1,-1.2,1.4)$, with zero transverse
        mean velocities. Its exact infinite-domain moments are
        $(\rho,u,T,\sigma,q)=(1,0.2,1.19,0.56,-1.071)$.

        Integrate the collision term with tensor Gauss-Legendre quadrature on
        $[-V,V]^3$. It should conserve mass, momentum and energy; stress relaxes
        at $-\sigma/\tau$ and heat flux at $-\Pr q/\tau$. Order and cutoff are
        different errors. The Shakhov target can have negative tails: count them
        without clipping, and check whether their mass is significant. This
        homogeneous experiment tests collision physics; it is not a shock solve.
    ''')
    code('w10-2-collision-audit', '''
        COLLISION_ORDER = 32
        VELOCITY_CUTOFF = 8.0
        PRANDTL = 2/3
        TAU = 1.0
        fresh = ks.collision_experiment(order=COLLISION_ORDER,cutoff=VELOCITY_CUTOFF,
                                         Pr=PRANDTL,tau=TAU)
        collision_rows = []
        for model,d in fresh['diagnostics'].items():
            collision_rows.append(dict(model=model,order=COLLISION_ORDER,cutoff=VELOCITY_CUTOFF,
                invariant_rate_max=float(np.max(np.abs(d['invariant_rate']))),
                stress_rate=d['stress_rate'],expected_stress_rate=d['expected_stress_rate'],
                q_rate=d['q_rate'],expected_q_rate=d['expected_q_rate'],
                minimum_target=d['minimum_target'],negative_nodes=d['negative_nodes'],
                negative_mass=d['negative_mass']))
        table_D = pd.DataFrame(collision_rows)
        print('Fresh mixture moments:',fresh['moments'])
        print('Table D: measured collision moments, without clipping or correction')
        display(table_D)
        assert fresh['input_positive'] and not fresh['is_shock_solution']
        assert all(not d['clipping_applied'] for d in fresh['diagnostics'].values())
    ''')
    code('w10-2-quadrature-sweep', '''
        # Exercise 3: change one parameter at a time; retain every measured row.
        QUADRATURE_CASES = [(12,8.0),(20,8.0),(32,8.0),(32,4.0),(32,6.0)]
        sweep = []
        for n,V in QUADRATURE_CASES:
            t0 = time.perf_counter()
            trial = ks.collision_experiment(order=n,cutoff=V,Pr=PRANDTL,tau=TAU)
            for model,d in trial['diagnostics'].items():
                sweep.append(dict(order=n,cutoff=V,model=model,
                    density_error=abs(trial['moments']['rho']-1),
                    heat_flux_error=abs(trial['moments']['q']+1.071),
                    invariant_rate_max=np.max(np.abs(d['invariant_rate'])),
                    q_rate_error=d['q_rate_absolute_error'],seconds=time.perf_counter()-t0))
        print('Table D continuation: quadrature and velocity-tail sensitivity')
        display(pd.DataFrame(sweep))
    ''')
    md('w10-2-pinn-and-errors', r'''
        ## 5. Reading a kinetic PINN result

        The source's transient network predicts **source-state** density, velocity
        and temperature (also heat flux for Shakhov). It does not directly output
        the nine plotted moments. The source state forms a collision target;
        a characteristic integral reconstructs the distribution, whose velocity
        integrals give the reported moments. Self-consistency between these two
        routes matters. Along an interior characteristic with constant relaxation
        time, before it meets an inflow boundary,
        $$f(t,x,\mathbf v)=e^{-t/\tau}f_0(x-v_xt,\mathbf v)
        +\frac1\tau\int_0^t e^{-(t-s)/\tau}
        \mathcal T_\theta(s,x-v_x(t-s),\mathbf v)\,ds.$$
        Here $\theta$ are network parameters and $s$ is integration time. If a
        characteristic enters through a boundary, start the integral at that
        entry time and use its prescribed incoming distribution. The stationary
        paper cases use different distribution/correction parameterizations and
        differential residual checks. Their losses are not interchangeable.

        Separate physics/self-consistency, initial/incoming conditions and optional
        observed-moment terms. A small weighted total loss is not a common
        accuracy unit, nor proof of accurate unobserved moments.

        We now evaluate **stored** fields. For field $a$ and matched reference
        $a_{ref}$, use a spatially weighted norm:
        $$E_a=100\sqrt{\frac{\int(a-a_{ref})^2dx}{\int a_{ref}^2dx}}.$$
        Integrals use trapezoidal weights on the declared comparison nodes.
        Plot the signed discrepancy $a-a_{ref}$ as well. Never divide pointwise
        by $a_{ref}(x)$ at a zero crossing. A zero reference norm makes relative
        error undefined; report an explicit physical scale instead.
    ''')
    code('w10-2-plot-tools', r'''
        FIELD_LABELS = {'rho':r'$\rho$','u':r'$u$','T':r'$T$','q':r'$q_x$',
                        'sigma':r'$\sigma_{xx}$','m':r'$m_{xxx}$','R':r'$R_{xx}$',
                        'Delta':r'$\Delta$','S':r'$S_{xx}$'}
        def selected_methods(case,include_intermediates=False):
            if 'five_field_S0' in case.methods and not include_intermediates:
                return [m for m in ('five_field_S0','data_assisted','equation_trained') if m in case.methods]
            return [m for m in case.methods if m!='reference']
        def score_profiles(case,include_intermediates=False):
            rows = []
            for method in selected_methods(case,include_intermediates):
                for field in case.columns:
                    e = ks.profile_metrics(case.x,case.field(method,field),case.field('reference',field))
                    rows.append(dict(method=method,field=field,E_percent=e['relative_l2_percent']))
            return pd.DataFrame(rows)
        def plot_profiles(case,fields=('rho','u','T','q','sigma','m','R','Delta','S')):
            fields = [f for f in fields if f in case.columns]
            rows = (len(fields)+2)//3
            fig,axes = plt.subplots(rows,3,figsize=(12,2.7*rows+0.6),squeeze=False)
            palette = ['#2355b5','#c52a32','#16834b','#8b4f9e']
            methods = ['reference']+selected_methods(case)
            for ax,field in zip(axes.flat,fields):
                for i,method in enumerate(methods):
                    ax.plot(case.x,case.field(method,field),label=method,
                            color='black' if i==0 else palette[(i-1)%4],
                            ls='-' if i==0 else ['--','-.',':','-'][(i-1)%4],lw=1.7)
                ax.set(xlabel='x',ylabel=FIELD_LABELS[field],xlim=(case.x[0],case.x[-1]))
                ax.grid(alpha=.18)
            for ax in list(axes.flat)[len(fields):]: ax.set_visible(False)
            handles,labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=min(3,len(methods)),frameon=False)
            fig.tight_layout(rect=(0,0,1,.9),h_pad=1)
            plt.show()
    ''')
    md('w10-2-transient', r'''
        ## 6. Table E - Short-time BGK and Shakhov tubes

        These two saved cases share $x\in[-0.5,0.5]$, $t_f=0.02$ and
        $\tau=\mathrm{Kn}=1$. Their Maxwellian initial macros interpolate between
        $(1,0,1)$ and $(0.125,0,0.8)$ with $H=(1+\tanh20x)/2$. Only incoming
        boundary populations are prescribed. This is a smoothed kinetic Riemann
        problem, not a discontinuous Euler Sod result.

        The tuples above mean $(\rho,u,T)$. Thus $p_L/p_R=10$ and
        $\rho_L/\rho_R=8$; $R_g=1$ and $\gamma=5/3$. Both reservoirs start at
        rest: **there is no prescribed stationary upstream Mach number**.
        BGK has $\Pr=1$ and Shakhov uses $\Pr=2/3$. These conditions apply to
        both profile figures directly below, each with its own 3200-cell DVM.

        The maximum collision weight over this short interval is
        $1-e^{-t_f/\tau}\approx0.0198$. Interpret differences as a predominantly
        short-time transport test. Compare each PINN with its own collision-model
        DVM. Do not swap the BGK and Shakhov reference columns.
    ''')
    code('w10-2-bgk-tube', '''
        tube_bgk = ks.load_case('tube_bgk')
        print('Table E-BGK: stored profiles rescored against matched DVM')
        display(score_profiles(tube_bgk).pivot(index='field',columns='method',values='E_percent'))
        plot_profiles(tube_bgk)
    ''')
    code('w10-2-shakhov-tube', '''
        tube_shakhov = ks.load_case('tube_shakhov')
        print('Table E-Shakhov: stored profiles rescored against matched DVM')
        display(score_profiles(tube_shakhov).pivot(index='field',columns='method',values='E_percent'))
        plot_profiles(tube_shakhov)
    ''')
    md('w10-2-steady-study', r'''
        ## 7. Table F - Independent stationary Mach-2 runs

        These are Shakhov shocks with $\Pr=2/3$ and $\tau=1/\rho$.
        Upstream $\rho_1=T_1=1$, $R_g=1$, $\gamma=5/3$, and
        $u_1=M_1\sqrt{\gamma R_gT_1}=2\sqrt{5/3}$. The exported comparison
        covers 81 core nodes in $[-10,10]$; this is the comparison interval,
        not a claim about the full solver domain. Unlike Table E, this case
        **does prescribe upstream $M_1=2$**.
        Three initializations (141001, 141002, 141003) were retained at Adam,
        SSB and split-half SSB stages. The stages change objective/quadrature and
        cumulative work: this is a recorded optimization path, not an equal-budget
        optimizer ranking. Compare heat flux and stress separately.

        The source's final runs pass flux, phase, quadrature and positive-macro
        checks, but fail tail and projection checks; two also fail kinetic residual
        checks. Small conserved-flux defects do not certify accurate distributions.
    ''')
    code('w10-2-stage-scores', '''
        stage_rows = []
        for seed in (141001,141002,141003):
            for stage in ('adam20000','ssb500','fixed500'):
                case = ks.load_case(f'm2_seed{seed}_{stage}')
                method = next(m for m in case.methods if m!='reference')
                row = {'seed':seed,'stage':stage}
                for field in ('q','sigma'):
                    row[field+'_error_percent'] = ks.profile_metrics(
                        case.x,case.field(method,field),case.field('reference',field))['relative_l2_percent']
                stage_rows.append(row)
        table_F = pd.DataFrame(stage_rows)
        print('Table F: all individual-seed errors')
        display(table_F)
        print('Mean individual-seed errors, not the error of a seed-mean curve')
        display(table_F.groupby('stage',sort=False)[['q_error_percent','sigma_error_percent']].mean())
    ''')
    code('w10-2-stage-profiles', '''
        SEED_TO_VIEW = 141001  # Repeat with 141002 and 141003.
        fig,axes = plt.subplots(2,2,figsize=(11,6),sharex=True)
        for j,field in enumerate(('q','sigma')):
            for i,stage in enumerate(('adam20000','ssb500','fixed500')):
                case = ks.load_case(f'm2_seed{SEED_TO_VIEW}_{stage}')
                method = next(m for m in case.methods if m!='reference')
                ref,pred = case.field('reference',field),case.field(method,field)
                if i==0: axes[0,j].plot(case.x,ref,'k-',label='Matched DVM',lw=2)
                axes[0,j].plot(case.x,pred,label=stage,ls=['--','-.',':'][i])
                axes[1,j].plot(case.x,pred-ref,ls=['--','-.',':'][i])
            axes[0,j].set_ylabel(FIELD_LABELS[field])
            axes[1,j].set(xlabel='x',ylabel='PINN - DVM')
            axes[1,j].axhline(0,color='0.5',lw=.6)
        for ax in axes.flat: ax.grid(alpha=.18)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
        fig.tight_layout(rect=(0,0,1,.91)); plt.show()
    ''')
    code('w10-2-steady-fluxes', '''
        case = ks.load_case(f'm2_seed{SEED_TO_VIEW}_fixed500')
        method = next(m for m in case.methods if m!='reference')
        upstream = ks.jump_states(2)['upstream']
        targets = ks.stationary_fluxes(upstream['rho'],upstream['u'],upstream['T'],0,0)
        fig,axes = plt.subplots(1,3,figsize=(12,3.6))
        for source,style in ((method,'--'),('reference','-')):
            fluxes = ks.stationary_fluxes(*(case.field(source,f) for f in ('rho','u','T','q','sigma')))
            for ax,key in zip(axes,('mass','momentum','energy')):
                ax.plot(case.x,100*(fluxes[key]/targets[key]-1),style,label=source)
                ax.set(xlabel='x',ylabel=key+' flux defect (%)')
                ax.grid(alpha=.18)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.88)); plt.show()
    ''')
    md('w10-2-supervision', r'''
        ## 8. Table G - Moment supervision and its reference-data roles

        **Figure conditions:** transient BGK tube, $t=0.02$,
        $x\in[-0.5,0.5]$, constant $\tau=\mathrm{Kn}=1$, $\Pr=1$,
        $R_g=1$, $\gamma=5/3$. The left/right $(\rho,u,T)$ states are
        $(1,0,1)$ and $(0.125,0,0.8)$ with the same $H(x)$ smoothing as
        Section 6. Both start at rest; no stationary upstream Mach is imposed.

        Compare the equation-trained BGK result, the five-field state S0 and
        the final eight-moment state against the **same 6400-cell DVM reference**
        on the declared common-overlap nodes. The paper's published comparison
        uses 401 nodes; this teaching extract excludes any endpoint outside the
        reference cell centres, rather than extrapolating. The metadata records
        the exact retained count and interpolation policy, so its recomputed
        values need not equal the paper's endpoint-inclusive last decimal.
        Table G also uses trapezoidal spatial weights; the corresponding paper
        table uses an unweighted norm on its 401 equally spaced nodes.
        The five-field loss constrains
        $\rho,u,T,q,\sigma$; the eight-moment loss also constrains $m,R,\Delta$.
        $S=R+\Delta/3$ is derived.

        Five-field S0 used a **400-cell** DVM for training observations, whereas
        the final eight-moment state used **6400-cell** targets, scales and some
        acceptance guards. The equation-trained case uses this 6400-cell reference
        for post-solve assessment. They share an evaluation reference here, not
        identical training information. Those roles prevent calling the whole
        comparison an independent held-out test. Points not explicitly observed can still
        be affected by selection using reference-derived guards. Read the source's
        admission flags: the final state retains failed physical checks and an
        attempt-cap stop. Report core and tails separately; small global L2 errors
        can hide wrong signs in small-amplitude tails.
    ''')
    code('w10-2-supervised-scores', '''
        supervised_ids = [c for c in ks.available_cases() if c.startswith('supervised')]
        assert len(supervised_ids)==1
        supervised = ks.load_case(supervised_ids[0])
        print('Table G: one declared reference/grid for every method')
        table_G = score_profiles(supervised).pivot(index='field',columns='method',values='E_percent')
        display(table_G)
        print('Case/data-role metadata:')
        display(supervised.metadata)
        plot_profiles(supervised)
    ''')
    code('w10-2-tail-profiles', '''
        # Compare signed quantities at the actual comparison nodes; no fitted shift.
        fig,axes = plt.subplots(1,3,figsize=(12,3.8))
        for ax,field,interval in zip(axes,('sigma','R','Delta'),((-0.5,-0.23),(0.19,0.5),(0.19,0.5))):
            mask = (supervised.x>=interval[0]) & (supervised.x<=interval[1])
            for method in ['reference']+selected_methods(supervised):
                ax.plot(supervised.x[mask],supervised.field(method,field)[mask],label=method,
                        lw=1.5,ls='-' if method=='reference' else '--')
            ax.axhline(0,color='0.45',lw=.6)
            ax.set(xlabel='x',ylabel=FIELD_LABELS[field]); ax.grid(alpha=.18)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.82)); plt.show()
        print('These are source-discussed tail regions, not newly selected independent test sets.')
        print('Elapsed time for the preceding audit sections (includes plotting/imports):',
              round(time.perf_counter()-START,2),'s')
    ''')
    md('w10-2-exercises', r'''
        ## 9. Submit a reproducible numerical investigation

        Keep a copy of the notebook and change only the stated controls. Include
        parameters, measured tables, physical profiles and an interpretation for
        every exercise. A neural method is allowed to lose.

        1. **Jump conditions - Table B.** Use $M=1.5,2,3$ separately. Report both
           endpoint states and the three flux differences. Deliberately omit
           $u\sigma$ when evaluating the saved Mach-2 interior, plot the resulting
           energy defect, then restore the correct formula. Explain why endpoints
           alone cannot reveal that mistake.
        2. **Moment information - Table C.** Recover $R_B=-2/3$, $\Delta_B=2$ and
           $S_A=S_B=0$. Repeat at orders 12, 24 and 64. Explain why supervising
           lower moments or $S$ cannot uniquely determine separate $R,\Delta$.
        3. **Quadrature and model physics - Table D.** At cutoff 8, compare orders
           12, 20 and 32; at order 32 compare cutoffs 4, 6 and 8. Report density/
           heat-flux errors, invariant defects, negative-target mass and runtime.
           Then change $\Pr$ from 1 to $2/3$ at fixed quadrature. Check the measured
           stress and heat-flux rates against their analytic laws.
        4. **Transient model comparison - Table E.** For each collision model,
           compare parent and final errors in all nine columns. Plot the signed
           $q,\sigma,R,\Delta$ discrepancies versus $x$. Identify the region of
           largest absolute error. Explain the significance of $t_f/\tau=0.02$;
           do not compare the two models against a swapped reference.
        5. **Optimization and seeds - Table F.** Inspect all three seeds. Quantify
           the mean change from SSB to split-half SSB for heat flux and stress.
           State the improvement and regression separately. Use the flux plot to
           explain why passing conservation does not imply an accurate profile.
        6. **Density thickness.** For one retained stationary Mach-2 stage, compute
           $\delta_\rho=(\rho_2-\rho_1)/\max|d\rho/dx|$ using `np.gradient(rho,x)`.
           Repeat on every second and fourth retained node, and on the matched
           DVM. Label this as sampling/derivative sensitivity, not a fresh DVM
           grid-refinement study. Never combine different Mach numbers in one plot.
        7. **Supervision and tails - Table G.** Compare all three methods against
           the same DVM, globally and over the stated tail intervals. Report
           relative L2 plus a signed discrepancy normalized by the full-reference
           maximum magnitude. Identify wrong signs and retained failed admission
           checks. List which reference information entered fitting, scaling and
           selection; removing observed positions is not enough to establish a
           new held-out test.

        **Assessment:** numerical setup and measured tables (35%), correct signed
        physical profiles (25%), interpretation of information/physics limits (30%),
        and reproducible environment/source identifiers (10%). No credit depends
        on beating a classical reference or making a failed gate pass.

        **Sources:** [paper v2](https://arxiv.org/abs/2605.29211v2),
        [pinned PINNs-DVM repository](https://github.com/Ehsan-Roohi/PINNs-DVM/tree/add5388f7da5f7eace95fff97d34d6ed58b90169),
        [extract provenance and reuse notice](../../data/week10_2_kinetic_shock/README.md).
    ''')
    from qa.week10_2_pinn_lesson import append_pinn_cells
    append_pinn_cells(md, code)
    doc = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'flowmllab_unit': '10.2'})
    if execute:
        from nbclient import NotebookClient
        from jupyter_client import AsyncKernelManager
        started = time.perf_counter()
        temporary_root = Path(os.environ.get('FLOWMLLAB_BUILD_TMP', str(ROOT/'build/week10_2')))
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='kernel-',dir=temporary_root) as runtime:
            manager = AsyncKernelManager(kernel_name='python3',connection_file=str(Path(runtime)/'connection.json'))
            manager.kernel_spec.argv = [sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
            NotebookClient(doc, timeout=300, km=manager, allow_errors=False,
                           resources={'metadata': {'path': str(ROOT)}}).execute(cleanup_kc=True)
        seconds = time.perf_counter() - started
        outputs = [item for cell in doc.cells if cell.cell_type == 'code'
                   for item in cell.get('outputs', [])]
        embedded_figures = sum('image/png' in item.get('data', {}) for item in outputs)
        if embedded_figures < 9:
            raise RuntimeError('Executed notebook must retain the original and fresh-training figures.')
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS/'execution.json').write_text(json.dumps({
            'mode': 'fresh CPU collision/moments, BGK PINN training and independent DVM; retained research rescoring',
            'jupyter_kernel_executed': True, 'seconds_in_build_environment': seconds,
            'code_cells': sum(c.cell_type=='code' for c in cells),
            'embedded_png_figures': embedded_figures,
            'python_version': sys.version.split()[0],
            'upstream_commit': ks.UPSTREAM_COMMIT,
            'new_shock_solve': True, 'new_neural_training': True,
            'DVM_used_in_training': False}, indent=2)+'\n', encoding='utf-8')
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(doc, BOOK)
    print('Wrote', BOOK.relative_to(ROOT), 'executed=', execute, flush=True)


def build_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    from PIL import Image as PILImage
    from qa.lecture_typography import register_reportlab, configure_matplotlib
    names = register_reportlab(prefix='KineticLectureTNR')
    configure_matplotlib()
    source = json.loads(LECTURE_SOURCE.read_text(encoding='utf-8'))
    width = A4[0] - 100
    styles = {
        'title': ParagraphStyle('title',fontName=names['bold'],fontSize=18,leading=21,spaceAfter=15,textColor=colors.black),
        'body': ParagraphStyle('body',fontName=names['regular'],fontSize=11,leading=14.5,spaceAfter=9,textColor=colors.black),
        'small': ParagraphStyle('small',fontName=names['regular'],fontSize=9.5,leading=12,spaceAfter=7,textColor=colors.black),
        'caption': ParagraphStyle('caption',fontName=names['italic'],fontSize=9.5,leading=12,spaceAfter=10,textColor=colors.black),
        'table': ParagraphStyle('table',fontName=names['regular'],fontSize=9.5,leading=11.8,textColor=colors.black),
    }
    story = []
    for number, page in enumerate(source['pages'], 1):
        flow = [Paragraph(f"{number}. {page['title']}", styles['title'])]
        for block in page['blocks']:
            kind = block['type']
            if kind == 'p':
                flow.append(Paragraph(block['text'], styles['body']))
            elif kind == 'list':
                for item in block['items']:
                    flow.append(Paragraph('• '+item, styles['body']))
            elif kind == 'equation':
                buffer = BytesIO()
                equation_figure = Figure(figsize=(1,1), dpi=200)
                FigureCanvasAgg(equation_figure)
                equation_figure.text(.5,.5,'$'+block['latex'].strip('$')+'$',
                    fontproperties=FontProperties(size=14),color='black',
                    horizontalalignment='center',verticalalignment='center')
                equation_figure.savefig(buffer,dpi=200,format='png',facecolor='white',
                                        bbox_inches='tight',pad_inches=.06)
                equation_figure.clear()
                buffer.seek(0)
                image = PILImage.open(buffer)
                w, h = image.size
                factor = min(72/200, width/w)
                buffer.seek(0)
                flow.extend([Spacer(1,3),Image(buffer,width=w*factor,height=h*factor),Spacer(1,8)])
                if block.get('explanation'):
                    flow.append(Paragraph(block['explanation'],styles['small']))
            elif kind == 'figure':
                path = ROOT/block['path']
                if not path.is_file(): raise FileNotFoundError(path)
                im = PILImage.open(path); w,h = im.size
                factor = min(width/w, 275/h)
                flow.append(Image(str(path),width=w*factor,height=h*factor))
                flow.append(Paragraph(block['caption'],styles['caption']))
            elif kind == 'table':
                rows = [[Paragraph('<b>'+str(v)+'</b>',styles['table']) for v in block['headers']]]
                rows.extend([[Paragraph(str(v),styles['table']) for v in row] for row in block['rows']])
                weights = block.get('widths',[1/len(rows[0])]*len(rows[0]))
                table = Table(rows,colWidths=[width*v/sum(weights) for v in weights],hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('TEXTCOLOR',(0,0),(-1,-1),colors.black),
                    ('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),
                    ('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),
                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                flow.extend([table,Spacer(1,12)])
            else:
                raise ValueError('Unsupported lecture block '+kind)
        available = A4[1]-115
        heights = sum(item.wrap(width,available)[1]+getattr(item,'spaceAfter',0)+getattr(item,'spaceBefore',0) for item in flow)
        if heights>available-10:
            raise ValueError(f'Lecture page {number} too dense: {heights:.1f} > {available-10:.1f} pt ({page["title"]})')
        story.extend(flow)
        if number<len(source['pages']): story.append(PageBreak())
    def decorate(canvas, doc):
        canvas.saveState();canvas.setFillColor(colors.black)
        canvas.setFont(names['regular'],9)
        canvas.drawString(50, A4[1]-30, 'FlowMLLab | Week 10.2 | Kinetic shocks')
        canvas.drawString(50,27,'Ehsan Roohi | Based on arXiv:2605.29211v2')
        canvas.drawRightString(A4[0]-50,27,str(doc.page))
        canvas.restoreState()
    SimpleDocTemplate(str(LECTURE),pagesize=A4,rightMargin=50,leftMargin=50,
                      topMargin=52,bottomMargin=53,title=source['title'],author='Ehsan Roohi',
                      allowSplitting=0).build(story,onFirstPage=decorate,onLaterPages=decorate)
    print('Wrote',LECTURE.relative_to(ROOT),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',action='store_true')
    parser.add_argument('--notebook',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--pdf',action='store_true')
    args = parser.parse_args()
    if not any((args.figures,args.notebook,args.pdf)): parser.error('Choose --figures, --notebook or --pdf')
    if args.figures: build_figures()
    if args.notebook: build_notebook(execute=args.execute)
    if args.pdf: build_pdf()


if __name__ == '__main__':
    main()
