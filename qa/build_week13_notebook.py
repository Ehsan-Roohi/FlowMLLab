#!/usr/bin/env python3
"""Build the Week 13 notebook in teaching order (learn, apply, audit).

Usage:
    python qa/build_week13_notebook.py            # write the notebook (unexecuted)
    python qa/build_week13_notebook.py --execute  # write and execute it in place (needs torch)

The notebook has three parts:

* Part A trains a small streamfunction PINN on CPU so that every step of the
  method (network, lifting, autodiff residual, loss, optimizer, held-out check)
  is visible and the student sees why a falling training loss is not proof of
  a solved flow;
* Part B inspects the retained Re = 1000, D/W = 2.2 deep-cavity field beside
  Nektar++ CFD;
* Part C audits the retained four-case A100 matrix (Re = 100/400, D = 1/2).

The retained evidence under results/ and data/ is read, never regenerated, except
for the figures that the notebook itself draws from that evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb"
COLAB = ("https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/"
         "notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb")


def cell(kind: str, source: str, cell_id: str) -> dict:
    source = source.strip("\n") + "\n"
    c = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


# --------------------------------------------------------------------------- code
BOOTSTRAP = r'''# FLOWMLLAB_COLAB_BOOTSTRAP_V1
# In Colab this cell obtains the complete public evidence tree. Locally it is a no-op.
from pathlib import Path as _FlowMLLabPath
import os as _flowmllab_os
import subprocess as _flowmllab_subprocess
import sys as _flowmllab_sys

if "google.colab" in _flowmllab_sys.modules or _flowmllab_os.environ.get("COLAB_RELEASE_TAG"):
    _flowmllab_root = _FlowMLLabPath("/content/FlowMLLab")
    if not (_flowmllab_root / ".git").is_dir():
        _flowmllab_subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(_flowmllab_root)],
            check=True,
        )
    _flowmllab_notebook_dir = _flowmllab_root / "notebooks/week13"
    _flowmllab_os.chdir(_flowmllab_notebook_dir)
    for _flowmllab_path in (_flowmllab_root, _flowmllab_notebook_dir):
        if str(_flowmllab_path) not in _flowmllab_sys.path:
            _flowmllab_sys.path.insert(0, str(_flowmllab_path))
    print("FlowMLLab evidence ready:", _flowmllab_root)
'''

SETUP = r'''from pathlib import Path
import hashlib, json
import matplotlib.pyplot as plt
import numpy as np

def find_root(start=Path.cwd()):
    for candidate in (start, *start.parents):
        if (candidate/'qa/WEEK13_PINN_MATRIX_PROTOCOL.md').is_file(): return candidate
    raise FileNotFoundError('Run inside a complete FlowMLLab checkout')

ROOT=find_root(); RESULT=ROOT/'results/week13_rectangular_pinn'
CASES=[(100,1),(400,1),(100,2),(400,2)]
print('FlowMLLab checkout located; evidence folder:', RESULT.relative_to(ROOT).as_posix())
'''

CPU_TRAIN = r'''import copy, time, torch
from torch import nn
torch.set_num_threads(1)
torch.manual_seed(13)
dtype=torch.float64
RE_CPU, D_CPU = 100., 1.          # square cavity, same Reynolds number as the Week 1 CFD reference
ADAM_STEPS, LBFGS_STEPS = 1000, 300

# 1. Network: (x, eta) -> (streamfunction correction, pressure)
cpu_pinn=nn.Sequential(nn.Linear(2,20),nn.Tanh(),nn.Linear(20,20),
                       nn.Tanh(),nn.Linear(20,2)).to(dtype=dtype)

def derivative(value, xy):
    # Automatic differentiation of a network output with respect to the coordinates.
    return torch.autograd.grad(value,xy,torch.ones_like(value),create_graph=True)[0]

# 2. Hard constraints ("lifting"): psi = psi_lid(x, eta) + (16 x(1-x) eta(1-eta))^2 * N(x, eta)
#    psi_lid carries the smoothed moving lid; the second factor vanishes quadratically at every wall,
#    so u = 0 on the three fixed walls and u = U_lid(x) on the lid hold exactly, whatever the weights are.
def cavity_fields(xy):
    raw=cpu_pinn(xy); x,e=xy[:,0:1],xy[:,1:2]
    lid=(1-torch.exp(-x*x/.001))*(1-torch.exp(-(1-x)**2/.001))          # U_lid(x): smoothed towards the corners
    psi=D_CPU*(e-1)*e**2*lid*torch.exp(-(1-e)**2/.01)                     # lifting term
    psi=psi+(16*x*(1-x)*e*(1-e))**2*raw[:,0:1]                            # learned correction, zero on all walls
    g=derivative(psi,xy)
    return g[:,1:2]/D_CPU,-g[:,0:1],raw[:,1:2],psi                        # u = psi_eta / D, v = -psi_x, p, psi

# 3. Residual of the steady momentum equations at arbitrary points (continuity is exact by construction)
def cavity_residual(points):
    xy=points.detach().clone().requires_grad_(True)
    u,v,p,_=cavity_fields(xy)
    gu,gv,gp=[derivative(z,xy) for z in (u,v,p)]
    def lap(g):
        return derivative(g[:,0:1],xy)[:,0:1]+derivative(g[:,1:2],xy)[:,1:2]/D_CPU**2
    rx=u*gu[:,0:1]+v*gu[:,1:2]/D_CPU+gp[:,0:1]-lap(gu)/RE_CPU
    ry=u*gv[:,0:1]+v*gv[:,1:2]/D_CPU+gp[:,1:2]/D_CPU-lap(gv)/RE_CPU
    return rx,ry,gu[:,0:1]+gv[:,1:2]/D_CPU

# 4. Collocation points (training) and an independent held-out set (never used by the optimizer)
train_xy=torch.rand(192,2,dtype=dtype)
heldout_xy=torch.rand(384,2,dtype=dtype)
x,e=train_xy[:,0:1],train_xy[:,1:2]
mask=(1-torch.exp(-(x*x+(1-e)**2)/.0001))*(1-torch.exp(-((1-x)**2+(1-e)**2)/.0001))   # down-weights the two lid corners
gauge=torch.tensor([[.5,.5]],dtype=dtype)

def training_loss():
    rx,ry,_=cavity_residual(train_xy)
    # Mean squared masked momentum residual + a pressure gauge (pressure is defined up to a constant).
    return ((mask*rx)**2+(mask*ry)**2).mean()+cpu_pinn(gauge)[0,1]**2

def heldout_rms():
    rx,ry,div=cavity_residual(heldout_xy)
    return float(torch.sqrt((rx**2+ry**2).mean()).detach()), float(div.detach().abs().max())

# 5. Stage 1: Adam (first-order, robust far from a minimum)
started=time.perf_counter()
optimizer=torch.optim.Adam(cpu_pinn.parameters(),lr=1e-3)
cpu_history=[]; best_loss=float('inf'); best_state=None
for step in range(ADAM_STEPS):
    optimizer.zero_grad()
    loss=training_loss()
    if loss.item()<best_loss:
        best_loss=loss.item(); best_state=copy.deepcopy(cpu_pinn.state_dict())
    loss.backward(); optimizer.step()
    cpu_history.append(loss.item())
cpu_pinn.load_state_dict(best_state)
adam_heldout,_=heldout_rms()

# 6. Stage 2: L-BFGS (quasi-Newton, like the SSBroyden2 stage of the research runs, from the best Adam state)
lbfgs=torch.optim.LBFGS(cpu_pinn.parameters(),max_iter=LBFGS_STEPS,history_size=50,
                        line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
def closure():
    lbfgs.zero_grad(); value=training_loss(); value.backward(); cpu_history.append(value.item()); return value
lbfgs.step(closure)
final_heldout,continuity_max=heldout_rms()
elapsed=time.perf_counter()-started

cpu_summary={'adam_initial_training_loss':cpu_history[0],'adam_best_training_loss':best_loss,
             'lbfgs_final_training_loss':cpu_history[-1],
             'heldout_residual_rms_after_adam':adam_heldout,'heldout_residual_rms_after_lbfgs':final_heldout,
             'continuity_max_heldout':continuity_max,'seconds':round(elapsed,1)}
print('Architecture:',cpu_pinn)
for key,value in cpu_summary.items(): print(f'{key:>36s}: {value:.4g}' if isinstance(value,float) else f'{key:>36s}: {value}')
assert np.isfinite(cpu_history).all() and best_loss<cpu_history[0]
assert continuity_max<1e-9      # continuity holds exactly because u, v come from one streamfunction
fig,ax=plt.subplots(figsize=(7,3.6)); ax.semilogy(cpu_history)
ax.axvline(ADAM_STEPS,color='#D55E00',lw=1); ax.text(ADAM_STEPS,max(cpu_history),'  L-BFGS starts',va='top')
ax.set(xlabel='optimizer step (Adam, then L-BFGS function evaluations)',ylabel='masked training loss'); ax.grid(alpha=.2); plt.show()
'''

CPU_EVAL = r'''# Evaluate the learned field on the 65 x 65 grid of the Week 1 reference and compare.
gx=np.linspace(0,1,65); ge=np.linspace(0,1,65); XX,EE=np.meshgrid(gx,ge)
xy=torch.tensor(np.c_[XX.ravel(),EE.ravel()],dtype=dtype,requires_grad=True)
uu,vv,pp,psi_net=cavity_fields(xy)
uu,vv,pp,psi_net=[q.detach().numpy().reshape(65,65) for q in (uu,vv,pp,psi_net)]

# Exact wall behaviour is checked independently of training quality.
expected=(1-np.exp(-gx**2/.001))*(1-np.exp(-(1-gx)**2/.001))
np.testing.assert_allclose(uu[-1],expected,atol=1e-10)
assert max(abs(vv).take([0,-1],axis=0).max(),abs(uu[0]).max(),abs(uu[:,[0,-1]]).max())<1e-9
print('Walls: lid profile reproduced exactly; zero velocity on the three fixed walls.')

# Week 1 finite-difference reference at the same Reynolds number (classical, unsmoothed lid).
reference=np.load(ROOT/'data/cavity_data.npz',allow_pickle=False)
k=int(np.argmin(abs(reference['Re']-RE_CPU))); uc,vc,psic=reference['u'][k],reference['v'][k],reference['psi'][k]
interior=(slice(1,-1),slice(1,-1))
velocity_error=np.sqrt(((uu-uc)[interior]**2+(vv-vc)[interior]**2).sum()/((uc[interior]**2+vc[interior]**2).sum()))
i_net,j_net=np.unravel_index(psi_net.argmin(),psi_net.shape); i_cfd,j_cfd=np.unravel_index(psic.argmin(),psic.shape)
print(f'Interior velocity relative L2 error against the Week 1 CFD reference: {100*velocity_error:.1f}%')
print(f'Primary vortex (streamfunction minimum): PINN at x={gx[j_net]:.3f}, y={ge[i_net]:.3f}; CFD at x={gx[j_cfd]:.3f}, y={ge[i_cfd]:.3f}')

fig,axs=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
im=axs[0].contourf(gx,D_CPU*ge,np.hypot(uu,vv),32,cmap='viridis')
axs[0].streamplot(gx,D_CPU*ge,uu,vv,color='white',density=.7,linewidth=.5)
fig.colorbar(im,ax=axs[0]); axs[0].set_title('Short CPU PINN: speed and streamlines')
im=axs[1].contourf(gx,D_CPU*ge,np.hypot(uc,vc),32,cmap='viridis')
axs[1].streamplot(gx,D_CPU*ge,uc,vc,color='white',density=.7,linewidth=.5)
fig.colorbar(im,ax=axs[1]); axs[1].set_title('Week 1 CFD reference (Re=100)')
im=axs[2].contourf(gx,D_CPU*ge,np.hypot(uu-uc,vv-vc),32,cmap='magma')
fig.colorbar(im,ax=axs[2]); axs[2].set_title('|velocity difference|')
for ax in axs: ax.set(xlabel='x/L',ylabel='y/L',aspect='equal')
plt.show()
'''

# --------------------------------------------------------------------------- markdown
INTRO = r'''# Week 13 - Physics-informed neural networks for the lid-driven cavity

<!-- MIE690A article-aligned validation v4 -->

[Open in Colab](COLAB_URL)

**Question.** Can a neural network solve the steady Navier-Stokes equations in a cavity without any CFD data, by
minimizing the residual of the equations, and how would you know that it did?

**What you will be able to do after this notebook**

1. Explain every component of a physics-informed neural network (PINN): the network, the hard boundary
   constraints, the automatically differentiated residual, the loss, and the two-stage optimizer.
2. Train a small streamfunction PINN on CPU and judge it with three independent checks: the training loss,
   a held-out residual, and a comparison with the Week 1 CFD reference.
3. Read a research-scale PINN result (Re = 1000 in a deep cavity, and a four-case matrix on an A100 GPU)
   with the same checks, and say which claims the evidence supports.

**Prerequisites:** Week 1 (streamfunction-vorticity cavity, Ghia validation), Week 2 (losses and optimizers),
the [PINN foundations reading](../../lectures/week04_2_pinn_cavity.pdf). Allow 75 minutes. CPU only;
the training cell in Part A takes about one minute.

**How the notebook is organized**

| Part | What happens | You run |
| --- | --- | --- |
| A. Learn | Build and train a small PINN for the square cavity at Re = 100; compare with CFD | a one-minute training cell |
| B. Apply | Inspect the retained Re = 1000, D/W = 2.2 deep-cavity PINN field beside Nektar++ CFD | data checks and plots |
| C. Audit | Read the retained four-case training matrix (Re = 100/400, D = 1/2) through its residual and CFD gates | evidence loaders and plots |

Parts B and C read retained results; they do not retrain the research models. The lecture is
[`lectures/week13_rectangular_cavity_pinn.pdf`](../../lectures/week13_rectangular_cavity_pinn.pdf).
'''

PART_A = r'''## Part A. What a PINN is, in the form used in this course

**Notation.** The cavity has width $L$ (reference length) and depth $H$; $D = H/L$ is the depth-to-width ratio
($D = 1$ is the square cavity). The lid moves with speed $U$ (reference velocity) and $Re = UL/\nu$. The network
works on the unit square $(x, \eta) \in [0,1]^2$ with $y/L = D\,\eta$, so that one architecture serves every depth;
every physical $y$-derivative therefore carries a factor $1/D$ ($\partial_y = D^{-1}\partial_\eta$).

**Streamfunction formulation.** Instead of predicting $u$ and $v$ separately, the network predicts a streamfunction
$\psi$ and a pressure $p$. Velocities follow from $u = \psi_\eta / D$ and $v = -\psi_x$, so continuity
$u_x + v_y = 0$ holds exactly and never needs to be learned. The equations left to satisfy are the two steady
momentum components,
$$r_x = u u_x + v u_y + p_x - Re^{-1}\nabla^2 u, \qquad r_y = u v_x + v v_y + p_y - Re^{-1}\nabla^2 v,$$
which are zero for the true flow. Every derivative in $r_x, r_y$ is computed by automatic differentiation of
the network output with respect to its inputs (the `derivative` function below), not by finite differences.

**Hard boundary constraints ("lifting").** The network output $N(x,\eta)$ is not used directly. We write
$$\psi = \psi_{lid}(x,\eta) + \big[16\,x(1-x)\,\eta(1-\eta)\big]^2 N(x,\eta),$$
where $\psi_{lid}$ is a fixed function that produces the moving lid, and the bracket vanishes quadratically on
all four walls. Consequently the wall velocities are exact for any weights: the optimizer only has to learn the
interior. The lid velocity is smoothed towards the two top corners over a width $\delta_x = \sqrt{10^{-3}} \approx 0.03$
(and in $\eta$ over $\delta_\eta = 0.1$); the classical cavity has a velocity jump there, which no smooth
network can represent, so this regularization is part of the problem definition and it is why later comparisons
with classical-lid CFD are called "near-matched".

**Collocation points and loss.** The residual is evaluated at $N_c$ random interior points (collocation points).
The loss is the mean of the squared, corner-masked residual plus a pressure gauge term (pressure is only defined
up to a constant). The mask $m = (1-e^{-d_1^2/r^2})(1-e^{-d_2^2/r^2})$ with $r = 0.01$ down-weights the two lid
corners, where the residual is dominated by the smoothed jump.

**Optimizer.** Stage 1 is Adam (first-order; robust when far from a minimum). Stage 2 is a quasi-Newton method
that uses an estimate of the inverse Hessian and a line search: L-BFGS here, SSBroyden2 in the research runs
(Part C). The second stage is where most of the residual reduction happens in PINN practice.

**Three checks, always.** (i) the training loss; (ii) the residual on an independent set of points the optimizer
never saw (the *held-out residual*); (iii) where a reference exists, the field error against it. Only (iii) speaks
about the flow; (i) and (ii) speak about the optimization. Keep this in mind while reading the numbers below.
'''

PART_A_EVAL = r'''### Evaluate the field and compare with the Week 1 CFD reference

The next cell evaluates the trained network on the same 65 x 65 grid as the Week 1 finite-difference reference at
Re = 100, checks the wall constraints, and reports (a) the interior velocity relative $L_2$ error and (b) the
position of the primary vortex (the streamfunction minimum) for both. The reference uses the classical lid
(velocity jump at the corners); the PINN uses the smoothed lid, so a few percent of difference near the top
corners is expected even for a converged PINN.
'''

PART_A_READING = r'''### What the short run shows

Read your printed numbers against these expectations (the retained execution had a training loss falling from
about 17 to about 0.2, a held-out residual that did **not** fall with it, and an interior velocity error of
roughly 50%; your values differ slightly with the PyTorch build):

* **The walls are exact and continuity is exact** (`continuity_max` at machine precision). These are properties
  of the representation, not achievements of training. A PINN paper that reports them as accuracy has reported nothing.
* **The training loss fell by almost two orders of magnitude**, and L-BFGS reduced it further than Adam did. On its
  own this is what a "successful" PINN training curve looks like.
* **The held-out residual did not follow.** With only 192 fixed collocation points, the optimizer can reduce the
  residual *at those points* without reducing it *between* them; the quasi-Newton stage, which fits the collocation
  set most aggressively, can even make the held-out residual worse. This is over-fitting of a differential equation.
* **The velocity error against CFD is large and the primary vortex sits in the wrong place.** After one minute the
  network produces a lid-driven recirculation of the right sense but the wrong shape and strength; compare the
  streamline patterns in the figure.

The remedies are known and you should try at least one: more collocation points, resampling the points every
step, a wider network, many more quasi-Newton iterations (the research runs in Part C use 4000 optimizer steps
in float64 on an A100 and still leave a measurable held-out residual), or curriculum in Reynolds number. What does
not change with any of them is the method of judgement: report the held-out residual and a field comparison next
to every training curve. Parts B and C apply exactly that discipline to research-scale runs.
'''

OPTIMIZER_NOTE = r'''### Why the optimizer and the residual convention matter

SSBroyden2 (used in Part C) is a self-scaled Broyden quasi-Newton method. With inverse-Hessian estimate $H_k$,
its search direction is $d_k = -H_k \nabla L_k$ and the update is $\theta_{k+1} = \theta_k + \alpha_k d_k$ with
a line search for $\alpha_k$; self-scaling rescales the inverse-Hessian update subject to the secant condition
$H_{k+1} y_k = s_k$. Full storage costs $O(P^2)$ for $P$ parameters, which is why L-BFGS (limited memory) is the
usual substitute on CPU. See [Urbán, Stefanou and Pons, JCP 523 (2025), 113656](https://arxiv.org/abs/2405.04230)
and [Kiyani et al.](https://arxiv.org/abs/2501.16371) for the self-scaling rules.

All residual diagnostics in Parts B and C use the vector root-mean-square over a point set $S$,
$R_S = \sqrt{\langle r_x^2 + r_y^2 \rangle_S}$; corner-history values that were stored as component RMS are
multiplied by $\sqrt{2}$ for display. Two physical facts shape the numbers you will see: a narrower lid transition
increases second velocity derivatives, and the viscous term scales as $1/Re$. Together they can make the Re = 100
residual *larger* than the Re = 400 residual even when the Re = 100 velocity field agrees *better* with CFD.
Residual magnitude and field error answer different questions.
'''

PART_B = r'''## Part B. A deep cavity at Re = 1000, D/W = 2.2

This retained author-model field was evaluated from checkpoint `restart-55118.ckpt` on a 301 x 661 grid
(the checkpoint's cluster path is recorded in `data/week13_deep_cavity/audit.json`). The export contains
$u$, $v$, $p$ and derived streamfunction and vorticity, but not the model weights and not a matching CFD field
from the same run. The next cell verifies the file hashes, checks the geometry and wall values, and rebuilds the
derived fields from $u$ and $v$ so that nothing in the plots depends on trusting the export. The short CPU model
of Part A and this network have different sizes and settings; one does not reproduce the other.
'''

PART_B_CHECKS = r'''### What these checks establish

The supplied field is finite, has the declared geometry and zero stationary-wall velocity, and its derived fields
reproduce the export. They do not certify convergence. The full-grid finite-difference divergence maximum is
about 4.3 and must remain visible alongside the regional diagnostics: a hard streamfunction formulation is
divergence-free under automatic differentiation, while an exported grid can show large finite-difference
divergence near sharp wall variations. Neither explanation alone proves that this checkpoint is accurate;
the matching CFD comparison below and the model's own residuals are the next checks. Full colour ranges are
retained, with explicitly labelled symmetric-log pressure and vorticity scales and no percentile clipping.
'''

PART_B_LOSS = r'''### The recovered training history is a separate provenance record

The Unity `training/data/loss.dat` was recovered together with a `resume.json` that identifies
`restart-65711.ckpt`: it is a resumed continuation, later than the field's checkpoint 55118, and its clock is
not total training from scratch. The 74 distinct logged steps (identical stage-boundary duplicates removed) show
the two momentum training MSEs and their sum. The original "test" columns repeated the training values and are
not treated as an independent test. Do not use this history as the field's matched residual audit, and remember
Part A: a small training loss establishes neither CFD agreement nor whole-domain convergence.
'''

PART_B_CASE_A = r'''### CFD beside PINN for the deep cavity

The two rows below are not two PINN renderings. The first is the retained Nektar++ CFD field at $t = 120$;
the second is PINN checkpoint 55118. Both are mapped to one grid, use the same speed and pressure scales, and
remove the same area-weighted pressure gauge. The reported velocity relative $L_2$ difference of about 3.6% is a
near-matched comparison: the lid profiles differ (`lid_max_difference` is of order one because the CFD lid is
classical and the PINN lid is smoothed), and the CFD archive does not establish mesh or long-time convergence.
'''

PART_C = r'''## Part C. The four-case research matrix as an audit

Four restartable float64 runs on an A100 (Re = 100 and 400, D = 1 and 2; Adam for 1000 steps, then SSBroyden2
to step 4000; exact streamfunction wall constraints; the same lifting and corner mask as Part A) were retained
with their optimizer histories, independent residual evaluations, wall checks and, for the two square cases,
frozen CFD gates. The protocol is `qa/WEEK13_PINN_MATRIX_PROTOCOL.md`. This part does not retrain anything; it
asks whether training loss, held-out residuals, corner behaviour, exact walls and CFD evidence support the same
conclusion.

**Physical-coordinate check.** The network sees $(x, \eta)$ on the unit square while physical $y/L = D\,\eta$, so
$\partial_y = D^{-1}\partial_\eta$, $u = D^{-1}\psi_\eta$, and every second $y$-derivative carries $D^{-2}$.
The $D = 1$ cases reduce to the square cavity and are the required implementation check.
'''

PART_C_LOSS = r'''### Training loss is not the verdict

Each panel plots the masked collocation residual beside the independent full-domain and top-corner residuals.
Adam occupies steps 1 to 1000 (left of the vertical line); SSBroyden2 continues from the same weights. This is
an optimizer trajectory, not a matched optimizer comparison: nothing here says that SSBroyden2 is better than
L-BFGS or Adam at equal cost.
'''

PART_C_TABLE = r'''### Independent evidence table

`R_full` and `R_corner` are the vector RMS residuals on an independent full-domain point set and on the
top-corner band; `wall max` is the largest wall-velocity error (exactly zero by construction). The square cases
are judged against frozen CFD gates (u and v centreline errors below 10% and 15%, interior velocity error below
15%, against the Week 1 reference). The deep cases report residuals only: `None` is the scientifically correct
entry when no matched field exists.
'''

PART_C_READING = r'''### Reading the table

* **The gap between training and held-out residuals is two to three orders of magnitude.** The masked training
  residual reaches about $5 \times 10^{-4}$ in all four cases, while the independent full-domain residual is
  0.05 to 0.5 and the corner band 0.3 to 3.4. This is the research-scale version of what Part A showed in one
  minute: the loss the optimizer sees is not the residual of the equations over the domain, mostly because the
  masked lid corners dominate the unmasked norm.
* **A large residual can coexist with a good field.** The Re = 100 square case has the largest full-domain
  residual (0.50) and passes every CFD gate with an interior velocity error of 3.3%; the Re = 400 square case has a
  ten-times smaller residual (0.047) and a larger field error (10.6%). The reasons were given above: the viscous
  term scales as $1/Re$ and the smoothed lid transition concentrates second derivatives near the corners, so the
  residual norm mostly measures the corner region while the CFD error measures the interior flow.
* **The deep cases are hypotheses.** With no matched CFD at $D = 2$, their residuals and their plausible topology
  are all the evidence there is. The 3.6% comparison of Part B is the closest thing to a field check for a deep
  cavity in this repository, and it is near-matched, not matched.
'''

PART_C_CASE_B = r'''### The square-case qualification figure

This figure comes from the earlier Re = 100, $D = 1$ qualification run (`results/week04_2_pinn_cavity`), which
preceded the four-case matrix and is a separate run from the `re100-d1` row above (its interior error was 3.10%;
the matrix run's is 3.34%). It retains the full PINN velocity fields, the pointwise PINN-minus-CFD velocity
error and two CFD/PINN centreline comparisons. It is a near-matched test because the PINN uses a smooth lid and
the CFD reference uses a classical lid. Frozen errors: 2.18% (u centreline), 4.42% (v centreline), 3.10%
(interior velocity).
'''

DECISION = r'''## Research decision and exercise

Answer before proposing any new run:

1. Did the held-out full-domain residual fall together with the masked training residual, in Part A and in each
   Part C case?
2. Did the corner band improve, stagnate or deteriorate during the quasi-Newton stage?
3. Which square cases pass every frozen CFD gate, and why can the case with the larger residual be the one with
   the smaller field error?
4. Which deep-case statements remain hypotheses because no matched raw field exists?
5. What factorial comparison would separate the effect of the representation (streamfunction versus primitive
   variables) from optimizer, budget and lid regularization?

**Exercise (Part A).** Change one thing in the training cell (collocation points 192 to 1024; resampling the
points every Adam step; L-BFGS iterations 300 to 1500; width 20 to 40) and report all three checks again.
State which check moved, by how much, and what that says about the change. Do not report the training loss alone.

The next paper-quality experiment must compare primitive/FOSLS and streamfunction representations at matched
physics, capacity and residual-evaluation budget, with several preregistered seeds. Failed seeds remain in the
denominator.

**References.** Raissi, Perdikaris and Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 378
(2019) 686-707. Służalec et al., J. Comput. Sci. 95 (2026) 102817 (primitive/FOSLS cavity PINNs). Ghia, Ghia
and Shin, J. Comput. Phys. 48 (1982) 387-411 (cavity benchmark). Urbán, Stefanou and Pons, J. Comput. Phys. 523
(2025) 113656 (self-scaled Broyden for PINNs).
'''


def build_cells() -> list[dict]:
    from w13_retained_cells import (CASE_A, CASE_B, DEEP_CHECKS, EVIDENCE_TABLE,  # noqa: PLC0415
                                    LOAD_MATRIX, LOSS_CONTINUATION, LOSS_CURVES)
    return [
        cell("markdown", INTRO.replace("COLAB_URL", COLAB), "w13-intro"),
        cell("code", BOOTSTRAP, "w13-bootstrap"),
        cell("code", SETUP, "w13-setup"),
        cell("markdown", PART_A, "w13-a-theory"),
        cell("code", CPU_TRAIN, "w13-a-train"),
        cell("markdown", PART_A_EVAL, "w13-a-eval-md"),
        cell("code", CPU_EVAL, "w13-a-eval"),
        cell("markdown", PART_A_READING, "w13-a-reading"),
        cell("markdown", OPTIMIZER_NOTE, "w13-a-optimizer"),
        cell("markdown", PART_B, "w13-b-intro"),
        cell("code", DEEP_CHECKS, "w13-b-checks"),
        cell("markdown", PART_B_CHECKS, "w13-b-checks-md"),
        cell("markdown", PART_B_LOSS, "w13-b-loss-md"),
        cell("code", LOSS_CONTINUATION, "w13-b-loss"),
        cell("markdown", PART_B_CASE_A, "w13-b-case-a-md"),
        cell("code", CASE_A, "w13-b-case-a"),
        cell("markdown", PART_C, "w13-c-intro"),
        cell("code", LOAD_MATRIX, "w13-c-load"),
        cell("markdown", PART_C_LOSS, "w13-c-loss-md"),
        cell("code", LOSS_CURVES, "w13-c-loss"),
        cell("markdown", PART_C_TABLE, "w13-c-table-md"),
        cell("code", EVIDENCE_TABLE, "w13-c-table"),
        cell("markdown", PART_C_READING, "w13-c-reading"),
        cell("markdown", PART_C_CASE_B, "w13-c-case-b-md"),
        cell("code", CASE_B, "w13-c-case-b"),
        cell("markdown", DECISION, "w13-decision"),
    ]


def build(target: Path = TARGET, execute: bool = False, keep_outputs: bool = False) -> Path:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    notebook = {"cells": build_cells(),
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3"}},
                "nbformat": 4, "nbformat_minor": 5}
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    if keep_outputs and target.exists():
        # Text-only rebuild: carry over executed outputs for code cells whose id and source are unchanged.
        previous = {c["id"]: c for c in json.loads(target.read_text(encoding="utf-8"))["cells"] if c.get("id")}
        for c in notebook["cells"]:
            old = previous.get(c["id"])
            if c["cell_type"] == "code" and old and old.get("cell_type") == "code" and old["source"] == c["source"]:
                c["outputs"] = old.get("outputs", [])
                c["execution_count"] = old.get("execution_count")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if execute:
        import nbformat
        from nbclient import NotebookClient
        nb = nbformat.read(target, as_version=4)
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(target.parent)}}).execute()
        nbformat.write(nb, target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="execute the notebook in place after writing it")
    parser.add_argument("--keep-outputs", action="store_true",
                        help="rebuild the text but keep existing outputs of unchanged code cells (no execution)")
    args = parser.parse_args()
    print(build(execute=args.execute, keep_outputs=args.keep_outputs).relative_to(ROOT))
