"""Build the independent Week-1.2 notebook, lecture and measured figures."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import nbformat as nbf
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab.pressure_velocity_lab import load_results,figures,write_report,LABELS
from qa.week01_2_teaching import flowchart,experiment,experiment_figure
from qa.week01_2_math import Equation
OUT=ROOT/'results/week01_2_pressure_velocity'
NOTE=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
PDF=ROOT/'lectures/week01_2_pressure_velocity.pdf'
LECTURE_PAGES=25


def notebook():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s,hidden=False):
        c=nbf.v4.new_code_cell(s)
        if hidden:c.metadata={'jupyter':{'source_hidden':True}}
        cells.append(c)
    md('''# Week 1.2 - Pressure-Velocity Coupling
## SIMPLE, PISO and PIMPLE in the Re=100 lid-driven cavity

This is an independent extension. The original Week-1 streamfunction-vorticity lab is unchanged.

[Lecture PDF](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/lectures/week01_2_pressure_velocity.pdf) | [Executed report](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/week01_2_pressure_velocity/README.md) | [Algorithm/source notes](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/README.md)

**Learning outcomes:** derive pressure correction, distinguish linear/coupling/time iterations, preserve mass, compare benchmark errors and measured cost.

**Qualified scope:** Re=100, 32 and 64 uniform cells per side, steady solutions. Startup time accuracy and larger-time-step benefits are future experiments.''')
    code('''from pathlib import Path
import os, sys, subprocess
if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
    ROOT = Path("/content/FlowMLLab")
    if not (ROOT / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(ROOT)], check=True)
    subprocess.run([sys.executable,"-m","pip","install","-q","-e",str(ROOT)],check=True)
else:
    ROOT = next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/"flowmllab/pressure_velocity.py").exists())
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from flowmllab.pressure_velocity import CavityConfig, run_cavity, pressure_matrix, divergence
from flowmllab.pressure_velocity_lab import load_results, figures, LABELS
RESULTS = ROOT / "results/week01_2_pressure_velocity"
manifest, recorded = load_results(RESULTS)
print("All recorded field hashes verified.")
print("Recorded source SHA-256:", manifest["source_sha256"])
''',True)
    md('''## 1. Begin with evidence
Every selected algorithm appears below. No method is silently removed after a failure.
The recorded SIMPLE/PISO/PIMPLE runs use the same central FV momentum equations, mesh spacing and steady stopping criteria.
The streamfunction-vorticity result is an independent FD reference, with its own stopping norm.
Sparse LU is only the linear-equation solver; it is independent of SIMPLE/PISO/PIMPLE.''')
    code('''summary = pd.DataFrame([{
    "method": LABELS[r["config"]["method"]], "cells/side": r["config"]["cells"],
    "iterations/steps": r.get("iterations",r.get("steps")), "CPU seconds": r["runtime_seconds"],
    "Ghia u rel L2": r["ghia_u_relative_l2"], "Ghia v rel L2": r["ghia_v_relative_l2"],
    "FV momentum Linf": r.get("momentum_residual_linf",np.nan),
    "FV divergence Linf": r.get("continuity_linf",np.nan),
    "converged": r["converged"]} for r in recorded])
display(summary)
for name in ["centerlines","speed","p","omega","convergence","runtime","grid_errors"]:
    display(Image(filename=str(RESULTS / "figures" / (name+".png")),width=950))
''')
    md(r'''## 2. Shared equations and grid
$$\partial_t\mathbf{u}+\nabla\cdot(\mathbf{u}\otimes\mathbf{u})=-\nabla p+Re^{-1}\nabla^2\mathbf{u},\qquad\nabla\cdot\mathbf{u}=0.$$
Unit square, density 1, lid speed 1, viscosity 0.01; impermeable no-slip walls.
Pressure is stored at cell centers, u on vertical faces and v on horizontal faces (MAC).
The lid-corner velocity discontinuity is retained. Tangential wall diffusion uses half-cell distances.

$$Aq=b-Gp,\quad d=1/\operatorname{diag}(A),\quad H(q)=b-\operatorname{offdiag}(A)q,\quad L=-D d G.$$
Here D is conservative cell divergence and G is the face pressure gradient.
Because the cavity is closed, L has a constant nullspace. We pin one cell during the solve and report zero-mean pressure.
Adding a constant to pressure must leave the velocity unchanged.''')
    code('''cfg = CavityConfig(method="simple",cells=32)
n = cfg.cells
print("p shape:",(n,n),"u shape:",(n,n+1),"v shape:",(n+1,n))
L = pressure_matrix(np.ones((n,n-1)),np.ones((n-1,n)),1/n)
print("Constant-pressure nullspace defect:",np.max(np.abs(L @ np.ones(n*n))))
assert np.max(np.abs(L @ np.ones(n*n))) < 1e-10
''')
    md(r'''## 3. SIMPLE - steady pressure correction
1. Build and under-relax the steady momentum equations.
2. Solve momentum using the current pressure to obtain tentative face velocity q*.
3. Solve $L p'=-Dq^*$.
4. Correct velocity by the full correction $q=q^*-dGp'$.
5. Update $p\leftarrow p+\alpha_p p'$ and repeat.

Recorded relaxation: alpha_u=0.7 and alpha_p=0.3. The iteration counter is not physical time.
The update size alone is not a reliable stopping residual when relaxation changes.''')
    md(r'''## 4. PISO - more momentum/pressure corrections within a time step
1. Keep the preceding physical-time velocities immutable.
2. Solve backward-Euler momentum for a tentative velocity.
3. Recompute H using the current velocity, solve $Lp=-D(dH)$, and set $q=d(H-Gp)$.
4. Recompute the off-diagonal momentum contribution and repeat the pressure/velocity correction.

The retained runs use two corrections and dt=0.05. Repeated iterations of one linear Poisson solve are **not** extra PISO corrections.
Backward Euler is first order in time. These retained runs qualify a steady limit, not a transient trajectory.''')
    md(r'''## 5. PIMPLE - outer momentum loops plus PISO corrections
Within one physical time step, rebuild and solve momentum, then perform the PISO correction loop.
Repeat this outer loop three times, preserving the same old-time fields throughout.

One outer loop reproduces the implemented PISO limit exactly.
Additional outer loops cost more; they can improve nonlinear consistency within a step.
A larger stable dt does not establish accurate startup dynamics.''')
    md('''## Algorithm flowcharts and what the literature compares

The [expanded teaching report](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/COMPARISON_AND_TEACHING.md) explains the pressure algebra, actual counters, frozen coefficients and immutable old-time fields. It includes primary paper references, limits of higher-Re claims and a classroom experiment sequence.

### SIMPLE: steady iteration
![SIMPLE control flow](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/week01_2_pressure_velocity/flowcharts/simple.png)

### PISO: inner coupling corrections
![PISO control flow](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/week01_2_pressure_velocity/flowcharts/piso.png)

### PIMPLE: outer momentum loops and inner corrections
![PIMPLE control flow](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/week01_2_pressure_velocity/flowcharts/pimple.png)

### A measured advantage at Re=100
Fifteen startup configurations measure the full nonlinear backward-Euler momentum defect after one physical step. At dt=0.05, PISO with two corrections leaves about 1.11; PIMPLE with three outer/two inner loops leaves about 0.00343, at extra solve cost. All satisfy continuity to pressure-solve precision.

![Within-step coupling experiment](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/week01_2_pressure_velocity/figures/coupling_step_study.png)

Extra PISO corrections can improve the frozen matrix defect while plateauing in the full nonlinear defect. PIMPLE rebuilds that matrix. This is **algebraic coupling evidence**, not a physical time-error estimate or proof that larger dt is accurate. [Raw measurements](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/week01_2_pressure_velocity/coupling_step_study.json).

The released solver remains qualified at Re=100. Re=400/1000, a variable lid and a matched temporal-error study are proposed extensions. High-Re spatial under-resolution cannot be fixed solely by adding pressure corrections.''')
    md('''## 6. Fresh student run
The default notebook displays executed results quickly. Set RUN_NEW=True to actually compute one or all algorithms.
Fresh outputs are compared visibly with the retained evidence. For the exact full comparison, run `python qa/run_week01_2.py`.
Avoid ranking one method by its number of iterations: the pressure/momentum solve count and measured elapsed time differ.''')
    code('''RUN_NEW = False
METHOD = "all"  # "simple", "piso", "pimple", or "all"
CELLS = 32
if RUN_NEW:
    chosen = ["simple","piso","pimple"] if METHOD == "all" else [METHOD]
    fresh = [run_cavity(CavityConfig(method=m,cells=CELLS),progress=True) for m in chosen]
    display(pd.DataFrame([{k:r[k] for k in ["converged","iterations","runtime_seconds","momentum_residual_linf","continuity_linf","ghia_u_relative_l2","ghia_v_relative_l2"]} | {"method":r["config"]["method"]} for r in fresh]))
    # Use one fresh result per selected method, preserving other recorded controls.
    active = [r for r in recorded if r["config"]["cells"] != CELLS or r["config"]["method"] not in chosen] + fresh
    for name,fig in figures(active,cells=CELLS).items():
        display(fig); plt.close(fig)
else:
    print("All three algorithms are available; recorded runs are displayed above. Set RUN_NEW=True to recompute.")
''')
    md('''## 7. Interpret the comparison
- Ghia velocity error is distinct from momentum and continuity residuals.
- All three FV methods should approach the same discrete steady solution.
- Agreement with the independent streamfunction formulation is a useful cross-check, not proof that either code is exact.
- Pressure mean must match before differences are compared. Ghia's velocity tables do not validate pressure.
- Two grids show a refinement trend; they do not establish an asymptotic order.
- FV and FD residual definitions differ, so their curves are not presented as the same norm.
- The singular lid corners need consistent boundary treatment; no cosmetic smoothing is applied.

## 8. Exercises
1. Show interior face-flux cancellation in the domain mass budget.
2. Add a constant to pressure and confirm unchanged gradients and velocities.
3. Compare one and two PISO corrections using the physical-step momentum defect.
4. Verify PIMPLE with one outer loop matches PISO.
5. Change SIMPLE relaxation while keeping its final momentum tolerance fixed.
6. Compare upwind and central convection with the same coupling method.
7. Propose a dt-refinement experiment before claiming a PIMPLE speed advantage.

## Attribution
Independent implementation informed by [PySIMPLE](https://github.com/VishalKandala/PySIMPLE), [NIST FiPy's pressure-correction derivation](https://pages.nist.gov/fipy/en/latest/generated/examples.flow.stokesCavity.html), and [OpenFOAM pressure-velocity algorithms](https://www.openfoam.com/documentation/user-guide/6-solving/6.3-solution-and-algorithm-control). The [source notes](README.md) document the additional audited GitHub examples and their limitations. Ghia, Ghia and Shin (1982), DOI: 10.1016/0021-9991(82)90058-4.
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python','version':'3.12'},'course_module':'1.2','reynolds':100})
    from qa.week01_2_higher_re_notebook import append_higher_re
    append_higher_re(nb)
    from qa.week01_2_formulation_notebook import append_formulations
    append_formulations(nb)
    from qa.week01_2_equation_lesson import improve_notebook
    improve_notebook(nb)
    nbf.write(nb,NOTE)
    from qa.notebook_review_updates import review_notebook_file
    review_notebook_file(NOTE)


def lecture(manifest):
    from qa.lecture_typography import register_reportlab
    register_reportlab()
    pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNRB',italic='TNRI',boldItalic='TNRB')
    W,H=A4;width=W-100;story=[]
    styles={'body':ParagraphStyle('body',fontName='TNR',fontSize=11,leading=15,spaceAfter=9),
        'title':ParagraphStyle('title',fontName='TNRB',fontSize=21,leading=26,spaceAfter=16),
        'sub':ParagraphStyle('sub',fontName='TNRB',fontSize=13,leading=17,spaceBefore=10,spaceAfter=8),
        'small':ParagraphStyle('small',fontName='TNR',fontSize=9,leading=12,spaceAfter=6),
        'cover':ParagraphStyle('cover',fontName='TNRB',fontSize=27,leading=33,alignment=1,spaceAfter=18),
        'center':ParagraphStyle('center',fontName='TNRI',fontSize=13,leading=18,alignment=1,spaceAfter=14)}
    def p(s,style='body'):return Paragraph(s,styles[style])
    def add(s,style='body'):story.append(p(s,style))
    def heading(s):
        story.extend([PageBreak(),p(s,'title')])
    def box(label,text):
        t=Table([[p(label,'sub')],[p(text)]],colWidths=[width]);t.setStyle(TableStyle([
            ('BOX',(0,0),(-1,-1),.6,colors.black),('LEFTPADDING',(0,0),(-1,-1),10),
            ('RIGHTPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,-1),(-1,-1),7)]))
        story.extend([t,Spacer(1,12)])
    def eq(expression):
        story.append(Equation(expression,width))
    def mathbox(label,*expressions):
        rows=[[p(label,'sub')]]+[[Equation(s,width-24)] for s in expressions]
        t=Table(rows,colWidths=[width])
        t.setStyle(TableStyle([('BOX',(0,0),(-1,-1),.6,colors.black),
            ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
            ('BOTTOMPADDING',(0,-1),(-1,-1),7)]))
        story.extend([t,Spacer(1,12)])
    def graphic(path,maxheight=430):
        im=Image(str(path));r=min(width/im.imageWidth,maxheight/im.imageHeight)
        im.drawWidth=im.imageWidth*r;im.drawHeight=im.imageHeight*r;story.extend([im,Spacer(1,12)])
    def fig(name,maxheight=430):graphic(OUT/'figures'/f'{name}.png',maxheight)
    def diagram(method):graphic(OUT/'flowcharts'/f'{method}.png',545)
    story.append(Spacer(1,50));add('FlowMLLab - Week 1.2','center')
    story.append(HRFlowable(width=width,color=colors.black,thickness=.8,spaceAfter=20))
    add('Pressure-Velocity Coupling<br/>in a Lid-Driven Cavity','cover')
    add('SIMPLE, PISO and PIMPLE<br/>A shared finite-volume Python implementation','center')
    story.append(HRFlowable(width=width,color=colors.black,thickness=.8,spaceAfter=20))
    add('Ehsan Roohi<br/>Revised 20 October 2026','center')
    box('The central question','How does pressure enforce incompressibility, and how can different coupling loops reach the same flow with different computational costs?')
    add('Initial qualified case: Re=100. Two uniform grids. Executed fields, benchmark comparisons, momentum and continuity checks. An independent companion to the Week 1 streamfunction-vorticity formulation.')
    heading('1. One physical problem, two formulations')
    mathbox('Continuity and momentum: incompressible Navier–Stokes',
        r'\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0',
        r'\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}=-\frac{\partial p}{\partial x}+\frac{1}{\mathrm{Re}}\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}\right)',
        r'\frac{\partial v}{\partial t}+u\frac{\partial v}{\partial x}+v\frac{\partial v}{\partial y}=-\frac{\partial p}{\partial y}+\frac{1}{\mathrm{Re}}\left(\frac{\partial^2v}{\partial x^2}+\frac{\partial^2v}{\partial y^2}\right)')
    add('Here u and v are horizontal and vertical velocity components, p is pressure divided by density times lid speed squared, x and y are positions divided by cavity side L, and t is time divided by L/U. Re = UL/nu, with lid speed U and kinematic viscosity nu. Momentum balances local acceleration and convection against pressure force and viscous diffusion. Continuity and both momentum equations must be solved together.')
    add('Use a unit square, unit lid speed and density, and viscosity 0.01. All walls are impermeable; tangential velocities are zero except for the moving lid.')
    add('Week 1 solves streamfunction and vorticity using the relations below. Pressure is recovered separately from steady momentum. Week 1.2 solves velocity and pressure together.')
    eq(r'u=\frac{\partial\psi}{\partial y},\qquad v=-\frac{\partial\psi}{\partial x},\qquad\nabla^{2}\psi=-\omega')
    box('Why add this extension?','Students can inspect pressure as the continuity constraint, shared-face mass budgets, under-relaxation, and physical-time versus coupling iterations. The pressure-velocity formulation also provides a route toward three-dimensional solvers.')
    add('Learning outcomes: derive the pressure operator; distinguish the three loops; verify mass and gauge invariance; compare measured errors and cost; state the limits of a steady benchmark.')
    from qa.week01_2_equation_lesson import lecture_lesson
    lecture_lesson(heading, add, eq, box)
    for method in ['simple','piso','pimple']:
        heading(method.upper() + ': follow the control flow')
        diagram(method)
        add('Read the symbol definitions and loop indices in the preceding lesson before following the arrows. Old-time velocity remains fixed inside a physical step.','small')
    heading('6. Benchmark velocity and two-grid evidence')
    fig('centerlines',350)
    add('Ghia, Ghia and Shin (1982) velocity markers are compared at their stated coordinates, including wall endpoints. MAC face locations and FD nodes are interpolated only for the declared centerline diagnostics.')
    box('Fair resolution comparison','32/64 FV cells per side correspond to 33/65 FD nodes: compare h, not array length. The three FV algorithms use identical spatial equations. The independent FD formulation is a cross-check with a different truncation error.')
    add('The FD reference errors decrease on refinement; FV errors against Ghia increase slightly between these two grids. This nonmonotone trend is retained explicitly. Coarse-grid agreement with discrete benchmark data does not establish continuum accuracy. Further refinement and another reference are needed; no asymptotic order is claimed. Ghia velocity tables do not validate pressure.')
    heading('7. Physical fields and pressure interpretation')
    fig('p',450)
    add('All pressure panels use the same zero-mean convention and common color limits. The three converged FV fields agree closely; the FD recovered pressure can differ near the singular lid corners.')
    box('Pressure evidence','Comparing pressure between independent formulations is informative but is not an exact-pressure validation. The notebook also shows speed, streamlines and vorticity on shared physical scales.')
    heading('8. Convergence, cost and stopping conditions')
    fig('convergence',330)
    add('The plotted steady FV momentum defect has units U²/L, with U=L=1. Continuity is the maximum cell divergence. The elapsed-time axis includes actual numerical work, so differences in loop cost remain visible.')
    add('The FD reference uses a vorticity-update stopping norm and is not overlaid as if it were the same FV residual. A small under-relaxed update is not a substitute for a momentum-equation residual.')
    box('Measured conclusion','All six new FV runs reach the declared steady tolerance. On each grid, the three methods agree within 2×10<super>-6</super> in u, v and zero-mean p. Timings are one local CPU solve per method, not a universal speed ranking.')
    heading('9. What comparative studies actually show')
    add('Different studies answer different questions. None of these sources is a controlled comparison of our three Python implementations. There is no universal fastest coupling method.')
    box('Issa (1986): the transient motivation','PISO approximates the implicitly discretized equations through staged pressure/velocity corrections. Its companion study reports lower effort for time-evolving flows at sufficiently small splitting error. These publisher abstracts do not justify arbitrary large steps or universal steady superiority.')
    add('<link href="https://doi.org/10.1016/0021-9991(86)90099-9" color="black">JCP 62, 40-65</link>; <link href="https://doi.org/10.1016/0021-9991(86)90100-2" color="black">companion paper, 66-82</link>.','small')
    box('Pascau and Garcia (2010): ranking depends on the case','Their collocated-grid cavity study favored standard PISO among the tested variants at Re=1000. At Re=5000, a SIMPLEC-based variant performed better over a broader relaxation range. This is not a PIMPLE test and does not transfer as a MAC-grid ranking.')
    add('<link href="https://congress2.cimne.com/eccomas/proceedings/cfd2010/papers/01330.pdf" color="black">Enhancement of PISO Scheme in Collocated Grids, ECCOMAS CFD 2010</link>.','small')
    box('Brogi et al. (2024): coupling and time-step accuracy','A modern OpenFOAM precision study discusses PISO/PIMPLE time-step choices in a compressible shock tube. Extra coupling can make larger steps useful, but increasing dt still affects accuracy. This is not an incompressible-cavity speed comparison.')
    add('<link href="https://doi.org/10.1016/j.future.2023.10.006" color="black">On floating point precision in CFD using OpenFOAM, FGCS 152, 1-16</link>. The 2022 preprint is an earlier version with different settings.','small')
    add('Our assessment rule: compare measured cost at a specified field-error and equation-defect target; retain failures. An iteration count alone is insufficient.')
    heading('Recorded Re=100 time: include the independent formulation')
    fig('runtime',320)
    add('All four recorded methods appear in this timing figure. Streamfunction-vorticity has its own stopping criterion and reconstructs pressure afterward. The preceding FV convergence curves omit it because the vorticity-update norm is not a momentum residual; omission from that residual axis must not omit its measured runtime.')
    heading('10. Higher Reynolds number: separate the causes')
    add('There is no universal Reynolds-number limit of SIMPLE, PISO or PIMPLE. Increasing Re can change spatial resolution requirements, nonlinear coupling and the physical/model behavior simultaneously.')
    mathbox('A. Spatial convection and mesh resolution',r'a_E=\frac{\nu}{h^2}-\frac{u_E}{2h},\qquad\mathrm{Pe}_h=\frac{|u_E|h}{\nu}',r'\mathrm{Pe}_h\leq2\quad\Longrightarrow\quad a_E\geq0')
    add('The displayed bound is a sufficient positivity condition for this central neighbor coefficient, not a universal solver-failure threshold. Near unit advective speed, Re/N is about 15.6 for Re=1000 on 64 cells. Pressure corrections alone cannot resolve missing boundary-layer or vortex scales.')
    box('B. Nonlinear equation consistency','Reducing viscosity strengthens convection relative to diffusion. Relaxation, smaller physical steps or outer coefficient updates can improve discrete equation convergence. Measure these changes at the same spatial scheme and accuracy target.')
    box('C. Physical/model behavior','A real higher-Re cavity can develop unsteady or three-dimensional motion. A 2D steady solver does not establish that physical state. Nonconvergence alone does not prove an algorithm bug or the need for a turbulence model.')
    add('<link href="https://doi.org/10.1016/j.jcp.2004.12.024" color="black">Albensoeder and Kuhlmann (2005), Accurate three-dimensional lid-driven cavity flow</link>.','small')
    box('Implemented study boundary','The baseline CavityConfig remains restricted to Re=100. The separate experimental wrappers retain executed Re=400/1000 calculations, native steady-equation gates, independent velocity errors and resolution studies. They do not establish turbulent, three-dimensional or transient accuracy.')
    heading('Executed higher-Re comparison: all four formulations')
    graphic(ROOT/'results/week01_2_formulations/runtime.png',245)
    add('Re=400 and 1000, h=1/32. FD uses 33x33 nodes; FV uses 32x32 cells. These are sequential local solver-call timings, including FD pressure recovery, with different native stopping norms. Compare reference error before interpreting seconds. They are not a universal speed ranking.')
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results/week01_2_formulations').glob('*-re*-n32.json'))]
    rows=[['Re','Method','Seconds','u error (%)','v error (%)']]
    for r in sorted(records,key=lambda q:(q['config']['reynolds'],q['config']['method'])):
        rows.append([r['config']['reynolds'],r['config']['method'],f"{r['end_to_end_seconds']:.2f}",f"{100*r['ghia_u_relative_l2']:.2f}",f"{100*r['ghia_v_relative_l2']:.2f}"])
    table=Table([[p(str(value),'small') for value in row] for row in rows],colWidths=[35,175,65,80,80],repeatRows=1)
    table.setStyle(TableStyle([('LINEBELOW',(0,0),(-1,0),.6,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(table)
    add('The Re=400 all-point v error retains the isolated published table entry. A separately labeled sensitivity omits that one value without replacing it. FD and FV residual magnitudes must not be compared as the same error.','small')
    heading('Resolution can matter more than the coupling label')
    graphic(ROOT/'results/week01_2_formulations/centerlines.png',390)
    add('The coarse FD field may satisfy its discrete steady equation and still be inaccurate. Refining FD from h=1/32 through 1/64 to 1/128 substantially changes the velocity profiles. At Re=1000 the 128-cell FD errors are about 2.0% and 1.7%. These are velocity-reference comparisons, not a pressure validation. Retained first attempts document a failed steady gate and an exhausted step budget.')
    heading('Recirculation and mesh refinement')
    graphic(ROOT/'results/week01_2_formulations/refinement.png',425)
    add('Each panel states its mesh spacing. The sampled primary vortex and resolved corner recirculation become more reliable under refinement. Streamlines follow the computed velocity; they are not pressure or vorticity isolines. Finer or higher-Re cases can take longer and require more memory. Begin with the coarse case before selecting a laptop or Colab refinement run.')
    heading('11. A measured within-step PIMPLE advantage')
    fig('coupling_step_study',260)
    add('Fifteen configurations start from rest at Re=100 on 32 cells per side. Each takes one backward-Euler step. The full nonlinear step defect rebuilds convection from the final velocity, unlike the frozen PISO predictor defect. Each panel has its own logarithmic y range.')
    study=json.loads((OUT/'coupling_step_study.json').read_text())
    values=[r for r in study['rows'] if r['dt']==.05]
    table=[[p('dt=0.05 configuration','small'),p('Nonlinear Linf','small'),p('Pressure solves','small'),p('Momentum solves','small')]]
    for r in values:
        label=f'PISO {r["inner"]} inner' if r['method']=='piso' else f'PIMPLE {r["outer"]} x {r["inner"]}'
        table.append([p(label,'small'),p(f'{r["full_nonlinear_step_momentum_linf"]:.3e}','small'),p(str(r['pressure_solves']),'small'),p(str(r['momentum_system_solves']),'small')])
    t=Table(table,colWidths=[width*.4,width*.2,width*.2,width*.2]);t.setStyle(TableStyle([('LINEBELOW',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,-1),(-1,-1),.5,colors.black),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.extend([t,Spacer(1,12)])
    add('All fifteen cell-divergence defects are below 10<super>-9</super>. Extra inner corrections can plateau in the nonlinear defect because the predictor coefficients remain frozen. PIMPLE updates those coefficients, at higher solve cost.')
    box('What this establishes','A demonstrated reduction in within-step algebraic error. It is not a physical time-error estimate, proof of an accurate larger dt, or a universal wall-time advantage.')
    heading('12. Classroom experiment and assessment plan')
    box('A. Already executed: common steady state','Re=100, 32/64 cells. Show that all three FV methods reach the same discrete steady field. SIMPLE is cheaper on the coarse grid; PISO is cheaper with the selected fine-grid settings. Keep the nonmonotone FV Ghia errors visible.')
    box('B. Already executed: one-step consistency','Re=100, 32 cells; vary inner and outer counts at each fixed dt. Compare the full nonlinear momentum defect, frozen predictor defect, mass balance and solve counts. More pressure corrections and more coefficient updates address different errors.')
    box('C. Proposed next: temporal accuracy at Re=100','Compare startup at matched final times; halve dt. Then add a smooth ramped/oscillating lid and a verified temporal reference. Compare field error, energy, amplitude/phase and cost. Use the same time scheme. Transient SIMPLE needs an outer solve inside every physical step; the current steady iteration is not a trajectory.')
    box('D. Proposed later: higher-Re qualification','Test Re=400, then 1000 with refined 64/128/256 meshes and independently validated convection. Inspect secondary vortices, centerlines and lid-wall stress. Separate spatial errors from coupling errors and physical unsteadiness.')
    add('Assess cost versus error, not a preferred winner. Keep the mesh, linear solver, initial fields and spatial equations fixed for a coupling comparison. Repeat wall-time measurements and retain failed configurations. A dt-refined numerical reference has its own error; document its refinement.')
    heading('13. Exercises and the next experiment')
    for s in ['1. Sum cell divergence and cancel shared interior face terms.','2. Add a pressure constant; verify unchanged velocity.','3. Compare one and two PISO corrections using the one-step momentum defect.','4. Check PIMPLE with one outer loop against PISO.','5. Change SIMPLE relaxation at fixed final residual tolerance.','6. Change only convection interpolation; separate spatial error from coupling effects.','7. Design a Δt-halving startup experiment before claiming a PIMPLE speedup.']:add(s)
    box('Next qualification','Start with a ramped or oscillating lid. Use the same temporal scheme and matched final times. Refine Δt; measure field and phase errors. Any transient streamfunction pressure reconstruction must include acceleration terms.')
    add('Only Re=100 steady behavior is qualified in this release. Higher-Re behavior, transient accuracy and stretched grids remain separate tasks.')
    heading('14. Reading, attribution and reproducibility')
    for title,url in [
        ('PySIMPLE: educational MAC finite-volume SIMPLE','https://github.com/VishalKandala/PySIMPLE'),
        ('NIST FiPy: pressure-correction derivation (Stokes example)','https://pages.nist.gov/fipy/en/latest/generated/examples.flow.stokesCavity.html'),
        ('OpenFOAM: solution and algorithm control','https://www.openfoam.com/documentation/user-guide/6-solving/6.3-solution-and-algorithm-control'),
        ('pyOpenFOAM: another inspectable Python loop organization','https://github.com/alanZee/pyOpenFOAM'),
        ('Ghia, Ghia and Shin (1982): benchmark tables','https://doi.org/10.1016/0021-9991(82)90058-4')]:
        add(f'<link href="{url}" color="black">{title}</link>')
    add('This independently written module uses original implementation and explanations. External examples inform derivation and organization; their validation claims are not inherited. The audited CFD-Python PISO-titled notebook had a commented-out second correction, illustrating why source inspection matters.')
    box('Reproduce and audit','qa/run_week01_2.py executes the steady cases; qa/week01_2_teaching.py executes the one-step sweep and draws flowcharts; qa/build_week01_2.py builds the figures and notebook. tests/test_pressure_velocity.py checks conservation, pressure gauge, PISO behavior, PIMPLE limits and the common steady solution. The expanded teaching supplement provides full source links and study boundaries.')
    add('The numerical reference data, solver settings, retained failures and reproduction commands accompany the notebook and report.','small')
    heading('Worked example: a conservative two-cell correction')
    add('Use two unit-volume cells, impermeable external faces and one shared face velocity q, positive to the right. Unit spacing and d=1 isolate the signs from mesh factors.')
    eq(r'D_1=1,\quad D_2=-1,\quad G_1=-1,\quad G_2=1')
    add('For q*=1 the two cell defects are +1 and -1. Their sum vanishes because the shared face enters with opposite signs; neither cell yet satisfies continuity.')
    eq(r'L=-DdG,\qquad Lp^{\prime}=-Dq^{*}')
    add('Here L has rows (1,-1) and (-1,1). Pin p2\'=0. The first row gives p1\'=-1; then Gp\'=1 and q_new=1-1=0. Adding the same constant to both pressures changes no face gradient. This is the constant pressure nullspace.')
    box('Exercise with solution','Change q*=2 and d=0.5. L becomes half the previous matrix. With p2\'=0, p1\'=-4, Gp\'=4 and q_new=2-0.5(4)=0. Applying only alpha_p of the face correction would leave a cell mass defect. In SIMPLE, alpha_p relaxes the pressure update, while the velocity correction remains conservative.')
    add('This is a nondimensional sign-and-conservation control, not a resolved cavity. Before using the sparse solver, write the same cancellation for three cells and verify D times a constant internal through-flux at the impermeable ends.')
    def footer(c,doc):
        c.saveState();c.setFont('TNR',9);c.setLineWidth(.4)
        if doc.page>1:
            c.drawString(50,H-28,'FlowMLLab - Week 1.2');c.drawRightString(W-50,H-28,'Pressure-Velocity Coupling');c.line(50,H-36,W-50,H-36)
        c.line(50,40,W-50,40);c.drawString(50,26,'Ehsan Roohi | Cavity flow');c.drawCentredString(W/2,26,str(doc.page));c.restoreState()
    temp=ROOT/'output/pdf';temp.mkdir(parents=True,exist_ok=True)
    target=temp/PDF.name
    SimpleDocTemplate(str(target),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=58,bottomMargin=52,
        title='FlowMLLab Week 1.2 - Pressure-Velocity Coupling',author='Ehsan Roohi').build(story,onFirstPage=footer,onLaterPages=footer)
    shutil.copy2(target,PDF)
    pages=len(PdfReader(PDF).pages)
    assert pages == LECTURE_PAGES,('Unexpected lecture layout',pages)
    return pages


def main():
    manifest,rows=load_results(OUT)
    target=OUT/'figures';target.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    for name,fig in figures(rows).items():
        fig.savefig(target/(name+'.png'),dpi=180,facecolor='white');plt.close(fig)
    study_path=OUT/'coupling_step_study.json'
    study=json.loads(study_path.read_text()) if study_path.exists() else experiment()
    assert study['source_sha256']==hashlib.sha256((ROOT/'flowmllab/pressure_velocity.py').read_bytes()).hexdigest()
    for method in ['simple','piso','pimple']:flowchart(method)
    experiment_figure(study)
    write_report(OUT,manifest,rows)
    notebook();pages=lecture(manifest)
    print(json.dumps({'physical_figures':8,'flowcharts':3,'lecture_pages':pages,'notebook':str(NOTE)},indent=2))


if __name__=='__main__':main()
