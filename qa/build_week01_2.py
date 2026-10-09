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
OUT=ROOT/'results/week01_2_pressure_velocity'
NOTE=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
PDF=ROOT/'lectures/week01_2_pressure_velocity.pdf'


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
for name in ["centerlines","speed","p","omega","convergence","grid_errors"]:
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
    nbf.write(nb,NOTE)


def lecture(manifest):
    for name,file in [('TNR','times.ttf'),('TNRB','timesbd.ttf'),('TNRI','timesi.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
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
    def fig(name,maxheight=430):
        im=Image(str(OUT/'figures'/f'{name}.png'));r=min(width/im.imageWidth,maxheight/im.imageHeight)
        im.drawWidth=im.imageWidth*r;im.drawHeight=im.imageHeight*r;story.extend([im,Spacer(1,12)])
    story.append(Spacer(1,50));add('FlowMLLab - Week 1.2','center')
    story.append(HRFlowable(width=width,color=colors.black,thickness=.8,spaceAfter=20))
    add('Pressure-Velocity Coupling<br/>in a Lid-Driven Cavity','cover')
    add('SIMPLE, PISO and PIMPLE<br/>A shared finite-volume Python implementation','center')
    story.append(HRFlowable(width=width,color=colors.black,thickness=.8,spaceAfter=20))
    add('Ehsan Roohi<br/>09 October 2026','center')
    box('The central question','How does pressure enforce incompressibility, and how can different coupling loops reach the same flow with different computational costs?')
    add('Initial qualified case: Re=100. Two uniform grids. Executed fields, benchmark comparisons, momentum and continuity checks. An independent extension; Week 1 remains unchanged.')
    heading('1. One physical problem, two formulations')
    box('Governing equations','∂u/∂t + div(u u<super>T</super>) = -grad(p) + Re<super>-1</super>Laplacian(u)<br/>div(u) = 0')
    add('Use a unit square, unit lid speed and density, and viscosity 0.01. All walls are impermeable; tangential velocities are zero except for the moving lid.')
    add('Week 1 solves streamfunction and vorticity: u=∂ψ/∂y, v=-∂ψ/∂x and Laplacian(ψ)=-ω. Pressure is recovered separately from steady momentum. Week 1.2 solves velocity and pressure together.')
    box('Why add this extension?','Students can inspect pressure as the continuity constraint, shared-face mass budgets, under-relaxation, and physical-time versus coupling iterations. The pressure-velocity formulation also provides a route toward three-dimensional solvers.')
    add('Learning outcomes: derive the pressure operator; distinguish the three loops; verify mass and gauge invariance; compare measured errors and cost; state the limits of a steady benchmark.')
    heading('2. Shared MAC finite-volume operators')
    add('Pressure p is stored at cell centers. Horizontal velocity u lies on vertical faces; vertical velocity v lies on horizontal faces. Arrays use y-first, x-second indexing.')
    box('Momentum and pressure algebra','Aq = b-Gp; d = 1/diag(A)<br/>H(q) = b-offdiag(A)q; q = d(H-Gp)<br/>L = -D d G; Lp = -D(dH)')
    add('D computes the net velocity flux divided by cell volume. A shared face contributes with opposite signs to its two neighbors. This gives exact interior flux cancellation.')
    add('Normal wall velocities are zero. Tangential wall diffusion uses half-cell distances. The lid-corner discontinuity is retained. Central convection and diffusion use the same frozen advecting face velocities for every coupling method.')
    box('Gauge and compatibility','L has a constant nullspace. The closed-cavity RHS must sum to zero. Pin one cell during the sparse solve; shift reported pressure to zero mean. A pressure constant cannot change any velocity correction.')
    add('Sparse LU is a linear solver, independent of the pressure-coupling algorithm. No multigrid or packaged CFD solver is used.')
    heading('3. SIMPLE: steady correction and relaxation')
    for s in ['1. Build and under-relax steady momentum using the current velocities.','2. Predict face velocities from the current pressure.','3. Solve Lp′=-Dq*.','4. Correct velocity fully: q=q*-dGp′.','5. Update pressure p←p+α<sub>p</sub>p′; repeat until momentum and continuity pass.']:add(s)
    box('Recorded settings','α<sub>u</sub>=0.7; α<sub>p</sub>=0.3. Steady momentum Linf &lt; 10<super>-7</super>; continuity uses the same declared tolerance.')
    add('<b>Advantage:</b> transparent steady pressure coupling and useful relaxation experiments.<br/><b>Limitation:</b> sensitivity to relaxation; iteration count is not physical time or a fair cost measure.')
    box('Worked mass correction','If a tentative field has a nonzero cell divergence, pressure correction changes shared face fluxes until the corrected net flux vanishes. Independent velocity clipping would generally destroy that conservation property.')
    heading('4. PISO: corrections inside a physical step')
    for s in ['1. Keep old-time velocities fixed.','2. Assemble and solve backward-Euler momentum for tentative velocity.','3. Recompute H from the current velocity; solve Lp=-D(dH).','4. Correct q=d(H-Gp).','5. Re-evaluate off-diagonal momentum contributions and repeat the pressure/velocity correction.']:add(s)
    box('Recorded settings','Δt=0.05; two pressure corrections per physical step. The old-time source remains fixed within both corrections.')
    add('<b>Advantage:</b> additional pressure-momentum consistency without a complete nonlinear outer loop.<br/><b>Limitation:</b> more pressure solves per step; temporal accuracy still depends on Δt and the time discretization.')
    box('Do not mislabel a solver','Two Jacobi/LU iterations on a single pressure system are not two PISO corrections. The extra correction must update the momentum coupling using the corrected velocity.')
    add('Backward Euler has first-order temporal accuracy. Marching to steady state does not validate startup trajectories.')
    heading('5. PIMPLE: outer momentum iteration plus PISO')
    add('Inside one physical time step, repeat momentum assembly and prediction, then the PISO correction loop. The new velocity changes nonlinear convection coefficients for the next outer loop.')
    box('Recorded settings','Three outer loops; two pressure corrections in each; Δt=0.05. Old-time velocities remain immutable across all six pressure corrections.')
    add('<b>Advantage:</b> additional nonlinear consistency within each time step; useful when a single PISO pass is insufficient.<br/><b>Limitation:</b> repeated momentum and pressure solves increase cost. Larger stable steps do not guarantee accurate transient dynamics.')
    box('An exact implementation check','Set the number of outer loops to one. With identical settings, the resulting PIMPLE step must match this PISO step exactly.')
    add('For this simple steady Re=100 cavity, extra outer loops need not be economical. A time-dependent lid and a matched temporal-error target are better tests of potential PIMPLE benefits.')
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
    heading('9. Exercises and the next experiment')
    for s in ['1. Sum cell divergence and cancel shared interior face terms.','2. Add a pressure constant; verify unchanged velocity.','3. Compare one and two PISO corrections using the one-step momentum defect.','4. Check PIMPLE with one outer loop against PISO.','5. Change SIMPLE relaxation at fixed final residual tolerance.','6. Change only convection interpolation; separate spatial error from coupling effects.','7. Design a Δt-halving startup experiment before claiming a PIMPLE speedup.']:add(s)
    box('Next qualification','Start with a ramped or oscillating lid. Use the same temporal scheme and matched final times. Refine Δt; measure field and phase errors. Any transient streamfunction pressure reconstruction must include acceleration terms.')
    add('Only Re=100 steady behavior is qualified in this release. Higher-Re behavior, transient accuracy and stretched grids remain separate tasks.')
    heading('10. Reading, attribution and reproducibility')
    for title,url in [
        ('PySIMPLE: educational MAC finite-volume SIMPLE','https://github.com/VishalKandala/PySIMPLE'),
        ('NIST FiPy: pressure-correction derivation (Stokes example)','https://pages.nist.gov/fipy/en/latest/generated/examples.flow.stokesCavity.html'),
        ('OpenFOAM: solution and algorithm control','https://www.openfoam.com/documentation/user-guide/6-solving/6.3-solution-and-algorithm-control'),
        ('pyOpenFOAM: another inspectable Python loop organization','https://github.com/alanZee/pyOpenFOAM'),
        ('Ghia, Ghia and Shin (1982): benchmark tables','https://doi.org/10.1016/0021-9991(82)90058-4')]:
        add(f'<link href="{url}" color="black">{title}</link>')
    add('This independently written module uses original implementation and explanations. External examples inform derivation and organization; their validation claims are not inherited. The audited CFD-Python PISO-titled notebook had a commented-out second correction, illustrating why source inspection matters.')
    box('Reproduce and audit','qa/run_week01_2.py executes all cases; qa/build_week01_2.py builds figures and the notebook. tests/test_pressure_velocity.py checks flux cancellation, pressure nullspace, PISO correction behavior, PIMPLE limits and the common steady solution. Results retain source and field hashes and runtime versions.')
    add('Lecture and code are English. Typography: Times New Roman. Prepared 09 October 2026.','small')
    def footer(c,doc):
        c.saveState();c.setFont('TNR',9);c.setLineWidth(.4)
        if doc.page>1:
            c.drawString(50,H-28,'FlowMLLab - Week 1.2');c.drawRightString(W-50,H-28,'Pressure-Velocity Coupling');c.line(50,H-36,W-50,H-36)
        c.line(50,40,W-50,40);c.drawString(50,26,'Ehsan Roohi | Re=100');c.drawCentredString(W/2,26,str(doc.page));c.restoreState()
    temp=ROOT/'output/pdf';temp.mkdir(parents=True,exist_ok=True)
    target=temp/PDF.name
    SimpleDocTemplate(str(target),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=58,bottomMargin=52,
        title='FlowMLLab Week 1.2 - Pressure-Velocity Coupling',author='Ehsan Roohi').build(story,onFirstPage=footer,onLaterPages=footer)
    shutil.copy2(target,PDF)
    pages=len(PdfReader(PDF).pages)
    assert pages==11,('Lecture overflow',pages)
    return pages


def main():
    manifest,rows=load_results(OUT)
    target=OUT/'figures';target.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    for name,fig in figures(rows).items():
        fig.savefig(target/(name+'.png'),dpi=180,facecolor='white');plt.close(fig)
    write_report(OUT,manifest,rows)
    notebook();pages=lecture(manifest)
    print(json.dumps({'figures':6,'lecture_pages':pages,'notebook':str(NOTE)},indent=2))


if __name__=='__main__':main()
