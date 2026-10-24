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
LECTURE_PAGES=30


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
    code('''cfg = CavityConfig(method="simple",cells=32)
n = cfg.cells
print("p shape:",(n,n),"u shape:",(n,n+1),"v shape:",(n+1,n))
L = pressure_matrix(np.ones((n,n-1)),np.ones((n-1,n)),1/n)
print("Constant-pressure nullspace defect:",np.max(np.abs(L @ np.ones(n*n))))
assert np.max(np.abs(L @ np.ones(n*n))) < 1e-10
''')
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
    checked=nbf.read(NOTE,4)
    improve_notebook(checked)
    nbf.write(checked,NOTE)


def lecture(manifest):
    """Build the authoritative mathematical lecture from its LaTeX source."""
    from qa.build_week01_2_lecture import build
    return build()


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
