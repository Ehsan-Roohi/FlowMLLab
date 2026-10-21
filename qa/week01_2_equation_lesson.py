"""Shared scalar-to-matrix lesson for the Week 1.2 notebook and lecture."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
ALGEBRA_CODE=r"""# Reduced algebraic examples, not cavity-flow measurements.
D = np.array([[1., -1.]])
G = -D.T
b = np.array([2., 0.])

def inner_corrections(A, pressure_guess, count=2):
    d = np.diag(1. / np.diag(A))
    R = A - np.diag(np.diag(A))
    L = -D @ d @ G
    q = np.linalg.solve(A, b - G[:, 0] * pressure_guess)
    records = [{"stage": "predictor", "q1": q[0], "q2": q[1],
                "pressure": pressure_guess, "mass defect": abs((D @ q)[0]),
                "frozen momentum defect": np.max(np.abs(A @ q - b + G[:, 0] * pressure_guess))}]
    for inner in range(count):
        H = b - R @ q
        pressure = np.linalg.solve(L, -D @ (d @ H))[0]
        q = d @ (H - G[:, 0] * pressure)
        records.append({"stage": f"inner {inner+1}", "q1": q[0], "q2": q[1],
                        "pressure": pressure, "mass defect": abs((D @ q)[0]),
                        "frozen momentum defect": np.max(np.abs(A @ q - b + G[:, 0] * pressure))})
    return q, pressure, records

A = np.array([[4., 1.], [1., 3.]])
_, _, piso_example = inner_corrections(A, 0.)
display(pd.DataFrame(piso_example))

q = np.zeros(2)
pressure = 0.
pimple_example = []
for outer in range(3):
    frozen_s = q.mean()
    A = np.array([[4.+frozen_s, 1.], [1., 3.+frozen_s]])
    q, pressure, _ = inner_corrections(A, pressure, count=2)
    final_s = q.mean()
    A_final = np.array([[4.+final_s, 1.], [1., 3.+final_s]])
    nonlinear_defect = np.max(np.abs(A_final @ q - b + G[:, 0] * pressure))
    pimple_example.append({"outer": outer+1, "frozen s": frozen_s, "final q1": q[0],
                           "final q2": q[1], "pressure": pressure,
                           "nonlinear momentum defect": nonlinear_defect,
                           "mass defect": abs((D @ q)[0])})
display(pd.DataFrame(pimple_example))
exact_s = (np.sqrt(97.)-9.) / 4.
print("Exact nonlinear q1=q2:", exact_s, "; pressure:", exact_s/2.-1.)
"""

REFERENCES="""## 2.18 Further reading

- [Patankar (1980), Numerical Heat Transfer and Fluid Flow](https://www.routledge.com/Numerical-Heat-Transfer-and-Fluid-Flow/Patankar/p/book/9780891165224): staggered grids and SIMPLE.
- [Versteeg and Malalasekera (2007), An Introduction to Computational Fluid Dynamics](https://www.pearson.com/en-gb/subject-catalog/p/Versteeg-An-Introduction-to-Computational-Fluid-Dynamics-The-Finite-Volume-Method-2nd-Edition/P200000005670?view=educator): finite-volume transport and pressure correction.
- [Ferziger, Peric and Street (2020), Computational Methods for Fluid Dynamics](https://doi.org/10.1007/978-3-319-99693-6): incompressible-flow methods and errors.
- [Moukalled, Mangani and Darwish (2016), Chapter 15](https://doi.org/10.1007/978-3-319-16874-6_15): incompressible finite-volume coupling.
- [Issa (1986), original PISO paper](https://doi.org/10.1016/0021-9991(86)90099-9).
- [Official OpenFOAM algorithm controls](https://doc.cfd.direct/openfoam/user-guide-v13/fvsolution): pressure reference, inner and outer correctors.

The equations above specify this notebook's staggered-grid implementation. Collocated-grid interpolation and Rhie-Chow stabilization are separate topics.
"""

def lesson_cells():
    import nbformat as n
    from qa.week01_foundations_lesson import markdown
    tex=(ROOT/'lectures/source/week01_2_pressure_velocity_core.tex').read_text(encoding='utf-8')
    cells=[]
    for i,part in enumerate(tex.split('\\newpage')):
        if not part.strip():continue
        part=re.sub(r'\\includegraphics\[[^\]]*\]\{week01_2_assets/([^}]*)\.pdf\}',lambda m:'\n\n![Pressure-velocity grid and loops](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/'+m[1]+'.png)\n\n',part)
        text=markdown(part).replace(r'\small','')
        text=re.sub(r'^### [^:]+:',f'## 2.{i+1}',text,count=1)
        c=n.v4.new_markdown_cell(text);c.id=f'pressure-lesson-{i+1}';c.metadata['pressure_coupling_lesson']=True;cells.append(c)
        if 'PIMPLE: a nonlinear coefficient example' in part:
            c=n.v4.new_code_cell(ALGEBRA_CODE);c.id='pressure-algebra-examples';c.metadata['coupling_algebra_example']=True;cells.append(c)
    c=n.v4.new_markdown_cell(REFERENCES);c.id='pressure-lesson-references';c.metadata['pressure_coupling_lesson']=True;cells.append(c)
    return cells


def improve_notebook(nb):
    import nbformat as n
    cached_example=next((c for c in nb.cells if c.metadata.get('coupling_algebra_example')),None)
    old_prefixes=('## 2.', '## 3. SIMPLE','## 4. PISO','## 5. PIMPLE','## 5.1','## Worked example: two control volumes','## Algorithm flowcharts')
    nb.cells=[c for c in nb.cells if not c.metadata.get('pressure_coupling_lesson') and not c.metadata.get('coupling_algebra_example')
              and not(c.cell_type=='markdown' and c.source.lstrip().startswith(old_prefixes))]
    setup=next(c for c in nb.cells if c.cell_type=='code' and 'manifest, recorded = load_results' in c.source)
    summary=next(c for c in nb.cells if c.cell_type=='code' and c.source.startswith('summary = pd.DataFrame'))
    grid=next(c for c in nb.cells if c.cell_type=='code' and c.source.startswith('cfg = CavityConfig'))
    nb.cells=[c for c in nb.cells if c is not summary and c is not grid and not(c.cell_type=='markdown' and c.source.startswith(('## 1. Begin with evidence','## 3. Recorded cavity evidence')))]
    new=lesson_cells()
    if cached_example is not None:
        for cell in new:
            if cell.metadata.get('coupling_algebra_example') and cell.source==cached_example.source:
                cell.outputs=cached_example.outputs
                cell.execution_count=cached_example.execution_count
    # The operator demonstration follows the scalar and matrix definitions.
    index=next(i for i,c in enumerate(new) if c.id=='pressure-lesson-4')+1
    new[index:index]=[grid]
    new += [n.v4.new_markdown_cell('## 3. Recorded cavity evidence\n\nThe retained runs compare velocity profiles, fields, native equation defects and solver-call times. All four formulations are included. Matching spatial spacing does not make the finite-volume and finite-difference truncation errors identical.'),summary]
    new[-2].id='pressure-recorded-evidence'
    index=nb.cells.index(setup)+1;nb.cells[index:index]=new
    nb.cells[0].source=nb.cells[0].source.replace('This is an independent extension. The original Week-1 streamfunction-vorticity lab is unchanged.','The lesson connects finite-volume conservation equations to SIMPLE, PISO and PIMPLE, with algebraic examples and recorded cavity comparisons.').replace('**Qualified scope:** Re=100, 32 and 64 uniform cells per side, steady solutions. Startup time accuracy and larger-time-step benefits are future experiments.','**Scope:** Re=100 baseline and retained Re=400/1000 experiments; steady velocity comparisons, mesh refinement and within-step coupling. Transient trajectory accuracy requires a separate time-step study.')
    from qa.notebook_math import normalize_notebook
    normalize_notebook(nb)
    return nb
