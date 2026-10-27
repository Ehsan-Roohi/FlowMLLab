"""Add the reviewed student case study without rebuilding existing notebook cells."""
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'qa'))
from notebook_math import normalize_markdown

COEFFICIENTS = r'''### Reading the correction coefficient in an integrated equation

Let $V_e$ be the horizontal-velocity control-volume size and $a_e$ the momentum
diagonal **after** division by $V_e$. The integrated row therefore has diagonal
$V_e a_e$. With the dimensionless pressure used here, the local SIMPLE correction is

$$
(V_e a_e)u'_e=-V_e\frac{p'_E-p'_P}{h}.
$$

$p'_E$ and $p'_P$ are the pressure corrections in adjacent pressure cells;
$h$ is their center-to-center distance. Dividing the momentum row by $V_e$ gives

$$
a_eu'_e=-\frac{p'_E-p'_P}{h},\qquad
d_e=\frac{V_e}{V_e a_e}=\frac{1}{a_e},\qquad
u'_e=-d_e\frac{p'_E-p'_P}{h}.
$$

Thus “volume divided by diagonal” and the code's “reciprocal diagonal” agree
when applied to the corresponding integrated and normalized rows. For
dimensional pressure, acceleration uses $p_d/\rho$; $\rho$ is density. The
matrix $D$ in the lesson denotes divergence, while $d_e$ is this face's
correction sensitivity. Pressure under-relaxation changes the stored pressure
update; the conservative face-velocity correction is still applied fully.'''

OLD_TIME = r'''### Which values remain fixed during PIMPLE?

At time $t^{n+1}$, backward Euler uses the completed field $u_e^n$ from $t^n$.
Let $a_e^{(s,k)}$ be the spatial momentum diagonal at outer pass $k$ and
$b_e^{(s,k)}$ its known boundary/source contribution. The normalized predictor
row is

$$
\left(a_e^{(s,k)}+\frac{1}{\Delta t}\right)u_e^*
=\sum_{g\in\mathrm{NB}(e)}a_{eg}^{(k)}q_g^*
+b_e^{(s,k)}+\frac{u_e^n}{\Delta t}-\frac{p_E^k-p_P^k}{h}.
$$

$\mathrm{NB}(e)$ lists neighboring velocity locations; $q_g^*$ is the
corresponding unknown predictor velocity. A star labels the current predictor,
$k$ labels an outer estimate, and $n$ labels the completed physical time.
Every transported neighbor in this implicit row belongs to the predictor
level. Convection coefficients can change at the next outer pass; $u_e^n$
does not. Replace the old-time field only after accepting the physical step.
Mixing an old-time west or south neighbor into an otherwise implicit predictor
changes the equation; it is not another PISO correction.'''

CASE_STUDY = r'''## 11. Worked study: a converged solution can still need a finer grid

The student's October 2026 cavity report compares SIMPLE, PISO and PIMPLE
on 32 and 64 cells per side at $Re=400$ and 1000. The table below preserves his
reported measurements. Separate fresh calculations check selected cases with
the current solver; the two sets of timings are not interchangeable.

The report's momentum defects are below $10^{-6}$ and cell divergences below
$10^{-9}$. Nevertheless, at $Re=1000$ on 32 cells it reports 8.67% error in
$u$ and 14.15% in $v$. On 64 cells those decrease to 2.59% and 3.46%.
The chosen 5% profile target is a classroom criterion, not a universal standard.
For $Re=400$, the scored $v$ norm excludes the published anomaly at
$x/L=0.9063$; its original value is retained and marked in the plots.

**Read the profiles as well as the numbers.** At fixed grid and spatial
scheme, the three converged coupling methods approach the same discrete
steady problem. Refining the grid changes how the flow is represented. Two
grids show a refinement trend, not a verified order of accuracy.

The fresh suite includes all three methods on 32 cells for each Reynolds
number, SIMPLE on 64 cells for each Reynolds number, and matched central/upwind
SIMPLE on 32 cells at $Re=400$. The four 64-cell PISO/PIMPLE submissions are
not independently rerun here. Original student arrays, full configuration and
machine specifications were not attached, so exact source-run replication and
timing replication cannot be claimed.

[Reviewed explanations, corrections and provenance](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/docs/reviews/student_cavity_20261026.md)

**Exercises**

1. From the table, distinguish passing the equation-defect targets from passing
   both profile-error targets. Explain the difference physically.
2. Divide pressure solves by reported iterations. Explain why the ratios are
   1 for SIMPLE, 2 for PISO and 6 for PIMPLE in these runs.
3. Locate where the fine-grid velocity profiles differ from the coarse ones.
   Compare this change with switching the coarse-grid coupling method.
4. Compare central and upwind at the same grid, Reynolds number and method.
   Do the measured profiles support the numerical-diffusion explanation?
5. Design a time-step refinement study before claiming a transient advantage
   for PIMPLE. Extra coupling loops do not increase backward Euler's time order.

The following cell loads small retained tables and figures; it does not launch
new CFD. Recomputing the nine-case suite with
`python qa/review_student_cavity.py --run` may take several minutes or longer on
a CPU. Begin with the retained evidence before running finer grids.'''

DISPLAY_CODE = '''from IPython.display import display, Image
study = ROOT / "results/week01_2_student_review"
submitted = pd.read_csv(study / "submitted_summary.csv")
fresh = pd.read_csv(study / "fresh_summary.csv")
print("Student-reported central-convection measurements (12 cases):")
display(submitted)
print("Fresh author-side calculations (9 cases; independent timings):")
display(fresh)
keys = ["method", "reynolds", "cells", "convection"]
checked = submitted.merge(fresh, on=keys, how="inner")
checked["fresh u error percent"] = 100 * checked["ghia_u_relative_l2"]
checked["fresh v error percent"] = 100 * checked["ghia_v_relative_l2"]
display(checked[keys + ["reported_u_error_percent", "fresh u error percent",
                       "reported_v_error_percent", "fresh v error percent"]])
print("Shared cases compare rounded reported errors with fresh field-derived errors.")
for name in ("mesh_and_methods", "coupling_histories", "convection_profiles"):
    display(Image(filename=str(study / (name + ".png")), width=1100))
'''

VORTICITY_NOTE = r'''### Check each coordinate spacing independently

The central vorticity formula uses the spacing of the differentiated coordinate:

$$
\omega_{j,i}\approx
\frac{v_{j,i+1}-v_{j,i-1}}{2\Delta x}
-\frac{u_{j+1,i}-u_{j-1,i}}{2\Delta y}.
$$

Rows $j$ follow $y$ and columns $i$ follow $x$. A square grid with
$\Delta x=\Delta y$ can hide an incorrect denominator. Use a manufactured
streamfunction $\psi=x^2y^3$ to check the derivatives independently:

$$
u=\frac{\partial\psi}{\partial y}=3x^2y^2,\qquad
v=-\frac{\partial\psi}{\partial x}=-2xy^3,\qquad
\omega=-2y^3-6x^2y.
$$

These velocities satisfy continuity. At $(x,y)=(0.4,0.6)$ the exact vorticity
is $-1.008$. Check it using $\Delta x=0.1$ and $\Delta y=0.2$, then deliberately
put $\Delta x$ in the y-derivative denominator. Explain why that mistake is
invisible when both spacings are equal. The central formulas differentiate
these polynomial terms exactly up to round-off; general smooth fields still
have truncation error. This is a derivative check, not a cavity-flow simulation.

This exercise was developed while reviewing The student's foundations notes.'''

VORTICITY_CODE = '''import numpy as np
check_x, check_y = 0.4, 0.6
check_dx, check_dy = 0.1, 0.2
check_u = lambda x, y: 3 * x**2 * y**2
check_v = lambda x, y: -2 * x * y**3
check_dv_dx = (check_v(check_x + check_dx, check_y) - check_v(check_x - check_dx, check_y)) / (2 * check_dx)
check_du_dy = (check_u(check_x, check_y + check_dy) - check_u(check_x, check_y - check_dy)) / (2 * check_dy)
omega_central = check_dv_dx - check_du_dy
omega_exact = -2 * check_y**3 - 6 * check_x**2 * check_y
omega_wrong_spacing = check_dv_dx - (
    check_u(check_x, check_y + check_dy) - check_u(check_x, check_y - check_dy)
) / (2 * check_dx)
np.testing.assert_allclose(omega_central, omega_exact, atol=1e-12, rtol=0)
assert abs(omega_wrong_spacing - omega_exact) > 0.5
print(f"Analytic vorticity:                  {omega_exact:.6f}")
print(f"Central differences, correct dx/dy: {omega_central:.6f}")
print(f"Using dx for the y derivative:       {omega_wrong_spacing:.6f}")
'''


def main():
    path=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
    book=nbformat.read(path,4)
    old={c.id:c for c in book.cells}
    ids=['student-review-coefficient-normalization','student-review-old-time-level',
         'student-review-refinement-study','student-review-refinement-evidence']
    book.cells=[c for c in book.cells if c.id not in ids]
    for anchor,identifier,text in [('pressure-lesson-5',ids[0],COEFFICIENTS),
                                  ('pressure-lesson-10',ids[1],OLD_TIME)]:
        cell=nbformat.v4.new_markdown_cell(normalize_markdown(text),id=identifier)
        index=next(i for i,c in enumerate(book.cells) if c.id==anchor)
        book.cells.insert(index+1,cell)
    book.cells.append(nbformat.v4.new_markdown_cell(normalize_markdown(CASE_STUDY),id=ids[2]))
    cell=nbformat.v4.new_code_cell(DISPLAY_CODE,id=ids[3])
    if ids[3] in old and old[ids[3]].source==cell.source:
        cell.outputs=old[ids[3]].outputs;cell.execution_count=old[ids[3]].execution_count
    book.cells.append(cell)
    nbformat.validate(book);path.write_text(nbformat.writes(book),encoding='utf-8',newline='\n')
    path=ROOT/'notebooks/week01/03_cavity_ghia.ipynb'
    book=nbformat.read(path,4);old={c.id:c for c in book.cells}
    identifiers=['student-review-rectangular-vorticity-note','student-review-rectangular-vorticity-check']
    book.cells=[c for c in book.cells if c.id not in identifiers]
    index=next(i for i,c in enumerate(book.cells) if c.cell_type=='code' and c.source.startswith('def compute_velocity('))
    note=nbformat.v4.new_markdown_cell(normalize_markdown(VORTICITY_NOTE),id=identifiers[0])
    code=nbformat.v4.new_code_cell(VORTICITY_CODE,id=identifiers[1])
    if code.id in old and old[code.id].source==code.source:
        code.outputs=old[code.id].outputs;code.execution_count=old[code.id].execution_count
    book.cells[index+1:index+1]=[note,code]
    nbformat.validate(book);path.write_text(nbformat.writes(book),encoding='utf-8',newline='\n')
    print('Added notation explanations, a refinement case study, and an unequal-spacing derivative check.')


if __name__=='__main__':main()
