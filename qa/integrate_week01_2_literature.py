"""Add the literature-driven exercises without replacing earlier lesson cells."""
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nbformat
from qa.refresh_week01_2_figures import CAPTIONS


def main():
    path=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
    book=nbformat.read(path,4)
    def md(id,text):return nbformat.v4.new_markdown_cell(text,id=id)
    def code(id,text):return nbformat.v4.new_code_cell(text,id=id)
    url='https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/'
    def picture(name,n):return f'![{name.replace("_"," ")}]({url}{name}.png)\n\n**Figure {n}.** {CAPTIONS[name]}'
    cells=[md('literature-reading-map',r'''## From reading to a controlled CFD experiment

**Predict, compute, explain.** Before running a cell, state what should change and what must remain fixed. Use the retained results first; the two fresh CPU studies below are optional.

| Reading | How we use it | Scope of the comparison |
|---|---|---|
| [Barba and Forsyth (2018), CFD Python](https://jose.theoj.org/papers/10.21105/jose.00021) | Start with a worked example, change one control, then answer a challenge. | Its finite-difference cavity/channel lessons do not rank our three FV algorithms. |
| [KTH SG2212/SG3114 (2017), Task 1, Q2-Q6](https://www.kth.se/social/files/58ad81f0f276542156c566ab/project.pdf) | Compare pressure-solver cost with the remaining mass defect; Q6 motivates Table L. | Our original Python experiment uses analytic trial fields and a fixed operator; it does not reproduce KTH's MATLAB cavity runs. |
| [Boroujerdi and Hamzeh (2019)](https://doi.org/10.1186/s40712-019-0104-7) | Separate coupling, grid and convection-scheme effects using Tables 6a-6c. | Their SIMPLE/SIMPLEC/SIMPLER, collocated/staggered and upwind-order comparisons differ from ours. Their ranking is configuration-specific. |
| [Barton (1998), transient SIMPLE/PISO comparison](https://onlinelibrary.wiley.com/doi/10.1002/%28SICI%291097-0363%2819980228%2926%3A4%3C459%3A%3AAID-FLD645%3E3.0.CO%3B2-U) | Compare physical profiles at the same time and examine the cost of reducing discrepancy. | The paper studies square-cylinder/step flows and several time schemes. Our cavity keeps backward Euler fixed and includes PIMPLE. |

### Table L: which part of the pressure solve costs time?

Freeze $A=-DdG$, with $d=0.05$, on an impermeable unit square. $D$ is cell divergence and $G$ is the face pressure gradient. For each analytic trial velocity $q^*$, solve
$$b=-Dq^*,\qquad Ap=b,\qquad q=q^*-dGp.$$
The full-system residual satisfies $r=b-Ap$ and **$Dq=-r$**, including the omitted pressure-gauge row. A relative GMRES tolerance is not an absolute continuity limit.

The retained experiment uses 16 and 32 cells/side, three analytic right-hand sides at each fixed matrix, and three timed repetitions after a warm-up. It compares LU rebuilt for each RHS, LU reused, and Jacobi-preconditioned GMRES at two tolerances. Setup and all solves are timed; assembly and diagnostics are outside the timer. These trial fields are **not cavity solutions**, so no Reynolds number or Ghia error is assigned to them.

### Table T2: the same cavity at the same physical time

All startup cases use **Re=100, 16 MAC cells/side, a unit square and unit lid speed, initially stationary fluid, central convection, backward Euler, and final time $tU/L=0.2$**. PISO uses two inner corrections; PIMPLE uses three outer passes with two inner corrections. The fine numerical comparator uses PIMPLE at $\Delta t=0.003125$, six outer passes and four inner corrections. A coarser comparator and doubled correction counts test its sensitivity.

The reported discrepancy is $100\|q_{\Delta t}(T)-q_{\mathrm{comparison}}(T)\|_2/\|q_{\mathrm{comparison}}(T)\|_2$ for native face velocities. It includes time and splitting/coupling effects on this mesh. **Steady Ghia data do not score startup fields.**

The earlier Exercise B (Table T) uses cell-centered velocities and a different comparison time step. Align the norm and comparison field before comparing its percentages with Table T2; its original controls and exercise remain available.'''),
    code('literature-retained-tables','''from qa.refresh_week01_2_literature import load as load_literature, tables as literature_tables
literature_data, literature_fields = load_literature(ROOT / "results/week01_2_literature")
linear_work_table, startup_evidence_table = literature_tables(literature_data)
display(Markdown("**Table L. Fixed-operator pressure-solver work and mass balance.** All timings are on the retained Unity CPU host; separate timing medians need not sum."))
display(linear_work_table)
display(Markdown("**Table T2. Startup at the same physical end time.** Reference rows are numerical sensitivity checks, not exact-solution errors."))
display(startup_evidence_table)
display(pd.DataFrame([{
    "reference time-step change %": 100*literature_data["reference_checks"]["reference_dt_change"],
    "reference correction-count change %": 100*literature_data["reference_checks"]["reference_loop_change"]}]))'''),
    md('literature-pressure-figure','### Read pressure and conservation together\n\n'+picture('linear_pressure_profiles',22)),
    md('literature-linear-exercise',r'''### Exercise L: change one linear-solver control

1. Predict which part of the cost LU reuse removes. Change `LINEAR_RHS_COUNT` from 3 to 6 while holding the matrix fixed. Report batch time and setup time for both counts.
2. Compare GMRES tolerances $10^{-4}$ and $10^{-8}$. Report worst cell divergence, pressure discrepancy from LU, and solver return status as well as time.
3. Set an absolute mass-defect target of your own. Decide which configurations meet it. Explain why nearly identical pressure curves need not imply equal conservation.
4. State when LU reuse is valid. In the full cavity solver $d=1/\mathrm{diag}(A_{\mathrm{momentum}})$ generally changes; changing the grid, time step, coefficients or boundary conditions may require a new factorization.

**Submit:** one table with the changed controls, median batch cost, setup cost, mass defect and solver status, plus a pressure/divergence profile. Do not generalize a small-matrix timing result to every CFD problem.'''),
    code('literature-linear-controls','''RUN_LINEAR_STUDY = False
LINEAR_CELLS = (16, 32)
LINEAR_RHS_COUNT = 3
LINEAR_REPEATS = 3
LINEAR_TOLERANCES = (1e-4, 1e-8)
if RUN_LINEAR_STUDY:
    from flowmllab.pressure_velocity_studies import pressure_work_study
    new_linear_rows, new_linear_fields = pressure_work_study(
        cells=LINEAR_CELLS, rhs_count=LINEAR_RHS_COUNT,
        repeats=LINEAR_REPEATS, tolerances=LINEAR_TOLERANCES)
    display(literature_tables({"pressure_rows":new_linear_rows,"time_rows":[]})[0])
    from qa.refresh_week01_2_literature import pressure_figure
    import matplotlib.pyplot as plt
    fresh_linear_plot = pressure_figure(new_linear_rows, {"pressure_"+k:v for k,v in new_linear_fields.items()})
    display(fresh_linear_plot)
    plt.close(fresh_linear_plot)
    print("Native pressure/divergence arrays are in new_linear_fields; raw repeated timings are in new_linear_rows.")
else:
    print("Table L is retained evidence. Set RUN_LINEAR_STUDY=True for your own CPU experiment.")'''),
    md('literature-startup-figure','### Read the physical profiles before ranking methods\n\n'+picture('startup_time_profiles',23)),
    md('literature-startup-exercise',r'''### Exercise T2: distinguish time error from coupling error

1. **Predict:** at fixed correction counts, what should halving $\Delta t$ change? Check both centerline velocity components and the native velocity-vector discrepancy.
2. **Change only the loops:** select a fixed $\Delta t$ and increase inner/outer corrections using the earlier Exercise B controls. Compare the full nonlinear step defect with the profile change. More corrections do not increase backward Euler's formal time order.
3. **Audit the comparator:** use the `reference_coarse`, `reference_fine` and `reference_loops` rows. Report both sensitivity percentages. If they are comparable to an alleged improvement, refine the comparison before choosing a winner.
4. **Work at a stated accuracy:** use the same host and thread count, repeat timings, retain failures and report pressure solves. Two settings with different errors are not a matched-accuracy speed comparison.

**Submit:** Table T2, both spatial velocity profiles and their signed discrepancies, the reference sensitivity, and a short qualified recommendation. The 16-cell impulsive-start cavity is a teaching example; these runs do not prove grid independence or a universal fastest algorithm.

### Exercise S: revisit the spatial experiment

Using Tables 6a-6c, change the grid at fixed method, then the convection scheme at fixed grid and method. Explain why a study that changes the algorithm and the spatial scheme simultaneously cannot isolate the algorithm's effect. Our central/upwind control is not a reproduction of the 2019 paper's first-/second-order-upwind comparison.'''),
    code('literature-startup-controls','''RUN_STARTUP_EXTENSION = False
EXT_CELLS = 16
EXT_END_TIME = 0.2
EXT_DT_VALUES = (0.05, 0.025, 0.0125)
EXT_REFERENCE_DT = 0.003125
EXT_REPEATS = 3
if RUN_STARTUP_EXTENSION:
    from flowmllab.pressure_velocity_studies import startup_study
    new_time_rows, new_time_fields, new_reference_checks = startup_study(
        cells=EXT_CELLS, end_time=EXT_END_TIME, steps=EXT_DT_VALUES,
        reference_dt=EXT_REFERENCE_DT, repeats=EXT_REPEATS)
    display(literature_tables({"pressure_rows":[],"time_rows":new_time_rows})[1])
    display(new_reference_checks)
    from qa.refresh_week01_2_literature import startup_figure
    import matplotlib.pyplot as plt
    fresh_startup_plot = startup_figure(new_time_rows, {"time_"+k:v for k,v in new_time_fields.items()})
    display(fresh_startup_plot)
    plt.close(fresh_startup_plot)
    print("Native fields are in new_time_fields. All earlier published results remain unchanged.")
else:
    print("Table T2 is retained evidence. Set RUN_STARTUP_EXTENSION=True to repeat the CPU experiment.")''')]
    ids={c.id for c in cells};book.cells=[c for c in book.cells if c.id not in ids]
    at=next(i for i,c in enumerate(book.cells) if c.id=='visual-caption-resources')
    book.cells[at:at]=cells
    for c in book.cells:
        if c.cell_type=='markdown':c.source=c.source.replace('Figures 1-20 share identifiers','Figures 1-23 share identifiers').replace('Figures 1-21 share identifiers','Figures 1-23 share identifiers')
    nbformat.validate(book);path.write_text(nbformat.writes(book),encoding='utf-8',newline='\n')
    index_path=ROOT/'lectures/source/week01_2_assets/figure_index.json';index=json.loads(index_path.read_text())
    for name,n in [('linear_pressure_profiles',22),('startup_time_profiles',23)]:
        index[name]={'number':str(n),'caption':CAPTIONS[name],'image':name+'.png','lecture':True}
    index_path.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Added literature exercises; earlier cell IDs retained.')


if __name__=='__main__':main()
