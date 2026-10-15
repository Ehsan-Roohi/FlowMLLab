"""Append an opt-in, evidence-first higher-Re student investigation."""
import nbformat as nbf

def append_higher_re(nb):
    if any(c.metadata.get('higher_re_extension') for c in nb.cells):return
    sources=[('markdown',r'''## 9. Higher-Reynolds-number investigation (experimental)

**Question:** how do mesh resolution, convection interpolation and coupling
settings change accuracy and work at $Re=400$ and $1000$?

Use the experimental wrapper below; the qualified `CavityConfig` still accepts
only Re=100. No external CFD solver is called. A small momentum residual proves
discrete equation convergence, not mesh independence or physical accuracy.

**Required sequence and deliverables**
1. Reproduce the retained Re=100 baseline before running this extension.
2. Run SIMPLE, PISO and PIMPLE at Re=400 on 32 cells, then repeat at Re=1000.
   Retain failures and report the exact settings and stopping reason.
3. Refine to 64 cells; use 128 only if the measured mesh error warrants its cost.
   Plot both centerlines against the Ghia table for the correct Reynolds number.
4. Plot speed with streamlines, zero-mean pressure and vorticity. Record the
   primary-vortex location and inspect secondary corner vortices.
5. Change one setting at a time: SIMPLE velocity relaxation (0.3/0.5/0.7),
   PISO corrections (1/2/4), or PIMPLE outer loops (1/2/3). Keep spatial equations
   and final momentum/mass targets fixed; count momentum and pressure solves.
6. Compare central and first-order upwind on the same mesh. Explain any smoother
   field using its benchmark error; numerical diffusion is not extra accuracy.

**Acceptance gates:** momentum Linf below $10^{-6}$; cell-divergence Linf below
$10^{-9}$; pressure mean near zero; finite fields; compare converged coupling
methods at fixed mesh/scheme (target max field difference $2\times10^{-5}$).
Aim for less than 5% centerline relative L2 error, then check refinement. Report
failure of any gate. Two meshes do not establish asymptotic order.

The Re=400 published v-table entry at x=0.9063 is isolated from its neighbors.
The all-point error is always retained; a separately labeled sensitivity metric
omits that one entry without replacing its value. Ghia does not validate pressure.

[Measured higher-Re report](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/week01_2_higher_re/README.md)
and [comparison study at Re=1000/5000](https://congress2.cimne.com/eccomas/proceedings/cfd2010/papers/01330.pdf).
The latter uses collocated grids and modified PISO/SIMPLEC variants, not our
PIMPLE implementation. No universal algorithm Reynolds-number ceiling follows.
'''),('code','''import json, hashlib
HIGH_RE = ROOT / "results/week01_2_higher_re"
experimental_records = [json.loads(f.read_text()) for f in sorted(HIGH_RE.glob("*.json"))
                        if f.name != "validation.json"]
display(pd.DataFrame([{
    "method": r["config"]["method"], "Re": r["config"]["reynolds"],
    "cells": r["config"]["cells"], "scheme": r["config"]["convection"],
    "status": r["status"], "iterations": r.get("iterations"),
    "momentum Linf": r.get("momentum_residual_linf"),
    "divergence Linf": r.get("continuity_linf"),
    "Ghia u rel L2": r.get("ghia_u_relative_l2"),
    "Ghia v rel L2 (all points)": r.get("ghia_v_relative_l2"),
    "v sensitivity: omit x=.9063": r.get("ghia_v_relative_l2_without_x09063"),
    "pressure solves": r.get("pressure_solves")
} for r in experimental_records]))
for name in ["centerlines", "convergence", "fields"]:
    figure = HIGH_RE / (name + ".png")
    if figure.exists(): display(Image(filename=str(figure)))
'''),('code','''# Run deliberately: finer grids and PIMPLE can take minutes on a CPU.
RUN_HIGHER_RE = False
HIGHER_RE = 400       # supported experimental values: 400, 1000
HIGHER_CELLS = 32     # refine to 64, then justify 128
HIGHER_METHOD = "all" # simple, piso, pimple, all
HIGHER_SCHEME = "central" # central or upwind
HIGHER_DT = 0.05      # steady marching control, not a transient-accuracy claim
if RUN_HIGHER_RE:
    from flowmllab.pressure_velocity_experiments import ExperimentalConfig, run_experiment
    from flowmllab.pressure_velocity import centerlines
    methods = ["simple", "piso", "pimple"] if HIGHER_METHOD == "all" else [HIGHER_METHOD]
    higher_runs, higher_summary = [], []
    save_root = ROOT / "output/week01_2_higher_re"
    save_root.mkdir(parents=True, exist_ok=True)
    for m in methods:
        try:
            r = run_experiment(ExperimentalConfig(
                method=m, cells=HIGHER_CELLS, reynolds=HIGHER_RE,
                convection=HIGHER_SCHEME, dt=HIGHER_DT, velocity_relaxation=0.5,
                pressure_relaxation=0.2, tolerance=1e-6, max_iterations=4000), progress=True)
            higher_runs.append(r)
            higher_summary.append({k:r[k] for k in ["converged", "iterations",
                "runtime_seconds", "pressure_solves", "momentum_residual_linf",
                "continuity_linf", "ghia_u_relative_l2", "ghia_v_relative_l2"]}
                | {"method":m, "Re":HIGHER_RE, "cells":HIGHER_CELLS, "scheme":HIGHER_SCHEME})
            np.savez_compressed(save_root / f"{m}-re{HIGHER_RE}-n{HIGHER_CELLS}-{HIGHER_SCHEME}.npz",
                **{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
        except Exception as error:
            higher_summary.append({"method":m, "Re":HIGHER_RE, "cells":HIGHER_CELLS,
                "scheme":HIGHER_SCHEME, "converged":False, "error":str(error)})
    higher_table = pd.DataFrame(higher_summary)
    display(higher_table)
    higher_table.to_csv(save_root / f"summary-re{HIGHER_RE}-n{HIGHER_CELLS}-{HIGHER_SCHEME}.csv",index=False)
    print("Saved student table and fields in",save_root)
    from flowmllab.pressure_velocity_experiment_plots import comparison_figures
    if higher_runs:
        for figure in comparison_figures(higher_runs).values():
            display(figure); plt.close(figure)
else:
    print("Showing retained experimental evidence; set RUN_HIGHER_RE=True for fresh runs.")
'''),('markdown',r'''### Submit your analysis

Submit one configuration/result CSV (including failed runs), the centerline,
field and convergence figures, and a short explanation addressing:
- Does refinement improve both centerlines? Does upwind change the answer?
- At the same discrete accuracy, do extra corrections repay their solve cost?
- Are low mass residuals accompanied by small momentum and benchmark errors?
- If a case fails, does changing dt, relaxation or the mesh isolate its cause?

Do not claim turbulence, a universal fastest method, or a maximum usable Re
from this two-dimensional steady study. A time-accurate comparison needs a
separate dt-refinement study with within-step convergence checks.
''')]
    for kind,source in sources:
        cell=nbf.v4.new_markdown_cell(source) if kind=='markdown' else nbf.v4.new_code_cell(source)
        cell.metadata['higher_re_extension']=True
        nb.cells.append(cell)
