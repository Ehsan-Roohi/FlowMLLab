"""Append the independent higher-Re formulation comparison to Week 1.2."""
import nbformat as nbf

def append_formulations(nb):
 if any(c.metadata.get('formulation_comparison') for c in nb.cells):return
 description=r'''## 10. Independent formulation: streamfunction-vorticity at Re=400 and 1000

**Question:** does the independent Week-1 formulation give the same physical
flow, and what accuracy and computation cost do we obtain?

The study below actually runs the Week-1-family streamfunction-vorticity solver,
unchanged, alongside fresh SIMPLE/PISO/PIMPLE runs from rest. All four use
central spatial differences and the same physical spacing h=1/32; FD has 33x33
nodes, FV has 32x32 pressure cells. They do not solve identical discrete equations.
Pressure in the FD method is recovered after velocity convergence.

The two-dimensional formulation uses
\[
u=\frac{\partial\psi}{\partial y},\qquad
v=-\frac{\partial\psi}{\partial x},\qquad
\nabla^2\psi=-\omega,
\]
\[
\frac{\partial\omega}{\partial t}
+u\frac{\partial\omega}{\partial x}
+v\frac{\partial\omega}{\partial y}
=\frac{1}{Re}\nabla^2\omega.
\]
Velocity is reconstructed from psi; pressure is absent from these evolution
equations. In the pressure-velocity methods, continuity instead supplies the
pressure-correction equation coupled to momentum.

**Convergence:** FV stops on momentum Linf <1e-6; FD stops on a relative
vorticity update <1e-12 at three checks and also passes a recomputed steady
vorticity RHS Linf <1e-6. Native residuals have different meanings and are
shown on separate panels. Ghia velocity errors provide the shared accuracy
comparison. Re=400 keeps the anomalous published v value and reports its
one-point omission sensitivity separately. Pressure has no Ghia benchmark.

**Timing:** sequential single-thread runs, one sample per setting; the timed
solver call includes FD pressure recovery. Different dt, equations, stopping
norms and unknown counts preclude a universal fastest-method claim. Compare
accuracy before interpreting seconds. The field panels show all four
methods with velocity streamlines and visible resolved corner recirculation.

**Retained failures:** the initial Re=400 update-only stop failed the steady
RHS check, and Re=1000 exhausted its first step budget. Both original fields
and logs remain in `attempts/`; reruns tighten the update tolerance without
relaxing the steady gate. The report includes a separate 64-cell SIMPLE/FD
mesh check plus an FD-only 128-cell refinement; PISO/PIMPLE are compared at
32 cells only. The fine-grid panels label each spacing explicitly.

**Student tasks:** compare both centerline errors and corner vortices; explain
why zero integrated/local mass error does not prove momentum accuracy; explain
why FD update and FV momentum histories cannot share an unlabeled residual
axis. Refine the FD mesh to investigate its own accuracy trend. Do not compare
fine-grid FD time to coarse-grid FV time as equal work. Finer and higher-Re
runs can take longer and need more memory/stronger hardware.

[Full report and raw results](https://github.com/Ehsan-Roohi/FlowMLLab/tree/main/results/week01_2_formulations)
'''
 code='''import json
FORMULATIONS = ROOT / "results/week01_2_formulations"
formulation_records = [json.loads(p.read_text()) for p in sorted(FORMULATIONS.glob("*-re*-n*.json"))]
if not formulation_records:
    raise FileNotFoundError("A complete current checkout needs results/week01_2_formulations")
display(pd.DataFrame([{
    "Re": r["config"]["reynolds"], "method": r["config"]["method"],
    "cells per side": r["config"]["cells"], "status": r["status"],
    "end-to-end seconds": r["end_to_end_seconds"],
    "steps / iterations": r.get("steps", r.get("iterations")),
    "u Ghia rel L2": r.get("ghia_u_relative_l2"),
    "v Ghia rel L2 (all)": r.get("ghia_v_relative_l2"),
    "v sensitivity omit x=.9063": r.get("ghia_v_relative_l2_without_x09063"),
    "FV momentum Linf": r.get("momentum_residual_linf"),
    "FD steady vorticity RHS Linf": r.get("steady_vorticity_linf"),
    "continuity Linf": r.get("continuity_linf"),
    "FD pressure-gradient discrepancy": r.get("pressure_grad_rel_residual"),
} for r in formulation_records]).fillna("n/a"))
# Keep inline images compact; full-resolution PNGs are retained in the report.
from PIL import Image as PILImage
from io import BytesIO
for name in ["p", "omega", "refinement"]:
    picture = PILImage.open(FORMULATIONS / (name + ".png"))
    picture.thumbnail((1400, 1100))
    buffer = BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    display(Image(data=buffer.getvalue()))

# Fresh independent FD run: opt in deliberately. These can take minutes.
RUN_STREAMFUNCTION = False
STREAM_RE = 400
STREAM_CELLS = 32
if RUN_STREAMFUNCTION:
    from flowmllab.streamfunction_experiments import run_streamfunction
    from flowmllab.pressure_velocity import centerlines, GHIA_Y
    from flowmllab.pressure_velocity_experiments import REFERENCE
    from flowmllab.cavity_visuals import RGB, flow_structure
    import hashlib
    sf = run_streamfunction(STREAM_RE, STREAM_CELLS, progress=True)
    folder = ROOT / "output/week01_2_streamfunction"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"streamfunction-re{STREAM_RE}-n{STREAM_CELLS}"
    archive = folder / (stem + ".npz")
    np.savez_compressed(archive, **{k:v for k,v in sf.items() if isinstance(v,np.ndarray)})
    metadata = {k:v for k,v in sf.items() if not isinstance(v,np.ndarray)}
    metadata["field_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (folder / (stem + ".json")).write_text(json.dumps(metadata,indent=2))
    display(pd.DataFrame([{k:sf[k] for k in ["status", "runtime_seconds", "steps",
        "steady_vorticity_linf", "continuity_linf", "ghia_u_relative_l2", "ghia_v_relative_l2"]}]))
    y,u,x,v = centerlines(sf)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axes[0].plot(u,y,label="Fresh FD")
    axes[0].plot(REFERENCE[STREAM_RE]["u"],GHIA_Y,"ko",label="Ghia")
    axes[0].set(xlabel="u / U",ylabel="y / L");axes[0].legend()
    image = axes[1].pcolormesh(sf["x"],sf["y"],np.hypot(sf["u"],sf["v"]),cmap=RGB,shading="nearest")
    fig.colorbar(image,ax=axes[1],label="Speed / U")
    flow_structure(axes[1],sf["x"],sf["y"],sf["u"],sf["v"])
    axes[1].set(aspect="equal",xlabel="x / L",ylabel="y / L",title="Fresh speed and streamlines")
    display(fig);plt.close(fig)
    print("Saved", folder, "; retain failed/nonconverged results too.")
'''
 for kind,text in [('markdown',description),('code',code)]:
  c=nbf.v4.new_markdown_cell(text) if kind=='markdown' else nbf.v4.new_code_cell(text)
  c.metadata['formulation_comparison']=True;nb.cells.append(c)

 from qa.notebook_math import normalize_notebook
 normalize_notebook(nb)
