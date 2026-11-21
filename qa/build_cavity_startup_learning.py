"""Build only the additive Week 1.2 startup companion and its lecture appendix.

No original lesson, lecture, exercise or steady benchmark is overwritten.
Usage: python qa/build_cavity_startup_learning.py notebook|evidence|appendix|all
"""
from pathlib import Path
import argparse
import json
import os
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/cavity_startup/matplotlib"))
RESULTS = ROOT / "results/cavity_startup"
MEDIA = RESULTS / "media"
REYNOLDS = [100, 400, 1000, 3200, 5000, 7500, 10000]


def presentation():
    """Select a retained native trajectory set explicitly, without moving it."""
    path = RESULTS / "presentation.json"
    if not path.is_file():
        raise FileNotFoundError("Select the published trajectory set in results/cavity_startup/presentation.json first")
    choice = json.loads(path.read_text())
    directory = (RESULTS / choice["trajectory_directory"]).resolve()
    if not directory.is_relative_to(RESULTS.resolve()):
        raise ValueError("Presentation trajectory_directory must stay within this result bundle")
    return choice, directory


def build_notebook():
    import nbformat as nbf
    md = lambda text: nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
    code = lambda text: nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    cells = [md(r"""
    # Week 1.2 companion - How cavity circulation forms

    **Aim.** Follow a fluid from rest after its top lid starts moving. Use real physical-time
    PISO/PIMPLE solutions to explain how speed, circulation, signed vorticity and pressure
    develop, and test whether a visible small vortex survives numerical refinement.

    This lesson supplements the existing [pressure-velocity notebook](W1_2_Cavity_Pressure_Velocity.ipynb)
    and [lecture](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/lectures/week01_2_pressure_velocity.pdf).
    It keeps the earlier steady validation and exercises intact.

    [Open this companion in Google Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb).

    **Run all on a CPU.** The default loads hash-verified retained trajectories and runs a small,
    fresh $16\times16$ experiment at $Re=100$. No GPU is used. The controls later in the notebook
    let you rerun longer, finer experiments. Recorded seven-case movies use a larger mesh;
    they are not recomputed by default. Wall-clock cost depends on the CPU, timestep,
    mesh and number of pressure/nonlinear corrections.

    **Question to answer.** When a new recirculation cell appears in a picture, is it a resolved
    feature of this two-dimensional startup, a transient feature, or numerical sensitivity?
    A smooth-looking movie alone cannot decide.

    **Evidence limit.** The displayed 128-cell high-$Re$ trajectories are provisional,
    mesh-dependent teaching examples. At $t^*=10$, matched 64-to-128-cell velocity differences
    range from about 17% to 70% for $Re=3200$ through 10000. They are not resolved flow
    predictions. The notebook shows these differences rather than hiding them behind a movie.
    """), code(r"""
    # FLOWMLLAB_CAVITY_STARTUP_BOOTSTRAP_V1: dedicated sparse checkout; native physical-time data.
    from pathlib import Path
    import os, sys, subprocess, json, importlib.util
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        ROOT = Path("/content/FlowMLLab-startup")
        if not (ROOT / ".git").exists():
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(ROOT)], check=True)
        subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "set", "flowmllab", "qa", "results/cavity_startup"], check=True)
        required = [ROOT / "flowmllab/cavity_startup.py", ROOT / "results/cavity_startup/presentation.json"]
        if not all(p.is_file() for p in required):
            dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
            if dirty:
                raise RuntimeError("This dedicated checkout has edits and lacks the new lesson. Preserve your edits and choose a new ROOT directory.")
            subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only"], check=True)
        if not all(p.is_file() for p in required):
            raise RuntimeError("The published startup companion is missing. Check the repository revision before continuing.")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)], check=True)
    else:
        ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "flowmllab/cavity_startup.py").exists())
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from dataclasses import replace
    from IPython.display import display, Video, Image, Markdown
    from IPython import get_ipython
    from flowmllab.cavity_startup import (observation_times, simulate_startup, frame_fields,
        case_summary, compare_trajectories, common_field_bounds, save_trajectory)
    from flowmllab.pressure_velocity_experiments import ExperimentalConfig
    from qa.build_cavity_startup import load_cases, comparison_figure, streamfunction_levels, sha256
    # The command-line renderer uses Agg; explicitly restore notebook image display.
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
    RESULTS = ROOT / "results/cavity_startup"
    RE_MENU = [100, 400, 1000, 3200, 5000, 7500, 10000]
    selection = json.loads((RESULTS / "presentation.json").read_text())
    retained = load_cases(RESULTS / selection["trajectory_directory"], RE_MENU)
    print("Verified native trajectory hashes:", RE_MENU)
    print("Each stored frame is an actual solution at its stated physical time.")
    print("Evidence scope:", selection["qualification"])
    """), md(r"""
    ## 1. Problem, units and what the plots mean

    The cavity is $0\le x/L,y/L\le1$. Initially the interior fluid is at rest. At $t=0^+$,
    the top wall moves right with constant speed $U_{\mathrm{lid}}$; the other walls remain
    stationary and all walls are impermeable. There is no lid ramp. The ideal top corners
    therefore have a velocity discontinuity. Their large gradients are retained.

    | Quantity | Definition and interpretation |
    |---|---|
    | $Re$ | $U_{\mathrm{lid}}L/\nu$, where $\nu$ is kinematic viscosity |
    | $t^*$ | $tU_{\mathrm{lid}}/L$, physical time in lid-travel units |
    | $\Delta t^*$ | Maximum actual time increment, shortened when necessary to hit a requested frame exactly |
    | $p^*$ | $(p-\langle p\rangle)/(\rho U_{\mathrm{lid}}^2)$, native transient pressure with zero area mean |
    | $\omega^*$ | $(\partial v/\partial x-\partial u/\partial y)L/U_{\mathrm{lid}}$, signed vorticity |
    | $\psi^*$ | $\psi/(U_{\mathrm{lid}}L)$, with $u=\partial\psi/\partial y$ and $v=-\partial\psi/\partial x$ in dimensionless units |

    A streamline is tangent to the velocity **at one instant**. In unsteady flow it is generally
    not the path followed by a fluid particle. The movies draw contours of the instantaneous
    streamfunction recovered from the same divergence-free face velocities. Solid contours
    indicate negative streamfunction and dashed contours positive streamfunction, with fixed
    levels across cases and times. These line styles identify streamfunction sign, not a new method.

    The seven Reynolds numbers above match the Ghia tables. In particular the value is **7500**,
    not 7000. Ghia's steady profiles do not supply a transient reference for these movies.
    """), md(r"""
    ## 2. Physical timesteps and pressure corrections are different loops

    In dimensionless form,

    $$\nabla\cdot\mathbf u=0,\qquad
    \frac{\partial\mathbf u}{\partial t^*}+(\mathbf u\cdot\nabla)\mathbf u
    =-\nabla p^*+Re^{-1}\nabla^2\mathbf u.$$

    The shared solver uses a staggered finite-volume (MAC) mesh: pressure at cell centers,
    normal velocity components at faces. Backward Euler approximates acceleration by
    $(\mathbf u^{n+1}-\mathbf u^n)/\Delta t^*$. For each new physical timestep, PIMPLE
    repeats momentum prediction and pressure correction while keeping $\mathbf u^n$ fixed.
    Increasing `outer_correctors` improves the nonlinear coupling **within the same timestep**;
    increasing `pressure_correctors` repeats pressure/velocity correction. Neither increases
    physical time. PISO uses one outer momentum prediction. SIMPLE is a steady iteration route
    and is deliberately rejected by this startup collector.

    **Sequence:** retain old-time velocity → predict momentum → solve pressure correction
    → correct face fluxes → repeat within-step corrections → advance physical time → save
    only when the actual solution lands on a requested frame.

    Continuity can be tiny while momentum coupling, timestep or mesh errors remain significant.
    We therefore record both continuity and the momentum defect of the actual backward-Euler
    step, as well as the steady momentum defect. A nonzero steady defect during startup is
    expected; it is not the same as a failed transient timestep.
    """), code(r"""
    rows = []
    for result in retained:
        s = case_summary(result)
        rows.append({"Re": s["config"]["reynolds"], "cells/side": s["config"]["cells"],
            "end t*": s["end_time"], "frames": s["frames"], "actual steps": s["actual_steps"],
            "solver seconds (recorded CPU)": s["runtime_seconds"],
            "max divergence": s["max_divergence_linf"],
            "max timestep momentum defect": s["max_time_step_momentum_defect_linf"]})
    display(pd.DataFrame(rows).set_index("Re"))
    """), md(r"""
    ## 3. Three movies, matched times and fixed scales

    All panels share the unit-square domain, physical frame times, initial condition, mesh,
    timestep and correction settings. Only $Re$ changes. The color range is fixed over every
    stored time and case. Speed uses a linear scale; signed vorticity uses a symmetric-log scale
    with a linear interval $|\omega^*|\le1$, and signed pressure uses a symmetric-log scale with
    a linear interval $|p^*|\le0.01$. This keeps the full extrema, including corner peaks,
    while making smaller signed interior structures visible. No field is clipped or smoothed.
    Plot contours interpolate between native samples for drawing; the saved scientific values
    are unchanged. Playback frame rate changes viewing speed, not the physical frame timestamps.

    Native transient pressure includes local acceleration through the time-dependent momentum
    solve. It is not the pressure recovered by a steady postprocessor. A changing gauge would
    create a false pressure flicker; subtracting the area mean at every step fixes that freedom.
    Vorticity is differentiated from cell-centered velocity, using one-sided derivatives at
    edge cells. Thin wall layers and the lid-corner discontinuity make these derivatives
    particularly sensitive to spatial resolution.
    """), code(r"""
    movie_manifest = json.loads((RESULTS / "media/movies.json").read_text())
    for field in ("speed", "omega", "p"):
        record = movie_manifest["fields"][field]
        movie = RESULTS / "media" / record["movie"]
        assert sha256(movie) == record["movie_sha256"], f"Movie checksum mismatch: {movie.name}"
        display(Markdown("**" + {"speed":"Speed and instantaneous streamlines", "omega":"Signed vorticity", "p":"Signed pressure"}[field] + "**"))
        # URL playback avoids copying three video payloads into the notebook file.
        url = "https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/cavity_startup/media/" + movie.name
        display(Video(url=url, width=950))
    print("Local MP4 files are also available in", RESULTS / "media")
    """), md(r"""
    ## 4. Stop the movie and measure the flow

    Set `TIME_TO_VIEW` to a stored time. The figure reports the nearest actual frame, not an
    interpolated solution. Change `FIELD_TO_VIEW` among `speed`, `omega` and `p`. The color
    limits still come from the complete seven-case trajectory set. This prevents a quiet early
    frame from being rescaled to appear as intense as a later one.
    """), code(r"""
    TIME_TO_VIEW = 10.0
    FIELD_TO_VIEW = "speed"
    index = int(np.argmin(np.abs(retained[0]["times"] - TIME_TO_VIEW)))
    actual = retained[0]["times"][index]
    print(f"Requested t*={TIME_TO_VIEW:g}; showing actual stored t*={actual:g}")
    fig, _, _ = comparison_figure(retained, FIELD_TO_VIEW, index,
        common_field_bounds(retained), streamfunction_levels(retained))
    display(fig); plt.close(fig)
    """), md(r"""
    Two useful quantitative observables are kinetic energy per unit area,
    $E^*=\tfrac12\int_0^1\int_0^1(u^{*2}+v^{*2})\,dx^*dy^*$, and the minimum of $\psi^*$,
    which measures the strength of the main clockwise circulation in this sign convention.
    The grid node containing that minimum estimates the primary vortex center; its position
    is discrete, so a one-cell jump need not represent a sudden physical jump.

    Comparing at equal $t^*$ is a controlled physical comparison, not a claim that every
    Reynolds number has completed the same fraction of its equilibration. The viscous
    diffusion time scales as $L^2/\nu$, or $Re$ in these time units. Convection and wall-layer
    formation also matter, so do not infer a steady state from that timescale alone.
    """), code(r"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for result in retained:
        data = [frame_fields(result, k) for k in range(len(result["times"]))]
        re = result["config"]["reynolds"]
        axes[0].plot(result["times"], [d["kinetic_energy"] for d in data], label=f"Re={re:g}")
        axes[1].plot(result["times"], [d["psi_min"] for d in data])
    axes[0].set_ylabel(r"$E^*$")
    axes[1].set_ylabel(r"$\min\psi^*$")
    for ax in axes:
        ax.set_xlabel(r"$t^*=tU_{\mathrm{lid}}/L$")
        ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.27, wspace=.28)
    display(fig); plt.close(fig)
    """), md(r"""
    ## 5. Run your own small startup on a CPU

    The next cell performs a genuinely fresh solve. Its $16\times16$ mesh and short duration
    are a quick learning experiment, **not** the seven-case retained production setting.
    Increase one control at a time. For all seven retained Reynolds numbers, set `RUN_FRESH`
    and choose a value from `RE_MENU`; high-$Re$ coarse runs may be visibly mesh sensitive.

    | Control | What changes | What to hold fixed when studying it |
    |---|---|---|
    | `CELLS` | Spatial resolution and cost | Physical end time, observation times, method, $Re$ |
    | `DT` | Temporal resolution | Mesh and sufficient within-step corrections |
    | `END_TIME` | Duration of startup | Do not compare different final times as identical states |
    | `FRAME_INTERVAL` | Saved/movie sampling | Use a multiple of `DT` for a pure storage-frequency change |
    | `OUTER`, `PRESSURE` | Within-step coupling work | Mesh, timestep and physical end time |
    | `METHOD` | `piso` or `pimple` coupling | Same physical timestep, mesh and end time |

    If a frame interval is not a multiple of the timestep, the solver shortens real steps to
    hit the frames; changing that interval then also changes the actual integration schedule.
    """), code(r"""
    RUN_FRESH = True
    RE, CELLS, DT, END_TIME, FRAME_INTERVAL = 100, 16, .02, .2, .02
    METHOD = "pimple"
    OUTER, PRESSURE = 3, 2
    if RUN_FRESH:
        config = ExperimentalConfig(method=METHOD, reynolds=RE, cells=CELLS,
            dt=DT, end_time=END_TIME, outer_correctors=OUTER,
            pressure_correctors=PRESSURE, max_iterations=100000)
        fresh = simulate_startup(config, observation_times(END_TIME, FRAME_INTERVAL))
        display(pd.DataFrame([case_summary(fresh)]).drop(columns=["config", "qualification"]).T)
        fig, _, _ = comparison_figure([fresh], "speed", -1)
        display(fig); plt.close(fig)
    """), md(r"""
    ## 6. Timestep, corrector and grid sensitivity

    The retained baseline audit at $Re=1000$ and $7500$ changes one control at a time: 32 versus
    64 cells, $\Delta t^*=0.02$ versus $0.01$, and 3 outer/2 pressure corrections versus
    6 outer/4 pressure corrections. Each pair has exactly matching physical frame times.
    Mesh comparison samples the fine-grid field at coarse cell centers solely to measure
    the difference. Each movie and retained scientific payload keeps its native grid.

    For velocity use $\delta_u=\sqrt{\langle(u_a-u_b)^2+(v_a-v_b)^2\rangle}$ and
    $\delta_u/\sqrt{\langle u_b^2+v_b^2\rangle}$ where the denominator is nonzero.
    At rest the relative value is undefined and is displayed as missing, not fabricated as zero.
    Pressure and vorticity RMS differences retain their dimensionless units and declared gauge.
    These are observed numerical sensitivities, not rigorous error bounds or independent truth.
    The 32-to-64-cell velocity differences at $t^*=10$ were approximately 13.2% at $Re=1000$
    and 77.6% at $Re=7500$, much larger than the corresponding timestep/corrector changes.
    This motivated finer-mesh trajectories, not a claim that 64 cells were converged.
    """), code(r"""
    sensitivity = json.loads((RESULTS / "sensitivity.json").read_text())
    table = []
    for label, report in sensitivity["comparisons"].items():
        row = report["frames"][-1]
        table.append({"study": label, "Re": report["candidate"]["reynolds"],
            "t*": row["time"], "velocity relative L2 difference": row["velocity_relative_l2"],
            "pressure RMS difference": row["pressure_rms_difference"],
            "vorticity RMS difference": row["vorticity_rms_difference"]})
    display(pd.DataFrame(table).set_index("study"))
    display(Image(filename=str(RESULTS / "media/sensitivity_profiles.png")))
    if (RESULTS / "media/grid_refinement_profiles.png").exists():
        display(Markdown("**Matched-timestep 64 versus 128 cells: physical profiles at t*=10.**"))
        display(Image(filename=str(RESULTS / "media/grid_refinement_profiles.png")))
    """), md(r"""
    ## 7. Export a new movie without changing scientific data

    FFmpeg is required only for MP4 export, not the solver. Colab normally provides it;
    a local machine can install FFmpeg or set `FLOWMLLAB_FFMPEG` to its executable.
    The command below is an optional longer rerun. Use a new directory for each experiment;
    the driver refuses to overwrite a completed scientific trajectory. Store configuration,
    native fields and checksums together. Reuse the existing MP4s for a presentation if you
    have not run your own matched experiment.

    ```python
    # Run from ROOT. This is a longer, optional computation, not the quick default.
    out = ROOT / "results/cavity_startup/student_experiment"
    subprocess.run([sys.executable, "qa/build_cavity_startup.py", "solve",
        "--reynolds", "100", "--cells", "32", "--dt", ".02",
        "--end-time", "4", "--interval", ".1", "--output", str(out)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "qa/build_cavity_startup.py", "render",
        "--input", str(out), "--output", str(out), "--reynolds", "100"], cwd=ROOT, check=True)
    ```

    ## 8. Student work - measurable deliverables

    1. **Timeline and mechanism.** At $Re=100,400,1000$, inspect $t^*=0.2,1,4,10$.
       Submit one matched speed/streamline plate, an energy-versus-time plot and a paragraph
       explaining how wall shear starts circulation. Use the actual frame time in each caption.
    2. **Vortex formation.** Track the primary streamfunction minimum and its $(x/L,y/L)$
       location. Mark a candidate secondary cell by closed contours and a local extremum of
       the opposite sign. Record its position, signed $\psi^*$ and first sampled appearance
       time; this time is only bracketed by the frame interval. Verify with mesh and timestep
       refinement before calling it resolved. A faint color patch is not sufficient evidence.
    3. **Timestep study.** At one moderate $Re$, compare `DT=.02` and `.01` at identical
       physical times and fixed mesh. Plot both centerline velocity profiles and the energy
       curves. Report the final relative velocity difference and measured CPU time.
    4. **Coupling study.** Hold `DT` fixed; compare PISO, PIMPLE 3/2 and PIMPLE 6/4 corrections.
       Compare momentum-step defects, continuity and physical profiles. Explain why small
       continuity alone did not establish accuracy and why more corrections cost time.
    5. **Grid study.** Compare 16,32 and, if affordable,64 cells at one common time. Overlay
       $u(x/L=0.5,y/L)$ and $v(x/L,y/L=0.5)$, record vortex locations and pressure extrema,
       and distinguish spatial sensitivity from timestep/coupling sensitivity. Preserve all
       three results even if the coarsest field looks poor.
    6. **Pressure and sign.** Plot one vertical pressure profile at two physical times and
       demonstrate that adding a spatial constant changes no pressure gradient. Explain
       why the shared zero-mean gauge is required before comparing plotted pressure values.
    7. **High-$Re$ claim audit.** Choose 3200,5000,7500 or10000. State the evidence needed
       before describing a small visible cell as resolved or a finite-time frame as steady.
       Compare the available 7500 refinement results. Do not label these 2D finite-time
       pictures as validated three-dimensional turbulence.
    8. **Submission.** Provide a short report, three short movies or selected physical-time
       frames, a table of exact controls/costs/sensitivities, and your configuration plus
       compact trajectory files. No bar chart is needed: show physical profiles and time histories.

    **Read further.** Ghia, Ghia & Shin (1982), *Journal of Computational Physics* 48,387-411,
    [DOI](https://doi.org/10.1016/0021-9991(82)90058-4), is the steady benchmark source.
    The preceding Week1.2 lecture derives the shared MAC pressure-correction algorithm.
    """)]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "flowmllab": {"lesson": "week01_2_startup_companion", "requires_gpu": False,
            "scope": "Additive physical-time experiment; existing notebooks preserved"}})
    path = ROOT / "notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb"
    path.write_text(nbf.writes(notebook) + "\n", encoding="utf-8", newline="\n")
    print(path)
    return path


def build_evidence():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flowmllab.cavity_startup import frame_fields, compare_trajectories
    from qa.build_cavity_startup import load_cases, write_json
    MEDIA.mkdir(parents=True, exist_ok=True)
    choice, directory = presentation()
    cases = load_cases(directory, REYNOLDS)
    base_cases = load_cases(RESULTS / "base", REYNOLDS)
    lookup = {int(c["config"]["reynolds"]): c for c in base_cases}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for case in cases:
        fields = [frame_fields(case, k) for k in range(len(case["times"]))]
        label = f"Re = {case['config']['reynolds']:g}"
        axes[0].plot(case["times"], [f["kinetic_energy"] for f in fields], label=label)
        axes[1].plot(case["times"], [f["psi_min"] for f in fields])
    axes[0].set_ylabel(r"$E^*$"); axes[1].set_ylabel(r"$\min\psi^*$")
    for ax in axes:
        ax.set_xlabel(r"$t^*=tU_{\mathrm{lid}}/L$"); ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.27, left=.08, right=.98, top=.96, wspace=.28)
    for suffix in ("png", "pdf"):
        fig.savefig(MEDIA / ("startup_observables." + suffix), dpi=160)
    plt.close(fig)
    comparisons = {}
    fig, grid = plt.subplots(2, 3, figsize=(13, 7.0))
    for row, re in enumerate((1000, 7500)):
        group = [("64 cells, dt=0.02, 3/2", lookup[re], "#151515", "-")]
        for variant, label, color, style in (
            ("mesh32", "32 cells", "#b72e20", "--"),
            ("dt_half", "dt=0.01", "#135ac4", "-."),
            ("correctors", "6/4 corrections", "#008369", ":")):
            case = load_cases(RESULTS / variant, [re])[0]
            candidate, reference = (case, lookup[re]) if variant == "mesh32" else (lookup[re], case)
            comparisons[f"Re{re}_{variant}"] = compare_trajectories(candidate, reference)
            group.append((label, case, color, style))
        for label, case, color, style in group:
            fields = [frame_fields(case, k) for k in range(len(case["times"]))]
            values = ([f["kinetic_energy"] for f in fields], [f["psi_min"] for f in fields],
                      [np.sqrt(np.mean(f["p"] ** 2)) for f in fields])
            for col in range(3):
                grid[row, col].plot(case["times"], values[col], color=color, ls=style, label=label)
        for col, quantity in enumerate((r"$E^*$", r"$\min\psi^*$", r"$\mathrm{RMS}(p^*)$")):
            grid[row, col].set(xlabel=r"$t^*$", ylabel=quantity, title=f"Re = {re}")
            grid[row, col].grid(alpha=.16)
    fig.legend(*grid[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=.07, right=.98, bottom=.12, top=.94, wspace=.32, hspace=.50)
    for suffix in ("png", "pdf"):
        fig.savefig(MEDIA / ("sensitivity_profiles." + suffix), dpi=160)
    plt.close(fig)
    if cases[0]["config"]["cells"] > 64:
        # Re=1000/7500 already have these exact controls in dt_half; reuse
        # their verified native files rather than duplicating scientific data.
        matched = [load_cases(RESULTS / ("dt_half" if re in (1000, 7500) else "matched64"), [re])[0]
                   for re in REYNOLDS]
        coarse_lookup = {int(c["config"]["reynolds"]): c for c in matched}
        fine_lookup = {int(c["config"]["reynolds"]): c for c in cases}
        for re in REYNOLDS:
            for key in ("dt", "convection", "method", "outer_correctors", "pressure_correctors"):
                if coarse_lookup[re]["config"][key] != fine_lookup[re]["config"][key]:
                    raise ValueError(f"Fine mesh comparison must keep {key} matched")
            comparisons[f"Re{re}_grid64_128"] = compare_trajectories(coarse_lookup[re], fine_lookup[re])
        fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2))
        for row, re in enumerate((1000, 7500)):
            for case, label, color, style in ((coarse_lookup[re], "64 cells", "#171717", "-"),
                    (fine_lookup[re], "128 cells", "#c12626", "--")):
                f = frame_fields(case, -1)
                up = np.array([np.interp(.5, f["x"], line) for line in f["u"]])
                vp = np.array([np.interp(.5, f["y"], line) for line in f["v"].T])
                pp = np.array([np.interp(.5, f["x"], line) for line in f["p"]])
                wp = np.array([np.interp(.5, f["x"], line) for line in f["omega"]])
                axes[row, 0].plot(up, f["y"], color=color, ls=style, label=label)
                axes[row, 1].plot(f["x"], vp, color=color, ls=style)
                axes[row, 2].plot(pp, f["y"], color=color, ls=style)
                axes[row, 3].plot(wp, f["y"], color=color, ls=style)
            for col, (xlabel, ylabel) in enumerate(((r"$u/U$", r"$y/L$"), (r"$x/L$", r"$v/U$"),
                                                    (r"$p^*$", r"$y/L$"), (r"$\omega^*$", r"$y/L$"))):
                axes[row, col].set(xlabel=xlabel, ylabel=ylabel, title=f"Re = {re}")
                axes[row, col].grid(alpha=.15)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(left=.06, right=.985, top=.94, bottom=.12, wspace=.40, hspace=.50)
        for suffix in ("png", "pdf"):
            fig.savefig(MEDIA / ("grid_refinement_profiles." + suffix), dpi=170)
        plt.close(fig)
    write_json(RESULTS / "sensitivity.json", {"comparisons": comparisons,
        "meaning": "Matched-time observed differences; neither a rigorous error bound nor steady Ghia validation.",
        "pressure": "Native transient pressure, zero area mean on each grid before fine-to-coarse sampling."})
    print(f"Built physical observables and {len(comparisons)} matched sensitivity comparisons.")
    return cases, comparisons


def build_appendix():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    from qa.lecture_typography import register_reportlab
    if os.name == "nt":
        os.environ.setdefault("FLOWMLLAB_TNR_DIR", str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"))
    fonts = register_reportlab("StartupTNR")
    math_font = Path(matplotlib.get_data_path()) / "fonts/ttf/STIXGeneral.ttf"
    pdfmetrics.registerFont(TTFont("StartupMath", str(math_font)))
    styles = {
        "title": ParagraphStyle("title", fontName=fonts["bold"], fontSize=21, leading=24, spaceAfter=14),
        "heading": ParagraphStyle("heading", fontName=fonts["bold"], fontSize=14, leading=17, spaceAfter=8),
        "body": ParagraphStyle("body", fontName=fonts["regular"], fontSize=11.5, leading=15.2, spaceAfter=8),
        "caption": ParagraphStyle("caption", fontName=fonts["regular"], fontSize=10.2, leading=12.5, spaceAfter=8),
    }
    def P(text, style="body"):
        # Times New Roman lacks these mathematical codepoints. Keep prose in
        # the verified font and embed STIX only for the missing math symbol.
        text = text.replace("⟨p⟩", "mean(p)")
        text = text.replace("∇", '<font name="StartupMath">∇</font>')
        return Paragraph(text, styles[style])
    target = RESULTS / "cavity_startup_lecture_appendix.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = landscape(A4)
    doc = SimpleDocTemplate(str(target), pagesize=(width, height), rightMargin=38, leftMargin=38,
                            topMargin=34, bottomMargin=34, title="Week 1.2: Physical-time cavity startup", author="FlowMLLab")
    story = []
    def page(title):
        if story: story.append(PageBreak())
        story.append(P(title, "title"))
    def picture(path, max_width=740, max_height=390):
        from PIL import Image as PILImage
        with PILImage.open(path) as im: iw, ih = im.size
        ratio = min(max_width/iw, max_height/ih)
        return Image(str(path), width=iw*ratio, height=ih*ratio)
    movie = json.loads((MEDIA / "movies.json").read_text())
    choice, directory = presentation()
    controls = movie["cases"][0]["config"]
    reports = json.loads((RESULTS / "sensitivity.json").read_text())["comparisons"]
    page("Week 1.2 companion: How cavity circulation forms")
    story += [P("Aim and route", "heading"), P("Start a two-dimensional cavity from rest, follow its physical-time evolution, and use numerical sensitivity to decide what a small visible vortex means. The companion notebook runs on a CPU and adds a transient experiment to the existing pressure-velocity lesson; all earlier steady examples and exercises remain available."),
        P("1. Define the startup and time scale. 2. Separate physical steps from correction iterations. 3. Read three matched movies. 4. Measure energy and circulation. 5. Repeat a small CPU solve. 6. Test timestep, coupling and mesh sensitivity. 7. Submit a quantitative flow-development report."),
        P("Physical problem", "heading"), P("The square has side L. The fluid starts from rest. At t = 0+, its top lid moves to the right at constant speed U; the other walls stay stationary and all four walls are impermeable. Density is constant and the fluid is Newtonian. No gradual lid ramp is applied. The ideal top corners contain a wall-velocity discontinuity, so large corner gradients are physical/numerical features to inspect, not pixels to hide."),
        P("Nondimensional variables", "heading"), P("Re = UL/ν; x* = x/L; y* = y/L; t* = tU/L; u* = u/U; v* = v/U. Pressure is p* = (p − ⟨p⟩)/(ρU²), where ⟨p⟩ is the cell-area mean at the current time. Signed vorticity is ω* = (∂v/∂x − ∂u/∂y)L/U. The retained cases are Re = 100, 400, 1000, 3200, 5000, 7500 and 10000, matching the seven Ghia table values."),
        P("Scope of the evidence", "heading"), P("These are finite-time solutions of the two-dimensional discrete equations. Ghia's steady velocity tables are not a transient reference. A finite-time frame at high Re is neither proof of a steady state nor validation of three-dimensional turbulence. Matched 64-to-128-cell velocity differences at t* = 10 range from about 17% to 70% for Re = 3200 through 10000. Those high-Re panels are provisional, mesh-dependent teaching examples, not resolved flow predictions. The sensitivity study below measures these differences explicitly.")]
    page("The algorithm: one physical step, several corrections")
    story += [P("The dimensionless momentum balance is ∂u/∂t* + (u·∇)u = −∇p* + Re⁻¹∇²u, with ∇·u = 0. Here u denotes the two-component dimensionless velocity. The MAC mesh stores pressure at cell centers and each normal velocity at its corresponding cell face; shared face fluxes enforce a conservative mass balance."),
        P("Backward Euler and PIMPLE", "heading"), P("Backward Euler replaces the time derivative by (uⁿ⁺¹ − uⁿ)/Δt*. At each physical step, keep the previous-time velocity uⁿ fixed. Predict momentum using the current pressure and convecting velocity. Solve a pressure-correction equation and correct face velocities. Repeat pressure corrections and, for PIMPLE, outer nonlinear momentum predictions. Only after these loops does physical time advance. PISO uses a single outer momentum prediction. SIMPLE is a steady iteration algorithm and is not used as a movie clock."),
        P("Old-time velocity → momentum prediction → pressure correction → corrected face fluxes → repeat within-step corrections → advance physical time → save an observation frame", "heading"),
        P(f"The displayed trajectories use central convection, backward Euler, {controls['cells']} × {controls['cells']} MAC cells, Δt* = {controls['dt']:g}, {controls['outer_correctors']} outer corrections and {controls['pressure_correctors']} pressure corrections per outer loop. The physical interval is 0 ≤ t* ≤ 10. Frames are saved every 0.1 time unit, including t* = 0. Each frame comes from an actual timestep, with a shortened step when necessary to land on the observation time. Frames are not interpolated between solutions."),
        P("Four different checks", "heading"), P("Continuity checks the corrected face-flux balance. The timestep momentum defect tests the backward-Euler equation at the current step. The steady momentum defect omits local acceleration and therefore need not be small during startup. Timestep and mesh studies test discretization sensitivity. A small continuity defect does not replace the other three checks."),
        P("CPU controls", "heading"), P("The notebook's fresh default is a small 16 × 16 Re = 100 solve to t* = 0.2. Retained trajectories are loaded for immediate inspection. Increasing cells, reducing Δt*, extending final time or increasing corrector counts all add work. Saving fewer frames reduces storage, but changing a noncommensurate frame interval also changes the shortened-step schedule; use frame intervals that are integer multiples of Δt* when isolating storage cost.")]
    page("Measure development, not just visual similarity")
    story.append(picture(MEDIA / "startup_observables.png", max_height=270))
    story += [P("Kinetic energy is E* = ½∫∫(u*² + v*²) dx*dy*. The minimum streamfunction measures primary clockwise circulation in the adopted sign convention. Both observables come from the same native face fields as the movies. At a common t*, larger Re generally has a thinner developing viscous layer, so identical elapsed time does not imply identical progress toward a steady state. The viscous diffusion scale L²/ν equals Re in lid-travel time units; convection also affects development."),
        P("A streamline is tangent to the velocity at one instant. During startup it is generally not a fluid-particle pathline. We draw contours of the instantaneous streamfunction: solid for negative values, dashed for positive values. Levels are fixed across times and cases. A candidate secondary vortex needs closed contours plus an extremum of the opposite sign, and then a refinement check. A colored spot alone is insufficient."),
        P("Pressure uses a consistent zero-mean gauge at every saved time. Adding a spatial constant changes no pressure force; changing that constant between frames would nevertheless create misleading apparent pressure variation. Native transient pressure is retained because a steady pressure-recovery equation would omit local acceleration.")]
    for field, title in (("speed", "Speed and instantaneous streamlines"), ("omega", "Signed vorticity and circulation"), ("p", "Native transient pressure")):
        page(title)
        story.append(picture(MEDIA / f"{field}_t10.png", max_height=390))
        info = movie["fields"][field]
        limits = info["limits"]
        scale = "linear" if field == "speed" else f"symmetric-log, linear within ±{info['linear_threshold']:g}"
        story.append(P(f"All seven Reynolds numbers at t* = 10, with the same {controls['cells']}-cell mesh and displayed controls. The {scale} scale is fixed over every case and all stored times; displayed range [{limits['vmin']:.5g}, {limits['vmax']:.5g}] retains the full recorded extrema. Black curves show instantaneous streamfunction contours. {choice['qualification']}", "caption"))
    page("What changed under numerical refinement?")
    story.append(picture(MEDIA / "sensitivity_profiles.png", max_height=345))
    story += [P("At Re = 1000 and 7500, each comparison changes one control: 32 versus 64 cells; Δt* = 0.02 versus 0.01; or 3 outer/2 pressure versus 6 outer/4 pressure corrections. Energy, circulation and pressure RMS are plotted at identical physical times. Agreement of integral quantities does not guarantee agreement of every small vortex. Inspect local profiles and extrema as well.", "caption"),
        P("For mesh differences, the finer velocity, pressure and vorticity are evaluated at coarse cell centers using bilinear interpolation solely for the reported norm. Native saved fields and visualizations are unaltered. Pressure has zero area mean on each original grid. These are observed numerical sensitivities, not an external accuracy certificate.", "caption")]
    page("Sensitivity numbers and an engineering interpretation")
    data = [["Re / changed control", "Velocity relative L2", "Pressure RMS", "Vorticity RMS"]]
    for name, report in reports.items():
        if "grid64_128" in name: continue
        final = report["frames"][-1]
        data.append([name.replace("_", " "), f"{final['velocity_relative_l2']:.4g}",
                     f"{final['pressure_rms_difference']:.4g}", f"{final['vorticity_rms_difference']:.4g}"])
    table = Table(data, colWidths=[280, 145, 145, 145])
    table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,0), fonts["bold"]),
        ("FONTNAME",(0,1),(-1,-1),fonts["regular"]), ("FONTSIZE",(0,0),(-1,-1),11),
        ("BOTTOMPADDING",(0,0),(-1,-1),8), ("TOPPADDING",(0,0),(-1,-1),8),
        ("LINEBELOW",(0,0),(-1,0),.7,colors.black), ("LINEBELOW",(0,-1),(-1,-1),.7,colors.black)]))
    story += [table, Spacer(1,15), P("Final differences at t* = 10. Velocity relative L2 uses the RMS vector difference divided by the comparator's RMS velocity; it is undefined at the initial rest state. Pressure and vorticity entries are absolute RMS differences in their dimensionless units. A smaller timestep or more correction loops is a numerical comparator, not exact truth."),
        P("Interpret the largest sensitivity first. If more correctors materially alter a field, improve within-step convergence before attributing changes to time discretization. If halving Δt* matters, refine time before interpreting a mesh trend. If 32-to-64 differences remain substantial, the 64-cell picture is a visualization of a mesh-dependent solution, not a mesh-independent result. Vorticity differentiates velocity and can be more sensitive than kinetic energy. Corner extrema need special care because the imposed lid velocity is discontinuous there."),
        P("The 32-to-64-cell velocity differences at t* = 10 were 13.2% (Re = 1000) and 77.6% (Re = 7500), much larger than the timestep/corrector changes. This motivated a finer mesh rather than a convergence claim. A useful result includes the limitations. Do not smooth away oscillations, clip an inconvenient corner peak, change a color range between frames or call iteration count physical time. Retain weak/coarse results and explain what a refinement actually changes.")]
    refinement = {name: report for name, report in reports.items() if "grid64_128" in name}
    if refinement:
        page("A finer mesh at a matched timestep")
        data = [["Re", "Velocity relative L2", "Pressure RMS", "Vorticity RMS"]]
        for name, report in refinement.items():
            final = report["frames"][-1]
            data.append([str(report["candidate"]["reynolds"]), f"{final['velocity_relative_l2']:.4g}",
                         f"{final['pressure_rms_difference']:.4g}", f"{final['vorticity_rms_difference']:.4g}"])
        table = Table(data, colWidths=[160, 185, 185, 185])
        table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,0),fonts["bold"]),
            ("FONTNAME",(0,1),(-1,-1),fonts["regular"]), ("FONTSIZE",(0,0),(-1,-1),11),
            ("BOTTOMPADDING",(0,0),(-1,-1),8), ("TOPPADDING",(0,0),(-1,-1),8),
            ("LINEBELOW",(0,0),(-1,0),.7,colors.black), ("LINEBELOW",(0,-1),(-1,-1),.7,colors.black)]))
        story += [table, Spacer(1,14), P("Final differences at t* = 10 between 64 and 128 cells, both at Δt* = 0.01 with identical pressure-velocity coupling settings. All seven Reynolds numbers have matched physical frame times. The larger mesh improves spatial resolution but is not an exact solution. The displayed trajectories remain finite-resolution evidence wherever the differences are substantial."),
            P(choice["qualification"]), P("Use the next page to localize differences. An integral diagnostic can agree while the velocity derivative, a narrow wall layer or a small secondary circulation differs. Do not infer mesh independence from prettier streamlines.")]
        page("Locate the mesh differences in physical space")
        story.append(picture(MEDIA / "grid_refinement_profiles.png", max_height=365))
        story.append(P("Re = 1000 (top) and 7500 (bottom), at t* = 10 and Δt* = 0.01. Black solid: 64 cells; red dashed: 128 cells. Columns show u/U on the vertical centerline, v/U on the horizontal centerline, native zero-mean pressure p* on the vertical centerline, and signed vorticity ω* on the vertical centerline. Centerline values are interpolated only between the two adjacent native grid lines. The saved fields are not smoothed; plotted endpoints are the actual cell-center coverage.", "caption"))
    page("Student tasks and report checklist")
    tasks = [
        "1. At Re = 100, 400 and 1000, compare t* = 0.2, 1, 4 and 10. Submit matched speed/streamline frames and energy histories. Explain the role of wall shear and why equal time need not mean equal equilibration.",
        "2. Track the primary streamfunction minimum and its grid-node location. Identify a candidate secondary cell by closed contours and the opposite-sign extremum. Record first sampled appearance time, position and signed streamfunction, then test refinement. State the sampling interval's uncertainty in appearance time.",
        "3. At one moderate Re, halve Δt* at fixed mesh and physical observation times. Overlay both centerline velocity profiles and energy histories; report numerical differences and actual CPU time.",
        "4. Compare PISO, PIMPLE 3/2 and PIMPLE 6/4 corrections at fixed Δt*. Plot timestep momentum defects and physical profiles. Explain why small continuity alone does not establish accuracy and why more corrections add work.",
        "5. Compare 16, 32 and, if affordable, 64 cells. Overlay u at x/L = 0.5 and v at y/L = 0.5. Record vortex locations, signed pressure extrema and norm changes. Preserve the coarse result even when it looks poor.",
        "6. Plot one pressure profile at two physical times. Show that adding a spatial constant leaves pressure gradients unchanged. Explain the zero-mean gauge and why steady pressure recovery is inappropriate during startup.",
        "7. Audit one high-Re claim. State which evidence is missing before calling a small cell resolved or a t* = 10 frame steady. Use the Re = 7500 sensitivity data. Do not infer three-dimensional turbulent accuracy from a two-dimensional calculation.",
        "Submit: a concise report, three short movies or selected physical-time frames, exact numerical settings, measured CPU cost, matched sensitivity results and compact native trajectories with configuration/checksums. Show physical profiles and time histories instead of bar charts."]
    story.extend(P(t) for t in tasks)
    story.append(P("Reference: U. Ghia, K. N. Ghia and C. T. Shin, High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method, J. Comput. Phys. 48 (1982) 387-411. DOI: 10.1016/0021-9991(82)90058-4. This reference supplies steady velocity benchmarks; the present appendix addresses startup and numerical sensitivity.", "caption"))
    def footer(canvas, document):
        canvas.saveState(); canvas.setFont(fonts["regular"], 9)
        canvas.drawString(38, 17, "FlowMLLab | Week 1.2 physical-time startup companion")
        canvas.drawRightString(width-38, 17, str(document.page)); canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)
    return target


def build_readme():
    from qa.build_cavity_startup import load_cases
    choice, directory = presentation()
    cases = load_cases(directory, REYNOLDS)
    c = cases[0]["config"]
    reports = json.loads((RESULTS / "sensitivity.json").read_text())["comparisons"]
    lines = [
        "# How cavity circulation develops", "",
        "A physical-time companion to Week 1.2: follow the lid-driven cavity from rest, inspect speed, signed vorticity and pressure, and test which structures survive changes in numerical resolution.", "",
        "[CPU notebook](../../notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb) · [Complete Week 1.2 lecture](../../lectures/week01_2_pressure_velocity.pdf) · [Standalone startup appendix](cavity_startup_lecture_appendix.pdf)", "",
        "## Watch the flow", "",
        "[Speed and instantaneous streamlines](media/cavity_startup_speed.mp4) · [Signed vorticity](media/cavity_startup_omega.mp4) · [Signed pressure](media/cavity_startup_p.mp4)", "",
        "All seven cases use Re = 100, 400, 1000, 3200, 5000, 7500 and 10000. The square starts from rest; the top lid starts impulsively and moves right at unit speed. The other walls are stationary and all walls are impermeable. There is no lid ramp.", "",
        "| Shared control | Displayed setting |", "|---|---|",
        f"| Mesh | {c['cells']} × {c['cells']} MAC cells |",
        f"| Physical interval | 0 ≤ tU/L ≤ {cases[0]['times'][-1]:g} |",
        f"| Maximum timestep | ΔtU/L = {c['dt']:g} |",
        f"| Coupling | {c['method'].upper()}, {c['outer_correctors']} outer / {c['pressure_correctors']} pressure corrections |",
        f"| Convection | {c['convection']} |",
        "| Saved frames | Every 0.1 physical-time unit, including rest |", "",
        choice["qualification"], "",
        "<img src=\"media/omega_t10.png\" width=\"1000\" alt=\"Signed vorticity and instantaneous streamlines for seven Reynolds numbers at physical time tU/L=10\">", "",
        "The domain, field limits and streamline levels stay fixed across cases and times. Speed has a linear scale. Signed vorticity uses a symmetric-log scale, linear within |ωL/U| ≤ 1; native zero-mean pressure uses a symmetric-log scale, linear within |p/(ρU²)| ≤ 0.01. Full recorded extrema are retained. Contour drawing interpolates between native samples; no scientific field is filtered or clipped. Solid streamfunction contours are negative and dashed contours positive. Instantaneous streamlines are not particle pathlines.", "",
        "## What the refinement tests showed", "",
        "The first 32-to-64-cell comparison changed the final velocity field by 13.2% at Re = 1000 and 77.6% at Re = 7500 in the stated relative L2 norm. Those changes were much larger than the timestep/corrector sensitivities, motivating a finer mesh. A smooth picture alone did not establish accuracy.", "",
        "| Matched 64-to-128-cell study, tU/L = 10 | Velocity relative L2 difference | Pressure RMS difference | Vorticity RMS difference |",
        "|---|---:|---:|---:|"]
    for name, report in reports.items():
        if "grid64_128" not in name: continue
        row = report["frames"][-1]
        lines.append(f"| Re = {report['candidate']['reynolds']:g} | {100*row['velocity_relative_l2']:.3g}% | {row['pressure_rms_difference']:.4g} | {row['vorticity_rms_difference']:.4g} |")
    lines += ["", "These comparisons keep ΔtU/L = 0.01 and coupling settings fixed. Fine fields are sampled at coarse cell centers solely for the numerical difference. Pressure has zero area mean on each original grid. These are observed sensitivities, not an external error bound or a steady Ghia validation.", "",
        "<img src=\"media/grid_refinement_profiles.png\" width=\"1000\" alt=\"Matched-time velocity, pressure and vorticity centerline profiles on 64 and 128 cells at Reynolds 1000 and 7500\">", "",
        "## Student work", "",
        "The notebook defines eight concrete tasks: measure circulation development and vortex appearance; halve the timestep; vary PISO/PIMPLE correction counts; compare grids; test the pressure gauge; audit a high-Re claim; and submit physical profiles, time histories, exact settings and measured CPU costs. Its default fresh solve uses 16 × 16 cells at Re = 100 to tU/L = 0.2, while the larger verified trajectories are loaded for immediate inspection. A GPU is not required.", "",
        "## Evidence and reproduction", "",
        f"- `{choice['trajectory_directory']}/`: displayed native staggered velocities, transient pressure, exact frame times, actual-step diagnostics and configuration in compressed NPZ files; matching JSON receipts contain SHA-256 hashes, solver-source hashes and recorded CPU times.",
        "- `base/`, `mesh32/`, `dt_half/`, `correctors/`, `matched64/`: retained controlled numerical comparisons; these files are scientific evidence, not disposable animation frames.",
        "- [Sensitivity values](sensitivity.json), [movie provenance](media/movies.json), [presentation selection](presentation.json), and [Unity run receipt](unity_runs.json).",
        "- [Frozen simulation source](simulation_source.tar.gz) and its [manifest](simulation_source_manifest.json) preserve the code actually run. Current rendering/admission hashes are identified separately from the simulation source hashes.", "",
        "From the repository root:", "", "```bash",
        "python qa/build_cavity_startup.py solve --reynolds 100 --cells 32 --dt .02 --end-time 4 --interval .1 --output results/cavity_startup/student_experiment",
        "python qa/build_cavity_startup.py render --input results/cavity_startup/student_experiment --output results/cavity_startup/student_experiment --reynolds 100",
        "python qa/build_cavity_startup_learning.py all",
        "python qa/build_cavity_startup_learning.py execute",
        "python qa/append_cavity_startup_lecture.py", "```", "",
        "Rebuilding the teaching documents requires the repository test extras, PyMuPDF (`python -m pip install -e \".[test]\" \"PyMuPDF>=1.24,<2\"`) and Times New Roman via `FLOWMLLAB_TNR_DIR`. The `all` action regenerates notebook source; `execute` then runs that notebook in a fresh CPU kernel and renews its execution receipt. The append command preserves and verifies all 74 existing lecture pages.", "",
        "The solve command refuses to overwrite existing scientific evidence. Use a new directory for a new configuration. FFmpeg is needed only for MP4 export. The published observation times correspond to actual solution steps; neither steady SIMPLE iterations nor interpolated steady solutions are used as a physical clock.", ""]
    path = RESULTS / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def execute_notebook():
    """Execute the actual CPU defaults in a fresh kernel and verify figure outputs."""
    from datetime import datetime, timezone
    import hashlib
    from time import perf_counter
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from qa.build_cavity_startup import write_json, sha256
    work = ROOT / "tmp/cavity_startup/notebook_runtime"
    for key, folder in (("JUPYTER_RUNTIME_DIR", "jupyter_runtime"),
                        ("JUPYTER_CONFIG_DIR", "jupyter_config"),
                        ("IPYTHONDIR", "ipython"), ("TEMP", "temp"), ("TMP", "temp")):
        location = work / folder; location.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(location)
    path = ROOT / "notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source_hash = hashlib.sha256(json.dumps([{"cell_type": c.cell_type, "source": c.source}
        for c in notebook.cells], sort_keys=True).encode()).hexdigest()
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=300, startup_timeout=300, resources={"metadata": {"path": str(ROOT)}})
    start = perf_counter()
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    elapsed = perf_counter()-start
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    outputs = [out for c in code_cells for out in c.outputs]
    png_count = sum("image/png" in out.get("data", {}) for out in outputs)
    video_count = sum("<video" in out.get("data", {}).get("text/html", "") for out in outputs)
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(out.output_type == "error" for out in outputs)
    assert png_count >= 5, f"Expected actual figure outputs, found {png_count}"
    assert video_count == 3, f"Expected three movie players, found {video_count}"
    path.write_text(nbformat.writes(notebook)+"\n", encoding="utf-8", newline="\n")
    receipt = {"schema": 1, "executed_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": str(path.relative_to(ROOT)).replace("\\", "/"),
        "notebook_sha256": sha256(path), "cell_source_sha256": source_hash,
        "code_cells": len(code_cells), "executed_code_cells": len(code_cells),
        "figure_outputs": png_count, "video_players": video_count, "error_outputs": 0,
        "runtime_seconds": elapsed, "kernel_python": sys.executable,
        "fresh_default": {"Re": 100, "cells": 16, "dt": .02, "end_time": .2,
            "method": "pimple", "outer_correctors": 3, "pressure_correctors": 2},
        "presentation": json.loads((RESULTS / "presentation.json").read_text()),
        "movie_manifest_sha256": sha256(MEDIA / "movies.json"),
        "source_sha256": {name: sha256(ROOT / name) for name in (
            "flowmllab/cavity_startup.py", "qa/build_cavity_startup.py", "qa/build_cavity_startup_learning.py")},
        "scope": "Fresh local CPU kernel; default cells executed. Hosted Colab was not separately claimed."}
    write_json(RESULTS / "notebook_execution.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["notebook", "evidence", "appendix", "readme", "execute", "all"])
    args = parser.parse_args()
    if args.action in ("notebook", "all"): build_notebook()
    if args.action in ("evidence", "all"): build_evidence()
    if args.action in ("appendix", "all"): build_appendix()
    if args.action in ("readme", "all"): build_readme()
    if args.action == "execute": execute_notebook()
