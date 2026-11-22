"""Publish an additive, CPU-only lesson from the closed cavity campaign.

Build never runs CFD. Execute checks the saved-data notebook in a fresh kernel.
The original startup lesson, movies, exercises and lecture remain intact.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / "tmp/cavity_settling"
os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf

from flowmllab.cavity_settling_results import ResultBundle, REYNOLDS, overview, grid_fields, late_profiles

BUNDLE = ROOT / "results/cavity_settling"
NOTEBOOK = ROOT / "notebooks/week01_2/W1_2_Cavity_Stationarity_and_Resolution.ipynb"
GITHUB = "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/"
COLAB = "https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def table(rows):
    head = "| Re | Baseline end t* | Last full window | Failed gates / 16 | 6/4 confirmation end t* | Conclusion |\n| ---: | ---: | --- | ---: | ---: | --- |"
    body = []
    for r in rows:
        confirmation = f"{r['confirmation_end_time']:g}" if r['confirmation_end_time'] is not None else "—"
        body.append(f"| {r['Re']} | {r['baseline_end_time']:g} | {r['last_window_start']:g}–{r['last_window_end']:g} | {r['failed_gates']} | {confirmation} | {r['conclusion']} |")
    return head + "\n" + "\n".join(body)


def notebook():
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s): cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
    md(r"""
    # Week 1.2 companion — When has a cavity flow settled?

    **Aim.** Use real saved CFD results to distinguish a finished run, a temporally
    stationary numerical solution, and a solution that is accurate with respect
    to grid spacing and time step. Inspect the physical flow before summarising numbers.

    This companion follows the [pressure–velocity lesson](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb)
    and [startup movies](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb).
    It adds the closed continuation campaign: 22 numerical configurations,
    seven long baseline trajectories, three confirmation branches, and 16
    controlled comparisons. The original lessons and exercises are unchanged.

    **CPU route:** Run all loads retained arrays, verifies each opened file, and
    produces tables and physical figures. It does not launch a CFD run, require
    a GPU, or repeat the long campaign. Downloaded data are about 44 MB before
    lesson figures and notebook outputs. Exact timing depends on your computer.

    **Learning outcomes:** explain the difference between an equation residual and
    temporal change; audit a full-window decision; compare profiles at exactly
    matched times; separate grid, timestep and coupling effects; write a qualified
    conclusion when higher-Re flow remains unresolved.
    """)
    md("## 1. Setup and data identity\n\nIn Colab, the cell obtains only the course code and this result bundle. A local checkout is used directly. Each scientific payload is checked against its recorded SHA-256 before use.")
    code(r"""
    from pathlib import Path
    import os, sys, subprocess
    if "google.colab" in sys.modules:
        ROOT = Path("/content/FlowMLLab_cavity_results")
        if not ROOT.exists():
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(ROOT)], check=True)
            subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "set",
                            "flowmllab", "qa", "results/cavity_settling"], check=True)
    else:
        ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / "results/cavity_settling/manifest.json").is_file())
    sys.path.insert(0, str(ROOT))
    from IPython.display import display, Markdown
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from flowmllab.cavity_settling_results import ResultBundle, REYNOLDS, overview, grid_fields, late_profiles
    from qa.plot_cavity_settling import history_figure, profile_figure
    get_ipython().run_line_magic("matplotlib", "inline")
    bundle = ResultBundle(ROOT / "results/cavity_settling")
    analysis = bundle.analysis
    print(f"Admitted configurations: {len(bundle.cases)}; controlled comparisons: {len(analysis['comparisons'])}")
    print("Decoded analysis SHA-256:", bundle.manifest["original_analysis_sha256"])
    """)
    md(r"""
    ## 2. Physical problem and numerical controls

    The square cavity has side length $L$ and an impulsively started lid moving
    in the positive $x$ direction at $U_{\mathrm{lid}}$. The fluid is initially at
    rest. The other three walls are stationary; all walls are impermeable. The
    problem is two-dimensional and incompressible, with
    $Re=U_{\mathrm{lid}}L/\nu$ and $t^*=tU_{\mathrm{lid}}/L$.

    Long baselines use $128\times128$ staggered finite-volume cells,
    $\Delta t^*=0.01$, central convection, and PIMPLE with 3 outer corrections
    and 2 pressure corrections. Pressure is stored in its zero-area-mean gauge:
    $p^*=(p-\langle p\rangle)/(\rho U_{\mathrm{lid}}^2)$.
    Signed vorticity is $\omega_z^*=L(\partial v/\partial x-\partial u/\partial y)/U_{\mathrm{lid}}$.

    A **candidate** must pass all 16 frozen engineering gates in **three
    consecutive full physical windows**. A partial first continuation window
    does not count. The window is 20 time units for $Re\le3200$ and 50 for
    $Re\ge5000$. A separate branch changes only PIMPLE corrections from 3/2 to
    6/4 and tests a further full window against the candidate's final window.
    It is an independent numerical confirmation, not an independent random seed.

    Passing a time-step momentum defect says the implicit step was solved well.
    Passing the **steady** momentum defect additionally tests the equations
    without the time derivative. Neither proves spatial accuracy. A picture
    that appears stationary is not a substitute for these checks.
    """)
    code(r"""
    # Change this to any member of REYNOLDS to inspect its frozen policy.
    RE = 1000
    policy = bundle.json(f"provenance/policies/re{RE}.json")
    display(pd.DataFrame([{"metric": k, "frozen_limit": v}
                          for k, v in policy["thresholds"].items()]))
    print("Window duration:", policy["window_duration"], "Required consecutive windows:", policy["consecutive_windows"])
    print(policy["criterion_origin"])
    """)
    md(r"""
    The field-change gates use the maximum in time of spatial RMS and
    $L_\infty$ changes relative to the window-start field. Adjacent-window means
    test slow drift. For a sampled scalar $q$, a window range is
    $\max_W q-\min_W q$. A spatial RMS of a velocity change is
    $[(\sum_{i=1}^{N_u}(\Delta u_i)^2+\sum_{j=1}^{N_v}(\Delta v_j)^2)/(N_u+N_v)]^{1/2}$,
    where $N_u$ and $N_v$ are the numbers of native staggered velocity samples.
    Full definitions are in the retained [evaluator](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/cavity_settling/provenance/source/flowmllab/cavity_startup_continuation.py).
    Gates are accumulated at every solver step. The lighter plotted history
    is sampled for display and must not be used to reconstruct an unobserved maximum.
    """)
    md("## 3. What actually finished?\n\nEvery scheduled trajectory in this campaign is closed. The table below distinguishes that administrative fact from numerical stationarity. A confirmed row is limited to its fixed grid, time step, two-dimensional model and tested coupling controls.")
    code("outcomes = pd.DataFrame(bundle.outcomes())\ndisplay(outcomes)\nassert outcomes.loc[outcomes.fixed_grid_confirmed, 'Re'].tolist() == [100, 400, 1000]")
    md("## 4. Inspect the native physical fields\n\nFirst compare the seven baselines at the same physical time, t* = 100. Then inspect their own final times, which differ. Common spatial bounds and one full-extrema color scale are used within each figure. Streamfunction contours show instantaneous streamlines, not particle trajectories. Pressure and vorticity retain their signs; symmetric-log color normalization only changes how colors are assigned.")
    code("fig = overview(bundle, 'speed', time=100.)\nplt.show()")
    code("# These are deliberately labelled with their different actual end times.\nfor field in ['speed', 'p', 'omega']:\n    fig = overview(bundle, field)\n    plt.show()")
    md("## 5. A late-time profile is evidence, not a pass/fail decision\n\nChoose a Reynolds number and inspect centerline velocities and both pressure profiles. The first trace is t* = 100; the second is the recorded baseline endpoint if different. A third trace is shown only when a separate 6/4 confirmation exists. Native cell centers are retained and no wall-pressure samples are invented.")
    code("RE = 1000  # try 3200, 5000, 7500 or 10000 next\nfig = late_profiles(bundle, RE)\nplt.show()")
    code(r"""
    case = bundle.baseline(RE)
    fig, history_record, legends = history_figure(case["id"], analysis["cases"][case["id"]])
    plt.show()
    windows = bundle.windows(case["id"])
    display(pd.DataFrame([{k: w[k] for k in ['start_time', 'end_time', 'partial_initial_window',
                                           'passed', 'consecutive_passes', 'failed_metrics']}
                          for w in windows]))
    """)
    md("The upper-left and central probes in the history are fixed positions, not tracked vortex centers. Their signed signals can change when a vortex moves past them. Use the full field and streamfunction extrema together with the probes.")
    code(r"""
    # Audit every metric in the last full baseline window; do not relax a limit.
    policy = bundle.json(f"provenance/policies/re{RE}.json")
    last = [w for w in windows if not w['partial_initial_window']][-1]
    gate_table = pd.DataFrame([{
        'metric': name, 'observed': last['metrics'][name], 'limit': limit,
        'passes': last['metrics'][name] is not None and last['metrics'][name] <= limit
    } for name, limit in policy['thresholds'].items()])
    display(gate_table)
    if RE in bundle.confirmations:
        report = bundle.json(bundle.confirmations[RE]['confirmation_file'])
        display(pd.DataFrame([{'metric': k, 'observed': report['metrics'][k], 'limit': v,
                               'passes': report['metrics'][k] <= v}
                              for k, v in report['thresholds'].items()]))
        print('Separate confirmation:', report['parent_time'], '→', report['actual_time'],
              'passed =', report['independent_confirmation_passed'])
    else:
        print('No stationarity confirmation was justified for this baseline.')
    """)
    md(r"""
    ## 6. All controlled comparisons use exactly t* = 10

    We cannot compare a coarse run at t* = 200 with a fine run at t* = 10 and
    call the difference grid error. The 16 controls below all contain actual
    samples at t* = 10. No nearest-time choice or temporal interpolation is used.

    For cell-centered velocity, the reported relative difference is
    $D_u=\|\mathbf{u}_c-I\mathbf{u}_r\|_2/\|I\mathbf{u}_r\|_2$.
    Here $c$ is the candidate, $r$ a numerical comparator, and $I$ is bilinear
    spatial sampling of fine cell-center values onto coarse cell centers when
    needed. At identical grid spacing, $I$ is the identity. The comparator is
    not an exact solution, so $D_u$ is sensitivity, not a certified error bound.
    Native centerline profiles do not use this fine-to-coarse remapping.
    """)
    code("controls = pd.DataFrame(bundle.comparisons())\ncontrols['velocity_difference_percent'] = 100 * controls.velocity_relative_l2\ndisplay(controls[['id', 'kind', 'Re', 'time', 'velocity_difference_percent', 'pressure_rms_difference']])\nassert len(controls) == 16 and (controls.time == 10).all()")
    code(r"""
    # Choose any ID from the table. Change only one numerical control at a time.
    pair = next(p for p in analysis['comparisons'] if p['kind'] == 'grid'
                and bundle.cases[p['candidate_id']]['config']['reynolds'] == 7500)
    fig, profile_record, legends = profile_figure(pair, analysis['cases'][pair['candidate_id']],
                                                 analysis['cases'][pair['reference_id']], 10.)
    plt.show()
    fig = grid_fields(bundle, 7500)  # also available: 3200, 5000, 10000
    plt.show()
    """)
    md(r"""
    ## 7. Interpretation of the closed campaign

    - **Re = 100, 400, 1000:** three full passing baseline windows followed by
      a passing 6/4 confirmation support fixed-grid temporal stationarity.
    - **Re = 3200, 5000, 7500, 10000:** the recorded endpoints remain unresolved.
      This is not proof that their physical flow must be unsteady. A longer
      integration, spatial resolution, coupling accuracy and the 2D model must
      be considered separately.
    - At t* = 10, 128 versus 256 cells gives velocity differences of about
      **4.23%, 7.77%, 32.00% and 55.31%**, respectively, for these four Reynolds
      numbers at the same time step, 0.0025. The large high-Re sensitivity is a
      reason to withhold an accuracy claim, even if a late movie looks settled.
    - The fine controls stop at t* = 10. They do not establish grid independence
      of the late endpoints. This campaign is not a new Ghia benchmark validation
      and does not establish 3D stability or transition.

    The configuration's legacy `end_time=10` describes the seed run. For continued
    data, use the native snapshot `time`, the status and window endpoints; do not
    mistake that seed configuration value for the actual continuation time.
    """)
    md(r"""
    ## 8. Student investigations and required deliverables

    Work with the retained data first. Keep the frozen limits unchanged.

    | Investigation | Action | Deliverable and decision |
    | --- | --- | --- |
    | A — Stationarity audit | Compare Re = 1000 and 3200. List each of the final three full windows and every failed gate. Inspect their late profiles. | One table, two physical-profile figures and a paragraph explaining why one case is confirmed and the other remains unresolved. |
    | B — Numerical coupling | For Re = 1000, compare 3/2 and 6/4 at t* = 10, then inspect the separate 240–260 confirmation. | State which variables were held fixed; explain why the matched-time test and the confirmation answer different questions. |
    | C — Timestep and mesh | At Re = 7500 and 10000, compare all available timestep controls on 128 cells and the 128/256 mesh pair at t* = 10. | Use velocity/pressure profiles and the sensitivity values; identify which control produces the larger observed difference without declaring the comparator to be truth. |
    | D — Upper-left vortex | Inspect speed/streamlines, signed vorticity, fixed-probe history and streamfunction extrema at Re = 3200 and 7500. | Explain what is evidence of a secondary vortex and what could be a probe crossing, discretization effect or unresolved evolution. A probe is not a moving vortex center. |
    | E — Engineering conclusion | Choose one unresolved case and propose the next discriminating numerical check. | Specify grid, time step, actual common comparison times, full-window length, cost implication and success criterion before any new run. No new solve is required for this notebook. |

    **Submission:** a short PDF with the five deliverables, the comparison table
    exported below, and the notebook with outputs. Include Re, grid, time step,
    correction counts, actual times and pressure gauge in every comparison.
    Explain an unsuccessful or unresolved test rather than hiding it.

    **Assessment:** 30% defensible stationarity audit; 30% matched controls and
    physical profiles; 20% interpretation of vortices and numerical limitations;
    20% reproducibility and the next-test proposal.
    """)
    code(r"""
    # Optional export: set SAVE_TABLE to True after choosing your investigation.
    SAVE_TABLE = False
    if SAVE_TABLE:
        destination = ROOT / 'student_outputs/cavity_stationarity'
        destination.mkdir(parents=True, exist_ok=True)
        controls.to_csv(destination / 'matched_controls.csv', index=False)
        outcomes.to_csv(destination / 'stationarity_outcomes.csv', index=False)
        print(destination)
    """)
    md("## 9. Complete evidence and further reading\n\n[All final figures, native samples and provenance](https://github.com/Ehsan-Roohi/FlowMLLab/tree/main/results/cavity_settling) · [Stationarity definitions and optional continuation commands](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/docs/CAVITY_STARTUP_STATIONARITY.md) · [Complete Week 1.2 lecture](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/lectures/week01_2_pressure_velocity.pdf)\n\nThe 0–10 startup lesson and 0–100 movie viewer remain separate. Every plotted final field in this companion is linked to a verified native snapshot; no field is inferred from a movie frame. Full solver restart sets remain in the research archive, outside the teaching repository.")
    for i, cell in enumerate(cells): cell['id'] = f'cavity-final-{i:02d}'
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "flowmllab": {"lesson": "week01_2_stationarity_results", "requires_gpu": False, "default_starts_cfd": False}})
    nbf.validate(nb)
    NOTEBOOK.write_text(nbf.writes(nb) + "\n", encoding="utf-8", newline="\n")


def _save(fig, stem):
    directory = BUNDLE / "lesson_figures"
    directory.mkdir(exist_ok=True)
    records = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        records.append({"file": str(path.relative_to(BUNDLE)).replace('\\', '/'),
                        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    return records


def readme(bundle, records):
    rows = bundle.outcomes()
    lines = ["# Cavity continuation: final results and numerical sensitivity", "",
        "[Open the CPU notebook](../../notebooks/week01_2/W1_2_Cavity_Stationarity_and_Resolution.ipynb) · "
        f"[Run in Colab]({COLAB}{NOTEBOOK.relative_to(ROOT).as_posix()}) · "
        "[Browse the figures](https://ehsan-roohi.github.io/FlowMLLab/cavity-settling/)", "",
        "All 22 configurations and three confirmation branches are closed. **Re = 100, 400 and 1000 pass the fixed-grid temporal stationarity checks and separate tighter-coupling confirmations.** The four higher-Re cases remain unresolved at their recorded endpoints. Completion of the scheduled work is not a claim that every flow became steady.", "",
        "## Baseline and confirmation outcomes", "", table(rows), "",
        "Baselines: 128 × 128 staggered cells, Δt* = 0.01, central convection, PIMPLE with 3 outer/2 pressure corrections. Confirmations change only the correction counts to 6/4. The 2D unit square starts from rest with an impulsive unit-speed lid; the other walls are stationary and all walls are impermeable. Time is t* = t U_lid/L. Pressure retains the native zero-area-mean gauge.", "",
        "A candidate requires all 16 frozen gates in three consecutive full windows (length 20 for Re ≤ 3200; 50 for Re ≥ 5000), then a further full confirmation window. Partial first windows do not count. Gates are accumulated at every solver step; sampled plot histories are not a replacement for the gate evaluator. These are frozen engineering tolerances, not thresholds quoted from Ghia.", "",
        "## Physical flow", "",
        "![Speed and streamlines at all seven own endpoints](lesson_figures/endpoints_speed.png)", "",
        "**Different endpoint times are written on the panels.** These panels show the attained state, not a matched-time Reynolds-number comparison. The separate [t* = 100 comparison](lesson_figures/matched_t100_speed.pdf) uses identical physical time. All figures retain full field extrema and native values; no smoothing is used.", "",
        "[Signed pressure](lesson_figures/endpoints_p.pdf) · [Signed vorticity](lesson_figures/endpoints_omega.pdf) · [Speed/streamlines PDF](lesson_figures/endpoints_speed.pdf)", "",
        "## All 16 matched numerical comparisons", "",
        "Every comparison below is evaluated at the exact recorded t* = 10. The grid controls use Δt* = 0.0025. A numerical comparator is not truth. Velocity differences are relative spatial L2 norms; fine-to-coarse bilinear cell-center sampling is used only for mesh difference norms. Plotted profiles preserve each grid's cell-center samples, and do not invent pressure at walls.", "",
        "| Control | Re | Relative velocity difference (%) | Profiles |", "| --- | ---: | ---: | --- |"]
    retained = bundle.json('provenance/plot_receipt.json')
    profile_records = {r['comparison_id']: r for r in retained['figures'] if r['kind'] == 'profiles'}
    for row in bundle.comparisons():
        record = profile_records[row['id']]
        pdf = next(r['name'] for r in record['files'] if r['name'].endswith('.pdf'))
        lines.append(f"| {row['id']} | {row['Re']} | {100*row['velocity_relative_l2']:.5f} | [PDF](figures/{pdf}) |")
    lines += ["", "Mesh sensitivity is about 4.23%, 7.77%, 32.00% and 55.31% at Re = 3200, 5000, 7500 and 10000. These early-time controls do **not** establish late-time grid independence. The large high-Re sensitivity prevents an accuracy claim based on an apparently settled movie. An unresolved endpoint is not proof of physical unsteadiness, 3D instability or turbulence.", "",
              "![Matched grid fields at Re 7500](lesson_figures/grid_re7500.png)", "",
              "Filled 256 × 256 reference field and 128 × 128 dashed contours use identical physical levels; both are at t* = 10 and Δt* = 0.0025. Native signed pressure uses symmetric-log normalization, linear within ±0.01. The fine grid is a numerical comparator.", "",
              "## Every baseline's histories and late profiles", "", "| Re | Signed probes, energy, streamfunction and equation defects | Late velocity and pressure profiles |", "| ---: | --- | --- |"]
    histories = {r['case_id']: r for r in retained['figures'] if r['kind'] == 'history'}
    for re in REYNOLDS:
        case = bundle.baseline(re)
        pdf = next(r['name'] for r in histories[case['id']]['files'] if r['name'].endswith('.pdf'))
        lines.append(f"| {re} | [History](figures/{pdf}) | [Profiles](lesson_figures/late_profiles_re{re}.pdf) |")
    lines += ["", "The upper-left and central probes are fixed spatial positions, not tracked vortex centers. Late profiles label each actual time and any tighter-coupling branch separately.", "",
        "## Reproducible evidence", "",
        "- [Summary JSON](summary.json), [stationarity outcomes CSV](stationarity_outcomes.csv) and [all matched controls CSV](matched_controls.csv).",
        "- [Scientific manifest](manifest.json): SHA-256 and sizes for all 248 exported scientific files; [full analysis, gzip JSON](data/analysis.json.gz).",
        "- [Native fields](data/fields): all 22 configurations at exact t* = 10, all seven baselines at t* = 100 and their own endpoints, and three confirmation endpoints. These selected snapshots are not full restart sets.",
        "- [All baseline/control windows and diagnostic histories](data/cases), [three confirmation reports](data/confirmations), and [frozen policies/source capsule](provenance).",
        "- [Retained 23 analysis figures](figures) and [additional native-field/late-profile figures](lesson_figures). Every plotted input has a verified source identity.", "",
        "The complete analysis is published losslessly; selected native fields make the figures reproducible without copying every frequent checkpoint into Git. Full recovery sets and intermediate native snapshots remain in the research storage. The configuration's legacy end_time = 10 identifies the original seed, while each native snapshot/status records its actual continuation time. This publication does not supersede the [original startup lesson](../cavity_startup/README.md) or the [0–100 movies](../cavity_continuation/README.md).", "",
        "```bash", "python qa/build_cavity_settling_lesson.py build", "python qa/build_cavity_settling_lesson.py execute", "python qa/build_cavity_settling_lesson.py check", "```", "",
        "These commands read and plot stored results. They do not submit or resume CFD jobs. The companion concludes with five specific student investigations and required deliverables.", ""]
    (BUNDLE / "README.md").write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def page(bundle, records):
    esc = html.escape
    rows = bundle.outcomes()
    outcome_rows = ''.join(f"<tr><td>{r['Re']}</td><td>{r['baseline_end_time']:g}</td><td>{r['failed_gates']}/16</td><td>{esc(r['conclusion'])}</td></tr>" for r in rows)
    figures = [('lesson_figures/matched_t100_speed', 'All seven baselines at the same t* = 100.'),
               ('lesson_figures/endpoints_speed', 'Own baseline endpoints. Times differ and are labelled.'),
               ('lesson_figures/endpoints_p', 'Signed native pressure, with zero area mean.'),
               ('lesson_figures/endpoints_omega', 'Signed vorticity. Full extrema are retained.')]
    figures += [(f'lesson_figures/grid_re{re}', f'Re = {re}: 128/256-cell comparison at exact t* = 10; Δt* = 0.0025.') for re in (3200,5000,7500,10000)]
    figures += [(f'lesson_figures/late_profiles_re{re}', f'Re = {re}: late physical profiles at the labelled actual times.') for re in REYNOLDS]
    retained = bundle.json('provenance/plot_receipt.json')
    figures += [('figures/' + Path(next(f['name'] for f in r['files'] if f['name'].endswith('.png'))).stem,
                 ('Baseline history: ' + r['case_id']) if r['kind'] == 'history' else ('Matched control: ' + r['comparison_id'])) for r in retained['figures']]
    gallery = ''.join(f'<figure><a href="{esc(stem)}.pdf"><img src="{esc(stem)}.png" alt="{esc(caption)}" loading="lazy"></a><figcaption>{esc(caption)} <a href="{esc(stem)}.pdf">PDF</a></figcaption></figure>' for stem,caption in figures)
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Cavity stationarity — FlowMLLab</title><style>
    body{{font:17px/1.65 system-ui,sans-serif;color:#172b40;background:#f6f8fa;margin:0}}main{{max-width:1160px;margin:auto;padding:32px 24px}}h1{{font-size:2.2rem;line-height:1.2}}a{{color:#075ea6}}figure{{margin:36px 0;padding:18px;background:white;border:1px solid #dbe3eb;border-radius:12px}}img{{width:100%;height:auto}}figcaption{{margin-top:14px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #dbe3eb;text-align:left}}.table{{overflow-x:auto}}nav{{display:flex;gap:20px;flex-wrap:wrap}}.note{{border-left:4px solid #b37417;padding:14px 20px;background:#fff7e8}}</style></head><body><main>
    <nav><a href="../">FlowMLLab</a><a href="{GITHUB}{NOTEBOOK.relative_to(ROOT).as_posix()}">Notebook</a><a href="{COLAB}{NOTEBOOK.relative_to(ROOT).as_posix()}">Run on CPU in Colab</a><a href="{GITHUB}results/cavity_settling/README.md">Data and provenance</a></nav>
    <h1>When has a cavity flow settled?</h1><p>Final results from 22 numerical configurations, seven long trajectories and three tighter-coupling confirmation branches. The companion notebook makes the stationarity audit and all 16 matched controls reproducible from saved data.</p>
    <p class="note">Re = 100, 400 and 1000 pass fixed-grid temporal stationarity checks. The higher-Re endpoints remain unresolved. All runs have finished, but this does not establish that every case is steady or spatially converged.</p>
    <div class="table"><table><thead><tr><th>Re</th><th>Baseline end t*</th><th>Failed final gates</th><th>Conclusion</th></tr></thead><tbody>{outcome_rows}</tbody></table></div>
    <p>Unit square; impulsive unit-speed lid; stationary other walls; initially stationary fluid. Baselines: 128 × 128 cells, Δt* = 0.01, PIMPLE 3/2, central convection. Confirmation branches change only PIMPLE corrections to 6/4. Native pressure has zero area mean. Signed pressure and vorticity use symmetric-log color scales with linear thresholds 0.01 and 1 respectively; speed is linear. No field is smoothed or clipped.</p>
    <p>The 256-cell controls reach t* = 10. They quantify early-time numerical sensitivity, not late-time grid independence. At Re = 3200, 5000, 7500 and 10000 the relative velocity differences are 4.23%, 7.77%, 32.00% and 55.31%. The finer grid is a comparator, not truth.</p>
    <h2>Native fields and complete physical comparisons</h2>{gallery}
    <p><a href="../cavity-continuation/">Earlier 0–100 movies</a> · <a href="{GITHUB}docs/CAVITY_STARTUP_STATIONARITY.md">Stationarity definitions</a></p></main></body></html>'''
    # All non-fragment links open separately, matching the course site convention.
    body = body.replace('<a href=', '<a target="_blank" rel="noopener noreferrer" href=')
    (BUNDLE / 'index.html').write_text(body, encoding='utf-8', newline='\n')


def build():
    bundle = ResultBundle(BUNDLE)
    records = []
    records += _save(overview(bundle, 'speed', time=100.), 'matched_t100_speed')
    for field in ['speed', 'p', 'omega']:
        records += _save(overview(bundle, field), 'endpoints_' + field)
    for re in [3200, 5000, 7500, 10000]:
        records += _save(grid_fields(bundle, re), f'grid_re{re}')
    for re in REYNOLDS:
        records += _save(late_profiles(bundle, re), f'late_profiles_re{re}')
    write_json(BUNDLE / 'lesson_figures/manifest.json', {'schema': 1, 'source_analysis_sha256': bundle.manifest['original_analysis_sha256'], 'files': records})
    outcomes, controls = bundle.outcomes(), bundle.comparisons()
    write_json(BUNDLE / 'summary.json', {'schema': 1, 'source_analysis_sha256': bundle.manifest['original_analysis_sha256'],
        'stationarity': outcomes, 'matched_controls': controls, 'limitation': 'Fixed-grid temporal confirmation is not late-time mesh convergence or Ghia validation.'})
    for name, data in [('stationarity_outcomes', outcomes), ('matched_controls', controls)]:
        with (BUNDLE / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    notebook(); readme(bundle, records); page(bundle, records)
    print('Built companion notebook, 15 physical figures, complete result index and site page.', flush=True)


def execute():
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    for key, name in [('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('JUPYTER_CONFIG_DIR','jupyter_config'),
                      ('IPYTHONDIR','ipython'),('TEMP','temp'),('TMP','temp')]:
        path = WORK / name; path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
    notebook()
    document = nbf.read(NOTEBOOK, as_version=4)
    source_hash = hashlib.sha256(json.dumps([{'cell_type':c.cell_type, 'source':c.source} for c in document.cells], sort_keys=True).encode()).hexdigest()
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(document, km=manager, timeout=600, startup_timeout=300, resources={'metadata':{'path':str(ROOT)}})
    started = perf_counter()
    try: client.execute()
    finally:
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    code = [c for c in document.cells if c.cell_type == 'code']
    outputs = [o for c in code for o in c.outputs]
    images = sum('image/png' in o.get('data', {}) for o in outputs)
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for o in outputs)
    assert images >= 8, f'Expected physical figure outputs, got {images}'
    NOTEBOOK.write_text(nbf.writes(document)+'\n', encoding='utf-8', newline='\n')
    write_json(BUNDLE / 'notebook_execution.json', {'executed_utc':datetime.now(timezone.utc).isoformat(),
        'notebook':NOTEBOOK.relative_to(ROOT).as_posix(), 'cell_source_sha256':source_hash,
        'notebook_sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        'code_cells':len(code), 'figure_outputs':images, 'error_outputs':0,
        'runtime_seconds':perf_counter()-started, 'python':sys.version.split()[0],
        'mode':'Fresh CPU kernel; saved-data loading and plotting; no CFD solves'})
    print(f'Notebook executed: {len(code)} code cells, {images} figure outputs, no errors.', flush=True)


def check():
    import copy
    import numpy as np
    from qa.analyze_cavity_settling import compare_fields
    bundle = ResultBundle(BUNDLE)
    count = bundle.verify_all()
    summary = json.loads((BUNDLE / 'summary.json').read_text())
    assert summary['stationarity'] == bundle.outcomes()
    assert summary['matched_controls'] == bundle.comparisons()
    for re in REYNOLDS:
        case = bundle.baseline(re)
        for rec in case['fields']: bundle.native(case, rec['time'])
    for entry in bundle.confirmations.values(): bundle.native(entry)
    for row in bundle.comparisons():
        for identity in ['candidate_id','reference_id']: bundle.native(bundle.cases[row[identity]], 10.)
    numerical_checks = []
    for pair in bundle.analysis['comparisons']:
        a, _ = bundle.native(bundle.cases[pair['candidate_id']], 10.)
        b, _ = bundle.native(bundle.cases[pair['reference_id']], 10.)
        fresh = compare_fields(a, b)
        retained = next(f for f in pair['frames'] if f['time'] == 10.)
        delta = max(abs(fresh[k] - retained[k]) for k in fresh)
        assert all(np.isclose(fresh[k], retained[k], rtol=1e-11, atol=1e-12) for k in fresh), pair['id']
        numerical_checks.append({'id': pair['id'], 'maximum_absolute_difference': delta})
    rejected = []
    for altered, time, label in [(bundle.baseline(100), 99.5, 'unavailable exact time'),
                                 (copy.deepcopy(bundle.baseline(100)), 100., 'configuration mismatch')]:
        if label == 'configuration mismatch': altered['config']['dt'] *= 2
        try: bundle.native(altered, time)
        except ValueError: rejected.append(label)
        else: raise AssertionError('Unsafe input was accepted: ' + label)
    for item in json.loads((BUNDLE / 'lesson_figures/manifest.json').read_text())['files']:
        assert hashlib.sha256((BUNDLE / item['file']).read_bytes()).hexdigest() == item['sha256']
    nbf.validate(nbf.read(NOTEBOOK, as_version=4))
    receipt = json.loads((BUNDLE / 'notebook_execution.json').read_text())
    assert receipt['notebook_sha256'] == hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    write_json(BUNDLE / 'validation.json', {'checked_utc':datetime.now(timezone.utc).isoformat(),
        'scientific_files_verified':count, 'native_frames_verified':sum(len(c['fields']) for c in list(bundle.cases.values())+list(bundle.confirmations.values())),
        'matched_control_recomputations':numerical_checks, 'rejected_invalid_inputs':rejected,
        'notebook_receipt_verified':True, 'scientific_analysis_sha256':bundle.manifest['original_analysis_sha256']})
    print(f'Validated {count} scientific files, all native fields/configurations, all 16 exact-time controls and executed notebook.', flush=True)


def export(site):
    target = Path(site) / 'cavity-settling'; target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLE / 'index.html', target / 'index.html')
    for directory in ['figures', 'lesson_figures']:
        shutil.copytree(BUNDLE / directory, target / directory, dirs_exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build','execute','check'])
    globals()[parser.parse_args().action]()
