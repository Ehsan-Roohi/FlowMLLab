#!/usr/bin/env python3
"""Pressure-driven N2 backward-facing-step pilot. Python standard library only."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

SPARTA_COMMIT = "95b9abaa8bd548991cc3c3f1c58b34722f7ade74"
DEFAULT_BASE = "/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-sparta-step"
KB = 1.380649e-23  # SI constant used by this pinned SPARTA revision
MASS = 4.65e-26
DIAM = 4.17e-10
OMEGA = 0.74
TREF = 273.0


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(out, smoke=False, ratio=0.5, kn=0.01, pressure_ratio=2.0,
             nx=None, ny=None, ppc=None, warmup=None, sample=None, seed=20260905):
    out = Path(out)
    if out.exists():
        raise ValueError(f"Refusing to overwrite case: {out}")
    nx = nx if nx is not None else (50 if smoke else 1000)
    ny = ny if ny is not None else (20 if smoke else 200)
    ppc = ppc if ppc is not None else (8 if smoke else 20)
    warmup = warmup if warmup is not None else (100 if smoke else 40000)
    sample = sample if sample is not None else (200 if smoke else 20000)
    block = 50 if smoke else 2000
    if not (0 < ratio < 1 and kn > 0 and pressure_ratio >= 1 and ppc >= 2):
        raise ValueError("Invalid geometry, Kn, pressure ratio or particle count")
    if nx < 10 or ny < 4 or nx % 10 or abs(ratio * ny - round(ratio * ny)) > 1e-9:
        raise ValueError("Use nx divisible by 10 and h/H aligned with the y grid")
    if warmup < block or warmup % block or sample < 4 * block or sample % block:
        raise ValueError("Warmup/sample must be block multiples; at least four sample blocks")
    if not 0 < seed < 900000000:
        raise ValueError("Invalid SPARTA seed")
    L, H, T = 85.47e-6, 85.47e-6 / 5, 300.0
    sx, h = .3 * L, ratio * H
    dx, dy = L / nx, H / ny
    lam = kn * H
    nout = (T / TREF) ** (OMEGA - .5) / (math.sqrt(2) * math.pi * DIAM**2 * lam)
    pout, pin = nout * KB * T, nout * KB * T * pressure_ratio
    # Bird94 eqs 1.38/4.75, identical-species form used by compute lambda/grid.
    tau_in = 1 / (2 * DIAM**2 * nout * pressure_ratio
                  * math.sqrt(4 * math.pi * KB * TREF / MASS)
                  * (T / TREF) ** (1 - OMEGA))
    dt = min(.2 * tau_in, .2 * min(dx, dy) / math.sqrt(8 * KB * T / (math.pi * MASS)))
    fnum = nout * dx * dy / ppc  # 2D unit-depth area, NOT L^3
    meta = {
        "status": "generated", "level": "smoke" if smoke else "cost_and_boundary_pilot",
        "sparta_commit": SPARTA_COMMIT, "seed": seed,
        "L_m": L, "H_m": H, "step_x_m": sx, "step_height_m": h, "h_over_H": ratio,
        "Kn_outlet_reference": kn, "lambda_outlet_reference_m": lam,
        "lambda_inlet_reference_m": lam / pressure_ratio, "tau_inlet_reference_s": tau_in,
        "pressure_ratio": pressure_ratio, "p_in_Pa": pin, "p_out_Pa": pout,
        "T_in_K": T, "T_out_injected_K": T, "T_wall_K": T,
        "nx": nx, "ny": ny, "dx_m": dx, "dy_m": dy, "ppc_outlet_reference": ppc,
        "fnum_unit_depth": fnum, "dt_s": dt, "warmup_steps": warmup,
        "sampling_steps": sample, "block_steps": block, "final_step": warmup + sample,
        "reference_max_cell_over_lambda_in": max(dx, dy) / (lam / pressure_ratio),
        "expected_fluid_cells": nx * ny - round(.3 * nx) * round(ratio * ny),
        "physics": {"gas": "N2", "collision_model": "VHS via VSS alpha=1",
                    "diameter_m": DIAM, "omega": OMEGA, "Tref_K": TREF,
                    "mass_kg": MASS, "rotational_dof": 2, "rotation_probability": .2,
                    "vibration": "disabled", "walls": "diffuse, full accommodation"},
        "provenance": "Step-19 manuscript, sections 3.1-3.2 and Figure 1; not original Bird input",
        "explicit_pilot_conventions": [
            "Kn=lambda/H uses outlet reference density at 300 K and Bird VHS mean free path.",
            "Both open boundaries inject at specified pressure and 300 K via emit/face subsonic.",
            "Outlet 300 K applies to incoming particles, not every outlet-cell particle.",
            "N2 rotation probability 0.2 and disabled vibration are explicit pilot choices.",
            "Paper does not identify inlet/outlet reference state of lambda or rotational relaxation.",
            "Pilot grid is coarser than the paper's stated lambda/3 criterion; no training release.",
        ],
        "bird_parity_validated": False, "training_data_approved": False,
    }
    out.mkdir(parents=True)
    write_json(out / "case.json", meta)
    (out / "nitrogen.species").write_text(
        "# ID molwt mass rotdof rotrel vibdof vibrel vibtemp weight charge\n"
        "N2 28.016 4.65e-26 2 0.2 0 0.0 0.0 1.0 0.0\n")
    (out / "nitrogen.vss").write_text("# species diameter omega Tref alpha (alpha=1 => VHS)\nN2 4.17e-10 0.74 273.0 1.0\n")
    # Open wall chain terminates on box boundaries. z cross tangent points into fluid.
    # Segment walls for resolved surface force/heat-flux output.
    nwall, nvertical = round(.3 * nx), round(ratio * ny)
    pts = [(i * L / nx, h) for i in range(nwall + 1)]
    pts += [(sx, (nvertical - j) * H / ny) for j in range(1, nvertical + 1)]
    surf = ["Backward-facing step; SI metres", "", f"{len(pts)} points", f"{len(pts)-1} lines", "", "Points", ""]
    surf += [f"{i} {x:.17g} {y:.17g}" for i, (x, y) in enumerate(pts, 1)]
    surf += ["", "Lines", ""] + [f"{i} {i} {i+1}" for i in range(1, len(pts))]
    (out / "step.surf").write_text("\n".join(surf) + "\n")
    deck = f"""# N2 pressure-driven BFS. See case.json for provenance and pilot conventions.
seed {seed}
dimension 2
global gridcut 0.0
boundary o s p
create_box 0 {L:.17g} 0 {H:.17g} -0.5 0.5
create_grid {nx} {ny} 1
global nrho {nout:.17g} fnum {fnum:.17g}
species nitrogen.species N2
mixture gas N2 temp {T} vstream 0 0 0
mixture initial N2 nrho {nout * (1 + pressure_ratio) / 2:.17g} temp {T} vstream 0 0 0
read_surf step.surf
surf_collide wall diffuse {T} 1.0
surf_modify all collide wall
bound_modify ylo yhi collide wall
balance_grid rcb cell
collide vss gas nitrogen.vss
collide_modify rotate smooth vibrate no
create_particles initial n 0
fix inlet emit/face gas xlo subsonic {pin:.17g} {T}
fix outlet emit/face gas xhi subsonic {pout:.17g} {T}
fix check grid/check {block} error
timestep {dt:.17g}
compute boundary boundary gas n press shx ke
variable exits_in equal c_boundary[1][1]
variable exits_out equal c_boundary[2][1]
variable inventory equal np
stats {block}
stats_style step cpu np ncoll nattempt nexit f_inlet[2] f_outlet[2]
print "SPARTA_STEP_WARMUP_BEGIN"
run {warmup}
print "SPARTA_STEP_SAMPLING_BEGIN"
# Every timestep contributes to mass balance; cell moments sampled every 10 steps.
fix flux ave/time 1 {block} {block} f_inlet[1] f_outlet[1] v_exits_in v_exits_out v_inventory file flux.blocks
compute gasfield grid all gas n nrho massrho u v w trot pxrho pyrho
compute thermal thermal/grid all gas temp press
fix blocks ave/grid all 10 {block // 10} {block} c_gasfield[*] c_thermal[*] ave one
fix average ave/grid all 10 {block // 10} {block} c_gasfield[*] c_thermal[*] ave running
dump blocks grid all {block} grid.block.*.gz id xc yc vol f_blocks[*]
dump_modify blocks first no format float %.16g
compute surface surf all gas n press shx shy ke
fix wallavg ave/surf all 1 {block} {block} c_surface[*] ave running
dump walls surf all {block} wall.running.*.gz id f_wallavg[*]
dump_modify walls first no format float %.16g
fix boxwalls ave/time 1 {block} {block} c_boundary[*] mode vector file boundary.blocks
run {sample - block}
# Last block: also write the cumulative post-warmup field and a restart.
dump final grid all {warmup + sample} grid.final.gz id xc yc vol f_average[*]
dump_modify final first no format float %.16g
run {block}
write_restart restart.final
print "SPARTA_STEP_SOLVER_COMPLETE"
"""
    (out / "in.step").write_text(deck)
    print(f"GENERATED={out} PARTICLE_WEIGHT={fnum:.8g} PIN={pin:.8g} POUT={pout:.8g}")
    return meta


def read_dump(path):
    """Stream one snapshot; reject missing headers/rows, truncation and extra snapshots."""
    with gzip.open(path, "rt") as f:
        if f.readline().strip() != "ITEM: TIMESTEP":
            raise ValueError(f"Invalid dump header: {path}")
        step = int(f.readline())
        if not f.readline().startswith("ITEM: NUMBER OF"):
            raise ValueError(f"Missing row count: {path}")
        count = int(f.readline())
        line = f.readline()
        if line.startswith("ITEM: BOX BOUNDS"):
            for _ in range(3):
                f.readline()
            line = f.readline()
        if not (line.startswith("ITEM: CELLS ") or line.startswith("ITEM: SURFS ")):
            raise ValueError(f"Missing field header: {path}: {line!r}")
        columns = line.split()[2:]
        yield step, columns
        for _ in range(count):
            row = [float(v) for v in f.readline().split()]
            if len(row) != len(columns) or not all(math.isfinite(v) for v in row):
                raise ValueError(f"Invalid/nonfinite data in {path}")
            yield row
        if f.read().strip():
            raise ValueError(f"Unexpected extra snapshot/data in {path}")


def report(out):
    out = Path(out)
    m = json.loads((out / "case.json").read_text())
    log = (out / "log.sparta").read_text()
    if "SPARTA_STEP_SOLVER_COMPLETE" not in log or re.search(r"^ERROR", log, re.M):
        raise ValueError("Solver completion absent or fatal error present")
    stuck = [int(v) for v in re.findall(r"Particles stuck\s*=\s*(\d+)", log)]
    if not stuck or any(stuck):
        raise ValueError(f"Missing/nonzero stuck-particle accounting: {stuck}")
    if not (out / "restart.final").is_file():
        raise ValueError("Final restart missing")
    final = read_dump(out / "grid.final.gz")
    step, cols = next(final)
    if step != m["final_step"]:
        raise ValueError(f"Wrong final step {step}; expected {m['final_step']}")
    seen, fluid_count, totals, pressure_edges = set(), 0, [0., 0., 0.], [[], []]
    min_lambda, max_dt_tau = math.inf, 0.
    profiles = {}
    with gzip.open(out / "fields.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "x_m", "y_m", "area_m2", "mean_particles", "n_m-3", "rho_kg_m-3", "u_m_s", "v_m_s", "w_m_s", "Trot_K", "rho_u", "rho_v", "Ttrans_K", "p_Pa"])
        for r in final:
            cell, x, y, area, count, n, rho, u, v, w, trot, rhou, rhov, temp, p = r
            if cell in seen:
                raise ValueError("Duplicate cell ID in final dump")
            seen.add(cell)
            solid = x < m["step_x_m"] and y < m["step_height_m"]
            if solid:
                if n != 0 or count != 0:
                    raise ValueError("Particles found inside the solid step")
                continue
            if area <= 0 or n <= 0 or temp <= 0 or p <= 0:
                raise ValueError("Nonpositive fluid area, density, temperature or pressure")
            if not math.isclose(p, n * KB * temp, rel_tol=1e-7):
                raise ValueError("Thermal pressure does not match n*kB*T")
            fluid_count += 1
            writer.writerow(r)
            totals[0] += rho * area
            totals[1] += rhou * area
            totals[2] += area
            lam = (temp / TREF) ** (OMEGA - .5) / (math.sqrt(2) * math.pi * DIAM**2 * n)
            tau = 1 / (2 * DIAM**2 * n * math.sqrt(4 * math.pi * KB * TREF / MASS) * (temp / TREF)**(1 - OMEGA))
            min_lambda = min(min_lambda, lam)
            max_dt_tau = max(max_dt_tau, m["dt_s"] / tau)
            if x < 1.01 * m["dx_m"]:
                pressure_edges[0].append(p)
            if x > m["L_m"] - 1.01 * m["dx_m"]:
                pressure_edges[1].append(p)
            ix = round(x / m["dx_m"] - .5)
            acc = profiles.setdefault(ix, [x, 0., 0., 0., 0.])
            acc[1] += rhou * area / m["dx_m"]
            acc[2] += rho * area
            acc[3] += p * area
            acc[4] += area
    if fluid_count != m["expected_fluid_cells"]:
        raise ValueError(f"Wrong fluid-cell count {fluid_count}; expected {m['expected_fluid_cells']}")
    expected_area = m["L_m"] * m["H_m"] - m["step_x_m"] * m["step_height_m"]
    if not math.isclose(totals[2], expected_area, rel_tol=1e-9):
        raise ValueError("Summed fluid area does not match the step geometry")
    with open(out / "axial_profiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "mass_flow_kg_per_m_s", "rho_mean_kg_m3", "p_mean_Pa"])
        for _, (x, flow, mass, pres, area) in sorted(profiles.items()):
            writer.writerow([x, flow, mass / area, pres / area])
    flux = [[float(v) for v in line.split()] for line in (out / "flux.blocks").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    expected_steps = list(range(m["warmup_steps"] + m["block_steps"], m["final_step"] + 1, m["block_steps"]))
    if [int(r[0]) for r in flux] != expected_steps or any(len(r) != 6 for r in flux):
        raise ValueError("Missing or unexpected mass-balance blocks")
    if not all(math.isfinite(v) for r in flux for v in r):
        raise ValueError("Nonfinite flux data")
    conversion = m["fnum_unit_depth"] * MASS / m["dt_s"]
    mass_in = [(r[1] - r[3]) * conversion for r in flux]
    mass_out = [(r[4] - r[2]) * conversion for r in flux]
    mi, mo = statistics.fmean(mass_in), statistics.fmean(mass_out)
    imbalance = abs(mi - mo) / max(abs(mi), abs(mo), 1e-100)
    # Independent nonoverlapping output blocks; not assumed statistically independent.
    block_summaries = []
    for s in expected_steps:
        data = read_dump(out / f"grid.block.{s}.gz")
        block_step, _ = next(data)
        if block_step != s:
            raise ValueError("Block timestamp mismatch")
        momentum, mass = 0., 0.
        for r in data:
            momentum += r[11] * r[3]
            mass += r[6] * r[3]
        block_summaries.append({"step": s, "mass_kg_per_m": mass, "bulk_u_m_s": momentum / mass})
        wall = read_dump(out / f"wall.running.{s}.gz")
        if next(wall)[0] != s or sum(1 for _ in wall) != round(.3*m["nx"]) + round(m["h_over_H"]*m["ny"]):
            raise ValueError("Missing/invalid surface output")
    bulk = [r["bulk_u_m_s"] for r in block_summaries]
    half = len(bulk) // 2
    drift = abs(statistics.fmean(bulk[:half]) - statistics.fmean(bulk[half:])) / max(abs(statistics.fmean(bulk)), 1e-100)
    p_edges = [statistics.fmean(v) for v in pressure_edges]
    p_errors = [abs(p_edges[0] / m["p_in_Pa"] - 1), abs(p_edges[1] / m["p_out_Pa"] - 1)]
    max_dx_lambda = max(m["dx_m"], m["dy_m"]) / min_lambda
    result = {
        "status": "smoke_complete" if m["level"] == "smoke" else "pilot_execution_complete",
        "final_step": step, "fluid_cells": fluid_count, "stuck_particles": sum(stuck),
        "mass_in_kg_per_m_s": mi, "mass_out_kg_per_m_s": mo,
        "mass_imbalance_fraction": imbalance, "bulk_velocity_half_drift_fraction": drift,
        "boundary_adjacent_mean_pressure_Pa": p_edges, "boundary_pressure_error_fraction": p_errors,
        "maximum_cell_over_minimum_sampled_lambda": max_dx_lambda,
        "maximum_dt_over_sampled_tau": max_dt_tau,
        "checks": {"positive_net_flow": mi > 0 and mo > 0,
                   "mass_imbalance_under_5_percent": imbalance < .05,
                   "bulk_half_drift_under_5_percent": drift < .05,
                   "boundary_pressure_errors_under_10_percent": max(p_errors) < .1,
                   "cell_under_lambda_over_3": max_dx_lambda <= 1/3,
                   "dt_under_tau_over_4": max_dt_tau < .25},
        "blocks": block_summaries,
        "block_bulk_u_std_m_s": statistics.stdev(bulk),
        "solver_loop_seconds": sum(float(v) for v in re.findall(r"Loop time of ([\d.eE+-]+)", log)),
        "particle_steps_per_second_per_rank": [float(v) for v in re.findall(r"Particle-moves/CPUsec/proc:\s*([\d.eE+-]+)", log)],
        "uncertainty_note": "Block spread is diagnostic, not an independence-corrected confidence interval.",
        "bird_parity_validated": False, "training_data_approved": False,
        "next_decision": "Review pressure, stationarity, cost and Bird conventions before refinement or additional geometries.",
    }
    write_json(out / "report.json", result)
    files = [p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(files)))
    print(json.dumps(result, indent=2))
    print("SPARTA_STEP_REPORT_COMPLETE TRAINING_DATA_APPROVED=False")
    return result


def submit(root, base, ref, account, module, ranks):
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("Full 40-character FlowMLLab commit required")
    if ranks < 2:
        raise ValueError("At least two ranks required for this MPI pilot")
    base, root = Path(base).resolve(), Path(root).resolve()
    (base / "runs").mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime("step-pilot-%Y%m%dT%H%M%SZ-", time.gmtime()), dir=base / "runs"))
    shutil.copytree(root, out / "code", ignore=shutil.ignore_patterns("__pycache__"))
    meta = {"flowmllab_commit": ref, "sparta_commit": SPARTA_COMMIT, "out": str(out),
            "openmpi_module": module, "account": account, "ranks": ranks, "jobs": {}}
    write_json(out / "submission.json", meta)
    (base / "LATEST_SPARTA_STEP_PILOT").write_text(str(out) + "\n")
    # Values exported structurally via subprocess env, never shell-interpolated.
    env = dict(os.environ, SPARTA_STEP_OUT=str(out), SPARTA_STEP_MPI_MODULE=module)
    common = ["sbatch", "--parsable", f"--account={account}", "--partition=cpu", "--nodes=1", "--export=ALL"]
    for phase, ntasks, cpus, memory, walltime, dependency in [
        ("build", 2, 4, "16G", "00:45:00", None),
        ("pilot", ranks, 1, "32G", "24:00:00", "build"),
        ("collect", 1, 1, "8G", "00:30:00", "pilot"),
    ]:
        args = common + [f"--job-name=step-sparta-{phase}", f"--ntasks={ntasks}",
                         f"--cpus-per-task={cpus}", f"--mem={memory}", f"--time={walltime}",
                         f"--chdir={out}", f"--output={out}/{phase}-%j.out", f"--error={out}/{phase}-%j.err"]
        if dependency:
            args += [f"--dependency=afterok:{meta['jobs'][dependency]}", "--kill-on-invalid-dep=yes"]
        raw = subprocess.check_output(args + [str(out / "code" / "unity_job.sh"), phase], env=env, text=True).strip()
        job = raw.split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Cannot parse sbatch job ID: {raw!r}")
        meta["jobs"][phase] = job
        write_json(out / "submission.json", meta)  # retain even a partial successful submission
        print(f"{phase.upper()}_JOB={job}", flush=True)
    print(f"OUT={out}\nSPARTA_STEP_PIPELINE_SUBMITTED")
    print(f"STATUS: python3 -I {out / 'code' / 'pilot.py'} status --out {out}")


def status(out, base):
    out = Path(out) if out else Path((Path(base) / "LATEST_SPARTA_STEP_PILOT").read_text().strip())
    meta = json.loads((out / "submission.json").read_text())
    print(f"OUT={out}")
    jobs = ",".join(meta["jobs"].values())
    if jobs:
        subprocess.run(["sacct", "-j", jobs, "--format=JobID%20,JobName%22,State%22,Elapsed,ExitCode,MaxRSS,NodeList%20"], check=False)
    for phase, job in meta["jobs"].items():
        print(f"\n{phase.upper()} JOB={job}")
        for ext in ("out", "err"):
            p = out / f"{phase}-{job}.{ext}"
            if p.exists():
                print(f"{p}:\n" + "\n".join(p.read_text(errors="replace").splitlines()[-15:]))
    p = out / "pilot" / "report.json"
    if p.exists():
        print(p.read_text())
    else:
        print("PILOT_REPORT_NOT_YET_AVAILABLE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--ratio", type=float, default=.5)
    g.add_argument("--kn", type=float, default=.01)
    g.add_argument("--pressure-ratio", type=float, default=2.)
    for key in ("nx", "ny", "ppc", "warmup", "sample"):
        g.add_argument("--" + key, type=int)
    g.add_argument("--seed", type=int, default=20260905)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--root", default=str(Path(__file__).resolve().parent))
    s.add_argument("--base", default=DEFAULT_BASE)
    s.add_argument("--ref", required=True)
    s.add_argument("--account", default="pi_roohie_umass_edu")
    s.add_argument("--module", default="openmpi/5.0.3")
    s.add_argument("--ranks", type=int, default=16)
    st = sub.add_parser("status")
    st.add_argument("--out")
    st.add_argument("--base", default=DEFAULT_BASE)
    args = vars(p.parse_args())
    action = args.pop("action")
    {"generate": generate, "report": report, "submit": submit, "status": status}[action](**args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"SPARTA_STEP_FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
