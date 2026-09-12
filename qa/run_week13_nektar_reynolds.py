"""Restartable D/W=5 Nektar++ Reynolds campaign with spectral vorticity.

Each accepted one-unit time chunk must have a clean solver exit, a finite
primitive-field VTU export, and a finite vorticity field produced by Nektar++
FieldConvert's spectral/hp ``vorticity`` module. Interrupted attempts are
retained but can never become restart sources.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

from prepare_week13_nektar import make_case


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_time(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        if element.tag.lower() == "time" and element.text:
            return float(element.text)
    raise ValueError("No Time metadata in final field")


def validate_ascii_vtu(path, *, require_vorticity=False):
    root = ET.parse(path).getroot()
    pieces = root.findall(".//Piece")
    if not pieces:
        raise ValueError(f"No VTU pieces in {path}")
    names = set()
    for piece in pieces:
        count = int(piece.get("NumberOfPoints", "0"))
        if count < 1:
            raise ValueError("Empty VTU piece")
        coordinates = piece.find("./Points/DataArray")
        if coordinates is None or coordinates.get("format", "ascii").lower() != "ascii":
            raise ValueError("Finite-field gate requires ASCII VTU coordinates")
        xyz = [float(value) for value in (coordinates.text or "").split()]
        if len(xyz) != 3 * count or not all(map(math.isfinite, xyz)):
            raise ValueError("Malformed/nonfinite VTU coordinates")
        for array in piece.findall("./PointData/DataArray"):
            name = array.get("Name", "")
            if array.get("format", "ascii").lower() != "ascii":
                raise ValueError(f"Finite-field gate requires ASCII field {name}")
            values = [float(value) for value in (array.text or "").split()]
            components = int(array.get("NumberOfComponents", "1"))
            if len(values) != count * components or not all(map(math.isfinite, values)):
                raise ValueError(f"Malformed/nonfinite VTU field {name}")
            names.add(name)
    if not {"u", "v", "p"}.issubset(names):
        raise ValueError(f"Missing primitive fields: {sorted(names)}")
    if require_vorticity:
        candidates = sorted(
            name for name in names
            if "vort" in name.lower() or name.lower().replace("_", "") in {"wz", "omegaz"}
        )
        if not candidates:
            raise ValueError(f"No recognizable spectral vorticity field: {sorted(names)}")
        return candidates
    return []


def run_process(command, cwd, log_path, deadline, reserve=60):
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=max(1, deadline - time.monotonic() - reserve))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def run(args):
    if args.re not in {100, 500, 1000}:
        raise ValueError("Declared Reynolds matrix is exactly 100, 500, 1000")
    if not math.isfinite(args.dt) or args.dt <= 0 or args.seconds <= 180:
        raise ValueError("Positive finite dt and wall budget above 180 seconds required")
    gate = json.loads(Path(args.gate).read_text())
    if not (
        gate.get("gate_passed") is True
        and gate.get("time_match_verified") is True
        and gate.get("boundary_operability_passed") is True
        and gate.get("corner_convention") == args.corner_convention
        and gate.get("Re") == 100
        and args.order in gate.get("orders", [])
        and args.dt in gate.get("dt", [])
        and gate.get("scope") == "short_time_restart_operability_and_agreement_not_steady_accuracy"
    ):
        raise ValueError("Reviewed boundary/restart-operability gate does not match")

    initial = None
    initial_path = getattr(args, "initial_state", None)
    if initial_path:
        initial = json.loads(Path(initial_path).read_text())
        for key, expected in (("re", args.re), ("nx", args.nx), ("ny", args.ny),
                              ("order", args.order), ("dt", args.dt),
                              ("depth", args.depth),
                              ("corner_convention", args.corner_convention)):
            if initial.get(key) != expected:
                raise ValueError(f"Initial-state mismatch: {key}")
        if initial.get("operability_passed") is not True:
            raise ValueError("Initial state has not passed mapping/operability checks")
        if sha(initial["field"]) != initial["field_sha256"]:
            raise ValueError("Initial-state field hash changed")
        initial_time = field_time(initial["field"])
        if abs(initial_time - initial["time"]) > 1e-7 or round(initial_time) < 1:
            raise ValueError("Initial-state physical time mismatch")
        if abs(initial_time - round(initial_time)) > 1e-7:
            raise ValueError("Initial time must be an integer chunk boundary")

    case = Path(args.output).resolve()
    case.mkdir(parents=True, exist_ok=True)
    config = {
        "re": args.re,
        "depth_over_width": args.depth,
        "order": args.order,
        "dt": args.dt,
        "chunk_time": 1.0,
        "corner_convention": args.corner_convention,
        "gate_re": 100,
        "gate_scope": "restart mechanics only; not Reynolds-specific accuracy",
        "gate_sha256": sha(args.gate),
        "end_time": args.end_time,
        "nx": args.nx,
        "ny": args.ny,
        "image_sha256": sha(args.image),
        "code_commit": args.commit,
        "vorticity_method": "Nektar++ FieldConvert -m vorticity (spectral/hp derivative)",
        "status": "transient_until_independent_steady_and_benchmark_audits_pass",
    }
    manifest = case / "campaign.json"
    if initial:
        config["initial_state_sha256"] = sha(initial_path)
        config["start_time"] = round(initial["time"])
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Campaign specification changed; use a new output directory")
    if not manifest.exists():
        manifest.write_text(json.dumps(config, indent=2) + "\n")

    deadline = time.monotonic() + args.seconds
    base = ["apptainer", "exec", "--cleanenv", "--bind", "/project", args.image]
    prior = Path(initial["field"]) if initial else None
    current = float(round(initial["time"])) if initial else 0.0
    chunk_count = round(args.end_time)
    if chunk_count != args.end_time or chunk_count < 1:
        raise ValueError("end-time must be a positive integer")

    if chunk_count <= current:
        raise ValueError("End time must exceed initial time")
    for index in range(round(current), chunk_count):
        chunk = case / f"chunk-{index:04d}"
        chunk.mkdir(exist_ok=True)
        marker = chunk / "accepted.json"
        if marker.exists():
            accepted = json.loads(marker.read_text())
            prior = chunk / accepted["attempt"] / "cavity.fld"
            for key, filename in (
                ("field_sha256", "cavity.fld"),
                ("primitive_vtu_sha256", "cavity.vtu"),
                ("vorticity_fld_sha256", "cavity-vorticity.fld"),
                ("vorticity_vtu_sha256", "cavity-vorticity.vtu"),
            ):
                if sha(chunk / accepted["attempt"] / filename) != accepted[key]:
                    raise ValueError(f"Accepted artifact hash changed: {filename}")
            current = accepted["time"]
            if abs(current - (index + 1)) > 1e-7:
                raise ValueError("Accepted time sequence mismatch")
            continue
        if time.monotonic() >= deadline - 180:
            return 75
        attempt_number = len(list(chunk.glob("attempt-*")))
        if attempt_number >= 4:
            raise RuntimeError("Four incomplete attempts: stop for review")
        attempt = chunk / f"attempt-{attempt_number:02d}"
        attempt.mkdir()
        steps = round(1.0 / args.dt)
        if abs(steps * args.dt - 1) > 1e-12:
            raise ValueError("dt must divide one nondimensional time unit")
        make_case(
            attempt, re=args.re, order=args.order, nx=args.nx, ny=args.ny,
            restart=str(prior) if prior else None, dt=args.dt, steps=steps,
            check_steps=steps, corner_convention=args.corner_convention,
            depth=args.depth,
            purpose="restartable_reynolds_matrix_not_yet_benchmark_certified",
        )
        try:
            status = run_process(
                base + ["IncNavierStokesSolver", "cavity.xml"], attempt,
                attempt / "solver.log", deadline,
            )
        except subprocess.TimeoutExpired:
            (attempt / "interrupted-budget.txt").write_text(
                "Unaccepted attempt; never used as restart.\n"
            )
            return 75
        if status:
            raise subprocess.CalledProcessError(status, "IncNavierStokesSolver")
        final = attempt / "cavity.fld"
        actual = field_time(final)
        if not math.isfinite(actual) or abs(actual - (current + 1.0)) > 1e-7:
            raise ValueError(f"Unexpected final time {actual}, expected {current + 1}")

        commands = [
            (base + ["FieldConvert", "cavity.xml", "cavity.fld", "cavity.vtu:vtu:uncompress"],
             attempt / "convert-primitive.log"),
            (base + ["FieldConvert", "-m", "vorticity", "cavity.xml", "cavity.fld", "cavity-vorticity.fld"],
             attempt / "convert-vorticity.log"),
            (base + ["FieldConvert", "cavity.xml", "cavity-vorticity.fld", "cavity-vorticity.vtu:vtu:uncompress"],
             attempt / "convert-vorticity-vtu.log"),
        ]
        for command, log_path in commands:
            try:
                status = run_process(command, attempt, log_path, deadline, reserve=20)
            except subprocess.TimeoutExpired:
                (attempt / "interrupted-budget.txt").write_text(
                    "Unaccepted export attempt; restart from last accepted chunk.\n"
                )
                return 75
            if status:
                raise subprocess.CalledProcessError(status, command)
        validate_ascii_vtu(attempt / "cavity.vtu")
        vorticity_names = validate_ascii_vtu(
            attempt / "cavity-vorticity.vtu", require_vorticity=True
        )
        record = {
            "attempt": attempt.name,
            "time": actual,
            "field_sha256": sha(final),
            "primitive_vtu_sha256": sha(attempt / "cavity.vtu"),
            "vorticity_fld_sha256": sha(attempt / "cavity-vorticity.fld"),
            "vorticity_vtu_sha256": sha(attempt / "cavity-vorticity.vtu"),
            "vorticity_fields": vorticity_names,
            "clean_exit": True,
            "numerical_accuracy": "requires temporal, p/h, and paper-reference audits",
        }
        temporary = chunk / "accepted.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, marker)
        prior, current = final, actual

    (case / "campaign-complete-needs-scientific-review").write_text(
        "Reached bounded end time. Benchmark and stationarity audits are still required.\n"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--initial-state", help="Verified mapped/continued state manifest")
    parser.add_argument("--re", type=int, choices=[100, 500, 1000], required=True)
    parser.add_argument("--corner-convention", choices=["stationary_endpoints"], required=True)
    parser.add_argument("--order", type=int, choices=[6], default=6)
    parser.add_argument("--dt", type=float, choices=[0.00025], default=0.00025)
    parser.add_argument("--end-time", type=float, default=80)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--depth", type=float, default=5.0)
    parser.add_argument("--seconds", type=float, default=6600)
    raise SystemExit(run(parser.parse_args()))
