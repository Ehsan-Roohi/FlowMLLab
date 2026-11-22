"""Export a closed native cavity campaign for teaching, without running CFD.

Run on the machine holding the campaign with --campaign-dir and --output.
The export keeps every admitted history/window and numerical comparison, and
byte-identical native fields at t*=10, t*=100 (where present) and each endpoint.
Recovery files and unsampled native frames remain in the original campaign.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def export(campaign_dir, output):
    base, out = Path(campaign_dir).resolve(), Path(output).resolve()
    if out.exists():
        raise FileExistsError(f"Preserve the existing export: {out}")
    queue = subprocess.run(["squeue", "-u", subprocess.check_output(["id", "-un"], text=True).strip(),
                            "-h", "-o", "%i|%T|%Z"], check=True, capture_output=True, text=True)
    if any(str(base) in line for line in queue.stdout.splitlines()):
        raise RuntimeError("The campaign still has a live/queued writer or consumer")
    analysis_bytes = (base / "qa/resolution_after_current.json").read_bytes()
    analysis = json.loads(analysis_bytes)
    if (analysis["coverage"] != {"verified_checkpoint": 22, "seed_only": 0, "missing": 0,
                                 "invalid": 0, "unstable_manifest": 0}
            or analysis["errors"] or len(analysis["comparisons"]) != 16):
        raise ValueError("Incomplete or invalid final campaign")
    for comparison in analysis["comparisons"]:
        if comparison["status"] != "matched" or comparison["unavailable_requested_profile_times"]:
            raise ValueError("A requested matched-time comparison is absent")
        if not any(frame["time"] == 10 for frame in comparison["frames"]):
            raise ValueError("Every comparison must include exact t*=10")
    manifest = {"schema": "flowmllab.cavity.final-results.v1",
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "original_analysis_sha256": digest(analysis_bytes),
                "original_analysis_bytes": len(analysis_bytes),
                "case_count": 22, "comparison_count": 16, "cases": [],
                "confirmations": [], "files": {},
                "qualification": "Closed finite-time 2D campaign; fixed-grid stationarity is separate from mesh/time accuracy.",
                "native_field_selection": "All 22 cases at exact t*=10; seven baselines at t*=100 and their own endpoints; three confirmation endpoints. No field/time interpolation."}
    out.mkdir(parents=True)

    def put(relative, payload, source=None):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if target.read_bytes() != payload:
            raise IOError("Export read-back mismatch")
        manifest["files"][relative] = {"bytes": len(payload), "sha256": digest(payload)}
        if source:
            manifest["files"][relative]["source_relative"] = source
        return relative

    def copy(source, relative, expected=None):
        payload = source.read_bytes()
        if expected and digest(payload) != expected:
            raise ValueError(f"Source checksum mismatch: {source}")
        return put(relative, payload, str(source.relative_to(base)))

    def json_copy(source, relative):
        raw = source.read_bytes()
        json.loads(raw)
        return put(relative, raw, str(source.relative_to(base)))

    put("data/analysis.json.gz", gzip.compress(analysis_bytes, compresslevel=9, mtime=0),
        "qa/resolution_after_current.json")
    campaign = json.loads((base / "campaign.json").read_text())
    json_copy(base / "campaign.json", "provenance/campaign.json")
    for p in sorted((base / "policies").glob("*.json")):
        json_copy(p, "provenance/policies/" + p.name)
    for name, expected in campaign["source_sha256"].items():
        copy(base / "source" / name, "provenance/source/" + name, expected)
    for name in ["analyze_cavity_settling.py", "plot_cavity_settling.py", "confirm_cavity_stationarity.py"]:
        relative = "qa/" + name
        if "provenance/source/" + relative not in manifest["files"]:
            copy(base / "source" / relative, "provenance/source/" + relative)

    def native_records(directory, entry_id, wanted):
        before = (directory / "restart.json").read_bytes()
        recovery = json.loads(before)["current"]
        checkpoint = (directory / recovery["file"]).read_bytes()
        if digest(checkpoint) != recovery["sha256"]:
            raise ValueError("Checkpoint identity changed")
        with np.load(io.BytesIO(checkpoint), allow_pickle=False) as archive:
            state = json.loads(str(archive["metadata_json"]))
            final = {key: archive[key].copy() for key in ["u_faces", "v_faces", "p"]}
        records = []
        for time in sorted(set(wanted)):
            candidates = [r for r in state["snapshots"] if r["time"] == time]
            if len(candidates) != 1:
                raise ValueError(f"Missing exact native frame {entry_id}, t={time}")
            record = candidates[0]
            raw = (directory / record["file"]).read_bytes()
            if digest(raw) != record["sha256"]:
                raise ValueError("Native snapshot checksum mismatch")
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata_json"]))
                if meta["config"] != state["config"] or meta["time"] != time or meta["contract_sha256"] != state["contract_sha256"]:
                    raise ValueError("Native frame configuration/time mismatch")
                if time == state["time"]:
                    if any(not np.array_equal(final[k], archive[k]) for k in final):
                        raise ValueError("Endpoint snapshot differs from complete recovery")
            name = f"data/fields/{entry_id}/t{time:g}.npz"
            put(name, raw, str((directory / record["file"]).relative_to(base)))
            records.append({"time": time, "file": name, "sha256": digest(raw), "step": meta["step"],
                            "contract_sha256": meta["contract_sha256"]})
        if (directory / "restart.json").read_bytes() != before:
            raise RuntimeError("A recovery changed while exporting")
        return state, records

    for case in campaign["cases"]:
        case_id = case["id"]
        directory = base / "runs" / case_id / "continuation"
        if (directory / "writer.lock").exists():
            raise RuntimeError("A writer lock is still present")
        status = json.loads((directory / "status.json").read_text())
        wanted = [10.]
        if case["purpose"] == "continue":
            wanted += [100., status["time"]]
        state, fields = native_records(directory, case_id, wanted)
        entry = {"id": case_id, "purpose": case["purpose"], "config": state["config"],
                 "time": state["time"], "candidate": state["candidate"],
                 "consecutive_passes": state["consecutive_passes"], "fields": fields}
        for filename in ["status.json", "windows.json"]:
            entry[filename[:-5] + "_file"] = json_copy(directory / filename, f"data/cases/{case_id}/{filename}")
        diagnostics = (directory / "diagnostics.csv").read_bytes()
        entry["diagnostics_file"] = put(f"data/cases/{case_id}/diagnostics.csv.gz",
                                       gzip.compress(diagnostics, compresslevel=9, mtime=0),
                                       str((directory / "diagnostics.csv").relative_to(base)))
        manifest["cases"].append(entry)

    for re in [100, 400, 1000]:
        directory = base / "confirmations" / f"re{re}"
        report = json.loads((directory / "confirmation.json").read_text())
        if not report["complete"] or not report["independent_confirmation_passed"] or report["failed_metrics"]:
            raise ValueError("Confirmation is not complete and passed")
        changes = {k for k in report["original_config"] if report["original_config"][k] != report["confirmation_config"][k]}
        if changes != {"outer_correctors", "pressure_correctors"}:
            raise ValueError("Confirmation changed another numerical/physical control")
        state, fields = native_records(directory, f"confirmation_re{re}", [report["actual_time"]])
        entry = {"reynolds": re, "config": state["config"], "fields": fields}
        for filename in ["confirmation.json", "status.json", "windows.json"]:
            entry[filename[:-5] + "_file"] = json_copy(directory / filename, f"data/confirmations/re{re}/{filename}")
        manifest["confirmations"].append(entry)

    figure_receipt = json.loads((base / "resolution_controls_current/plot_receipt.json").read_text())
    if len(figure_receipt["figures"]) != 23 or figure_receipt["skipped"]:
        raise ValueError("Expected all 23 scientific figures")
    for figure in figure_receipt["figures"]:
        for file in figure["files"]:
            copy(base / "resolution_controls_current" / file["name"], "figures/" + file["name"], file["sha256"])
    json_copy(base / "resolution_controls_current/plot_receipt.json", "provenance/plot_receipt.json")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive_path = out.with_suffix(".tar.gz")
    if archive_path.exists():
        raise FileExistsError(archive_path)
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                archive.add(p, arcname=str(p.relative_to(out)), recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            payload = archive.extractfile(member).read()
            if payload != (out / member.name).read_bytes():
                raise ValueError("Archive read-back mismatch")
    return {"directory": str(out), "archive": str(archive_path), "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": digest(archive_path.read_bytes()), "files": len(manifest["files"]),
            "native_frames": sum(len(c["fields"]) for c in manifest["cases"] + manifest["confirmations"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.campaign_dir, args.output), indent=2))
