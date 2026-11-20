"""Create a small source-only Unity verification bundle, with exact SHA256s."""
from pathlib import Path
import tarfile, hashlib, json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
files = list((ROOT/"flowmllab").rglob("*.py"))
files += [ROOT/name for name in (
    "pyproject.toml", "README.md", "data/cavity_data.npz",
    "notebooks/week04/cavity_project_utils.py",
    "notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb",
    "qa/week04_lab2_build.py", "qa/week04_lab2_execute.py",
    "qa/week04_lab2_gradient_check.py", "qa/week04_lab2_requirements.txt",
    "tests/test_cavity_lab2.py")]
files = sorted(set(files))
assert all(p.is_file() and p.is_relative_to(ROOT) for p in files)
manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "One isolated CPU Run All verification; no new CFD cases or model sweep.",
            "workspace": "/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115",
            "workspace_expiry": "2026-12-15 14:04:59",
            "run_directory": "week04-lab2-qa-20261120", "files": {}}
for p in files:
    manifest["files"][p.relative_to(ROOT).as_posix()] = {
        "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
target = ROOT/"tmp/week04_lab2/unity_inputs.tar.gz"
target.parent.mkdir(parents=True, exist_ok=True)
receipt = ROOT/"qa/week04_lab2_staging.json"
receipt.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
with tarfile.open(target, "w:gz") as archive:
    for p in files + [receipt]:
        archive.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
print(json.dumps({"archive": str(target), "bytes": target.stat().st_size,
                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "staged_files": len(files)+1}, indent=2))
