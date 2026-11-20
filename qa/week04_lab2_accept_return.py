"""Validate the Unity return before replacing this lab's owned output files."""
from pathlib import Path
from datetime import datetime, timezone
import tarfile, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
archive_path = ROOT/"tmp/week04_lab2/verified_outputs.tar.gz"
expected_archive_sha = "013debcfbd03311f36e05e6ce1d59ae751579347aca7de8dd9d2c0620aa579e2"
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == expected_archive_sha
with tarfile.open(archive_path, "r:gz") as archive:
    manifest_name = "qa/week04_lab2_return_manifest.json"
    manifest_bytes = archive.extractfile(manifest_name).read()
    manifest = json.loads(manifest_bytes)
    allowed_notebook = "notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb"
    payload = {}
    for rel, expected in manifest.items():
        assert rel == allowed_notebook or rel.startswith(("results/week04_lab2/", "qa/week04_lab2_")), rel
        destination = (ROOT/rel).resolve()
        assert destination.is_relative_to(ROOT.resolve()), rel
        member = archive.getmember(rel)
        assert member.isfile(), rel
        content = archive.extractfile(member).read()
        assert len(content) == expected["bytes"], rel
        assert hashlib.sha256(content).hexdigest() == expected["sha256"], rel
        payload[rel] = content
    execution = json.loads(payload["qa/week04_lab2_execution.json"])
    frozen = json.loads(payload["results/week04_lab2/frozen_models.json"])
    assert execution["helper_source_sha256"] == hashlib.sha256((ROOT/"flowmllab/cavity_lab2.py").read_bytes()).hexdigest()
    assert frozen["data_sha256"] == hashlib.sha256((ROOT/"data/cavity_data.npz").read_bytes()).hexdigest()
    local_nb = json.loads((ROOT/allowed_notebook).read_text(encoding="utf-8"))
    local_code = "\n".join("".join(c["source"]) for c in local_nb["cells"] if c["cell_type"] == "code")
    assert hashlib.sha256(local_code.encode()).hexdigest() == execution["executed_code_sha256"]
    # Keep the compact baseline timing receipt alongside its already retained
    # numerical evidence; its generated figures are replaced by the final ones.
    old_execution = ROOT/"qa/week04_lab2_execution.json"
    baseline_receipt = ROOT/"results/week04_lab2/product_lifting_baseline/initial_execution.json"
    if old_execution.exists() and not baseline_receipt.exists():
        baseline_receipt.write_bytes(old_execution.read_bytes())
    payload[manifest_name] = manifest_bytes
    for rel, content in payload.items():
        destination = ROOT/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name+".verified_return")
        temporary.write_bytes(content)
        temporary.replace(destination)
    for rel, expected in manifest.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected["sha256"], rel
receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "job_id": "2048425",
           "slurm_state": "COMPLETED", "slurm_exit_code": "0:0",
           "slurm_elapsed": "00:01:08", "return_archive_bytes": archive_path.stat().st_size,
           "return_archive_sha256": expected_archive_sha, "verified_returned_files": len(manifest),
           "input_archive_sha256": "fd0f1f6cb9b31519de353b8ebe161d047fb30013c8f2e4df908c8e59de549376",
           "local_data_and_helper_match_executed_inputs": True,
           "local_notebook_code_matches_executed_code": True,
           "all_retained_copies_byte_verified": True,
           "destination": str(ROOT)}
(ROOT/"qa/week04_lab2_transfer.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
