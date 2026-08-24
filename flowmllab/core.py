"""Repository discovery and lightweight scientific validation for FlowMLLab."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


EXPECTED_DATA_SHA256 = "09b96b744ee4d18126d8dcc92feb60e128774a1b4d41bb3d8c90a63ccfbabc36"
REQUIRED_FIELDS = ("x", "y", "Re", "u", "v", "p", "psi", "omega", "split", "accepted")


class ValidationError(RuntimeError):
    """Raised when a FlowMLLab scientific asset fails its declared contract."""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_repository_root(start: str | Path | None = None) -> Path:
    """Find a checkout containing the fixed FlowMLLab data and QA program."""
    candidates: list[Path] = []
    if os.environ.get("FLOWMLLAB_ROOT"):
        candidates.append(Path(os.environ["FLOWMLLAB_ROOT"]))
    if start is not None:
        candidates.append(Path(start))
    candidates.extend((Path.cwd(), Path(__file__).resolve().parent.parent))
    for origin in candidates:
        resolved = origin.expanduser().resolve()
        for candidate in (resolved, *resolved.parents):
            if (candidate / "data" / "cavity_data.npz").is_file() and (
                candidate / "qa" / "validate_course_release.py"
            ).is_file():
                return candidate
    raise ValidationError(
        "Could not locate a FlowMLLab checkout. Run inside the repository or set FLOWMLLAB_ROOT."
    )


def validate_core_assets(root: str | Path | None = None) -> dict[str, Any]:
    """Validate the fixed cavity archive, dimensional contract, and wall conditions."""
    repository = discover_repository_root(root)
    archive_path = repository / "data" / "cavity_data.npz"
    actual_hash = _digest(archive_path)
    if actual_hash != EXPECTED_DATA_SHA256:
        raise ValidationError(f"Dataset SHA-256 mismatch: {actual_hash}")

    with np.load(archive_path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_FIELDS if name not in archive.files]
        if missing:
            raise ValidationError("Missing dataset fields: " + ", ".join(missing))
        re_values = np.asarray(archive["Re"], dtype=float)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        u = np.asarray(archive["u"], dtype=float)
        v = np.asarray(archive["v"], dtype=float)
        p = np.asarray(archive["p"], dtype=float)
        expected_shape = (len(re_values), len(y), len(x))
        for name, field in (("u", u), ("v", v), ("p", p)):
            if field.shape != expected_shape:
                raise ValidationError(f"{name} has shape {field.shape}; expected {expected_shape}")
            if not np.isfinite(field).all():
                raise ValidationError(f"{name} contains non-finite values")
        if not np.all(np.diff(re_values) > 0):
            raise ValidationError("Reynolds-number cases are not strictly increasing")
        if not np.allclose(u[:, 0, :], 0.0) or not np.allclose(v[:, 0, :], 0.0):
            raise ValidationError("Bottom-wall no-slip condition failed")
        if not np.allclose(v[:, -1, :], 0.0):
            raise ValidationError("Moving-lid normal velocity condition failed")
        if not np.allclose(u[:, -1, 1:-1], 1.0):
            raise ValidationError("Moving-lid tangential velocity condition failed")
        accepted = np.asarray(archive["accepted"], dtype=bool)
        if not accepted.all():
            raise ValidationError("The fixed archive contains an unaccepted case")

    return {
        "status": "pass",
        "root": str(repository),
        "dataset_sha256": actual_hash,
        "cases": int(len(re_values)),
        "grid": [int(len(y)), int(len(x))],
        "reynolds_numbers": re_values.tolist(),
    }


def run_repository_qa(root: str | Path | None = None) -> int:
    """Run the full notebook, evidence, PDF, hash, and numerical release gate."""
    repository = discover_repository_root(root)
    completed = subprocess.run(
        [sys.executable, str(repository / "qa" / "validate_course_release.py")],
        cwd=repository,
        check=False,
    )
    return int(completed.returncode)


def generate_validation_figures(target: str = "all", root: str | Path | None = None) -> int:
    """Generate article-facing Ghia, pressure, or DSMC validation figures."""
    if target not in {"ghia", "pressure", "dsmc", "all"}:
        raise ValidationError(f"Unsupported figure target: {target}")
    repository = discover_repository_root(root)
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "common" / "article_validation.py"),
            target,
            "--root",
            str(repository),
        ],
        cwd=repository,
        check=False,
    )
    return int(completed.returncode)


def format_report(report: dict[str, Any]) -> str:
    """Return deterministic JSON for CLI and CI logs."""
    return json.dumps(report, indent=2, sort_keys=True)

