"""Read-only access to the retained FlowMLLab blind-case demonstration.

The public demo deliberately replays the archived three-seed POD--DeepONet
ensemble predictions.  It does not retrain a model, tune on blind cases, or
interpolate an unvalidated Reynolds number.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .core import ValidationError, discover_repository_root


@dataclass(frozen=True)
class BlindDemoCase:
    """Reference and retained ensemble fields for one blind Reynolds case."""

    reynolds: float
    x: np.ndarray
    y: np.ndarray
    reference_u: np.ndarray
    reference_v: np.ndarray
    reference_p: np.ndarray
    prediction_u: np.ndarray
    prediction_v: np.ndarray
    prediction_p: np.ndarray
    inference_ms: float
    cfd_seconds: float

    @property
    def reference_speed(self) -> np.ndarray:
        return np.hypot(self.reference_u, self.reference_v)

    @property
    def prediction_speed(self) -> np.ndarray:
        return np.hypot(self.prediction_u, self.prediction_v)

    @property
    def vector_error(self) -> np.ndarray:
        return np.hypot(
            self.prediction_u - self.reference_u,
            self.prediction_v - self.reference_v,
        )

    @property
    def relative_l2_uv(self) -> float:
        numerator = np.linalg.norm(
            np.concatenate(
                (
                    (self.prediction_u - self.reference_u).ravel(),
                    (self.prediction_v - self.reference_v).ravel(),
                )
            )
        )
        denominator = np.linalg.norm(
            np.concatenate((self.reference_u.ravel(), self.reference_v.ravel()))
        )
        return float(numerator / denominator)

    @property
    def maximum_vector_error(self) -> float:
        return float(np.max(self.vector_error))

    @property
    def relative_l2_p(self) -> float:
        reference = self.reference_p - np.mean(self.reference_p)
        prediction = self.prediction_p - np.mean(self.prediction_p)
        return float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))

    @property
    def mae_p(self) -> float:
        reference = self.reference_p - np.mean(self.reference_p)
        prediction = self.prediction_p - np.mean(self.prediction_p)
        return float(np.mean(np.abs(prediction - reference)))

    @property
    def maximum_pressure_error(self) -> float:
        reference = self.reference_p - np.mean(self.reference_p)
        prediction = self.prediction_p - np.mean(self.prediction_p)
        return float(np.max(np.abs(prediction - reference)))

    @property
    def wall_rms_error(self) -> float:
        error = self.vector_error
        wall_values = np.concatenate(
            (error[0, :], error[-1, :], error[1:-1, 0], error[1:-1, -1])
        )
        return float(np.sqrt(np.mean(wall_values**2)))


def available_blind_reynolds(root: str | Path | None = None) -> tuple[float, ...]:
    """Return the Reynolds numbers present in the retained prediction archive."""
    repository = discover_repository_root(root)
    prediction_path = repository / "results" / "pod_deeponet" / "deeponet_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        values = np.asarray(archive["Re"], dtype=float)
    return tuple(float(value) for value in values)


def load_blind_demo_case(
    reynolds: float, root: str | Path | None = None
) -> BlindDemoCase:
    """Load one frozen blind case and verify its archive-level data contract."""
    repository = discover_repository_root(root)
    data_path = repository / "data" / "cavity_data.npz"
    prediction_path = repository / "results" / "pod_deeponet" / "deeponet_predictions.npz"
    timing_path = (
        repository
        / "results"
        / "pod_deeponet"
        / "deeponet_protocol_and_timing.json"
    )

    with np.load(data_path, allow_pickle=False) as reference_archive:
        reference_re = np.asarray(reference_archive["Re"], dtype=float)
        reference_matches = np.flatnonzero(np.isclose(reference_re, float(reynolds)))
        if reference_matches.size != 1:
            raise ValidationError(f"Reference archive has no unique Re={reynolds:g} case")
        reference_index = int(reference_matches[0])
        x = np.asarray(reference_archive["x"], dtype=float).copy()
        y = np.asarray(reference_archive["y"], dtype=float).copy()
        reference_u = np.asarray(reference_archive["u"][reference_index], dtype=float).copy()
        reference_v = np.asarray(reference_archive["v"][reference_index], dtype=float).copy()
        reference_p = np.asarray(reference_archive["p"][reference_index], dtype=float).copy()

    with np.load(prediction_path, allow_pickle=False) as prediction_archive:
        prediction_re = np.asarray(prediction_archive["Re"], dtype=float)
        prediction_matches = np.flatnonzero(np.isclose(prediction_re, float(reynolds)))
        if prediction_matches.size != 1:
            available = ", ".join(f"{value:g}" for value in prediction_re)
            raise ValidationError(
                f"Re={reynolds:g} is not a retained blind case; choose one of {available}"
            )
        prediction_index = int(prediction_matches[0])
        prediction_u = np.asarray(
            prediction_archive["u"][prediction_index], dtype=float
        ).copy()
        prediction_v = np.asarray(
            prediction_archive["v"][prediction_index], dtype=float
        ).copy()
        if "p" not in prediction_archive.files:
            raise ValidationError(
                "Blind prediction archive predates the direct-pressure POD-DeepONet"
            )
        prediction_p = np.asarray(
            prediction_archive["p"][prediction_index], dtype=float
        ).copy()

    expected_shape = (len(y), len(x))
    fields = (
        reference_u, reference_v, reference_p,
        prediction_u, prediction_v, prediction_p,
    )
    if any(field.shape != expected_shape for field in fields):
        raise ValidationError("Blind demo field shape does not match the reference grid")
    if any(not np.isfinite(field).all() for field in fields):
        raise ValidationError("Blind demo archive contains a non-finite value")

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    return BlindDemoCase(
        reynolds=float(reynolds),
        x=x,
        y=y,
        reference_u=reference_u,
        reference_v=reference_v,
        reference_p=reference_p,
        prediction_u=prediction_u,
        prediction_v=prediction_v,
        prediction_p=prediction_p,
        inference_ms=float(timing["POD_DeepONet_ensemble_inference_ms"]),
        cfd_seconds=float(timing["CFD_Re275_seconds"]),
    )
