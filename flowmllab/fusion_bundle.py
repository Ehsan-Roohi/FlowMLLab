"""Persist fitted affine preprocessing alongside an identified model ensemble."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_bundle(directory, models, scalers, provenance):
    directory = Path(directory)
    payload = {"schema": 1, "provenance": provenance, "models": {}, "scalers": []}
    for name in models:
        if Path(name).name != name:
            raise ValueError("Model names must be local basenames")
        payload["models"][name] = sha256(directory / name)
    for scaler in scalers:
        kind = type(scaler).__name__
        if kind == "MinMaxScaler":
            offset, scale = scaler.min_, scaler.scale_
        elif kind == "StandardScaler":
            scale = 1.0 / scaler.scale_
            offset = -scaler.mean_ * scale
        else:
            raise ValueError(f"Unsupported transform: {kind}")
        payload["scalers"].append({"kind": kind, "scale": scale.tolist(), "offset": offset.tolist()})
    # Written last: an interrupted training run never looks like a complete bundle.
    with (directory / "bundle.json").open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


class FrozenAffine:
    def __init__(self, state):
        self.scale = np.asarray(state["scale"], dtype=float)
        self.offset = np.asarray(state["offset"], dtype=float)
        if (self.scale.ndim != 1 or self.offset.shape != self.scale.shape
                or not np.all(np.isfinite(self.scale))
                or not np.all(np.isfinite(self.offset)) or np.any(self.scale == 0)):
            raise ValueError("Invalid affine preprocessing")

    def _array(self, values):
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != len(self.scale):
            raise ValueError("Input feature count does not match saved preprocessing")
        return values

    def transform(self, values):
        return self._array(values) * self.scale + self.offset

    def inverse_transform(self, values):
        return (self._array(values) - self.offset) / self.scale


def load_bundle(directory):
    directory = Path(directory)
    payload = json.loads((directory / "bundle.json").read_text())
    if payload["schema"] != 1 or not payload["models"] or len(payload["scalers"]) != 3:
        raise ValueError("Invalid ensemble bundle")
    for name, expected in payload["models"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"Model identity mismatch: {name}")
    return payload, tuple(FrozenAffine(s) for s in payload["scalers"])
