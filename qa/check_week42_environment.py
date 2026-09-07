#!/usr/bin/env python3
"""Fail-fast dependency and GPU gate for the Week 4.2 Unity run."""
from importlib.metadata import version

import joblib
import matplotlib
import numpy
import scipy
import sklearn
import skopt
import torch
from pytorch_optimizer.optimizer.soap import SOAP


EXPECTED = {
    "torch": "2.5.1",
    "scikit-optimize": "0.10.2",
    "pytorch-optimizer": "3.10.1",
    "sympy": "1.13.1",
}


def main():
    found = {name: version(name) for name in EXPECTED}
    if found != EXPECTED:
        raise RuntimeError(f"Version gate failed: {found} != {EXPECTED}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError(f"GPU lacks the requested Ampere-or-newer capability: {torch.cuda.get_device_capability(0)}")
    x = torch.arange(8, device="cuda", dtype=torch.float64)
    if (x * x).sum().item() != 140.0:
        raise RuntimeError("CUDA float64 arithmetic gate failed")
    # Resolve the exact APIs imported or reachable by the pinned upstream module.
    assert SOAP is not None and skopt.sampler.Hammersly is not None and skopt.sampler.Sobol is not None
    print({
        "status": "READY", "versions": found,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
    })


if __name__ == "__main__":
    main()
