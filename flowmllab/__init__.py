"""Public Python interface for FlowMLLab."""

from .core import ValidationError, discover_repository_root, validate_core_assets
from .cylinder_lbm import (
    CylinderLBMConfig,
    grid_convergence_diagnostics,
    recommended_parameters,
    simulate_cylinder,
)
from .gas_dynamics import validate_week8_evidence
from .mahdavi_deeponet import validate_week9_evidence

__all__ = [
    "CylinderLBMConfig",
    "ValidationError",
    "discover_repository_root",
    "grid_convergence_diagnostics",
    "recommended_parameters",
    "simulate_cylinder",
    "validate_core_assets",
    "validate_week8_evidence",
    "validate_week9_evidence",
]
__version__ = "1.2.0"
