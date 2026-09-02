"""Public Python interface for FlowMLLab."""

from .core import ValidationError, discover_repository_root, validate_core_assets
from .cylinder_lbm import CylinderLBMConfig, recommended_parameters, simulate_cylinder

__all__ = [
    "CylinderLBMConfig",
    "ValidationError",
    "discover_repository_root",
    "recommended_parameters",
    "simulate_cylinder",
    "validate_core_assets",
]
__version__ = "1.2.0"
