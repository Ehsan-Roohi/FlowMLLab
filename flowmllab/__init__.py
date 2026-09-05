"""Public Python interface for FlowMLLab."""

from .core import ValidationError, discover_repository_root, validate_core_assets
from .aescte_dsmc import validate_aescte_evidence
from .cylinder_lbm import (
    CylinderLBMConfig,
    grid_convergence_diagnostics,
    recommended_parameters,
    simulate_cylinder,
)
from .gas_dynamics import validate_week8_evidence
from .hypersonic_cylinder import validate_hypersonic_cylinder_evidence
from .mahdavi_deeponet import validate_week9_evidence
from .probabilistic_uq import (
    fit_bayesian_linear_regression,
    fit_pod_gaussian_process,
    validate_probabilistic_uq_evidence,
)
from .step_geom_deeponet import (
    StepDomain,
    build_step_geom_deeponet,
    infer_step_domain,
    step_geom_deeponet_inputs,
    step_signed_distance,
)

__all__ = [
    "CylinderLBMConfig",
    "StepDomain",
    "ValidationError",
    "build_step_geom_deeponet",
    "discover_repository_root",
    "fit_bayesian_linear_regression",
    "fit_pod_gaussian_process",
    "grid_convergence_diagnostics",
    "infer_step_domain",
    "recommended_parameters",
    "simulate_cylinder",
    "step_geom_deeponet_inputs",
    "step_signed_distance",
    "validate_core_assets",
    "validate_aescte_evidence",
    "validate_hypersonic_cylinder_evidence",
    "validate_probabilistic_uq_evidence",
    "validate_week8_evidence",
    "validate_week9_evidence",
]
__version__ = "1.3.0"
