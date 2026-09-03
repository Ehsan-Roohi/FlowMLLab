"""Exact gas-dynamics relations and the retained Week-8 evidence contract.

The analytical relations are adapted from the author's GasDynamicsSciML
repository at commit 374431a1033138f56e2752bf8bbf9b75a454d80c.  They are
kept here so the FlowMLLab teaching notebooks remain executable from one
checkout.  GasDynamicsSciML remains the authoritative research implementation.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar


GAMMA = 1.4
GASDYNAMICS_SCIML_COMMIT = "374431a1033138f56e2752bf8bbf9b75a454d80c"
INTRO_COMPRESSIBLE_COMMIT = "fc14721ae80f48da63e55955c1caf096d8448f7b"
SU2_DIAMOND_COMMIT = "cd36c78f027a6c096c62839063e5e91c930e0c7c"


def rayleigh_ratios(mach: np.ndarray | float, gamma: float = GAMMA) -> np.ndarray:
    """Return ``T/T*``, ``P/P*``, ``rho/rho*``, ``u/u*``, ``T0/T0*``, ``P0/P0*``."""
    m = np.asarray(mach, dtype=float)
    if np.any(m <= 0.0):
        raise ValueError("Mach number must be positive.")
    denominator = 1.0 + gamma * m**2
    temperature = (1.0 + gamma) ** 2 * m**2 / denominator**2
    pressure = (1.0 + gamma) / denominator
    density = denominator / ((1.0 + gamma) * m**2)
    velocity = (1.0 + gamma) * m**2 / denominator
    stagnation_factor = (
        1.0 + 0.5 * (gamma - 1.0) * m**2
    ) / (0.5 * (gamma + 1.0))
    stagnation_temperature = temperature * stagnation_factor
    stagnation_pressure = pressure * stagnation_factor ** (gamma / (gamma - 1.0))
    return np.stack(
        [temperature, pressure, density, velocity,
         stagnation_temperature, stagnation_pressure],
        axis=-1,
    )


def rayleigh_inverse_t0(
    stagnation_temperature_ratio: float,
    branch: str,
    gamma: float = GAMMA,
) -> float:
    """Invert ``T0/T0*`` on a declared subsonic or supersonic branch."""
    target = float(stagnation_temperature_ratio)
    if not 0.0 < target <= 1.0:
        raise ValueError("T0/T0* must lie in (0, 1].")
    if abs(target - 1.0) < 1.0e-13:
        return 1.0
    residual = lambda mach: float(rayleigh_ratios(mach, gamma)[4] - target)
    if branch == "subsonic":
        return float(brentq(residual, 1.0e-5, 1.0 - 1.0e-9))
    if branch == "supersonic":
        return float(brentq(residual, 1.0 + 1.0e-9, 30.0))
    raise ValueError("branch must be 'subsonic' or 'supersonic'.")


def fanno_ratios(mach: np.ndarray | float, gamma: float = GAMMA) -> np.ndarray:
    """Return ``T/T*``, ``P/P*``, ``rho/rho*``, ``P0/P0*``, and ``4fL*/D``."""
    m = np.asarray(mach, dtype=float)
    if np.any(m <= 0.0):
        raise ValueError("Mach number must be positive.")
    denominator = 2.0 + (gamma - 1.0) * m**2
    temperature = (gamma + 1.0) / denominator
    pressure = np.sqrt((gamma + 1.0) / denominator) / m
    density = np.sqrt(denominator / (gamma + 1.0)) / m
    stagnation_pressure = (
        (denominator / (gamma + 1.0))
        ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    ) / m
    friction_length = (1.0 - m**2) / (gamma * m**2)
    friction_length += (gamma + 1.0) / (2.0 * gamma) * np.log(
        ((gamma + 1.0) * m**2) / denominator
    )
    friction_length = np.maximum(friction_length, 0.0)
    return np.stack(
        [temperature, pressure, density, stagnation_pressure, friction_length],
        axis=-1,
    )


def fanno_inverse_friction_length(
    friction_length: float,
    branch: str,
    gamma: float = GAMMA,
) -> float:
    """Invert ``4fL*/D`` on a declared subsonic or supersonic branch."""
    target = float(friction_length)
    if target < 0.0:
        raise ValueError("4fL*/D must be non-negative.")
    if target < 1.0e-14:
        return 1.0
    residual = lambda mach: float(fanno_ratios(mach, gamma)[4] - target)
    if branch == "subsonic":
        return float(brentq(residual, 1.0e-4, 1.0 - 1.0e-10))
    if branch == "supersonic":
        branch_limit = float(fanno_ratios(1.0e4, gamma)[4])
        if target >= branch_limit:
            raise ValueError("Supersonic Fanno target exceeds the finite branch limit.")
        return float(brentq(residual, 1.0 + 1.0e-10, 1.0e4))
    raise ValueError("branch must be 'subsonic' or 'supersonic'.")


def mach_angle(mach: np.ndarray | float) -> np.ndarray:
    """Return the Mach angle in radians."""
    m = np.asarray(mach, dtype=float)
    if np.any(m < 1.0):
        raise ValueError("Mach angle requires M >= 1.")
    return np.arcsin(1.0 / m)


def oblique_theta(
    mach: np.ndarray | float,
    beta: np.ndarray | float,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Evaluate the theta-beta-M relation; all angles are in radians."""
    m, shock_angle = np.broadcast_arrays(
        np.asarray(mach, dtype=float), np.asarray(beta, dtype=float)
    )
    numerator = 2.0 / np.tan(shock_angle) * (
        m**2 * np.sin(shock_angle) ** 2 - 1.0
    )
    denominator = m**2 * (gamma + np.cos(2.0 * shock_angle)) + 2.0
    return np.arctan(numerator / denominator)


@lru_cache(maxsize=4096)
def oblique_detachment(mach_rounded: float, gamma: float = GAMMA) -> tuple[float, float]:
    """Return shock angle and maximum turn angle at attached-shock detachment."""
    mach = float(mach_rounded)
    if mach <= 1.0:
        raise ValueError("Oblique shocks require M > 1.")
    mu = math.asin(1.0 / mach)
    optimum = minimize_scalar(
        lambda beta: -float(oblique_theta(mach, beta, gamma)),
        bounds=(mu + 1.0e-8, 0.5 * math.pi - 1.0e-8),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    beta = float(optimum.x)
    return beta, float(oblique_theta(mach, beta, gamma))


def oblique_beta(
    mach: float,
    theta: float,
    branch: str,
    gamma: float = GAMMA,
) -> float:
    """Invert the attached theta-beta-M relation on the weak or strong branch."""
    if mach <= 1.0:
        raise ValueError("Oblique shocks require M > 1.")
    mu = math.asin(1.0 / mach)
    beta_peak, theta_max = oblique_detachment(round(float(mach), 10), gamma)
    if theta < -1.0e-12 or theta > theta_max + 1.0e-10:
        raise ValueError("theta is outside the attached-shock domain.")
    if theta <= 1.0e-12:
        return mu if branch == "weak" else 0.5 * math.pi
    residual = lambda beta: float(oblique_theta(mach, beta, gamma) - theta)
    if branch == "weak":
        return float(brentq(residual, mu + 1.0e-9, beta_peak - 1.0e-9))
    if branch == "strong":
        return float(
            brentq(residual, beta_peak + 1.0e-9, 0.5 * math.pi - 1.0e-9)
        )
    raise ValueError("branch must be 'weak' or 'strong'.")


def area_mach(mach: np.ndarray | float, gamma: float = GAMMA) -> np.ndarray:
    """Return the isentropic area ratio ``A/A*``."""
    m = np.asarray(mach, dtype=float)
    if np.any(m <= 0.0):
        raise ValueError("Mach number must be positive.")
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    core = (2.0 / (gamma + 1.0)) * (
        1.0 + 0.5 * (gamma - 1.0) * m**2
    )
    return core**exponent / m


def mach_from_area(area_ratio: float, branch: str, gamma: float = GAMMA) -> float:
    """Invert ``A/A*`` on the subsonic or supersonic branch."""
    target = float(area_ratio)
    if target < 1.0:
        raise ValueError("A/A* must be at least one.")
    if abs(target - 1.0) < 1.0e-13:
        return 1.0
    residual = lambda mach: float(area_mach(mach, gamma) - target)
    if branch == "subsonic":
        return float(brentq(residual, 1.0e-8, 1.0 - 1.0e-10))
    if branch == "supersonic":
        return float(brentq(residual, 1.0 + 1.0e-10, 50.0))
    raise ValueError("branch must be 'subsonic' or 'supersonic'.")


def normal_shock_downstream_mach(mach_upstream: float, gamma: float = GAMMA) -> float:
    """Return the downstream Mach number of a normal shock."""
    m1 = float(mach_upstream)
    if m1 <= 1.0:
        raise ValueError("Normal-shock upstream Mach number must exceed one.")
    return math.sqrt(
        (1.0 + 0.5 * (gamma - 1.0) * m1**2)
        / (gamma * m1**2 - 0.5 * (gamma - 1.0))
    )


def normal_shock_total_pressure_ratio(
    mach_upstream: float,
    gamma: float = GAMMA,
) -> float:
    """Return ``P02/P01`` through a normal shock."""
    m1 = float(mach_upstream)
    if m1 <= 1.0:
        raise ValueError("Normal-shock upstream Mach number must exceed one.")
    first = ((gamma + 1.0) * m1**2) / ((gamma - 1.0) * m1**2 + 2.0)
    second = (gamma + 1.0) / (2.0 * gamma * m1**2 - (gamma - 1.0))
    return first ** (gamma / (gamma - 1.0)) * second ** (1.0 / (gamma - 1.0))


def isentropic_pressure_ratio(mach: float, gamma: float = GAMMA) -> float:
    """Return static-to-stagnation pressure ``P/P0``."""
    return (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (
        -gamma / (gamma - 1.0)
    )


def nozzle_back_pressure(
    exit_area_ratio: float,
    shock_area_ratio: float,
    gamma: float = GAMMA,
) -> float:
    """Return ``Pb/P01`` for a normal shock inside a C-D nozzle."""
    area_exit = float(exit_area_ratio)
    area_shock = float(shock_area_ratio)
    if not 1.0 <= area_shock <= area_exit:
        raise ValueError("Shock area must satisfy 1 <= As/At <= Ae/At.")
    mach_1 = mach_from_area(area_shock, "supersonic", gamma)
    mach_2 = normal_shock_downstream_mach(mach_1, gamma)
    shock_over_downstream_star = float(area_mach(mach_2, gamma))
    exit_over_downstream_star = (
        area_exit / area_shock
    ) * shock_over_downstream_star
    mach_exit = mach_from_area(exit_over_downstream_star, "subsonic", gamma)
    return normal_shock_total_pressure_ratio(
        mach_1, gamma
    ) * isentropic_pressure_ratio(mach_exit, gamma)


def nozzle_shock_area(
    exit_area_ratio: float,
    back_pressure: float,
    gamma: float = GAMMA,
) -> float:
    """Invert the bounded internal-shock relation using a bracketed root."""
    area_exit = float(exit_area_ratio)
    pressure = float(back_pressure)
    pressure_at_throat = nozzle_back_pressure(area_exit, 1.0 + 1.0e-8, gamma)
    pressure_at_exit = nozzle_back_pressure(
        area_exit, area_exit - 1.0e-8, gamma
    )
    lower, upper = sorted((pressure_at_throat, pressure_at_exit))
    if not lower - 1.0e-9 <= pressure <= upper + 1.0e-9:
        raise ValueError("Back pressure is outside the internal-shock domain.")
    residual = lambda area: nozzle_back_pressure(area_exit, area, gamma) - pressure
    return float(
        brentq(residual, 1.0 + 1.0e-8, area_exit - 1.0e-8)
    )


def shock_tube_pressure_ratio_general(
    driver_pressure_ratio: float,
    driver_temperature_ratio: float = 1.0,
    gamma_driven: float = GAMMA,
    gamma_driver: float = GAMMA,
    gas_constant_ratio: float = 1.0,
) -> float:
    """Return ``P2/P1`` for an ideal shock tube with possibly distinct gases."""
    p4_p1 = float(driver_pressure_ratio)
    t4_t1 = float(driver_temperature_ratio)
    gamma_1 = float(gamma_driven)
    gamma_4 = float(gamma_driver)
    r4_r1 = float(gas_constant_ratio)
    if p4_p1 <= 1.0 or t4_t1 <= 0.0 or r4_r1 <= 0.0:
        raise ValueError("Require P4/P1 > 1 and positive T4/T1 and R4/R1.")
    if gamma_1 <= 1.0 or gamma_4 <= 1.0:
        raise ValueError("Heat-capacity ratios must exceed one.")
    sound_speed_ratio = math.sqrt(gamma_1 / (gamma_4 * r4_r1 * t4_t1))
    coefficient = (gamma_4 - 1.0) * sound_speed_ratio

    def residual(p2_p1: float) -> float:
        delta = p2_p1 - 1.0
        denominator = math.sqrt(
            2.0 * gamma_1 * (2.0 * gamma_1 + (gamma_1 + 1.0) * delta)
        )
        base = 1.0 - coefficient * delta / denominator
        if base <= 0.0:
            return math.inf
        log_prediction = math.log(p2_p1) - (
            2.0 * gamma_4 / (gamma_4 - 1.0)
        ) * math.log(base)
        if log_prediction > 700.0:
            return math.inf
        return math.exp(log_prediction) - p4_p1

    delta_zero = gamma_1 * (
        (gamma_1 + 1.0)
        + math.sqrt((gamma_1 + 1.0) ** 2 + 4.0 * coefficient**2)
    ) / coefficient**2
    upper = min(p4_p1, 1.0 + (1.0 - 1.0e-8) * delta_zero)
    return float(
        brentq(residual, 1.0 + 1.0e-12, upper, xtol=1.0e-12, rtol=1.0e-12)
    )


def shock_tube_pressure_ratio(
    driver_pressure_ratio: float,
    driver_temperature_ratio: float = 1.0,
    gamma: float = GAMMA,
) -> float:
    """Return ``P2/P1`` for the equal-gas ideal shock tube."""
    return shock_tube_pressure_ratio_general(
        driver_pressure_ratio,
        driver_temperature_ratio,
        gamma,
        gamma,
        1.0,
    )


def shock_tube_residual_general(
    p2_p1: np.ndarray | float,
    p4_p1: np.ndarray | float,
    t4_t1: np.ndarray | float,
    gamma_1: np.ndarray | float = GAMMA,
    gamma_4: np.ndarray | float = GAMMA,
    r4_r1: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Return the compatibility-equation residual for the ideal shock tube."""
    p2, p4, t4, g1, g4, gas_ratio = np.broadcast_arrays(
        np.asarray(p2_p1, dtype=float),
        np.asarray(p4_p1, dtype=float),
        np.asarray(t4_t1, dtype=float),
        np.asarray(gamma_1, dtype=float),
        np.asarray(gamma_4, dtype=float),
        np.asarray(r4_r1, dtype=float),
    )
    sound_speed_ratio = np.sqrt(g1 / (g4 * gas_ratio * t4))
    delta = p2 - 1.0
    term = (g4 - 1.0) * sound_speed_ratio * delta
    term /= np.sqrt(2.0 * g1 * (2.0 * g1 + (g1 + 1.0) * delta))
    base = 1.0 - term
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = p2 * np.where(
            base > 0.0,
            base ** (-2.0 * g4 / (g4 - 1.0)),
            np.nan,
        )
    return predicted - p4


def load_week8_evidence(root: str | Path) -> dict[str, Any]:
    """Load the frozen educational copy of GasDynamicsSciML evidence."""
    result_dir = Path(root) / "results" / "gas_dynamics_week8"
    return {
        "primary": pd.read_csv(result_dir / "primary_metrics.csv"),
        "baselines": pd.read_csv(result_dir / "baseline_comparison.csv"),
        "range_generalization": pd.read_csv(result_dir / "range_generalization.csv"),
        "high_dimensional": pd.read_csv(result_dir / "high_dimensional_scaling.csv"),
        "physical": pd.read_csv(result_dir / "physical_diagnostics.csv"),
        "application": json.loads(
            (result_dir / "application_audit_summary.json").read_text(encoding="utf-8")
        ),
        "provenance": json.loads(
            (result_dir / "provenance.json").read_text(encoding="utf-8")
        ),
    }


def validate_week8_evidence(root: str | Path) -> dict[str, Any]:
    """Fail closed if the frozen Week-8 evidence or its interpretation drifts."""
    evidence = load_week8_evidence(root)
    primary = evidence["primary"]
    baselines = evidence["baselines"]
    high_dimensional = evidence["high_dimensional"]
    application = evidence["application"]
    physical = evidence["physical"]
    provenance = evidence["provenance"]

    expected_problems = {
        "Rayleigh inverse",
        "Fanno inverse",
        "Oblique inverse",
        "Nozzle inverse",
        "Shock tube implicit",
    }
    if set(primary["problem"]) != expected_problems or len(primary) != 5:
        raise ValueError("Week-8 primary benchmark set has drifted.")
    if float(primary["rel_l2"].max()) >= 3.5e-3:
        raise ValueError("A retained primary relative-L2 gate failed.")
    if set(baselines["model"]) != {"classical_interpolation", "physics_guided_mlp"}:
        raise ValueError("Matched Week-8 baselines are incomplete.")
    mlp_rows = baselines.loc[baselines["model"] == "physics_guided_mlp"]
    if len(mlp_rows) != 5 or not np.allclose(mlp_rows["coverage"], 1.0):
        raise ValueError("The retained MLP coverage gate failed.")
    if high_dimensional["dimension"].astype(int).tolist() != [2, 3, 4, 5]:
        raise ValueError("High-dimensional audit must contain dimensions 2 through 5.")
    five_dimensional = high_dimensional.loc[
        high_dimensional["dimension"] == 5
    ].iloc[0]
    if not float(five_dimensional["mlp_rel_l2"]) < float(
        five_dimensional["interpolation_rel_l2"]
    ):
        raise ValueError("The declared five-dimensional matched-budget result failed.")
    if int(application["shock_tube_queries"]) != 100_000:
        raise ValueError("The application workload must retain 100,000 shock-tube states.")
    if float(application["shock_tube_speedup"]) <= 10.0:
        raise ValueError("The retained application speedup gate failed.")
    if set(physical["problem"]) != {
        "Rayleigh", "Fanno", "Oblique", "Nozzle", "Shock tube"
    }:
        raise ValueError("The retained physical diagnostics are incomplete.")
    if provenance["sources"]["GasDynamicsSciML"]["commit"] != GASDYNAMICS_SCIML_COMMIT:
        raise ValueError("GasDynamicsSciML source commit does not match the frozen evidence.")
    if provenance["sources"]["Introduction-to-Compressible-Flows"]["commit"] != INTRO_COMPRESSIBLE_COMMIT:
        raise ValueError("Classical-notebook source commit has drifted.")
    if provenance["sources"]["SU2-Diamond-Airfoil-Verification"]["commit"] != SU2_DIAMOND_COMMIT:
        raise ValueError("SU2 bridge source commit has drifted.")
    article = provenance.get("article_alignment", {})
    if article.get("manuscript_identifier") != "AITF-D-26-00044R1":
        raise ValueError("The Week-8 article alignment record has drifted.")
    if "does not infer editorial acceptance" not in article.get("status", ""):
        raise ValueError("The article status boundary must remain explicit.")

    return {
        "status": "pass",
        "problems": len(primary),
        "max_primary_relative_l2": float(primary["rel_l2"].max()),
        "mlp_full_coverage_problems": int(len(mlp_rows)),
        "five_dimensional_interpolation_relative_l2": float(
            five_dimensional["interpolation_rel_l2"]
        ),
        "five_dimensional_mlp_relative_l2": float(five_dimensional["mlp_rel_l2"]),
        "shock_tube_queries": int(application["shock_tube_queries"]),
        "shock_tube_speedup": float(application["shock_tube_speedup"]),
        "source_commit": GASDYNAMICS_SCIML_COMMIT,
    }
