"""Classical binary scattering lab: deflection angles, transport cross-sections and a
Maxwellian-pair audit of a learned (E, b) -> cos(chi) surrogate.

Teaching potential: Lennard-Jones 12-6 in reduced units (epsilon = sigma = 1).
This is a transparent CPU analog of the ab initio angle-prediction problem in
Roohi, Shoja-Sani and Stefanov, Phys. Fluids 38, 057123 (2026). It does not use
the Jaeger Ar-Ar potential, the article's DeepONet, or any research checkpoint.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

B_MAX_DEFAULT = 3.0  # reduced impact-parameter cutoff used for the finite collision disk


def lennard_jones(r):
    """Reduced Lennard-Jones 12-6 potential V/epsilon with r in units of sigma."""
    r = np.asarray(r, float)
    inv6 = (1.0 / r) ** 6
    return 4.0 * (inv6 * inv6 - inv6)


def coulomb_repulsive(k):
    """Repulsive V(r) = k / r; its classical deflection is analytic (Rutherford)."""
    return lambda r: k / np.asarray(r, float)


def radial_function(r, b, energy, potential):
    """f(r) = 1 - b^2/r^2 - V(r)/E; the turning point is its outermost zero."""
    return 1.0 - (b / r) ** 2 - potential(r) / energy


def turning_point(b, energy, potential, r_start=1e-3, r_far=60.0, points=8000):
    """Outermost root of the radial function: scan a geometric grid inward from r_far,
    then bracket the first sign change with Brent's method."""
    grid = np.geomspace(r_start, r_far, points)
    with np.errstate(over="ignore", invalid="ignore"):
        values = radial_function(grid, b, energy, potential)
    values = np.where(np.isfinite(values), values, -np.inf)
    positive = np.where(values > 0.0)[0]
    if positive.size == 0:
        raise ValueError("No classically allowed region found in the scanned interval")
    idx = positive[-1]
    while idx > 0 and values[idx - 1] > 0.0:
        idx -= 1
    if idx == 0:
        raise ValueError("Radial function never changes sign; decrease r_start")
    lo, hi = grid[idx - 1], grid[idx]
    return brentq(lambda r: radial_function(r, b, energy, potential), lo, hi, xtol=1e-14, maxiter=200)


def deflection_angle(b, energy, potential=lennard_jones):
    """Classical deflection angle chi(b, E) in radians.

    chi = pi - 2 b * int_{r_min}^{inf} dr / (r^2 sqrt(f(r))).
    The substitution r = r_min / (1 - u^2) removes the inverse-square-root
    singularity at the turning point and maps r in [r_min, inf) to u in [0, 1).
    """
    if energy <= 0.0:
        raise ValueError("Energy must be positive")
    if b < 0.0:
        raise ValueError("Impact parameter must be non-negative")
    if b == 0.0:
        return float(np.pi)
    r_min = turning_point(b, energy, potential)

    def integrand(u):
        r = r_min / (1.0 - u * u)
        f = radial_function(r, b, energy, potential)
        if u < 1e-8:
            # limit of the transformed integrand at the turning point
            h = 1e-6
            f_prime = (radial_function(r_min + h, b, energy, potential)
                       - radial_function(r_min, b, energy, potential)) / h
            if f_prime <= 0.0:
                raise ValueError("Non-simple turning point (orbiting); outside the lab's range")
            return 2.0 * b / (r_min * np.sqrt(f_prime * r_min))
        if f <= 0.0:
            f = 1e-300
        dr_du = 2.0 * r_min * u / (1.0 - u * u) ** 2
        return (b / r ** 2) / np.sqrt(f) * dr_du

    integral, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-11, epsrel=1e-10)
    return float(np.pi - 2.0 * integral)


def rutherford_angle(b, energy, k):
    """Analytic deflection for V = k/r: tan(chi/2) = k / (2 E b)."""
    return 2.0 * np.arctan(k / (2.0 * energy * b))


def hard_sphere_angle(b, diameter=1.0):
    """Analytic deflection for a rigid sphere of the given diameter."""
    b = np.asarray(b, float)
    return np.where(b < diameter, 2.0 * np.arccos(np.clip(b / diameter, -1.0, 1.0)), 0.0)


@dataclass
class ScatteringTable:
    energies: np.ndarray      # reduced energies E/epsilon
    impact: np.ndarray        # reduced impact parameters b/sigma
    cos_chi: np.ndarray       # shape (len(energies), len(impact))

    def as_samples(self):
        e, b = np.meshgrid(self.energies, self.impact, indexing="ij")
        return np.column_stack([np.log(e.ravel()), b.ravel()]), self.cos_chi.ravel()


def build_table(energies, impact, potential=lennard_jones):
    cos_chi = np.empty((len(energies), len(impact)))
    for i, e in enumerate(energies):
        for j, b in enumerate(impact):
            cos_chi[i, j] = np.cos(deflection_angle(float(b), float(e), potential))
    return ScatteringTable(np.asarray(energies, float), np.asarray(impact, float), cos_chi)


def transport_cross_sections(impact, cos_chi):
    """Diffusion Q1 = 2 pi int (1 - cos chi) b db and viscosity Q2 = 2 pi int (1 - cos^2 chi) b db."""
    b = np.asarray(impact, float)
    q1 = 2.0 * np.pi * np.trapezoid((1.0 - cos_chi) * b, b, axis=-1)
    q2 = 2.0 * np.pi * np.trapezoid((1.0 - cos_chi ** 2) * b, b, axis=-1)
    return q1, q2


def omega22(energies, q2, reduced_temperature):
    """Collision integral Omega^(2,2)(T*) = (1/2) int_0^inf exp(-x) x^3 Q2(x T*) dx with
    x = E/(k T), by trapezoid over the tabulated energies (units of epsilon and sigma^2).
    The table must cover x up to about 15 for the truncation error to be negligible."""
    x = np.asarray(energies, float) / reduced_temperature
    weights = np.exp(-x) * x ** 3
    return 0.5 * np.trapezoid(weights * q2, x)


def omega22_star(energies, q2, reduced_temperature):
    """Omega^(2,2) normalized by its rigid-sphere value 2 pi (unit diameter), the quantity
    tabulated for the Lennard-Jones potential in Hirschfelder, Curtiss and Bird."""
    return omega22(energies, q2, reduced_temperature) / (2.0 * np.pi)


def fit_surrogate(table, hidden=(48, 48), seed=0, max_iter=2000):
    """Small MLP (log E, b) -> cos chi, standardized inputs, trained on the table only."""
    x, y = table.as_samples()
    scaler = StandardScaler().fit(x)
    model = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", solver="lbfgs",
                         max_iter=max_iter, random_state=seed, tol=1e-9, max_fun=60000)
    model.fit(scaler.transform(x), y)
    return scaler, model


def predict_surrogate(fitted, energies, impact):
    scaler, model = fitted
    e, b = np.meshgrid(np.asarray(energies, float), np.asarray(impact, float), indexing="ij")
    x = np.column_stack([np.log(e.ravel()), b.ravel()])
    pred = np.clip(model.predict(scaler.transform(x)), -1.0, 1.0)
    return pred.reshape(e.shape)


def sample_collision_states(reduced_temperature, count, seed, b_max=B_MAX_DEFAULT):
    """Relative kinetic energy of Maxwellian pairs (E/kT ~ Gamma(3/2)) and equal-area impact parameters."""
    rng = np.random.default_rng(seed)
    energies = reduced_temperature * rng.gamma(1.5, 1.0, size=count)
    impact = b_max * np.sqrt(rng.uniform(0.0, 1.0, size=count))
    return energies, impact


def collision_weighted_audit(fitted, reduced_temperature, count=4000, seed=1, potential=lennard_jones,
                             energy_bounds=(0.5, 40.0), b_max=B_MAX_DEFAULT):
    """Conditional Maxwellian-pair error statistics; not accepted DSMC collisions.

    The historical function name is retained. Gamma(3/2) samples random pairs,
    without the relative-speed weighting required for collision-event sampling."""
    energies, impact = sample_collision_states(reduced_temperature, count, seed, b_max)
    keep = (energies >= energy_bounds[0]) & (energies <= energy_bounds[1])
    energies, impact = energies[keep], impact[keep]
    exact = np.array([np.cos(deflection_angle(float(b), float(e), potential)) for e, b in zip(energies, impact)])
    scaler, model = fitted
    pred = np.clip(model.predict(scaler.transform(np.column_stack([np.log(energies), impact]))), -1.0, 1.0)
    err = pred - exact
    return {
        "reduced_temperature": float(reduced_temperature),
        "states": int(energies.size),
        "discarded_out_of_range": int(np.count_nonzero(~keep)),
        "rms_cos_error": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_cos_error": float(np.max(np.abs(err))),
        "fraction_abs_error_above_0p1": float(np.mean(np.abs(err) > 0.1)),
        "mean_momentum_transfer_exact": float(np.mean(1.0 - exact)),
        "mean_momentum_transfer_surrogate": float(np.mean(1.0 - pred)),
    }
