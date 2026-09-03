#!/usr/bin/env python3
"""Build the Week-2.1 probabilistic-UQ teaching notebook deterministically."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "week02_1" / "Probabilistic_UQ_CFD.ipynb"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def markdown(identifier: str, text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": identifier,
        "metadata": {},
        "source": _source(text),
    }


def code(identifier: str, text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": identifier,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


CELLS = [
    markdown(
        "uq-00-title",
        r"""
        # Week 2.1 — Probabilistic uncertainty for CFD surrogates

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02_1/Probabilistic_UQ_CFD.ipynb)

        [Lecture PDF](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/lectures/week02_1_probabilistic_uq.pdf)

        A point prediction answers *what value does the model return?* A
        predictive distribution also asks *how wide should our uncertainty be,
        what source of uncertainty does it represent, and does that width agree
        with held-out errors?*

        This Week-2 increment develops those questions from an exact Bayesian
        linear model to a POD--Gaussian-process cavity surrogate. It does not
        modify the frozen Week-4 POD--DeepONet evidence.
        """,
    ),
    markdown(
        "uq-01-boundary",
        r"""
        ## Scope and claim boundary

        By the end of the lab you should be able to:

        1. write a Gaussian observation model and identify its likelihood;
        2. distinguish parameter, observation, numerical, and distribution-shift
           uncertainty;
        3. compute an exact posterior and posterior predictive distribution;
        4. evaluate interval coverage, sharpness, Gaussian NLL, and Gaussian CRPS;
        5. keep model fitting, calibration, and blind testing separate; and
        6. explain why coverage over correlated CFD nodes is descriptive rather
           than a finite-sample guarantee.

        This is an educational uncertainty workflow, not uncertainty
        certification for a production CFD solver.
        """,
    ),
    code(
        "uq-02-bootstrap",
        r"""
        # FLOWMLLAB_COLAB_BOOTSTRAP_V1
        from pathlib import Path as _FlowMLLabPath
        import os as _flowmllab_os
        import subprocess as _flowmllab_subprocess
        import sys as _flowmllab_sys

        if "google.colab" in _flowmllab_sys.modules:
            _flowmllab_root = _FlowMLLabPath("/content/FlowMLLab")
            if not (_flowmllab_root / ".git").is_dir():
                _flowmllab_subprocess.run(
                    [
                        "git", "clone", "--depth", "1",
                        "https://github.com/Ehsan-Roohi/FlowMLLab.git",
                        str(_flowmllab_root),
                    ],
                    check=True,
                )
            _flowmllab_subprocess.run(
                [
                    _flowmllab_sys.executable, "-m", "pip", "install", "-q",
                    "-e", str(_flowmllab_root),
                ],
                check=True,
            )
            _flowmllab_os.chdir(_flowmllab_root / "notebooks" / "week02_1")

        def _find_root(start=_FlowMLLabPath.cwd()):
            for candidate in (start, *start.parents):
                if (candidate / "flowmllab" / "probabilistic_uq.py").is_file():
                    return candidate
            raise FileNotFoundError("Run inside a complete FlowMLLab checkout")

        ROOT = _find_root()
        if str(ROOT) not in _flowmllab_sys.path:
            _flowmllab_sys.path.insert(0, str(ROOT))
        print("FlowMLLab root:", ROOT)
        """,
    ),
    code(
        "uq-03-imports",
        r"""
        import json

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Image, display

        from flowmllab.probabilistic_uq import (
            GaussianPrediction,
            calibration_curve,
            fit_bayesian_linear_regression,
            fit_pod_gaussian_process,
            gaussian_crps,
            gaussian_negative_log_likelihood,
            interpolate_complete_cases,
            predict_bayesian_linear_regression,
            predict_pod_gaussian_process,
            relative_l2_per_case,
            rescale_prediction,
            validation_scale_factor,
        )
        """,
    ),
    markdown(
        "uq-04-model",
        r"""
        ## 1. Observation model before loss function

        Suppose a nondimensional velocity sensor reports

        $$y_i = f(x_i;w) + \epsilon_i, \qquad
        \epsilon_i \sim \mathcal N(0,\sigma_n^2).$$

        The Gaussian negative log likelihood is, up to constants,

        $$\frac{1}{2\sigma_n^2}\sum_i(y_i-f_i)^2 + n\log\sigma_n.$$

        MSE therefore encodes a constant-variance Gaussian assumption; it is not
        a neutral choice. If a model also predicts $\sigma_i$, the logarithmic
        term prevents it from making every interval arbitrarily wide.

        We will keep four sources separate:

        - **observation/statistical:** repeated measurements or particle samples;
        - **parameter/model:** limited data or uncertain fitted functions;
        - **numerical:** discretization, convergence, sampling window, and labels;
        - **distribution shift:** a new Reynolds number, geometry, regime, or sensor layout.
        """,
    ),
    markdown(
        "uq-05-bayes",
        r"""
        ## 2. Exact Bayesian update for a laminar velocity profile

        For a fully developed parabolic profile, write

        $$u(r)=U_{max}\left[1-(r/R)^2\right].$$

        Treat $U_{max}$ as unknown. With a Gaussian prior and Gaussian sensor
        noise, the posterior is available by linear algebra:

        $$S_N^{-1}=S_0^{-1}+\Phi^T\Phi/\sigma_n^2,$$
        $$m_N=S_N\left(S_0^{-1}m_0+\Phi^Ty/\sigma_n^2\right).$$

        This example and its sensor locations are newly generated for FlowMLLab.
        """,
    ),
    code(
        "uq-06-bayes-code",
        r"""
        rng = np.random.default_rng(690)
        radius = np.linspace(0.05, 0.9, 10)
        design = (1.0 - radius**2)[:, None]
        true_umax = 1.45
        sensor_noise = 0.04
        measured_velocity = true_umax * design[:, 0] + rng.normal(
            0.0, sensor_noise, radius.size
        )

        posterior = fit_bayesian_linear_regression(
            design,
            measured_velocity,
            prior_mean=np.array([1.0]),
            prior_covariance=np.array([[0.25**2]]),
            noise_std=sensor_noise,
        )
        dense_radius = np.linspace(0.0, 1.0, 200)
        predictive = predict_bayesian_linear_regression(
            posterior,
            (1.0 - dense_radius**2)[:, None],
            include_observation_noise=True,
        )
        lower, upper = predictive.interval(0.95)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(dense_radius, lower, upper, alpha=0.25, label="95% predictive interval")
        ax.plot(dense_radius, predictive.mean, label="posterior mean")
        ax.plot(dense_radius, true_umax * (1.0 - dense_radius**2), "k--", label="truth")
        ax.scatter(radius, measured_velocity, color="tab:red", label="noisy sensors")
        ax.set(xlabel="r/R", ylabel="u", title="Bayesian update for a parabolic profile")
        ax.legend()
        plt.show()

        print(f"Posterior Umax = {posterior.mean[0]:.4f}")
        print(f"Posterior standard deviation = {np.sqrt(posterior.covariance[0,0]):.4f}")
        """,
    ),
    markdown(
        "uq-07-gp-theory",
        r"""
        ## 3. From a POD field to probabilistic coefficient functions

        A centered velocity field is approximated by

        $$q(Re) \approx \bar q + \sum_{k=1}^{r} a_k(Re)\,\phi_k.$$

        The POD basis is fitted from complete training cases only. Each
        coefficient function $a_k(Re)$ receives an independent Gaussian-process
        model. Its posterior mean reconstructs the field mean; coefficient
        variances propagate through the squared basis values to marginal field
        variances.

        Independence between coefficients, a fixed kernel, finite POD rank, and
        deterministic CFD labels are explicit approximations. The returned
        standard deviation does not include grid error, solver-form discrepancy,
        or geometry shift.
        """,
    ),
    markdown(
        "uq-08-split",
        r"""
        ## 4. Freeze the physical-case protocol

        | Role | Reynolds numbers | Permitted use |
        | --- | --- | --- |
        | Fit | 100, 150, 200, 225, 250, 350, 400 | POD basis and GP coefficients |
        | Calibration | 300 | one multiplicative interval-width factor |
        | Blind | 175, 275, 375 | final evidence only |

        Rank 4, normalized-Re length scale 0.75, and GP noise level $10^{-8}$
        are declared before the blind cases are opened. Linear complete-case
        interpolation sees exactly the same fit cases.
        """,
    ),
    code(
        "uq-09-fit",
        r"""
        with np.load(ROOT / "data" / "cavity_data.npz", allow_pickle=False) as archive:
            reynolds = np.asarray(archive["Re"], float)
            split = np.asarray(archive["split"]).astype(str)
            velocity = np.stack([archive["u"], archive["v"]], axis=1)

        train_re = np.array([100.0, 150.0, 200.0, 225.0, 250.0, 350.0, 400.0])
        validation_re = 300.0
        blind_re = np.array([175.0, 275.0, 375.0])

        def case_indices(values):
            return np.array([np.flatnonzero(np.isclose(reynolds, value))[0] for value in values])

        train_index = case_indices(train_re)
        validation_index = case_indices([validation_re])[0]
        blind_index = case_indices(blind_re)
        assert np.all(split[blind_index] == "test")
        assert not np.any(np.isin(train_re, blind_re))

        model = fit_pod_gaussian_process(
            train_re,
            velocity[train_index],
            rank=4,
            length_scale=0.75,
            noise_level=1.0e-8,
        )
        print("Cases used by the fitted object:", model.train_reynolds)
        """,
    ),
    markdown(
        "uq-10-calibration",
        r"""
        ## 5. Calibration is separate from fitting

        A single scale factor multiplies the raw GP standard deviation so that a
        central 90% interval contains 90% of the *validation* component values.
        The field interior is used because no-slip and lid transforms can make
        boundary uncertainty exactly zero.

        Reusing one correlated spatial field this way is a transparent classroom
        diagnostic, not a distribution-free coverage theorem. The decisive test
        is whether the frozen factor transfers to untouched Reynolds cases.
        """,
    ),
    code(
        "uq-11-calibration-code",
        r"""
        validation_raw = predict_pod_gaussian_process(model, validation_re)
        interior = (..., slice(1, -1), slice(1, -1))
        scale_factor = validation_scale_factor(
            velocity[validation_index][interior],
            GaussianPrediction(
                mean=validation_raw.mean[interior],
                std=validation_raw.std[interior],
            ),
            target_level=0.9,
        )
        validation_calibrated = rescale_prediction(validation_raw, scale_factor)
        validation_curve = calibration_curve(
            velocity[validation_index][interior],
            GaussianPrediction(
                mean=validation_calibrated.mean[interior],
                std=validation_calibrated.std[interior],
            ),
            levels=(0.5, 0.8, 0.9, 0.95),
        )
        display(pd.DataFrame({
            "nominal": validation_curve.nominal,
            "validation_coverage": validation_curve.observed,
            "mean_width": validation_curve.mean_width,
        }))
        print("Frozen standard-deviation scale:", scale_factor)
        """,
    ),
    markdown(
        "uq-12-scores",
        r"""
        ## 6. Accuracy, calibration, and sharpness are different

        Relative $L_2$ measures the predictive mean. Coverage asks how often
        observations fall inside declared intervals. Mean interval width measures
        sharpness. Gaussian NLL and CRPS are proper scores that jointly penalize
        location and spread.

        A narrow interval can look decisive and still have poor coverage. An
        extremely wide interval can cover everything and still be uninformative.
        No single metric is sufficient.
        """,
    ),
    code(
        "uq-13-score-sandbox",
        r"""
        score_rng = np.random.default_rng(691)
        sensor_repeats = score_rng.normal(0.0, 1.0, 20_000)
        for label, predicted_std in [("calibrated", 1.0), ("overconfident", 0.4), ("too wide", 2.0)]:
            prediction = GaussianPrediction(
                mean=np.zeros_like(sensor_repeats),
                std=np.full_like(sensor_repeats, predicted_std),
            )
            curve = calibration_curve(sensor_repeats, prediction, levels=(0.9,))
            print(
                label,
                "coverage=", round(float(curve.observed[0]), 3),
                "NLL=", round(gaussian_negative_log_likelihood(sensor_repeats, prediction.mean, prediction.std), 3),
                "CRPS=", round(gaussian_crps(sensor_repeats, prediction.mean, prediction.std), 3),
            )
        """,
    ),
    markdown(
        "uq-14-gate",
        r"""
        ## 7. Blind gate

        Before changing `OPEN_BLIND`, record the fixed rank, kernel, calibration
        level, baseline, and expected failure mode. If blind coverage is poor, do
        not return to the blind fields to tune the scale and continue calling
        them blind.
        """,
    ),
    code(
        "uq-15-blind-code",
        r"""
        OPEN_BLIND = False

        if not OPEN_BLIND:
            print("Blind cases remain closed. Freeze and record the protocol first.")
        else:
            blind_truth = velocity[blind_index]
            blind_raw = predict_pod_gaussian_process(model, blind_re)
            blind_prediction = rescale_prediction(blind_raw, scale_factor)
            baseline = interpolate_complete_cases(train_re, velocity[train_index], blind_re)
            comparison = pd.DataFrame({
                "Re": blind_re,
                "POD_GP_relative_L2": relative_l2_per_case(blind_truth, blind_prediction.mean),
                "interpolation_relative_L2": relative_l2_per_case(blind_truth, baseline),
            })
            display(comparison)
        """,
    ),
    markdown(
        "uq-16-evidence",
        r"""
        ## 8. Retained evidence and the useful failure

        The public evidence was generated once with the frozen protocol by
        `qa/run_probabilistic_uq_validation.py`. Validation scaling improves the
        interval width on `Re=300`, but its nominal 90% coverage does not fully
        transfer to the three blind fields. That negative result is retained:
        one held-out field and one global scale cannot certify a shifted,
        spatially correlated flow family.
        """,
    ),
    code(
        "uq-17-evidence-code",
        r"""
        evidence_dir = ROOT / "results" / "probabilistic_uq"
        summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(evidence_dir / "blind_metrics.csv")
        calibration = pd.read_csv(evidence_dir / "calibration.csv")
        display(pd.Series(summary, name="retained summary"))
        display(metrics)
        display(Image(filename=str(evidence_dir / "probabilistic_uq_validation.png")))
        """,
    ),
    markdown(
        "uq-18-questions",
        r"""
        ## Interpretation prompts

        1. Why is the GP standard deviation not numerical CFD uncertainty?
        2. Which assumption lets coefficient variances add through squared POD modes?
        3. Why may thousands of grid nodes still provide only weak calibration evidence?
        4. When should interpolation be preferred even if the GP mean has lower average error?
        5. Design a new calibration set that tests extrapolation without reusing a blind case.
        6. What additional model is required when experimental observations disagree systematically with CFD?

        A strong report separates point accuracy, interval calibration, physical
        validity, and the boundary of the training distribution.
        """,
    ),
    markdown(
        "uq-19-sources",
        r"""
        ## Public sources and originality

        - Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT
          Press (2006), [open book site](https://gaussianprocess.org/gpml/).
        - Gneiting & Raftery, “Strictly Proper Scoring Rules, Prediction, and
          Estimation” (2007), [DOI](https://doi.org/10.1198/016214506000001437).
        - Kennedy & O'Hagan, “Bayesian Calibration of Computer Models” (2001),
          [DOI](https://doi.org/10.1111/1467-9868.00294).
        - Lakshminarayanan, Pritzel & Blundell, “Simple and Scalable Predictive
          Uncertainty Estimation Using Deep Ensembles” (2017),
          [paper](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847fffa5a85bce38-Abstract.html).

        All prose, equations, code, figures, sensor locations, and exercises in
        this Week-2.1 module were independently created for FlowMLLab from these public
        sources and FlowMLLab-owned data. No restricted course handout, solution,
        figure, or code was incorporated. See `THEORY_SOURCE_POLICY.md`.

        <!-- MIE690A article-aligned validation v4 -->

        **Article relationship:** additive Week-2.1 educational increment. It does not own
        or alter a manuscript-facing figure in `ARTICLE_FIGURE_MAP.md`.
        """,
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "flowmllab_increment": {
            "name": "probabilistic-uq",
            "course_week": "2.1",
            "source_policy": "THEORY_SOURCE_POLICY.md",
            "evidence": "results/probabilistic_uq",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
