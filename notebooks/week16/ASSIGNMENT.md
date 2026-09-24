# Week 16 assignment: design, predict, recompute

**Time:** two 75-minute sessions plus 3-5 hours of independent work. CFD data generation is instructor preparation; students use the retained dataset and may reproduce CFD as an extension.

**Prerequisites:** compressible flow and oblique shocks (Week 8), supervised regression, train/validation/test separation, and NumPy. POD from Week 4 is helpful; the notebook includes a short derivation.

## Scientific question

Can a learned model propose a fixed-volume body of revolution with a lower maximum off-body pressure coefficient at r/L=0.5 without increasing pressure drag by more than 2%? Does that proposal survive a fresh Euler calculation and mesh refinement?

## Dataset and model identity

Use `results/week16_lowboom/reference/clean_dataset_v801.npz` for student training: all 44 labels were recomputed with official SU2 8.0.1 and passed the retained convergence and full-field physical plausibility checks. Preserve the 24/6/8/6 training/validation/test/extrapolation split. These checks reject obvious numerical failures; they do not establish experimental accuracy or continuum mesh independence.

The primary portable model is `reference/clean_model_v801.npz`; verify it with `python qa/week16/clean_model_v801.py --check-only`. Its fixed architecture was fitted once using only the 24 training cases. On eight finer-mesh reference cases its aggregate waveform, peak and drag errors are 7.20%, 2.24% and 2.27%; the worst individual waveform error is 15.37%. The extrapolation waveform error is 34.29%, so aggregate acceptance inside this family does not justify extrapolation. This is a retrospective evaluation on known geometries, not a prospective blind test. Read `clean_model_audit_v801.json` and `clean_model_training_v801.json` for exact values, identities and settings.

Keep the original `dataset.npz`, historical checkpoints and optimizer results as historical evidence. The original SU2 8.5.0 labels failed the later full-field energy plausibility gate and must not be used as the primary validated training set. The retained optimized shape was proposed by the historical optimization pipeline; it was independently recomputed and checked using SU2 8.0.1. It is not an optimization result produced by the new clean model. New student optimizations need their own CFD confirmation before they can be accepted.

## Deliverables and rubric (100 points)

1. **Geometry and governing equations (15).** Derive the volume normalization of the radius function and the axisymmetric Euler source term. Explain why a planar 2D solution changes the physical problem. Plot three admissible shapes with identical axis scales. Check volume by independent quadrature.
2. **CFD evidence (20).** Explain the slip wall, symmetry axis and farfield conditions. Integrate the Taylor-Maccoll cone ODE and reproduce the cone comparison. For NASA SEEB-ALR, derive the pressure normalization, identify the prescribed coordinate shifts and compare both experiments separately with SU2 and LAVA; include the three-mesh waveform difference. Tabulate baseline mesh and domain differences. Explain why the number of iterations alone is not a convergence measure.
3. **Learning (20).** Train ridge, POD-MLP and POD-GP models using the frozen training geometry IDs. Use validation only for selecting the design model. Report waveform relative L2, peak error and pressure-drag error separately for test and extrapolation. Explain any case in which a neural model loses to a simpler baseline. Load the portable checkpoint without retraining and distinguish its retrospective evaluation from the historical frozen-prediction audit. Report the worst individual error as well as the aggregate error.
4. **Design (20).** Reproduce constrained optimization using the clean training labels, plot the baseline and proposed shapes, and tabulate predicted versus recomputed objectives. Clearly separate a new proposal from the retained historical candidate and its SU2 8.0.1 confirmation. Report pressure drag with the documented reference area. A surrogate-only improvement earns no CFD-validation credit. Apply performance constraints to each candidate separately: an alternative can pass physical checks but fail the drag constraint (retained alternative 1 exceeds the baseline drag by about 5.15%, above the allowed 2%).
5. **Stress test (15).** Inspect M=1.7 and M=1.9 reruns and all three extraction radii. Explain which conclusions transfer and which remain condition-specific. Do not call radial-distance checks an atmospheric uncertainty study.
6. **Interpretation and reproducibility (10).** Supply software versions, seeds and the dataset SHA-256. Explain why near-field peak reduction cannot be translated into a PLdB benefit without propagation and a validated loudness calculation.

## Suggested investigations

- Reduce training labels to 6, 12 and 24, keeping validation and test unchanged. Plot error versus number of actual CFD runs, including training cost.
- Replace pressure-peak optimization with pressure-signature L2 optimization. Compare both using their own CFD-confirmed designs.
- Find two parameter vectors with similar predicted pressure signatures and different drag. Check the forward solutions before describing them as equivalent designs.
- Repeat the optimization with a stricter shape-slope or radius constraint.

## Submission

One executed notebook, a two-page engineering memo, and a small CSV of baseline/predicted/recomputed metrics. State unsuccessful designs and failed checks explicitly. Do not upload the paper PDF or claim reproduction of the Beihang aircraft.
