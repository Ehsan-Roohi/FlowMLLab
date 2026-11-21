# CPU lesson results

These figures accompany the current default classroom experiments. They are separate from the retained research campaigns linked inside each notebook.

| Lesson | Current output | What the student can change |
|---|---|---|
| Week 7 modal forecasting | `week07_dmd_probe.png` compares a CFD vorticity history, autonomous DMD and a same-rank POD projection on the Re = 110 validation window. The lower panel retains the signed discrepancy. | POD rank 2, 4 or 8; horizon 10, 25 or 50 frames; fixed training window and initial state. |
| Week 7.3 masked completion | `week07_masked_quick_metrics.json` records the bounded CPU experiment. Its figures are stored in the executed notebook. | Masking, target-label budget and equal-work controls; the quick run does not replace the full retained study. |
| Week 7.4 force decoding | `week07_lift_control.png` shows a fresh POD-ridge prediction of instantaneous lift at validation Re = 105 and its signed error. | Rank, label count and ridge penalty. This uses the current velocity field; it is not an autonomous forecast. |
| Week 8 branch learning | `week08_branch_predictions.npz` and `week08_branch_profiles.png` compare current single-network and branch-specific Rayleigh-flow predictions, with signed Mach-number errors. | Width, training coverage and branch labels. Reference targets come from the exact gas-dynamics relation. |
| Week 9.3 nozzle pairing | `week09_pairing_profiles.png` shows Mach, normalized pressure and coordinate-assignment errors for three analytical nozzles. | Throat location and sensor/query counts. This is a data-contract check, not neural-network training. |
| Week 10.1 collision map | `week10_collision_profiles.png` shows reference and surrogate deflection-map profiles at two unseen energies, retaining signed errors and the full plotted range. | Impact-parameter spacing, cutoff and network width. The demonstration is not a complete DSMC collision model. |
| Week 14 closure regression | `week14_gap_predictions.npz` and `week14_gap_profiles.png` contain the current MLP/PCHIP predictions across the withheld wall-distance interval. | Gap width, features, width and regularization. A good coefficient regression does not establish convergence of a coupled RANS solve. |

`week07_4_retained_lift.png` is a presentation-only replot of the original Re = 115 time/value arrays in `results/week07_4_diverse_pretraining/metrics.json`. The legend was moved outside the curves; no training, smoothing or resampling was performed.

The notebooks distinguish these current CPU calculations from historical model checkpoints, long inverse-PINN runs and retained RANS restarts. Validation selects settings; previously published test cases should not be described as a new independent blind test. Execution identities and individual figure/PDF reviews are recorded in `qa/course_refresh_middle.json`.
