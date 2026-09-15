# What our measurements do, and do not, say about Davidson's paper

Primary reading: [arXiv v3, 16 May 2026](https://arxiv.org/abs/2511.12493v3),
corresponding to the 2026 Journal of Turbulence paper. The downloaded code
distribution carries a later 11 September 2026 update. Matching those dates
and assets is part of reproduction, not an assumption.

| Item | Paper | Current local evidence | Permitted conclusion |
| --- | --- | --- | --- |
| Classical channel model | Underpredicts k while mean velocity is good | True unmodified 5200 case gives about 42.85% k error and 1.73% U error under our pointwise norm | Qualitatively consistent |
| Spatial PINN correction | Figure 5, before local-feature NN deployment | Released table-based 5200 case gives about 8.95% k error and 4.99% U error | Source-package result; not yet a verified reproduction of Figure 5 |
| Final PINN-NN at 5200 | Section 5.1 / Figure 8: good velocity with both models, improved k | The 4.99% value above is not from this final model | Do not attribute that particular trade-off to Figure 8 |
| Velocity overshoot | Section 5.1 explicitly discusses the 550 case | Our earlier percentages concern 5200 | Do not conflate Reynolds numbers |
| Iteration sensitivity | Paper describes slow coefficient oscillations and exponential averaging | NN-10000 run does not reach 1e-6 within its 40,000-iteration budget | Report actual failure of the gate; not proof the physical model is invalid |
| 10000 mesh | Table 1 reports 150 wall-normal cells | Current distributed NN-10000 case uses 120 | Not a mesh-identical reproduction of Figure 9 |
| Reference DNS | Lee-Moser channel statistics | All numerical values of bundled stress data exactly match the public primary-source table | No reference-data replacement caused the observed discrepancy |

Our percentages use an unweighted relative Euclidean norm on the RANS points.
They are not percentages quoted in the paper. A qualitative statement that a
profile is well predicted does not mathematically imply that every error norm
must decrease. Conversely, a newly calculated higher norm cannot automatically
be advertised as a published conclusion.

## Correct classroom wording

"The distributed table-based PINN example improves k under our chosen metric,
while its U error increases. These numbers characterize this package-stage
comparison. They are not yet verified reproductions of the paper's final
PINN-NN comparison and do not establish a contradiction with the paper."

The final NN-5200 check is an explicitly constructed case: original 5200 grid,
setup and restart, source NN deployment and checkpoints, and m=500 averaging
specified for this Reynolds number in the paper. It is not falsely labeled as
an unchanged NN-5200 run directory supplied in the archive. Its actual gate and
metrics appear in `results/week14_validation/summary.json`. The 40,000-iteration
run finished at residual 2.386e-3, above 1e-6: it is not a converged verification
of the paper's Figure 8.

## Before asserting paper-level numerical agreement

Match the actual figure and model stage; source revision; mesh; normalization;
coefficient/checkpoint set; averaging and clipping; reference data; restart
history; and achieved convergence. The known target-regeneration mismatch
should be resolved with an identified source version, not hidden by replacing
files until a visually pleasing answer appears.

No assertion is made that Davidson's paper is wrong. The current limitation
is traceability between the distributed files, their regeneration scripts,
and the reported paper figures.

One concrete state mismatch: the released c_k MinMaxScaler records nu_t/y
bounds [0.00018832, 0.36794152], whereas reconstructing that feature from the
current training-script input gives approximately [0.000036688, 0.38402563].
Thus the released scaler and current profile are not the same fitted data
state. We do not infer which state generated each paper curve from filenames.

The full inverse-PINN run also distinguishes best and final loss. All 200,000
epochs completed, reaching a minimum printed loss of about 1.24 but finishing
around 231, not the paper's stated final value of 4. The released scheduler's
80,000/800,000 milestone difference and environment differences remain explicit;
we do not claim either is the sole cause without a controlled experiment.
