# Week 10.2 - Kinetic-shock teaching evidence

[Week guide](../../weeks/week10_2/README.md) · [Executed notebook](../../notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb) · [Lecture](../../lectures/week10_2_bgk_shakhov_shock_pinn_dvm.pdf)

This unit uses a pinned extract of Ehsan Roohi's
[PINNs-DVM study](https://arxiv.org/abs/2605.29211v2), with source hashes and case
bindings in the [data manifest](../../data/week10_2_kinetic_shock/manifest.json).
The new teaching code is a NumPy implementation of moment, quadrature and
conservation diagnostics. The saved PINN profiles remain research outputs;
the module does not describe them as newly trained or independently held out.

## Physical figures

| Figure | What it shows |
| --- | --- |
| [Kinetic model schematic](kinetic_model_schematic.png) | One physical coordinate, three molecular velocity components and two collision targets |
| [Moment hierarchy](moment_hierarchy.png) | Why the transverse fourth-moment marginal is needed for separate R and Delta |
| [Homogeneous relaxation](collision_rates.png) | Analytic stress and heat-flux time histories; fresh collision-rate quadrature is checked in Table D |
| [BGK tube](tube_bgk_profiles.png) | Parent and final equation-trained profiles against their matched BGK DVM |
| [Shakhov tube](tube_shakhov_profiles.png) | Parent and final profiles against the Shakhov reference, with signed heat flux and stress |
| [Mach-2 optimizer path](m2_optimizer_profiles.png) | Seed141001 physical profiles and signed discrepancies; Table F retains all three seeds |
| [Moment-supervision comparison](supervised_bgk_profiles.png) | Equation-trained, five-field S0 and final eight-moment states against the same 6400-cell BGK reference |

Every Mach-2 plot concerns the same stationary Shakhov case. The transient tubes
are separately labelled and do not use a stationary constant-flux acceptance
test. Scientific legends are outside the plotted curves; signs and original
comparison coverage are retained.

## Metrics and interpretation

[profile_metrics.csv](profile_metrics.csv) records relative L2 percentages using
trapezoidal spatial weights on each case's declared nodes. Reference norms are
global over the comparison interval; no pointwise division at a zero crossing
is used. The notebook also plots unnormalized signed differences.

The supervised comparison uses the strict common overlap with the 6400-cell
reference, without endpoint extrapolation, alignment fitting or smoothing.
The source paper's corresponding 401-node table uses an unweighted norm. These
are explicitly different scoring conventions; the teaching CSV is not presented
as an exact reproduction of every final decimal in that table.

The five-field S0 was trained against a 400-cell reference. The final eight-moment
state uses 6400-cell targets and reference-derived guards. Both can be assessed
against the same 6400-cell reference here, but they are not an experiment in
which only the number of supervised moments changed. Intermediate eight-moment
states remain available in the extract and are labelled separately.

For Mach 2, split-half SSB improves the mean heat-flux error while slightly
worsening mean stress error relative to the preceding SSB stage. The stages
change objective/quadrature and cumulative work. The final runs retain failed
tail/projection checks; conservation alone does not certify the whole kinetic
solution. The final supervised tube likewise retains failed physical checks and
small-amplitude wrong-sign tails.

## Reproduction

Run the notebook on CPU to perform the fresh calculations and rescore saved
profiles. [execution.json](execution.json) records its measured build-environment
runtime and code-cell count. Installation and hosted-Colab execution times are
not inferred from that local measurement.

[validation.json](validation.json) records the final notebook/PDF hashes,
embedded-figure check, 27 focused tests, page-by-page review and preservation
of all 62 previously numbered notebooks. The repository release gate passes;
its unrelated Week 9 upstream-source check still requires an external cache.

To rebuild these figures and the new lesson only, from the repository root:

```sh
python qa/build_week10_2_materials.py --figures --notebook --execute --pdf
```

The PDF build needs licensed Times New Roman locally; font files are not bundled.
Source provenance and the distinction between the paper's CC BY 4.0 license and
the upstream repository's unspecified software license are retained in the
[extract notice](../../data/week10_2_kinetic_shock/README.md).
