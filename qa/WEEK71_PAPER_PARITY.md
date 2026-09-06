# Week 7.1 paper-to-course evidence audit

Audited against FlowMLLab `bebee083f67169e25ab2aeeeb4df085adfa8a565` on 2026-09-06.
This closes the missing text-comparison task for the Mach-cylinder teaching case;
quantitative paper reproduction remains open. No new DSMC or model training was performed.

## Article anchors

Source: [publisher-provided article text](https://www.researchgate.net/publication/404728916_Neural_networks_for_rarefied_gas_dynamics_Relaxation_problem_polyatomic_shock_waves_and_hypersonic_cylinder_flow), DOI [10.1063/5.0334590](https://doi.org/10.1063/5.0334590), Section V.

| Dimension | Article statement and location | Course evidence / consequence |
| --- | --- | --- |
| Physical case | V.A: argon, Mach 5–15, Kn based on diameter 0.01, diameter 0.3048 m, freestream 200 K, diffuse wall 500 K. | The compact archive supplies fields; exact producing input decks are not verified here. |
| Simulation vs export grid | V.A: DS2V, 194×100 divisions, 1.5 million particles, adaptive collision cells. | Manifest records a 400×400 structured export, then a 50×50 selection. Export dimensions cannot establish collision-cell resolution or grid independence. |
| Sampling | V.B: 50,000 random spatial points per snapshot. | 2,225 deterministic retained points per case; 44,500 total. Different evaluation support. |
| Model | V opening names FNO/U-Net; V.B describes DeepONets, alternately five physical outputs and three MA/TOV/P outputs. | Text is inconsistent. Resolve using figure-specific source and checkpoints; do not silently select an architecture. Course uses one 3×96 tanh multilayer perceptron (MLP). |
| Split | V: separate interpolation and Mach-15 holdout protocols. | Course training uses integer Mach 5–14; validation 8.25/8.75/9.25/9.75; interpolation tests 5.5/6.5/7.5/8.5/9.5; extrapolation 15. |
| Linear baseline | V.E/Fig.31: Mach-15 stagnation/surface profiles include linear extrapolation. | Course compares global compact fields. It cannot establish superiority over the published model. |
| Cost | V.F/Table I: CPU-hours and GPU inference costs. | Course timing measures its own MLP fit; hardware, support, and total offline cost require matching. |

## Retained course results

Values below are percentages, transcribed programmatically from
[mlp_metrics.json](../results/hypersonic_cylinder_week7_1/mlp_metrics.json).
Relative L2 is the unweighted Euclidean error norm divided by the reference norm,
computed separately for each variable over all retained points in the split.
These are historical tests, not newly generated blind cases.

| Split | Variable | Linear baseline (%) | Teaching MLP (%) |
| --- | --- | ---: | ---: |
| interpolation | Local Mach | 0.3979 | 1.5302 |
| interpolation | Source temperature TOV | 0.6086 | 2.6975 |
| interpolation | Source pressure P | 0.7786 | 2.4959 |
| extrapolation | Local Mach | 0.5382 | 1.5250 |
| extrapolation | Source temperature TOV | 0.8768 | 3.3546 |
| extrapolation | Source pressure P | 0.8482 | 3.0903 |

The linear baseline wins on all six comparisons. At Mach 15 it uses the
Mach-13 and Mach-14 fields at matching source rows. The stored MLP was selected
at epoch 182 using validation loss, with seed 760. Neither result reproduces
the paper model. NPZ names ending in `_ratio` are legacy names, not verified
freestream normalizations.

## Evidence needed to close quantitative parity

1. Identify the exact checkpoint and source revision for each cylinder figure,
   including whether each is an ensemble, separate-output model, or shared model.
2. Recover its train/validation/test case lists, spatial sample indices, masks,
   fitted input/output scalers, physical units and preprocessing order.
3. Evaluate the recovered model and the linear baseline on identical points and
   profiles. Report per-variable full-field, shock-region and wall errors separately;
   preserve the original published metric definitions alongside new diagnostics.
4. Link the DSMC input decks and sampling history to the field archive. Keep
   simulation resolution separate from structured-export resolution. Assess
   particle, time-step and sampling sensitivity before claiming numerical independence.
5. Record inference hardware, batch size and point count with training and data-generation
   costs. Do not turn CPU-hours into elapsed speedup without processor counts.

The paper's Knudsen-parameterized and diatomic studies are outside this compact
Mach-only data contract and remain unassessed. No manuscript correction or
coauthor decision is implied by this audit.
