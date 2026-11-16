# Fresh label-free Ghia PINN baseline

These are case-specific PDE-trained networks. The existing supervised POD-DeepONet is a different experiment.

The checkpoint-bound sampling audit finds 14/21 collar pairs with identical collocation tensors and discrete objectives. Seeds 7 and 61 add no annulus points; seed 23 adds only one. Of those identical-objective pairs, 13 also have identical frozen models. The remaining Re1000/seed61 weight difference is an execution-sensitivity observation, not a collar response; its cause is not established.

42 frozen endpoints were scored. Of those, 0 meet all four recorded numeric targets; 13 of 21 matched collar pairs meet the 2% sensitivity target. A completed optimizer budget is not evidence of accurate flow.

| Re | Scored | Ghia u/v target | Interior vector target | All four targets | Interior vector error range |
|---:|---:|---:|---:|---:|---:|
| 100 | 6/6 | 0/6 | 3/6 | 0/6 | 3.32% to 12.94% |
| 400 | 6/6 | 0/6 | 0/6 | 0/6 | 68.93% to 118.57% |
| 1000 | 6/6 | 0/6 | 0/6 | 0/6 | 103.04% to 115.46% |
| 3200 | 6/6 | 0/6 | 0/6 | 0/6 | 97.24% to 108.91% |
| 5000 | 6/6 | 0/6 | 0/6 | 0/6 | 92.46% to 109.88% |
| 7500 | 6/6 | 0/6 | 0/6 | 0/6 | 95.32% to 107.03% |
| 10000 | 6/6 | 0/6 | 0/6 | 0/6 | 95.11% to 107.52% |

All three predeclared seeds and both collars appear in the centerline figures. Markers on error curves are the original Ghia positions, evaluated directly by the frozen network. Flagged printed entries remain visible. The field figures use predeclared seed 7 and collar 0.02 at Re1000 and Re7500; they are not a best-seed selection.

The native scores compare PINN automatic-differentiation vorticity with retained CFD transported vorticity at 511 x 511 interior nodes. The additional visualization applies one centered curl operator to both retained velocity fields at 509 x 509 common inner nodes; it never changes those original scores.

A zero collar difference is not sufficient sensitivity evidence: sparse fixed sampling can place no points in the removed annulus, leaving the training sets and trajectory identical. Inspect sample identities and annulus counts. The retained high-Re CFD fields have unresolved grid sensitivity. Collar sensitivity does not establish full-domain strong-residual convergence. No CFD pressure truth is available. Do not infer unseen-Re generalization or an optimizer ranking from this fixed-baseline campaign.

## Figures

[All seven Reynolds numbers: signed centerlines and Ghia errors](all_reynolds_centerlines.pdf)

[Re1000 physical fields](re1000_fields.png) · [Re1000 streamline overlay](re1000_streamlines.png)

[Re7500 physical fields](re7500_fields.png) · [Re7500 streamline overlay](re7500_streamlines.png)

Figure provenance and fixed scales are recorded in `figure_manifest.json`. The numeric ledger remains `../evaluation_manifest.json`.
