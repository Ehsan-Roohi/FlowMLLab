# Fixed sampling intervention: matched cavity comparison

The intervention replaces fixed uniform collocation with conditional-uniform strata and global area weights. Each of the 12 runs keeps its own original physical/network configuration, seeded initialization and closure/point-work cap. The original 42-case failed baseline remains unchanged.

| Re | Seed | Collar | Closures per method | Uniform vector error | Stratified vector error | Change |
|---:|---:|---:|---:|---:|---:|:---|
| 100 | 7 | 0.01 | 3017 | 3.316% | 10.563% | worsened |
| 100 | 7 | 0.02 | 3017 | 3.316% | 10.458% | worsened |
| 100 | 23 | 0.01 | 3029 | 4.698% | 14.159% | worsened |
| 100 | 23 | 0.02 | 3011 | 6.837% | 4.741% | improved |
| 100 | 61 | 0.01 | 3005 | 12.936% | 19.572% | worsened |
| 100 | 61 | 0.02 | 3005 | 12.936% | 6.102% | improved |
| 7500 | 7 | 0.01 | 3023 | 95.322% | 102.718% | worsened |
| 7500 | 7 | 0.02 | 3023 | 95.322% | 99.708% | worsened |
| 7500 | 23 | 0.01 | 3049 | 107.026% | 111.409% | worsened |
| 7500 | 23 | 0.02 | 3015 | 105.179% | 112.123% | worsened |
| 7500 | 61 | 0.01 | 3008 | 99.392% | 106.777% | worsened |
| 7500 | 61 | 0.02 | 3008 | 99.392% | 100.590% | worsened |

The signed centreline sheets contain all three seeds, both collars and the actual printed Ghia positions. Predeclared seed 7 / collar 0.02 supplies the physical field examples; it was not chosen by its reference error.

Speed uses one linear scale across Re100/Re7500 and all methods. Vorticity uses the same centered curl of stored u/v on 509 x 509 nodes and one symmetric-log scale; its signed error uses a linear scale. Original native AD-versus-transported-vorticity scores are retained separately. No CFD pressure truth exists in this experiment.

A reduction in a reported error is not full-domain convergence, universal optimizer superiority, or proof of corner integrability. The fit uses a finite punctured domain. The high-Re CFD reference has its own grid uncertainty. Three paired seeds are descriptive; report worsened observables as well as improved ones. Different collar/seed pairs can have different work caps. Optimizer-step seconds do not measure campaign wall time or controlled hardware speed, and no speed ranking is made here.

[All seeds: signed profiles and errors](all_pairs_centerlines.pdf)  |  [Matched physical field comparisons](physical_comparison.pdf)

Hashes, fixed scales and plotting choices are recorded in `figure_manifest.json`. The authoritative score comparison remains `../comparison/comparison_manifest.json`.

## Evaluation chronology

The earlier failed evaluation scored 4 cases and accessed targets. All 12 original trained endpoints had already frozen then. The repaired completed pass restores exact saved states, freezes all native projections before its own reference read, and performs no fitting or model selection.

The repaired inference path restores exact saved samples and trained state. Cross-host regeneration can shift a few annular coordinates in the last bits; this repair does not establish cross-platform optimizer continuation.
