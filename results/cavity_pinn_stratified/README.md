# Completed matched sampling intervention

All twelve frozen method pairs are scored: Re=100 and 7500, seeds 7, 23 and 61, collars 0.01 and 0.02. The intervention replaces uniform fixed collocation by 512 upper-corner-annulus points, 512 remaining-wall-band points and 1024 interior points, with correct global area weights. Physics, network, initialization and each original counterpart's closure/residual-point budget are preserved. Different seeds or collars can have different work caps.

Interior velocity error improves in **2/12** cases and worsens in **10/12**. Both gains are at Re100, collar 0.02, seeds 23 and 61; every Re7500 case worsens. Re100 stratified errors span 4.74–19.57%, and Re7500 spans 99.71–112.12%. The independent outer residual can decrease while field error increases. This intervention did not establish accurate high-Re flow or full-domain corner convergence. The [original 42-case failed baseline](../cavity_pinn_campaign/README.md) remains unchanged.

- [Verified pair comparison and chronology](comparison/comparison_manifest.json)
- [All physical paired metrics](comparison/paired_metrics.csv) · [Signed profiles at actual Ghia positions](comparison/paired_signed_profiles.csv)
- [All seeds and collars: physical centreline figures](figures/all_pairs_centerlines.pdf)
- [Matched speed, vorticity, signed errors and streamline overlays](figures/physical_comparison.pdf)
- [Field gallery and figure provenance](figures/README.md) · [Figure hashes](figures/figure_manifest.json)
- [Experiment bridge](experiment_manifest.json) · [Scoring ledger](evaluation_manifest.json)
- [Notebook](../../notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb) · [Lecture](../../lectures/week04_cavity_surrogates_deeponet.pdf) · [Protocol](../../docs/CAVITY_PINN_STRATIFIED_PROTOCOL.md)

Physical field examples use the predeclared seed 7/collar 0.02, never the best reference score. Speed shares one linear scale. The extra vorticity visualization uses the same centered curl on stored u/v on 509×509 nodes; frozen native AD-versus-CFD-transported-vorticity scores stay separate. There is no CFD pressure truth in this experiment.

## Evaluation repair and prior access

The earlier failed evaluation scored four cases and read their targets. All twelve original trained endpoints were already frozen. Cross-host regeneration shifted a few annular coordinates by at most about 1.11e−16; the saved tensors and complete trained state remained intact. A separate inference-only adapter restored those exact saved samples without resampling, fitting or model choice. All twelve native projections in the repaired pass froze before that pass's target reads. This does not erase the earlier access or certify cross-platform optimizer continuation; the exact low-level platform cause was not isolated.

Optimizer-step timings exclude failed allocations, queue/requeue time, checkpoint I/O and audits. They are not campaign wall time or a controlled hardware speed comparison. Read the protocol and runtime-event provenance before making cost claims.
