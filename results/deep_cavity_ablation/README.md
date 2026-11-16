# Completed controlled D/W=5 optimizer study

All six fresh runs are frozen and scored: Re=1000, D/W=5, paired seeds 12345, 23456 and 34567, Adam→L-BFGS versus SOAP→self-scaled Broyden. Each route used the same paired initialization, 8192 fixed authored balanced points, six hidden layers of width 64, float64 and 2000+8000 function/gradient evaluations. Completion did **not** produce an accurate CFD solution.

Interior velocity NRMSE is 97.06–116.70% for Adam→L-BFGS and 101.41–106.59% for SOAP→SSB. SOAP→SSB lowers interior vorticity error in all three seeds, but lowers interior velocity error in only one and raises interior pressure error in all three. Near the upper-left corner it lowers velocity and vorticity errors in all three, while the errors remain large. This is a comparison of two whole pipelines; it cannot isolate the contribution of either stage alone.

The floor reference velocities are extremely small. Absolute errors near 1e-8 to 1e-7 of lid speed still correspond to roughly 100% regional error and missed weak circulation. Pressure uses one global interior gauge. The floor pressure profiles additionally remove each cut mean to expose shape; those plots must not replace signed regional offsets, which are about −0.0074 to −0.0079 in density-times-lid-speed-squared units. Vorticity error uses the same sampled derivative on both fields, separately from each frozen native AD field.

- [Full scoring manifest and paired controls](evaluation_manifest.json)
- [Every regional metric](regional_metrics.csv)
- [Complete physical figure set](lesson_figures/controlled_pilot_figures.pdf)
- [Objective histories: all seeds](optimizer_objective_history.pdf)
- [Upper profiles, first predeclared seed](profiles_upper_s12345.pdf) · [Floor profiles](profiles_floor_s12345.pdf) · [Streamlines](streamlines_s12345.pdf)
- [RGB presentation copies and hashes](lesson_figures/figure_manifest.json): scientific RGB pixels are unchanged; only fully opaque alpha channels were removed. All three seeds are shown in the notebook.
- [Teaching notebook](../../notebooks/week13/W13_D5_Controlled_Optimizer_Study.ipynb) · [Lecture](../../lectures/week13_rectangular_cavity_pinn.pdf)
- [Protocol](../../docs/DEEP_CAVITY_ABLATION_PROTOCOL.md) · [Attributed original source](../../data/week13_d5/author_source/NOTICE.md)

The separate [cost-only study](cost_study/README.md) remains an unscored admission experiment. Its 50+10-call budget is not this full pilot. Historical transferred/refined author workflows retain their own provenance and are not relabeled as these fresh runs. Three seeds and finite work establish neither a universal optimizer ranking nor full-domain convergence.

## Page index for the combined figure PDF

The [combined PDF](lesson_figures/controlled_pilot_figures.pdf) contains visible black Times New Roman footers, page labels and outline bookmarks identifying every seed and region. All pages are Re=1000, D/W=5. A 24-point footer margin was added without resampling or modifying the original scientific plot contents.

| Page | Seed | Content |
| --- | --- | --- |
| 1 | All three | Objective histories |
| 2 | 12345 | Upper profiles |
| 3 | 12345 | Floor profiles |
| 4 | 12345 | Streamline overlays |
| 5 | 23456 | Upper profiles |
| 6 | 23456 | Floor profiles |
| 7 | 23456 | Streamline overlays |
| 8 | 34567 | Upper profiles |
| 9 | 34567 | Floor profiles |
| 10 | 34567 | Streamline overlays |
