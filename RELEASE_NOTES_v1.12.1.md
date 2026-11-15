# FlowMLLab v1.12.1 - Show our CFD validation and neural predictions

Lab 5 previously showed only Ghia reference markers in its default plots.
It now loads actual FlowMLLab CFD solutions at Re=100,400,1000 and displays
computed streamlines, grid-dependent velocity profiles, signed sample errors
and recomputed raw/screened Ghia norms. Eight native archives and metadata are
checked against their hashes; all earlier metrics are reproduced.

The homepage adds the CFD/Ghia figure and the two held-out neural-profile
comparisons, alongside the existing full-field POD-DeepONet evidence. The lesson
explains the different labels, Reynolds ranges and test cases of these models.
Only Lab 5 is rebuilt. Earlier notebooks, source fields, frozen neural models,
existing lecture PDFs and other homepage sections are preserved.

The selected profile network improves over Adam, but interpolation wins both
held-out cases. Re=3200,5000,7500,10000 CFD fields remain missing. Iteration
convergence is not a claim of mesh/time-step independence or pressure validation.

Validation before publication: native archive/metric checks; original-notebook
preservation; three fresh Jupyter kernels; targeted numerical tests; course
release checks; distribution build and isolated-wheel smoke test. New comparison
figures were inspected at full resolution, with legends outside the axes.

Archived release: [Zenodo DOI 10.5281/zenodo.24247466](https://doi.org/10.5281/zenodo.24247466).
