# Hypersonic-cylinder data license

Copyright (c) 2026 Ehsan Roohi.

Ehsan Roohi licenses `cylinder_teaching_subset.npz` under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**:
<https://creativecommons.org/licenses/by/4.0/legalcode>.

On 2026-09-05, Ehsan Roohi confirmed ownership of the cylinder data and
authorized proceeding with the open-data release proposed for FlowMLLab.
This data-specific grant is recorded separately from the article's open-access
license and the software's MIT license.

## Attribution and changes

Credit Ehsan Roohi and cite the associated research by E. Roohi, A. Shoja-Sani
and F. Ebrahimzadeh Azghadi, "Neural networks for rarefied gas dynamics:
Relaxation problem, polyatomic shock waves, and hypersonic cylinder flow,"
*Physics of Fluids* 38, 057108 (2026),
<https://doi.org/10.1063/5.0334590>.

Identify FlowMLLab's teaching derivative and link to this license. The
derivative selects a deterministic 50-by-50 grid from each original 400-by-400
field, excludes invalid/solid sentinel entries, and stores selected columns
and source-row identifiers in NPZ form. It retains 44,500 points in 20 cases.
The source and derivative hashes are in `manifest.json`. State any further
changes you make; attribution does not imply endorsement.

The license permits sharing and adaptation, including commercial reuse,
subject to its terms. No warranty of scientific accuracy is supplied.

## Scope

This notice applies to the named committed dataset. It does not relicense all
scripts, model checkpoints, logs, third-party works or other datasets in
`AllMachNNCylinder.zip`. It does not change micro-step or micro-nozzle rights.
It is effective for the current repository from the authorization date;
the immutable v1.4.0 archive has not been rewritten.
