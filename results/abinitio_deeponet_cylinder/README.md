# Week 10.1 — Jäger Ar–Ar collision DeepONet: cylinder fields and surface profiles

[Week 10](../../notebooks/week10/README.md) ·
[Lecture companion](../../lectures/week10_1_abinitio_collision_deeponet.md)

## Research attribution

Ehsan Roohi, Ahmad Shoja-sani and Stefan Stefanov,
*Physics constrained neural collision operators for hard sphere surrogates
and ab initio angle prediction in direct simulation Monte Carlo*,
**Physics of Fluids 38, 057123 (2026)**,
[10.1063/5.0328463](https://doi.org/10.1063/5.0328463).

These pre-existing research fields were supplied by Ehsan Roohi from the
July 2026 Jäger/DSMC DeepONet package. The related article describes an MLP;
this later DeepONet package is **not a reproduction of that checkpoint or
the paper's speedup**. FlowMLLab adds AI-assisted verification and plotting,
not new simulations. This is separate from Week 7.1's Fusion-DeepONet paper.

## Colored fields

![Temperature](temperature_exact_deeponet.png)

![Mach number](mach_exact_deeponet.png)

![Pressure](pressure_exact_deeponet.png)

![Number density](density_exact_deeponet.png)

[Overview](contours_overview.png) · [Vector temperature figure](temperature_exact_deeponet.svg)
· [Provenance and rendering checks](contour_manifest.json)

PNG pairs are 3840 × 1280 pixels, 320 dpi. The script also generates SVG
versions of every pair; the temperature SVG is retained here. Each quantity
has common colors across both runs. Density is divided by the upstream
4.247e20 m^-3 and uses logarithmic colors.

## Surface pressure and heat flux

![Reconstructed cylinder surface pressure and heat flux](surface_comparison.png)

[Vector surface figure (SVG)](surface_comparison.svg). This is the previously
prepared, author-requested surface comparison, retained without smoothing.
The figure's **neural-derived table** is the July **DeepONet-derived table**,
not the original article's MLP. Pressure is in Pa and heat flux in kW/m².
The archived angle convention places the upstream stagnation region near 180°.

These profiles use increments ending at **NOUT 88–98** in both compact
surface archives, not the single NOUT 98/95 full-field pair above.

| Sampling window | Exact-derived table | DeepONet-derived table |
| --- | --- | --- |
| Initial tU/D | 8.45938 | 8.46433 |
| Final tU/D | 9.52651 | 9.53072 |
| Reconstructed incident samples | 4,280,969 | 4,278,976 |

| Surface profile | Relative spatial L2 difference, DeepONet versus exact |
| --- | --- |
| Pressure | 0.4451% |
| Heat flux | 1.0919% |

The norm uses the same 90 equal surface intervals. Cumulative statistics
were converted to non-overlapping increments: subtract the previous
exposure-weighted profile when sampling continues, and do not subtract it
after a sampling reset. Increment means are then weighted by exposure,
estimated from the median incident-count/number-flux ratio under the
archived planar, equal-segment, unweighted sampling conditions. This avoids
counting overlapping cumulative samples repeatedly.

Shading is the range of **four non-overlapping block means**, not a confidence
interval. The first-versus-last-half differences are 0.6484% / 1.8440%
(pressure / heat flux) for exact and 0.5643% / 1.2757% for DeepONet.
Temporal variability is therefore comparable to the between-run differences.
This nearly time-matched, single-seed comparison does **not** isolate network
error, establish statistical significance or certify steady convergence.
Its percentages must not be used as errors of the colored fields.

Source archives supplied by the author:
`DS2V_UNIFIED_M10_COMPACT_RESULTS_20260726_230916.zip` (DeepONet) and
`DS2V_UNIFIED_EXACT_M10_LONG_COMPACT_20260726_233749.zip` (exact).
The 2500 × 950 PNG and vector SVG are derived teaching figures from the
pre-existing research outputs; no new DSMC runs were made. The research
attribution and reuse boundaries on this page apply to them as well.
These surface figures were added to the working course after v1.4.1;
the immutable v1.4.1 Zenodo archive is not retroactively changed.

## Interpretation limits

DeepONet learns a collision-angle map; DSMC produces the macroscopic flow.
Both runs use table lookup with different angular tables, not a direct
neural prediction of the plotted fields.

| Run | Output | tU/D | Cell-centre points |
| --- | --- | --- | --- |
| Exact-derived table | 98 | 9.526506 | 214,233 |
| DeepONet-derived table | 95 | 9.242032 | 214,233 |

**Different times and sampling windows: qualitative comparison only.**
No synchronized error map, convergence certificate, confidence interval or
speedup is claimed. Earlier surface-profile errors concern other reconstructed
windows and must not be used as accuracy measures of these fields.

## Verification and reproduction

Source: author-supplied `DEEPO_NET_CYLINDER_CONTOURS_308d9892.zip`, pinned by
SHA-256 in the manifest. ZIP CRC and all listed member hashes passed. Both
fields have identical coordinates, finite values, no duplicate centres and
no centres inside the cylinder.

Only the first Tecplot zone is used; subsequent boundary zones are excluded.
Linear interpolation onto an 851 × 401 display grid uses a triangulation
with solid-intersecting triangles masked. No extrapolation, statistical
smoothing, mirrored lower-half flow or synthetic samples are added. All
native values fall within the common color ranges. Display resolution is
not simulation resolution.

From the repository root, with the original ZIP available locally:

```bash
python qa/plot_deeponet_cylinder_contours.py /path/to/DEEPO_NET_CYLINDER_CONTOURS_308d9892.zip
```

Outputs default to ignored `tmp/abinitio_final/contours`, leaving retained
evidence untouched. The original archive and checkpoint are not redistributed.
Publication of these derived teaching figures was requested by Ehsan Roohi;
this does not grant a blanket license for the dataset, checkpoint or
third-party DS2V source. Cite the paper and FlowMLLab for this case.

This is a Week 10.1 supplemental reading case, not a new training notebook.
The archived v1.4.0 tag and DOI remain unchanged. This companion is included
in the separately requested v1.4.1 maintenance release.
