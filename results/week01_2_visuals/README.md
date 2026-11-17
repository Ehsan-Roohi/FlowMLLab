# Week 1.2: RGB field atlas

[Notebook](../../notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb) ·
[Lecture](../../lectures/week01_2_pressure_velocity.pdf)

The unit-square cavity has a unit-speed upper lid and stationary remaining
walls. The retained cases cover Re=100, 400 and 1000. The plots use unchanged,
checksum-verified computed fields from the existing 26-case archive; no CFD
calculation is repeated for this presentation update.

| Quantity | Nondimensional definition | Common color range | Color normalization |
| --- | --- | --- | --- |
| Speed | sqrt(u²+v²)/U_lid | 0 to 1 | Linear |
| Pressure | p/(rho U_lid²), zero spatial mean | -2 to 2 | Symmetric log; linear for absolute pressure <= 0.01 |
| Vorticity | omega L/U_lid | -250 to 250 | Symmetric log; linear for absolute vorticity <= 1 |

RGB colors pass through blue, cyan, green, yellow and red. The colorbar labels
are physical values. Rounded limits contain every native extremum in the
retained archive. Filled contours interpolate native samples without
filtering; nearest samples fill half-cell edge bands. Thin black curves are
computed velocity streamlines. Their spacing is a plotting setting, not a
measure of circulation strength or an accuracy diagnostic.

The original field panels and their identifiers are retained. Figures 24–25
add pressure at Re=400 and 1000 with h/L=1/64, matching Figure 15 at Re=100.
FV uses 64 pressure cells per side; the independent streamfunction-vorticity
method uses 65 boundary-inclusive nodes and reconstructs pressure after its
velocity solve. The notebook also retains all 32-cell comparisons. Ghia
validates sampled velocity profiles, not pressure. Changing color or contour
interpolation does not improve the numerical solution.

The [render receipt](render_validation.json) records the unchanged native-array
digest, common scales, RGB export checks and hashes of all 13 field figures.
The [release check](release_validation.json) records notebook preservation
and the manual inspection of the rebuilt PDF.

Rebuild from the repository root:

```sh
python qa/refresh_week01_2_contours.py
python qa/build_week01_2_lecture.py
python qa/validate_week01_2_figure_guide.py
python qa/update_course_inventory.py --write
```

PNG files are RGB at 300 dpi; paired PDFs preserve vector labels and
streamlines with rasterized color fills. Source plotting functions also apply
to the notebook's optional fresh field comparisons.
