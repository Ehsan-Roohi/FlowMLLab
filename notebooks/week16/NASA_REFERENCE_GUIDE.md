> **Week 16 scope (24 September 2026):** the release covers the verified teaching-body CFD, neural model, design checks and Taylor–Maccoll study. NASA SEEB-ALR CFD failed convergence/physical checks and remains a deferred research extension; no successful NASA validation is claimed.

# NASA SEEB-ALR: geometry, reference data and a reproducible comparison

This guide adds an experimental reference to Week 16. Start with the retained NASA data; run new CFD only after the geometry, coordinates and pressure normalization agree. The original sources are in `cases/week16_lowboom/reference/`, with download locations and SHA-256 hashes in `source_manifest.json`.

## Three bodies, three purposes

| Body | Question it answers | What it cannot establish |
|---|---|---|
| 7-degree cone, Taylor–Maccoll solution | Does axisymmetric Euler CFD reproduce the conical pressure solution? | Accuracy of a finite-body off-body pressure signature |
| Week 16 two-parameter, fixed-volume body | Can a surrogate propose a design that survives fresh CFD? | Full-aircraft lift, trim, propulsion or ground noise |
| NASA SEEB-ALR, including as-built nose and sting | How does a computed near-field signature compare with a measured signature? | Experimental validation of the two-parameter neural model |

The cone solution is a separately integrated ordinary differential equation. NASA's LAVA file is a CFD prediction, not an exact solution. The two `shiftavg` files contain wind-tunnel measurements and their supplied uncertainty information. Keep those sources distinct in every figure legend.

## 1. Inspect the geometry before meshing

Open `SEEB-ALR-as-built.stp` in Gmsh or a CAD viewer. Show the meridian with equal coordinate scales, then magnify the nose. Preserve the finite nose rather than closing it with an invented point; include the cylindrical sting and document where a numerical extension ends. A visually small change to the nose can alter the first compression wave.

The STEP file contains an INCH conversion-based unit with a millimetre base unit. CAD software can convert the coordinates on import. Therefore inspect the imported bounding box and explicitly record its unit; do not infer units from the filename. For a normalized body length of 17.667 inches, use 17.667 for inch coordinates or 17.667 × 25.4 for millimetre coordinates. Show both the dimensional model and the normalized model as a sanity check. The reference pressure line at H=21.2 inches corresponds to H/L≈1.200.

For an axisymmetric solve, extract a meridian and solve with the cylindrical source terms enabled. A planar 2D solve of that same outline is a different physical body. Label slip wall, exposed symmetry axis, inflow/farfield and outlet; plot the pressure-sampling line on the mesh. A useful mesh figure has an overall view, a nose inset and an inset where the shock crosses the sampling line.

## 2. Match the reference conditions and variables

NASA's retained README specifies Euler flow at Mach 1.6, zero angle of attack and zero sideslip. The LAVA submission identifies its first zone as H=21.2, PHI=0.0. Do not reuse the teaching campaign's Mach 1.8 configuration unchanged.

| Quantity | NASA comparison | Teaching campaign |
|---|---|---|
| Mach number | 1.6 | 1.8 |
| Geometry | SEEB-ALR as built and sting | Two-parameter fixed-volume body |
| Pressure ordinate | Δp/p∞ | Cp |
| Observation location | H=21.2 inches for the first LAVA zone | r/L=0.5 for the design objective |
| Flow model | Euler reference condition | Axisymmetric Euler |

For an ideal gas, q∞=γp∞M∞²/2 and therefore

\[
\frac{\Delta p}{p_\infty}=\frac{\gamma M_\infty^2}{2}C_p.
\]

At γ=1.4 and Mach 1.6 the multiplier is 1.792. Plotting Cp directly against Δp/p∞ would introduce a large artificial amplitude mismatch. Nominal teaching pressure and temperature can be used for inviscid nondimensional similarity, but must not be described as measured tunnel conditions.

## 3. Read the experimental records faithfully

The NASA Tecplot macros are included because the transformations are part of the data provenance. Using zero-based Python column numbers, they specify:

```python
import numpy as np
from pathlib import Path
source = Path('cases/week16_lowboom/reference')  # run from repository root
run195 = np.loadtxt(source / 'shiftavg_195_219-221.39.out')
run553 = np.loadtxt(source / 'shiftavg_553_578-580.out')
x195 = run195[:, 0] + 124.942 + 26.2945
x553 = run553[:, 0] + 124.748 + 26.2945
p195, u195 = run195[:, 1], run195[:, 4]
p553, u553 = run553[:, 1], run553[:, 4]
```

The macros plot the supplied uncertainty as V2±V5. Do not assign a confidence level that is not documented in the source. These coordinate shifts come from the NASA macros; they are not adjustable fit parameters. Apply no further horizontal translation, amplitude rescaling or constant pressure offset to make CFD agree. If a transformation is necessary because a solver uses another origin, derive it geometrically and document it before inspecting errors.

The LAVA file is Tecplot text with multiple `ZONE` sections. Parse the desired zone explicitly; blindly loading all numeric rows mixes observation locations. Its four columns are X, Y, Z and Δp/p∞. The small nonzero Y coordinates in the first zone do not change its stated H=21.2 extraction condition.

## 4. Build the comparison before judging the solver

First overlay both measured signatures with uncertainty bands and the retained LAVA first zone. Use x in inches and Δp/p∞ on the vertical axis. Then add your SU2 result as a separate curve. Retain the full available signals and declare a common comparison interval before computing errors; a possible fixed interval for this exercise is x=25–46 inches. Never silently extrapolate beyond a curve's sampled support.

A pointwise relative error is unstable near zero pressure. Use a waveform norm instead. On declared comparison points xᵢ, with reference samples pᵢ and interpolated CFD samples p̂ᵢ, define

\[
E_2=100\sqrt{\frac{\sum_i(\hat p_i-p_i)^2}{\sum_i p_i^2}},\qquad
E_{\rm peak}=100\frac{|\max_i\hat p_i-\max_i p_i|}{|\max_i p_i|}.
\]

These are unweighted discrete metrics. Nonuniformly spaced points receive unequal physical weighting per unit x; if using quadrature-weighted norms instead, state the weights and do not compare them as if they were the same metric. Report each experimental run separately, along with peak position and a plot of the complete waveform. Define explicitly whether “peak” means maximum positive pressure or maximum absolute amplitude; the formula above uses maximum positive pressure.

A fraction of computed points inside supplied uncertainty bands is descriptive coverage, not a statistical hypothesis test. Experimental uncertainty does not remove discretization or modelling error.

## 5. Reproduce CFD in a controlled sequence

1. Verify geometry units, nose closure, sting and observation line. Save the meridian and mesh-generation settings.
2. Run axisymmetric Euler at the NASA condition. Save the SU2 version, exact configuration, cell count, solver log, residual history, extracted pressure and exit status.
3. Check positive pressure and density, residual reduction and stabilization of physical outputs. Iteration count alone is insufficient.
4. Compare against the original experimental coordinates and a separately identified LAVA curve.
5. Refine the mesh while keeping geometry, boundaries, extraction and numerical settings fixed. Record any intentionally changed local refinement.
6. Repeat the comparison with the same window, samples and definitions. Archive unsuccessful runs as well as successful ones.

A residual converged run can still diffuse a shock or misrepresent a finite nose. Test geometry fidelity and shock resolution before attributing a mismatch to the physical model. Mesh alignment can reduce numerical diffusion, but choosing a grid solely because it matches one curve risks tuning to the benchmark.

Report mesh sensitivity as the difference between named mesh levels. Two similar answers are not automatically grid independence, and a formal Grid Convergence Index requires additional assumptions and a suitable refinement sequence. A nose cap held at a fixed number of cells is not uniformly refined with the rest of the mesh; disclose that limitation.

The resolved geometry generator remains in `seeb_resolved.py`. Its initial SU2 8.5.0 coarse solve failed convergence and physical checks. The controlled version replay uses exactly that archived mesh/configuration with official SU2 8.0.1, before the two finer meshes are permitted. Consult the actual report and publication status; this sequence is not itself a claim of success.

```bash
python qa/week16/install_su2_801.py
export SU2_CFD="$PWD/.tools/week16_su2_801/bin/SU2_CFD"
# Fresh checkout/output directories; keep the original failed reference intact.
gh run download 35941629706 --name nasa-resolved-level-1 --dir nasa_original
tar -xzf nasa_original/nasa-resolved-level-1.tar.gz
python qa/week16/seeb_version_v801.py --reference-folder results/week16_lowboom/reference/seeb_resolved_level_1
# Continue only if the coarse replay passes physical and numerical checks.
python qa/week16/seeb_family_v801.py --level 1.5
python qa/week16/seeb_family_v801.py --level 2
python qa/week16/reference_report.py
```

The three accepted-candidate levels are 1, 1.5 and 2, with 10, 15 and 20 cells on the finite nose cap. The original CAD sting ends at x/L≈1.656; a cylindrical extension reaches the outlet. Pressure extraction retains the keys `x_inches` and `dp_pinf`. All three meshes must converge and pass full-field thermodynamic checks; the finest must have waveform error below 20% and peak error below 10% against both records, with last-two waveform change below 5%. These thresholds are not relaxed when a run fails.

The earlier `seeb_reference.py` mesh family is retained for provenance but rejected as validation evidence. Its residual-converged coarse solution had an incorrect local stagnation state. Increasing limiter-freeze time cannot by itself establish physical correctness on an underresolved nose. See [the nose physics audit](NOSE_PHYSICS_AUDIT.md) for the observed failure, analytical checks and predeclared acceptance allowances. Original pilot evidence is in GitHub Actions run 35935655513; the subsequent mesh-scaled-freeze attempt is 35937834499.

No independently checkable new SU2-versus-NASA error table is asserted by this guide alone. A release claiming such validation must include its actual run evidence and a script that recomputes the table from that evidence. The archived NASA files remain useful without a new solver run.

## 6. Classroom route and assessment

**First 30 minutes:** identify the three bodies; derive the Cp conversion; plot both experimental signatures and the LAVA first zone. Check the coordinate shifts against the macros.

**Next 30 minutes:** inspect CAD and mesh, label boundaries and the sampling line, and explain why a pointed replacement nose can affect the signal. Compare a pressure peak metric with the waveform norm.

**Final 15 minutes:** distinguish equation verification, experimental CFD validation, neural test accuracy and design verification. Explain which evidence is needed for each claim.

Optional instructor extension: generate and run at least three meshes, record cost, and compare the same pressure signal on all levels. Keep this computational extension separate from the short student path.

Questions for a short engineering memo:

- Why does agreement with LAVA alone not count as experimental validation?
- Can two signals have equal pressure peaks but substantially different waveform error? Construct an example by shifting a signal, without modifying the scored reference comparison.
- Why does NASA CFD validation not experimentally validate a neural network trained only on the two-parameter design family?
- What happens to an error metric if Cp and Δp/p∞ are confused?
- Which additional equations and measurements are needed before claiming a ground-level PLdB improvement?

## Sources and reuse

- [NASA SEEB-ALR benchmark description](https://lbpw.larc.nasa.gov/sbpw1/test-cases/seeb-alr/).
- [NASA reference conditions and geometry instructions](https://lbpw-ftp.larc.nasa.gov/lbpw1/seeb-alr/README), retained unchanged as `NASA_README.txt`.
- [Original STEP geometry](https://lbpw-ftp.larc.nasa.gov/lbpw1/seeb-alr/geometry/SEEB-ALR-as-built.stp).
- [NASA workshop data archive](https://lbpw-ftp.larc.nasa.gov/lbpw1/workshop/seeb.tgz). The manifest identifies the exact retained archive members.
- Morgenstern et al., *Advanced Concept Studies for Supersonic Commercial Transports Entering Service in the 2018 to 2020 Period*, NASA CR-2013-217820, Section 4.2. [NASA record](https://ntrs.nasa.gov/citations/20130010174).
- [NASA archive public-domain statement](https://lbpw-ftp.larc.nasa.gov/lbpw1/). The original data remain unchanged; this explanatory guide is authored for FlowMLLab.

The motivating Beihang study is Zheng et al., *Aerospace Science and Technology* 178 (2026), 113218, [doi:10.1016/j.ast.2026.113218](https://doi.org/10.1016/j.ast.2026.113218). This guide and reduced Week 16 model do not reproduce its full aircraft, its training data or its neural weights. They also do not demonstrate TMS-10 performance.
