# Supersonic Shape Optimization with Verified CFD

FlowMLLab | Week 16 | Ehsan Roohi | MIE 690A

## 1. The engineering question

A supersonic body produces compression and expansion waves. Their strength and location depend on how the body displaces the fluid along its length. This module asks whether a compact learned model can propose a quieter-looking near-field signature while preserving volume and controlling pressure drag. A complete sonic-boom prediction requires another step: propagation through the atmosphere, followed by a perceptual loudness calculation. Our computed evidence stops at the off-body pressure signature.

The motivating study is Zheng et al., Aerospace Science and Technology 178 (2026), 113218, DOI 10.1016/j.ast.2026.113218. It combines CFD, atmospheric propagation and neural networks for a full aircraft configuration. Its sequential inverse networks illustrate an important difficulty: a far-field signature does not uniquely determine geometry, and errors can accumulate between learned stages. Our original teaching experiment uses a much smaller design space so that every proposed improvement can be recomputed with CFD.

Learning outcomes: formulate a constrained geometric design problem; distinguish axisymmetric from planar flow; inspect numerical credibility; train and compare surrogates; and accept a design only after its forward calculation has been checked.

## 2. Geometry with a meaningful constraint

Set the length L=1 m and let s=x/L. Define an unnormalized radius shape f(s)=sin(pi*s)*exp[a*(2*s-1)+b*cos(2*pi*s)] on 0<=s<=1. The first parameter redistributes radius fore and aft; the second changes the balance between the central and end portions. The radius is r(s)=c*f(s), where c is chosen from V=pi*L*c^2*integral(f^2 ds). This fixes V=pi*(0.06 m)^2*L/2 for every design.

The shape remains positive between its pointed endpoints. Volume normalization prevents an optimizer from obtaining a trivial pressure reduction merely by shrinking the body. The bounded design box is -0.45<=a<=0.45 and -0.25<=b<=0.25. A separate extrapolation set extends b to 0.30-0.40. Positive volume alone does not make a transport aircraft feasible: cabin arrangement, structural thickness, lift, trim and propulsion are not represented.

## 3. Axisymmetric Euler equations

For zero swirl, use U=(rho,rho*u,rho*v,rho*E), where u is axial and v radial velocity. In cylindrical coordinates, the conservation form is d(rU)/dt+d(rFx)/dx+d(rFr)/dr=(0,0,p,0). Axial flux is Fx=(rho*u,rho*u^2+p,rho*u*v,u*(rho*E+p)); radial flux is Fr=(rho*v,rho*u*v,rho*v^2+p,v*(rho*E+p)). Close the system with p=(gamma-1)*rho*(E-(u^2+v^2)/2), gamma=1.4.

The factor r and the radial pressure source represent the three-dimensional spreading of an axisymmetric flow. Omitting them gives a planar body of infinite span. SU2 is explicitly configured with AXISYMMETRIC=YES. The solution has pressure drag but no skin friction or boundary-layer separation. The fixed reference values p_inf=101325 Pa and T_inf=288.15 K define this nondimensional teaching experiment; they are not a supersonic cruise atmosphere.

## 4. Meshes, boundaries and pressure extraction

Gmsh creates three connected transfinite quadrilateral blocks upstream of the body, alongside it and downstream. The default domain is -0.5<=x/L<=3 and 0<=r/L<=1.5. The body follows a spline through independently generated radius samples. Mesh lines cluster toward the body and axis. Physical groups identify the body, axis and farfield boundaries.

The body has an inviscid slip condition; the exposed axis has the symmetry condition; outer boundaries use SU2's characteristic farfield treatment. The flow is initialized at M=1.8 and zero incidence. Roe fluxes, MUSCL reconstruction and the Venkatakrishnan limiter are used. The limiter is frozen after 300 iterations, and the convergence stop begins after 500. All solver settings are retained with each case.

Dimensional pressure is reconstructed from conservative variables. Cp=(p-p_inf)/(0.5*gamma*p_inf*M^2). Interpolation extracts signatures on three lines at r/L=0.25, 0.5 and 0.75. The design objective is the maximum Cp on the middle line. A lower peak on this line does not guarantee lower noise at another distance or on the ground.

## 5. Numerical credibility before learning

An independently integrated Taylor-Maccoll ODE solution supplies the pressure on a 7-degree cone at M=1.8. The benchmark uses a separate cone mesh and measures its surface pressure away from the apex and outlet. This comparison tests the axisymmetric pressure solution; it does not independently validate an entire sonic-boom prediction chain.

The shaped-body baseline is also checked on several meshes, and its outer domain is enlarged. The dataset uses approximately 45,000 cells per case. A refined calculation uses approximately 81,000 cells. We report actual differences in peak Cp, waveform norm and pressure drag. Residual convergence and spatial convergence answer different questions: one tests convergence of the discrete equations; the other tests sensitivity to discretization.

A retained data label requires positive finite pressure and density, at least five orders of density-residual reduction, and a relative range of pressure drag below 1e-4 over the last 100 iterations. These are numerical acceptance criteria for this exercise. Their satisfaction is not proof of exact physical accuracy.

## 6. Pressure drag and design objectives

Pressure force is integrated on the surface of revolution. The axial contribution is Dp=2*pi*integral[(p-p_inf)*r*dr/dx dx]. Subtracting p_inf removes cancellation of a large constant pressure term without changing the closed-body force. Cp and CDp are dimensionless; CDp=Dp/(q_inf*L^2). Using frontal area instead would change the numerical coefficient, so the reference area must accompany every comparison.

The primary design objective is min(max_x Cp(x,r/L=0.5)) subject to CDp<=1.02*CDp_baseline and the fixed-volume geometry. We use a penalty during numerical search and then check the constraint explicitly with CFD. A low objective from a learned model is a proposal. CFD establishes the achieved objective and whether the constraint was actually met.

## 7. Dataset, POD and learned models

The campaign contains 24 training geometries, six validation geometries, eight test geometries and six extrapolation geometries, with frozen Latin-hypercube seeds. All conditions except shape are fixed in the training campaign. The ordinary test set is unseen geometry inside the same parameter box. It is not unseen topology or a new geometry family.

Fit all scalers and the POD basis on training data only. The pressure signature is represented by its mean plus a linear combination of up to 12 principal modes. A model maps the two shape parameters to the POD coefficients and log pressure-drag coefficient. The logarithm ensures a positive reconstructed drag. Ridge regression, a two-hidden-layer tanh MLP and a Matern Gaussian process use the same training cases and output representation.

Architectures are fixed in advance. Select the design model by the sum of validation peak and drag mean relative errors. The test and extrapolation errors are reported after this decision. Report waveform relative L2, peak error and drag error separately because a visually good waveform can still have an important error at a shock peak. A single fixed-architecture training experiment is not a statistical ranking of architecture families. Refit variability is discussed in Section 13.

## 8. Optimization and independent recomputation

Differential evolution searches the bounded two-parameter box using the selected surrogate. This derivative-free choice is convenient because a maximum-over-samples objective can be nonsmooth. The search has a fixed seed and bounded evaluation budget. Additional candidates are selected only if the surrogate predicts comparable pressure peaks and compliance with the drag limit, while their shape parameters remain separated.

Each accepted proposal is meshed again and solved independently using SU2. Compare predicted and achieved pressure signatures and drag, and refine the best candidate. An optimizer can exploit small surrogate errors, so ordinary test error is not an adequate substitute for this final check. If the proposed optimum violates its constraint after CFD, report failure and enrich the training set before another iteration.

This forward-model optimization avoids requiring the network to identify a unique inverse geometry. It does not prove global uniqueness or provide a general solution to all inverse sonic-boom problems. Similar objective values also do not imply identical waveforms: inspect both.

## 9. Changes in conditions and the route to ground noise

Recompute the baseline and selected design at M=1.7 and M=1.9. These are direct CFD stress tests, not predictions by a network trained on Mach variation. Compare all extraction radii, noting that they are different physical observation locations. A single-point optimum may lose some benefit elsewhere.

To extend the module to actual ground sonic boom, the next research stage must couple a suitable off-body signal to a validated propagation solver, account for geometric spreading and the atmosphere, and compute an appropriate loudness metric. The augmented Burgers framework can include nonlinear steepening, thermoviscous absorption and molecular relaxation. Wind, humidity, temperature profiles and ray geometry matter. No PLdB value is inferred from the present peak-Cp reduction.

NASA's sonic-boom prediction workshops provide a useful independent validation route for near-field and propagation components. The public workshop material includes pressure signatures, grids, wind-tunnel data and propagated signals. Access and reuse conditions must be checked for each chosen dataset.

## 10. Research extensions and evidence claims

A research extension can ask how much CFD is needed to discover reliable low-boom designs. Compare active sampling with a fixed design-of-experiments budget, and include the cost of generating labels. Introduce additional shape families, a lift-producing configuration, realistic flight constraints and atmospheric scenarios only when the simpler components have been checked.

A multi-solution inverse method should be assessed through forward recomputation, feasible-shape diversity, constraint satisfaction and performance under held-out conditions. An MLP-to-operator-model substitution alone does not establish scientific novelty. The useful claim is an observed improvement in reliability, sample efficiency or physical understanding under matched computational budgets.

The present module offers a reproducible educational experiment: new Gmsh meshes, actual SU2 data, independent pressure benchmarking, transparent learned baselines and freshly recomputed design proposals. It does not reproduce TMS-10 or the full Beihang aircraft, and it does not demonstrate community-level noise reduction.

## 11. Reading the NASA geometry and pressure records

The NASA SEEB-ALR benchmark serves a different purpose from both the Taylor-Maccoll cone and the design body. The cone has a similarity solution for wall pressure. SEEB-ALR has an as-built CAD geometry and off-body wind-tunnel pressure records. Our design body has two parameters chosen for a tractable learning exercise. The network is not trained on the NASA geometry.

Start from NASA's original STEP file, not a traced silhouette. The file is imported in millimetres; the physical reference length is 17.667 inches, or 448.7418 mm. Divide all CAD coordinates by this length. Retain the finite nose and the downstream sting. Extending the sting to the computational outflow is a numerical domain choice that must be documented separately from the supplied model.

The comparison condition is Mach 1.6 at zero incidence. The pressure extraction line is 21.2 inches from the axis, so r/L = 21.2/17.667. NASA provides pressure excess normalized by freestream pressure. Our design plots use Cp, normalized by dynamic pressure. The conversion is dp/p_inf = (gamma M_inf^2/2) Cp. Comparing these two quantities without conversion gives the wrong amplitude even if the CFD pressure is correct.

The original NASA Tecplot macros specify longitudinal coordinate shifts. Preserve those shifts and show the comparison window. Do not shift a calculated shock onto a measured shock to reduce the error. A student should be able to regenerate every x coordinate from the distributed raw record and macro.

## 12. Comparing CFD with experiment

Plot three kinds of evidence distinctly: NASA wind-tunnel measurements, NASA's archived LAVA calculation, and our SU2 calculation. LAVA is a separate numerical solution, not an exact answer. Agreement between two solvers is useful, but agreement with an experiment addresses a different question. The NASA data files and their uncertainty columns are retained unchanged alongside a hash manifest.

For a reference vector y and interpolated prediction y_hat at the same coordinates, the waveform error is norm(y_hat-y)/norm(y). Also report the relative error in the positive pressure peak. A small peak error can coexist with a large waveform error when the shock location or expansion width is wrong. Plot the signed difference as well as the pressure curves, and declare the comparison window before interpreting the error.

Residual convergence concerns the algebraic solution on one mesh. Refine the mesh and compare the extracted waveforms to assess discretization sensitivity. An observed difference between the last two meshes is not a formal grid convergence index. Inspect the nose resolution, mesh alignment and shock region as well as the total cell count. A very small density residual cannot repair an incorrectly represented nose.

The executable NASA report and its JSON output contain the current measured differences. Follow the NASA_REFERENCE_GUIDE.md reproduction instructions to regenerate the geometry, solve the flow and recompute the comparison. These comparisons assess the CFD benchmark within its stated assumptions; they do not experimentally validate the learned two-parameter model. The finite nose cap remains two cells across levels, so this is not uniform refinement of the entire domain.

## 13. Auditing the learned model with frozen predictions

The recovered audit contains eight test geometries, archived model predictions, and CFD waveforms recomputed on a finer mesh. The report verifies the original dataset and recovered file hashes, geometry ordering, coordinate equality, and exact equality of the prediction arrays before computing errors. No retraining occurs in the audit report.

The aggregate relative waveform error is 7.15%, the mean relative positive-peak error is 3.45%, and the mean relative pressure-drag error is 1.69%. The largest individual waveform error is 15.80%. Thus the declared aggregate 10% thresholds pass, but a statement that every prediction is within 10% would be false. Against the original coarse CFD, the same frozen predictions have 5.05% aggregate waveform error; the coarse-to-finer CFD difference is 3.86%.

This audit represents a refit of the published architecture. The original trained checkpoint was not retained. The PCA automatic solver can use randomized SVD, so a fixed MLP seed alone does not fix the whole fitted pipeline. Use the archived predictions for reproducing these audit numbers; treat a newly fitted model as a new realization. The recovered compact arrays support the numerical comparisons, while their original raw solver logs are not part of the recovered audit evidence.

The Beihang journal study uses a full aircraft, 36 geometric descriptors, 3,480 samples, forward and inverse networks, and atmospheric propagation. This educational module does not implement that complete chain. Reproducing it faithfully requires the authors' geometry and data, as well as the propagation and training settings. A lower near-field pressure peak alone is not evidence of lower ground PLdB.

## 14. A portable checkpoint and its limits

A separate later refit is distributed as model_checkpoint.npz, containing the training scalers, full-SVD POD basis and all network weights. The NumPy-only FrozenSurrogate class evaluates these arrays without retraining. The audit verifies the 24 training identities, scalers, POD training subspace, layer dimensions and eight test coordinates. It does not infer a training chronology from the arrays alone.

On the retained finer-mesh cases this checkpoint has 6.39% aggregate waveform error, 1.93% mean peak error and 1.57% mean drag error; the worst waveform error is 14.63%. The comparison is retrospective because the finer references already existed. On six extrapolation geometries, waveform error rises to 34.16% and drag error to 22.14%. These failures matter when an optimizer searches beyond the training region.

Keep the historical prediction audit, this portable refit and newly fitted notebook models distinct. A model that reproduces a pressure waveform within a two-parameter family has not thereby learned general aircraft aerodynamics. The historical design improvement belongs to the model that proposed that design; every new optimized candidate still needs fresh CFD.

## 15. Reading convergence evidence

The NASA run driver saves the complete iteration history and checks positive density and pressure, a final log10 density residual at or below -9, a residual decrease of at least five decades, and a relative drag range below 1e-4 over the final 100 iterations. A solver exit code of zero alone does not satisfy these conditions. The limiter is frozen at iteration 2000 and acceptance checks start at 2500, so an earlier residual reduction is not mistaken for the final discrete solution.

The three meshes use the same Roe flux, entropy fix 0.05 and fixed CFL 5. Examine the drag as well as the density residual. Even if these algebraic checks pass, the experimental pressure and inter-mesh waveform criteria must be assessed separately. A numerical residual is not an error bar on the experimental comparison.

## 16. Reading the actual mesh

The figure shows the exported coarse computational grid, including the finite nose and the pressure-sampling line. Each inset uses equal physical coordinate scales. The outer blocks are sheared to provide resolution along the downstream propagation direction; this is a mesh choice, not an additional physical model. The radial axis ahead of the body uses symmetry, the body and sting use a slip wall, and the remaining outer boundaries use the prescribed farfield condition.

Gmsh also stores unused geometric control points. They must not be counted as fluid nodes. The mesh audit reads all exported element, node and boundary records, verifies connectivity and compares physical node counts before launching SU2. These file-integrity checks prevent truncated meshes from being accepted as numerical evidence.

## References

Zheng, Q., Liang, Y., Yang, Y. and Pan, C. (2026). Research on low-drag low-boom supersonic transport configuration using an MDO framework and deep learning methods. Aerospace Science and Technology 178, 113218. https://doi.org/10.1016/j.ast.2026.113218

SU2 8.5.0 source and configuration: https://github.com/su2code/SU2/tree/v8.5.0 . Governing equations: https://su2code.github.io/docs_v7/Theory/ . Supersonic wedge tutorial: https://su2code.github.io/tutorials/Inviscid_Wedge/ .

Gmsh reference manual: https://gmsh.info/doc/texinfo/gmsh.html . Geometry and meshing implementation in this module is independently authored.

Taylor, G. I. and Maccoll, J. W. (1933). The air pressure on a cone moving at high speeds. Proceedings of the Royal Society A, 139, 278-297. The cone ODE benchmark in this module is independently implemented.

AIAA/NASA Sonic Boom Prediction Workshop: https://lbpw.larc.nasa.gov/ . Workshop summary and data description: https://lbpw-ftp.larc.nasa.gov/lbpw1/presentations/21a_park-summary.pdf .

scikit-learn documentation: https://scikit-learn.org/stable/ . See PCA, MLPRegressor and GaussianProcessRegressor for the fitted models and conventions.

Earlier Chinese aerodynamic optimization article: https://doi.org/10.7638/kqdlxxb-2025.0081 . This is distinct from the 2026 neural-network article above.

NASA SEEB-ALR test case and source files: https://lbpw.larc.nasa.gov/sbpw1/test-cases/seeb-alr/ and https://lbpw-ftp.larc.nasa.gov/lbpw1/ . See cases/week16_lowboom/reference/source_manifest.json for exact source URLs and hashes.
