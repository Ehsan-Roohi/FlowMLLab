# Week 11 - Physics-audited shock and vortex identification

## From flow fields to defensible labels

Learning outcomes: distinguish rotation from shear and compression; explain why shock and vortex labels may overlap; fit and judge a small learned detector without pixel leakage; separate weak-reference agreement from physical validation.

Suggested 90-minute class: 15 minutes of diagnostic controls, 20 minutes of physical framing, 30 minutes in the notebook, 15 minutes of evidence critique, and 10 minutes for the exit assessment. Prerequisites are velocity gradients, supervised learning, cylinder wakes and compressible-flow branches from Weeks 1, 2, 7 and 8.

Research basis: Ehsan Roohi, Physics-audited joint neural segmentation of shocks and vortex cores: cross-solver transfer and controlled airfoil--cylinder studies (author-supplied research manuscript, 2026). The source describes joint detection in airfoil and cylinder fields. Do not describe the manuscript as a published JCP article or invent a journal DOI.

The executable lab starts with an original CPU teaching analog. Its final section reads real airfoil and cylinder research fields through retained figures and a provenance ledger: six fresh forward passes with a frozen research checkpoint. No new CFD or research-model training is claimed. The analytic warm-up and research evidence remain explicitly separate.

## Vorticity is necessary context, not a core label

Let A be the in-plane velocity-gradient tensor with rows (u_x,u_y) and (v_x,v_y). Its symmetric part describes strain and its antisymmetric part describes local rotation. Planar vorticity is omega_z = v_x - u_y; dilatation is theta = u_x + v_y. A large derivative can also result from wall shear, a grid artifact or amplified sampling noise.

For the trace-free in-plane tensor, Q_d = -(u_x-v_y)^2/4 - u_y*v_x. The imaginary part of its complex eigenvalues gives swirling strength: lambda_ci = sqrt(max(Q_d,0)). In two dimensions these are algebraically linked, not two independent pieces of evidence. This is not the full three-dimensional vortex-classification problem.

Work three controls by hand: solid rotation u=-y, v=x gives omega=2 and lambda_ci=1; simple shear u=y, v=0 gives omega=-1 but lambda_ci=0; isotropic compression u=-x, v=-y gives theta=-2 and no swirling strength. The notebook verifies these identities numerically.

These gradients are unchanged by constant velocity translations on a fixed grid. That does not make the detector objective under arbitrary time-dependent rotating frames.

## A shock is more than a bright gradient

Density or pressure gradients indicate edges, not uniquely shocks. Contact layers, solid boundaries, expansion regions and numerical ringing can also produce strong signals. Compression, pressure rise and front geometry are complementary evidence. A velocity-gradient sensor alone cannot verify a gas-dynamic jump.

Across a locally planar stationary shock, audit the normal mass, momentum and total-energy fluxes: rho*u_n; rho*u_n^2+p; and rho*u_n*(h+|u|^2/2). A moving front requires velocity relative to that front. Use upstream and downstream states outside the numerically thick transition; oblique fronts require a local normal and consistent orientation.

The teaching field uses a tanh compression layer superposed with rotation and shear. Its labels are known construction regions. The superposition is not a Navier-Stokes or Euler solution and is not required to satisfy these jump conditions. Call it a shock-like diagnostic control, not a verified physical shock.

Discussion: why would a sensor appear accurate if its own thresholded gradient map were also used as ground truth? Design a separate jump audit and a solid-wall negative control before proposing an improved classifier.

## Overlapping structures need independent decisions

A vortex can intersect a shock. An exclusive softmax over background, shock and vortex prohibits this overlap by construction. Independent sigmoid foreground outputs allow two labels at one location; a background mask can then be derived from their complement. Ambiguous regions and exact solid geometry require explicit treatment rather than silent relabelling.

The research framework uses separate learned, physics-only and hybrid products, with task-restricted adaptation. If the shared encoder and the complete vortex pathway are frozen, changing only the shock branch preserves vortex outputs. Freezing the vortex decoder alone is insufficient when its input encoder continues to change. Output equality is a testable claim about computational dependencies, not a generic property of multitask learning.

For conflicting shared gradients g_s and g_v, projection methods can remove a component when their dot product is negative. Such methods can help one task and hurt another; retain both signed changes. Physical diagnostic channels do not by themselves constitute a PDE-residual loss or a PINN.

Our local MLP has eight primitive/derivative features and two independent output decisions. It teaches overlap and evaluation, but has no spatial encoder, dual convolutional decoder or research adaptation mechanism.

## Freeze the experiment before opening the test

The notebook generates eight training cases, two validation cases and two held-out cases from different construction seeds. Entire fields define groups. Subsampling training pixels is allowed; randomly splitting adjacent pixels from one field between training and testing is not a test of case generalization. Research trajectories, grids and solver families need analogous grouping and duplicate-input checks.

Fit the feature scaler on training pixels only. The 32-by-24 tanh MLP uses a fixed optimizer budget; it is compared with compression and swirling-strength thresholds. Select each method's two operating thresholds on validation mean case-wise Dice, then freeze them. No test-dependent quantiles, loss-weight changes or mask cleanup are permitted.

Report convergence warnings rather than hiding them. A low training loss is not physical fidelity. A physical baseline can win; that is a valid result. The notebook's two test cases provide a classroom comparison, not a statistical generalization claim across flow families.

Assignment: remove rotational inputs, rebuild the model with a fresh predeclared split, and report both task metrics and changes in fragmentation. Never reuse the already-inspected classroom test as a new blind benchmark.

## Read the colored outputs critically

[RESEARCH_FIGURE]

Real airfoil field from Roohi's ShockVortexML study: native-density schlieren, learned shock mask and learned vortex mask. The fresh forward pass uses frozen thresholds 0.97 and 0.85. Geometry-crossing display-gradient stencils are excluded. These ML-only masks are not human labels or a hybrid. Thin missed branches and merged cores remain visible.

Dice = 2*|prediction intersect reference|/(|prediction|+|reference|). Empty-versus-empty is assigned one in this lab; state the convention. A front shifted by one pixel can have poor Dice while remaining close physically. Conversely, a thick predicted envelope can gain overlap without locating the centerline accurately.

The gallery retains first/middle/last eligible times for both airfoil and cylinder, excluding all human-review frames. These are previously inspected development-test trajectories; do not infer independent accuracy from appearance or mask counts. The notebook retains the synthetic Dice exercise separately. All six full-size comparisons and checkpoint provenance are in results/week11_research/.

## Reference provenance and transfer gates

The manuscript uses physics-derived proposals as weak reference masks, not an independent exact oracle. If a hybrid uses those same proposals as support, its score measures reference-conditioned agreement. Learned-only and hybrid results must therefore remain separate. Development-inspected test fields are not untouched external tests.

Before transferring to research CFD, identify the source solver and case, native coordinates, solid mask, derivative stencil, nondimensional scales, reference construction, training-family membership and checkpoint. Preserve full-field arrays as well as visualizations. Walls and missing values must not become artificial vortices through interpolation or differentiation across the solid.

Add new controls for grid changes, noise and solver changes. Recompute derivatives from perturbed primitive velocities; perturbing only the already-derived diagnostic channels tests a different problem. A shock-overlap region must not automatically veto a real core. Thresholds expressed in raster pixels change physical size with resolution.

The research repository is https://github.com/Ehsan-Roohi/ShockVortexML. This lecture paraphrases the author-supplied manuscript and teaches its audit principles without copying a training pipeline or redistributing unpublished CFD archives.

## Reconstruct the field before identifying structures

The real-field companion asks a different question from the manufactured classification exercise. Suppose a sensor or storage pipeline supplies a spatially coarsened velocity field. Can a learned reconstruction recover the velocity gradients needed to identify a vortex? Our independent adaptation of the supplied Ricardo-course super-resolution exercise compares three paths: interpolation followed by a physical diagnostic; U-Net velocity reconstruction followed by that same diagnostic; and direct U-Net mask prediction. U-Net is the shared architecture, not a separate rival to a so-called Ricardo algorithm.

A U-Net contracts the spatial representation to learn broader context, then expands it while concatenating encoder features through skip connections. The original architecture is described by Ronneberger, Fischer and Brox (2015), https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/. The course archive illustrates single-component turbulent-field reconstruction. Here we independently implement a smaller two-level network with two velocity inputs and either two reconstructed velocity outputs or one mask logit. No archive source code, weights or figures are redistributed, and this is not an exact reproduction of its four-level network.

For reconstruction, minimize the mean squared difference between predicted and reference velocities after per-component training-only normalization. For direct segmentation, minimize binary cross entropy against the velocity-derived mask. These losses answer different questions. A velocity error with wavelength h produces a derivative error proportional to its amplitude divided by h; low velocity MSE therefore need not imply accurate swirling strength. Thresholding a diagnostic adds another discontinuity: a small change near the cutoff can switch many pixels.

The real inputs are previously generated FlowMLLab LBM cylinder wakes, not newly simulated data and not the hypersonic cylinder article. The 32-by-78 downstream ROI is area-reduced to 8-by-19 and bilinearly restored before either neural path sees it. Both components are available. The physical coordinates retain their true aspect ratio. The native ROI is coarse and subsampled; it is not grid-independent DNS, and its boundary is not a physical wall.

## A matched protocol and a limited reference

Entire Re90 and Re110 cases provide training data; Re100 provides validation; previously inspected Re105 is the retained test. Every normalization statistic and the physical diagnostic threshold come from training only. The threshold is the 90th percentile of interior training swirling strength. Compute the reference and reconstructed diagnostics using the same ROI derivative operator, not the separately archived pre-subsampling vorticity. Omit a two-cell border from the metrics. Pure rotation and simple-shear controls verify this diagnostic independently of training.

Both networks use Adam at 0.001, 60 epochs, batch size 32 and three fixed seeds. Select each checkpoint by minimum validation loss, and the direct mask probability threshold by validation Dice on a predeclared grid. Report all seeds, not the best retained-test seed. The first seed and midpoint frame are fixed for illustration. Parameter counts differ slightly because the output heads differ; this is matched input and optimization budget, not exact parameter equality.

Read the companion notebook's table in two parts. Velocity and vorticity relative L2 errors assess reconstructed fields; divergence RMS is an additional consistency diagnostic, not an enforced constraint. Frame-macro Dice and IoU assess agreement with the thresholded native swirling-strength reference. Direct segmentation cannot have a velocity error because it produces no velocity. Empty-versus-empty masks score one. Temporal frames are correlated, so seed dispersion is not a confidence interval over independent flows.

These wakes contain vortices but no shocks. The comparison cannot establish shock detection or identify every dynamically important coherent structure. Agreement with a derivative-based weak reference is not human-validated physical accuracy. Additional higher-resolution, independent flows and threshold-sensitivity experiments are required before broader research claims. The companion protocol records the source archive hash and explains the adaptation; the archive folder name alone does not establish code authorship or redistribution permission.

Run notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb to audit the retained comparison and view loss histories. Its optional retraining cell runs all six fits into a fresh scratch directory, never into retained results. The synthetic warm-up and the separate research-image gallery remain available and retain their original evidence limitations.

## Read the retained reconstruction comparison

[RECONSTRUCTION_TABLE]

The neural rows average three independently initialized fits on the same retained case; they are not confidence intervals across flows. The full per-seed table and color-field comparisons are in the companion notebook and results/week11_reconstruction/. All seven metric rows were recomputed from the saved models. The direct mask model scores higher against this particular weak reference, while the reconstruction model provides velocities for downstream diagnostics. Neither observation demonstrates shock detection or cross-solver generalization.

[RECONSTRUCTION_LOSS]

Solid curves show training loss and dashed curves validation loss. The two panels have different objectives and cannot be ranked by comparing their absolute heights. Several best checkpoints occur near the end of the fixed budget: this is a completed controlled run, not a proof of optimization convergence. Extending the budget is a new predeclared experiment, not an opportunity to select a better retained-test result retrospectively.

## Exit assessment and reproducibility

Submit one page with: the split IDs, frozen thresholds, both methods' case-wise Dice, at least one failure, and a labeled comparison figure. Explain why vorticity does not imply a vortex, why independent sigmoid labels are appropriate, and why physical-label agreement is not independent physical validation.

Propose a front-location metric and a component-matching rule before looking at a new research case. State how the rules handle geometry, ambiguity, missing structures and empty references. Describe the stronger evidence needed for cross-solver claims and for temporal vortex tracking.

Run notebooks/week11/W11_Shock_Vortex_Identification.ipynb from a complete FlowMLLab checkout or use its Colab launcher. It runs on CPU and keeps experiment outputs in memory. The authoring builder can regenerate the retained teaching notebook and figure; a student's Run All does not overwrite results/.

Source attribution: Ehsan Roohi, Physics-audited joint neural segmentation of shocks and vortex cores: cross-solver transfer and controlled airfoil--cylinder studies, author-supplied 2026 manuscript; ShockVortexML repository above. Original FlowMLLab lecture wording, synthetic fields and code were AI-assisted and checked. These additions are not claims of original research-data generation or full-paper reproduction.
