# Week 12 - Geometry-aware reconstruction of noisy DSMC moments

## Reconstruction, not cosmetic smoothing

Learning outcomes: aggregate additive particle moments correctly; distinguish sampling error from genuine flow structure; explain prior-plus-observation reconstruction; preserve measured zero-frequency content; use a support warning without overstating its guarantees.

Suggested 90-minute class: 20 minutes of moment algebra, 15 minutes of the inverse problem, 30 minutes of the executable experiment, 15 minutes of physical audits and 10 minutes for assessment. Prerequisites are DSMC sampling, supervised splits, POD and the Week 10 moment hierarchy. Week 11 motivates why noisy derivatives can destroy feature identification.

Research source: Ehsan Roohi, Geometry-native machine learning reconstruction of DSMC moment fields with support monitoring, arXiv:2609.01637 (2026), https://doi.org/10.48550/arXiv.2609.01637. The author confirms submission to Journal of Computational Physics. This lecture does not call it a published JCP article.

The paper's cavity prior is a trained MambaIR restoration model; its cylinder estimator uses geometry-adapted, coupled heat-flux reconstruction. The notebook starts with a scalar spectral analog, then audits real author-supplied JCP2 cavity results. Archive method names are retained: this is not a claim that each archived stage equals the final paper estimator. No research model is retrained.

## From molecular velocities to the moment hierarchy

DSMC estimates a distribution in two spatial dimensions with three molecular-velocity components. Retained fields include number density, two bulk velocities, translational temperature, three pressure-tensor components and two heat-flux components. Heat flux is a central third-order moment, not simply a smoothed temperature gradient.

For simulator weights w and molecular velocity xi, accumulate C0=sum(w), C_i=sum(w*xi_i), C_ij=sum(w*xi_i*xi_j), E2=sum(w*|xi|^2), and F_i=sum(w*|xi|^2*xi_i). Sum these additive quantities over the declared blocks first. Then compute u_i=C_i/C0 and J_i=F_i-u_i*E2-2*sum_j(u_j*C_ji)+2*C0*u_i*|u|^2.

The physical heat flux requires the molecular-mass, cell-volume and sampling-time normalization. Separately centralising each block inserts different velocities into a nonlinear expression; averaging those central quantities is not generally equivalent to pooling raw accumulators first. The notebook demonstrates the discrepancy and verifies the pooled expression against direct centralisation of all particles.

Large raw terms may nearly cancel in J_i. A small bias in velocity or energy flux can therefore become a large relative heat-flux error. Visually smooth fields can still have the wrong amplitude or sign.

## Observations are noisy; references are too

Raw(B) denotes an estimate formed from B declared additive sampling blocks. For independent equal-variance blocks, averaging reduces variance by B. Correlation, unequal exposure, sampling resets and nonstationarity invalidate a naive block-count argument. Do not treat neighboring cells or overlapping cumulative outputs as independent replicates.

A high-budget independent reference is not the exact Boltzmann solution. Compare every method against the same reference and retain observation/reference independence. Raw(3) may be nested inside Raw(10); that should be disclosed, while neither should overlap its evaluation reference. Keep development priors, gain selection and support thresholds separate from held-out evaluation pairs.

For native cell areas A, NRMSE=sqrt(sum(A*(estimate-reference)^2)/sum(A*reference^2)). Declare masks and denominators. A near-zero reference norm makes relative error unstable. Report seed-level results, not only a favorable average. The ratio of mean errors differs from the mean of individual ratios.

The notebook uses explicitly synthetic independent Gaussian blocks and an independent 80-block-equivalent noisy reference. Analytic truth remains available only to diagnose this manufactured experiment. These fields cannot establish the noise covariance or convergence rate of a real DSMC solver.

## A prior must listen to the current observation

Let T_g be a geometry-adapted linear transform, z_obs=T_g*m_obs and z_prior the prior coefficients. Reconstruct z_hat=z_prior+H_g*(z_obs-z_prior). The operator's singular values are bounded between zero and one so it does not amplify the coefficient residual. Zero transfer returns the prior; identity transfer returns the observation before additional constraints.

For a scalar cavity DCT, a Wiener-type gain is G=signal_power/(signal_power+noise_power/B). Set the zero-frequency gain to one. A trusted historical shape can suppress sampling noise, but the current observation anchors changes in amplitude and mean. Replacing that observation with the prior can erase real departures from development data.

The paper uses a learned MambaIR prior for the cavity and a separately fitted cylinder transfer. Our notebook estimates a mean prior and scalar spectral gains from development draws. That is an interpretable training exercise, not a MambaIR implementation. Gains and Gaussian-filter widths are frozen before held-out evaluation.

For equal-area cells, restore the observation's arithmetic mean after reconstruction. For native unequal-area cells, restore its area-weighted Cartesian mean. This constraint preserves a measurement, not an exact truth; it does not guarantee positivity, full moment consistency or conservation by itself.

## Geometry and identifiable information

Around a cylinder, define the outward normal n=(cos(theta),sin(theta)) and tangent t=(-sin(theta),cos(theta)). Rotate q_n=q_x*cos(theta)+q_y*sin(theta) and q_t=-q_x*sin(theta)+q_y*cos(theta). The inverse is q_x=q_n*cos(theta)-q_t*sin(theta), q_y=q_n*sin(theta)+q_t*cos(theta). Test the round trip numerically before filtering.

The paper transforms fluid-only normal/tangential fields in cylinder-centred coordinates, learns bounded two-component spectral transfer, returns to native Cartesian fields and restores the measured area-weighted means. Do not interpolate through the solid or confuse a smooth visualization grid with solver resolution. The Cartesian classroom grid does not implement this full cylinder treatment.

Energy conservation constrains divergence of heat flux, not both components uniquely. For any smooth psi, delta_q=(d(psi)/dy,-d(psi)/dx) has zero divergence. Adding it changes heat flux without changing its energy-residual contribution. The notebook verifies this derivative identity.

Thus a small energy residual cannot uniquely identify heat flux. Use current observations, independent references, geometry and physical diagnostics together. Do not enforce Fourier's law as ground truth in strongly non-equilibrium rarefied flow.

## Compare fields and failure modes

[RESEARCH_FIGURE]

Real DSMC cavity qy, Kn=0.085 and lid speed 350 m/s: independent reference, three-block observation and archived observation-conditioned estimator. The first archive seed is illustrated, not the best seed. A common color range includes all displayed values; no interpolation hides noise. Axes are normalized array positions; values use archive units.

Across eight seeds, mean qy relative L2 error is 17.61% for Raw(3), 9.80% for Raw(10), 12.63% for prior-only, and 4.34% for the conditioned estimator. Corresponding qx errors are 11.87%, 6.57%, 4.90%, and 3.15%. These are direct errors against a finite-budget reference, not noise-deconvolved scores. Raw(3) and Raw(10) are disjoint in this archive.

All 64 archived comparison scores were independently recomputed within 2e-6; 16 fixed-width Gaussian comparisons were added without reference-based tuning. All-seed scores, profiles and error maps are retained in results/week12_research/. This is a historical prospective archive now being inspected, not a newly blind trial. Noise reduction alone does not establish speedup.

## Support monitoring and abstention

A new field may lie outside the range in which the prior and residual transfer were qualified. The paper monitors nine components in three geometry zones using observation residual power relative to development noise. Its gain-envelope score is calibrated on development units; the stated rule is a heuristic, not a universal finite-sample coverage guarantee.

The notebook deliberately implements a simpler one-field residual/noise score. The maximum validation score is frozen before a shifted observation is examined. No test reference or test error enters the decision. Label this an illustrative heuristic, not the paper's 27-component-zone monitor or a calibrated probability of failure.

If the score exceeds the threshold, abstain from presenting the reconstructed field as supported. Keep the raw estimate, collect additional independent samples, or build new in-condition development evidence. Do not silently retune the prior after looking at the test reference. A within-envelope score alone does not prove accuracy.

Discussion: a mean-preserving estimator can carry a noisy or out-of-condition mean unchanged. What distinct roles do data consistency, support monitoring and independent physical validation play? Why is an attractive contour insufficient evidence for any of them?

## Assessment and research handoff

Submit the pooled-versus-separate central-moment check, the frozen experiment protocol, all six synthetic test-case metrics, a contour comparison, and the support decision for a shifted observation. Include a case where the simplest baseline is competitive or better. Explain which information the prior supplies and which information the observation preserves.

Extensions: introduce temporal correlation and measure the failure of the 1/B noise law; add a narrow layer and quantify smoothing bias; implement normal/tangential rotation and area-weighted mean restoration; design independent references for a real DSMC run. These are extensions, not already-executed article reproductions.

Run notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb using CPU dependencies in a current complete checkout or use its main-branch Colab launcher. A normal run leaves retained evidence unchanged. Synthetic controls, a fresh real-data Noise2Noise fit and historical research predictions occupy separate sections.

Reading: Ehsan Roohi, Geometry-native machine learning reconstruction of DSMC moment fields with support monitoring, arXiv:2609.01637, https://doi.org/10.48550/arXiv.2609.01637. Lecture wording and classroom code are original, AI-assisted FlowMLLab additions grounded in the author's manuscript. No original research data, solver or checkpoint is fabricated or silently substituted.

## Noise2Noise: learn from noisy targets

Suppose Y=f+e and Y'=f+e' are two noisy measurements of the same underlying field. For squared loss, training against Y' has the same population optimum as training against f if the target perturbation has zero conditional mean given the input and underlying field: E[e' | Y,f]=0. The cross term then vanishes and the target-noise energy is independent of the predictor. Independence and zero conditional mean are sufficient, not something guaranteed by two different filenames.

The idea comes from Lehtinen et al., Noise2Noise: Learning Image Restoration without Clean Data, ICML 2018, https://arxiv.org/abs/1803.04189. The FlowMLLab patch MLP is an original teaching implementation, not copied NVlabs code or the article's architecture. Roohi's existing real DSMC observations supply the data; no Gaussian noise is added and no research-network predictions enter training.

DSMC heat flux is a nonlinear central moment. Finite-sample bias, correlated blocks, transient evolution and mismatched underlying fields can invalidate the simple argument. An unbiased observation of a coarse discrete estimator would also not prove agreement with the continuum kinetic solution. This exercise tests practical denoising against a finite independent reference, not those stronger claims.

Ask before running: why does training a noisy input against itself reward an identity map? Why does averaging independent targets reduce training-target noise without making that target exact? What fails when the input seed contributes to its own target?

## Real-data training protocol and fair baselines

Use the JCP2 cavity at Kn=0.085 and lid speed 350 m/s. Eight independent-seed identifiers are available on the same 100x100 grid. The source archive documents separate observation/reference seeds and disjoint Raw(3)/Raw(10); the course does not independently reconstruct RNG streams or original block ancestry. Coordinates and field values retain normalized array positions and archive units.

Freeze training seeds 26082101-104, validation 105-106 and evaluation 107-108. For each training input, form a noisy target from the other three training seeds. Its three input blocks do not appear in the nine-block target according to the archive. The four pairs share training data; spatial patches and pair reuse do not create independent experimental replicates.

A 5x5 patch of qx and qy supplies 50 features to a fixed 64x32 tanh MLP. Training-only scales, initialization 12, 3,000 patch locations per seed and a 160-epoch budget are fixed. No spatial coordinates or random-pixel early-stopping split are used. Reflection padding is numerical, not a physical boundary condition; edge-band errors expose its limitations.

Compare Raw(3), a Gaussian width chosen by noisy validation pairs, a training-fitted DCT filter, the 12-development-block training mean and the MLP. Report a separate mean-restored MLP, never select it using test errors. Raw(10) is an additional larger-budget comparator. Open the evaluation reference only after fitting and selection. These previously inspected archive seeds are held out from this fit, not a newly blind scientific test.

## Fresh fit: improvement and a neural failure

[NOISE2NOISE_FIGURE]

The fresh patch MLP reduces mean qy reference error from 17.60% for Raw(3) to 6.31% across two evaluation seeds. Gaussian filtering gives 8.20%, the spectral filter 7.98%, and the training mean 9.05%. These are course-model results, distinct from the historical research estimator's 4.34% across eight seeds. Do not merge their protocols or claim reproduction of the research network.

For qx the MLP gives 10.37%, versus 11.87% raw, 8.76% Gaussian, 7.17% spectral and 5.98% training mean. The simple baseline wins. The optimizer reached 160 epochs without satisfying its convergence criterion; its warning remains in the notebook and retained manifest. This is a budget-limited demonstration, not a fully optimized network.

Every Run All fits the model from raw observations and recomputes the scores. No stored predictions are used to replace the fresh teaching fit. Data and retained results remain read-only. Increasing training effort is a future development experiment: declare its selection rule before looking at new evaluation results and retain this initial outcome.

## Inspect profiles, then state what was learned

[NOISE2NOISE_AUDIT]

Both components and both evaluation seeds are retained. Inspect column-49 profiles, direct relative L2 error, gradient error and the outer-10%-array edge-band error. Gradients of the finite reference contain noise too; they are not exact derivative truth. The common contour scale includes all displayed values without interpolation or clipping.

Submit all 28 component/method/seed scores and explain why the training mean is so competitive for a fixed underlying field. Compare mean-restored and unconstrained MLP outputs: measured-mean preservation need not reduce reference error. Two seeds cannot justify statistical significance, calibrated uncertainty or a simulation-speedup claim. A stronger next experiment needs independently qualified realizations and entire-condition holdouts, not more random patches from the same cavity.
