# Physics-Informed Neural Networks for Rectangular Cavity Flow

## Representation, optimization, verification, and a reproducible research design

FlowMLLab | Week 13 | Ehsan Roohi | Graduate research lecture

This capstone lecture asks a narrower and harder question than “can a neural network draw a cavity vortex?” The question is whether a stated neural representation, trained with a stated numerical budget, approximates the same boundary-value problem as a matched CFD reference—and whether the conclusion survives changes in Reynolds number, cavity depth, initialization, and evaluation support. The lid-driven cavity is useful precisely because a plausible streamline plot is easy to obtain while trustworthy velocity, pressure, wall and corner behavior are not.

Week 13 builds on the verified Re=100 square-cavity pilot in Week 4.2 and on Christopher J. McDevitt’s permitted DeepPlasma implementation. It adds rectangular coordinates, a staged Adam-to-SSBroyden2 optimizer, restartable A100 jobs, independent full-domain residuals, a top-corner audit, and a predeclared multi-case protocol. The retained four-case experiment is a feasibility pilot. It is not yet a definitive comparison of PINN formulations and is not presented as a reproduction of every result in McDevitt, Fowler and Roy.

### 1. The cavity is a family of problems, not one picture

Let the cavity width be L and its physical depth be H. Define D=H/L and use normalized computational coordinates x in [0,1] and eta in [0,1], with y=D eta. The lid speed U and width L define Re=UL/nu. This convention deliberately keeps Re tied to the width while D changes the geometry; choosing H instead would define a different parameterization and must not be mixed into the same table.

@equation mapping

At D=1 the domain is square. For D>1 it is deep, and additional vertically stacked recirculation cells may appear as Re and D increase. Cheng and Hung’s systematic CFD study shows that vortex count and the depth at which the lower flow becomes nearly symmetric depend jointly on Re and D. Those published topology transitions are valuable qualitative checks, but a digitized streamline figure is not a pointwise reference field.

@figure geometry

The steady two-dimensional incompressible equations in physical coordinates are

@equation ns

The words “steady, two-dimensional” are part of the mathematical problem, not innocent plotting choices. At sufficiently demanding conditions, a steady solution may be unstable or nonunique. A converged residual can then identify a mathematical branch that is not the physical long-time state. A research claim must distinguish optimization failure, approximation error, solution multiplicity and model inadequacy.

### 2. Primitive variables and streamfunction are competing representations

A primitive-variable PINN represents u, v and p directly. Momentum and continuity appear as separate residuals, so finite loss weights decide how strongly the optimizer balances them. Boundary conditions may be imposed softly or by an analytic output transform. A first-order-system least-squares formulation can add auxiliary outputs for velocity derivatives and thereby reduce the highest derivative order evaluated through the base network.

A streamfunction-pressure PINN instead represents psi and p, then defines u=psi_y and v=-psi_x. For a smooth psi, incompressibility follows from equality of mixed derivatives. This removes continuity as an optimization objective but raises the differentiation order: viscous momentum terms require third spatial derivatives of psi.

@equation streamfunction

Neither representation is “the Navier–Stokes method” by itself; both approximate Navier–Stokes solutions. Their difference is the admissible function class, constraint structure, derivative graph and optimization landscape. A fair comparison holds the physical boundary-value problem fixed and reports capacity, precision, residual evaluations and wall time, not merely outer iterations.

@table representations

### 3. What the 2026 Journal of Computational Science paper actually tested

Służalec and coauthors studied a square cavity at Re=1, 10, 100 and 1000. Their PINN has seven outputs: u, v, p and four auxiliary first-derivative fields. It uses a FOSLS-inspired residual and hard output constraints; their CRVPINN variant replaces the ordinary residual norm with a discrete robust variational construction. They compare against a 200 by 200 Taylor–Hood FEM calculation and report unacceptable errors from Re=100 upward.

That result is an important challenge, not a universal impossibility theorem for all PINNs. Their paper does not evaluate McDevitt’s streamfunction-pressure representation, the SSBroyden2 continuation used here, or rectangular aspect ratios. Conversely, a successful run of the present method would not show that their implementation was “wrong.” It would show that formulation, optimizer, sampling or regularization can change the outcome; a factorial experiment is needed to isolate which factor matters.

@table literature

The sharp research question is therefore: under a matched physical problem and matched computational budget, does exact incompressibility through a hard streamfunction representation improve the probability of obtaining the correct cavity branch relative to a primitive/FOSLS representation? The aspect-ratio extension then asks whether any advantage persists as vortex topology becomes richer.

### 4. Nondimensionalization on a rectangular computational domain

Automatic differentiation acts on x and eta, but the equations are written in x and y. Because y=D eta, each y derivative contributes a factor 1/D. Therefore u=psi_eta/D, v=-psi_x, the convective y derivatives carry 1/D, the y diffusion terms carry 1/D squared, and p_y is p_eta/D.

@equation rectangular

Omitting even one of these metric factors silently solves a different PDE. The error can be invisible at D=1 and appear only when the aspect-ratio sweep begins. This is why D=1 must be used as a reduction test and why manufactured or analytic derivative checks precede expensive cluster work.

The coordinate transform does not itself change the network input range: x and eta remain in the unit square for every geometry. That improves numerical scaling, but it also means a fixed network has to represent increasingly long physical structures through the D-dependent output transform and residual. Capacity equivalence across D is therefore a testable design choice rather than a guarantee of equal difficulty.

### 5. Hard velocity boundaries by streamfunction lifting

Write psi_theta=psi_lid+B squared times q_theta, where B=16x(1-x)eta(1-eta). Squaring B makes both the correction and its first normal derivative vanish on every wall. The known lifting supplies all boundary velocity. With the physical depth factor included, the rectangular lifting is

@equation lifting

On the top wall, u=g(x) and v=0. The other walls have u=v=0. The smooth g(x) approaches one away from the top corners and returns continuously to zero at the corners. This regularization avoids prescribing two incompatible velocities at one point, but it is part of the physical specification: changing its width changes the problem.

@figure lid

A hard boundary transform can satisfy every sampled wall point before training while the interior momentum balance is poor. Exact incompressibility can coexist with a wrong vortex. Structural constraints remove failure modes; they do not certify a solution.

### 6. Why Adam and SSBroyden2 are staged

Adam rescales stochastic first-order updates using running moment estimates. It is inexpensive per update and often moves a random network into a useful basin. It does not exploit deterministic curvature and can stall at a residual level that is visually convincing but quantitatively inadequate.

SSBroyden2 is a full-memory quasi-Newton method with line search. It builds an inverse-Hessian approximation and can reduce a deterministic residual much more sharply, but one outer step may call the closure several times and its dense state is expensive. Comparing “1000 Adam iterations” with “1000 SSBroyden2 iterations” is therefore not an optimizer benchmark. In this lecture they form one declared continuation path.

@equation objective

The collocation set remains fixed during the quasi-Newton stage so each line search evaluates the same objective. Random resampling inside a line search would mix optimization and Monte Carlo noise. If adaptive sampling is later introduced, it should occur between optimizer stages with its own ablation.

@figure losses

The training curve contains four distinct signals: masked collocation momentum residual, held-out full-domain momentum residual, held-out continuity, and top-corner momentum residual. The corner curve is not cosmetic. A global residual can improve while a small high-error region worsens, especially when the training objective deliberately tapers the upper corners.

### 7. Restartability is part of the numerical method

On Unity, gpu-preempt jobs can stop before a long optimization finishes. A restart that reloads only network weights is not an exact continuation. Adam carries moment estimates; SSBroyden2 carries its inverse-Hessian state and current parameter vector. Fixed collocation points and random-number states also affect the subsequent trajectory.

The Week 13 runner checkpoints model parameters, optimizer state, collocation points, configuration and CPU/GPU random states through an atomic replace. SLURM sends SIGUSR1 before the two-hour limit; the job saves, exits with a dedicated code, requeues and resumes. The audit records the SLURM job, A100 model, float64 precision, upstream commit and source digest.

@table restart

This engineering detail is scientifically material. A figure assembled from an undocumented restart cannot be assumed equivalent to one uninterrupted deterministic run.

### 8. Evidence must be independent of the optimized objective

The first evidence layer is algebraic: D=1 reduces to the square equations, the hard transform satisfies the velocity boundaries for arbitrary q_theta, and divergence is zero to floating-point precision. The second layer is numerical: evaluate unmasked momentum residuals at independently seeded points, including a separately reported top-corner subset.

The third layer is a matched field comparison. For square cases where the retained CFD archive contains the same Re, the protocol freezes three gates before inspecting the new model: 10 percent relative L2 for u on the vertical centerline, 15 percent for v on the horizontal centerline, and 15 percent for the vector velocity at 8192 seeded interior points.

@equation errors

For deep cases no matching raw field is currently retained. They may be residual-audited and checked for wall satisfaction and plausible topology, but they are not called field-validated. Published Cheng–Hung vortex counts constrain interpretation without supplying fabricated pointwise data.

@table gates

### 9. Reading the four-case feasibility pilot

The initial matrix contains Re=100 and 400 at D=1 and 2. Every case uses the same three-layer, 50-unit tanh network, 16384 collocation points, float64, 1000 Adam steps and 3000 SSBroyden2 steps. The quasi-Newton target was extended from 1000 to 3000 only after the initial residual audit, using exact optimizer-state continuation; that extension is therefore a disclosed post-pilot decision, not a preregistered paper protocol. All jobs start from the same declared seed. This controls a first experiment but does not measure seed-to-seed success probability.

@figure fields

@table results

The square cases have a retained CFD comparison and therefore can pass or fail the frozen field gates. The deep cases have no matched raw reference and remain residual-audited pilots regardless of how attractive their streamlines appear. Large top-corner residuals are retained and discussed rather than hidden by the smooth mask.

### 10. Vortex topology in deep cavities

At D=2 a vertical stack of recirculation cells is possible because momentum transmitted from the moving lid weakens with depth while the closed geometry still requires recirculation. Vortex count should be extracted algorithmically from interior extrema of psi after smoothing only at a declared scale, then checked for persistence under grid refinement.

Cheng and Hung report that at Re=1000 a cavity with D between roughly 1.2 and 2.2 contains two large vortices. That is a useful prior for a future D=2, Re=1000 case. It is not a license to force two extrema during training or to declare any two visible loops correct. Vortex locations, signs and strengths must be compared with a raw matched reference.

A topology-aware metric complements field norms. It can detect a missing secondary cell whose contribution to global velocity energy is small. It can also be unstable when a weak extremum is close to the detection threshold, so report persistence versus grid resolution and threshold.

### 11. From a teaching pilot to a publishable study

A publishable paper requires a matched factorial design. At minimum, compare primitive/FOSLS and streamfunction-pressure representations; Adam-only and Adam-to-quasi-Newton schedules; several Reynolds numbers crossing the reported failure region; several depth ratios; and multiple independent seeds. Match the regularized lid, collocation support, network capacity or parameter count, float64 precision and total residual-evaluation or wall-time budget.

@table paperdesign

The primary outcome is not the lowest loss from the best seed. It is the distribution of qualified errors and the fraction of seeds that reach the correct branch under frozen gates. Failed runs stay in the denominator. Secondary outcomes include H1-type velocity error, pressure after gauge alignment, full and corner residuals, vortex topology, runtime and peak memory.

At least one conventional reference family must be regenerated with the identical smooth lid and aspect ratios and checked by grid refinement. The existing classical-lid archive supports near-matched screening, not the final paper. If raw fields for the earlier aspect-ratio study are recovered, their solver configuration, convergence history, units and ownership must be documented before reuse.

### 12. What could falsify the proposed explanation?

If the streamfunction method succeeds only when given more parameters, more residual evaluations or an easier lid profile, representation has not been isolated. If both formulations succeed after SSBroyden2, the optimizer rather than incompressibility structure may be decisive. If both fail at the same aspect ratio and the reference is multistable, the steady model or branch selection may be the true issue.

If low residual and exact walls coexist with a large matched field error, a residual-only notion of reliability is falsified for that case. If a method achieves low field error but large unmasked corner residual, the result may still be useful away from the singular zone, but the claim must name that restricted domain.

Negative outcomes remain publishable when the design isolates a mechanism. A broad statement that “PINNs cannot solve cavity flow” is weaker than a phase diagram showing where particular representations, optimizers and constraints fail under controlled budgets.

### 13. Research notebook workflow

The companion notebook is an audit notebook, not an interactive replacement for the GPU runner. It verifies hashes and metadata, loads retained JSON and loss histories, redraws convergence curves, assembles the case table, checks frozen gates and displays geometrically faithful fields. It never modifies the retained results directory.

Students first identify which claims are available for each case. They then compare the Adam endpoint with the final quasi-Newton endpoint, inspect whether held-out and corner residuals follow the training objective, and decide whether each square case passes the frozen CFD gates. For deep cases they record topology observations as hypotheses until matched raw references exist.

The final exercise is to write a preregistration for the factorial paper study. It must specify primary metrics, seeds, stopping rule, failure definition, computing budget and analysis of incomplete or preempted runs before the first comparison is launched.

### 14. Conclusions

The central lesson is that PINN reliability is a property of a complete numerical experiment: physical specification, representation, constraints, optimizer, support, reference and audit. Streamfunction lifting removes continuity and wall-velocity errors by construction, but does not remove momentum error, corner sensitivity, optimization failure or branch ambiguity.

The Week 13 pilot is intentionally modest. It demonstrates a restartable multi-geometry research pipeline and exposes both successes and limitations. The proposed paper becomes credible only after adding matched primitive/FOSLS baselines, multiple seeds, regularized-lid CFD/FEM references and Re=1000 or higher. Until then, the correct language is “feasibility evidence,” not “PINNs solve rectangular cavities.”

### References and provenance

[1] C. J. McDevitt, DeepPlasma, LDC module, pinned commit fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b. https://github.com/cmcdevitt2/DeepPlasma/tree/fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b/LDC . McDevitt gave Ehsan Roohi permission to use the cavity case for this course; the upstream source remains attributed and is not silently vendored.

[2] C. J. McDevitt, E. Fowler, and S. Roy, Physics-Constrained Deep Learning of Incompressible Cavity Flows, arXiv:2211.06375, 2022. https://doi.org/10.48550/arXiv.2211.06375 . Related parameterized cavity research; not identical to the fixed-Re module used by the pilot.

[3] T. Służalec, D. Wójcik, C. Uriarte, M. Łoś, A. Paszyńska, and M. Paszyński, Reliable physics-informed neural networks for Navier–Stokes simulations. Can we trust AI-generated numerical simulations? Journal of Computational Science 95 (2026), 102817. https://doi.org/10.1016/j.jocs.2026.102817 . Primitive/FOSLS PINN and CRVPINN comparison at Re=1–1000.

[4] M. Cheng and K. C. Hung, Vortex structure of steady flow in a rectangular cavity, Computers & Fluids 35 (2006), 1046–1062. https://doi.org/10.1016/j.compfluid.2005.08.006 . Aspect-ratio and Reynolds-number topology reference.

[5] U. Ghia, K. N. Ghia, and C. T. Shin, High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method, Journal of Computational Physics 48 (1982), 387–411. https://doi.org/10.1016/0021-9991(82)90058-4 . Classical square-cavity benchmark.

[6] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019), 686–707. https://doi.org/10.1016/j.jcp.2018.10.045 .

The FlowMLLab figures are generated from the retained Unity artifacts of the exact public code commit recorded in each audit. Local cavity files supplied by the instructor were used to locate provenance and design checks; unpublished files and private correspondence are not redistributed. The JCS comparison above is based on the published article, not on an inferred description from email.
