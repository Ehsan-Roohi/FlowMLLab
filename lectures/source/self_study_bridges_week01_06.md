# Self-study worked bridges, Weeks 1.2–6

These are additional derivations and exercises; they do not replace any original lesson or retained result.

## Worked example: two control volumes and one shared face
Take two unit-volume cells with impermeable exterior faces and one interior
face velocity q, positive from left to right. With unit spacing,
\[
D=\begin{bmatrix}1\\-1\end{bmatrix},\quad
G=\begin{bmatrix}-1&1\end{bmatrix},\quad d=1,\quad
L=-DdG=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}.
\]
For the predictor q*=1, Dq*=(1,−1): each cell fails continuity, but total
mass defect is zero because the shared flux cancels. Solve Lp'=−Dq* with gauge
p'_2=0. The first equation gives p'_1=−1. Then Gp'=1 and
q_new=q*−dGp'=0, so both cell defects vanish. Adding any constant to p' leaves
Gp' unchanged. The nullspace is therefore physical pressure-gauge freedom,
not an excuse to omit a continuity equation.

This is a nondimensional algebra control, not a resolved cavity. It checks
signs and cancellation before sparse assembly. With q*=2 and d=.5, find p'
and q_new. **Answer:** p'=(-4,0), Gp'=4, q_new=0. In SIMPLE, applying only
alpha_p times this face correction would leave a mass defect; alpha_p relaxes
the pressure update, while the velocity receives the full conservative correction.

---

## Self-study bridge A: exact scalar Bayes before POD or GP
Complete this section after Week 2. The field/POD–GP section is bridge B,
best studied after Week 4's SVD/POD lesson. Prerequisites for bridge A are a
Gaussian density, variance, and scalar differentiation; bridge B additionally
needs matrix covariance, projection, and positive-definite linear systems.

Let the Poiseuille amplitude w have prior N(1,.25²), and let two normalized
observations at unit basis value be y=(.9,1.1), with independent known noise
σ=.1. Completing the square in the prior times likelihood gives
\[
S_n^{-1}=S_0^{-1}+\Phi^T\Phi/\sigma^2,
\quad m_n=S_n(S_0^{-1}m_0+\Phi^Ty/\sigma^2).
\]
Here Φ is 2×1, w is scalar, S0=.0625, precision=16+200=216;
mn=1 and Sn=1/216=.00462963. A new amplitude observation has variance
σ²+Sn=.01462963, whereas the latent amplitude has variance Sn only. The
Gaussian 95% predictive interval is approximately 1±.2371. Neither includes
CFD model discrepancy. With a field basis φ(x), replace Sn by φ(x)²Sn.

**Exercise:** discard the second observation. **Answer:** precision116,
mean106/116=.913793, variance1/116. Explain why less data increase parameter
uncertainty but do not change the declared measurement-noise variance.

## Self-study bridge B: a two-observation GP calculation
A GP is a joint Gaussian model of function values, with a kernel specifying
covariance. Take latent covariance K=[[1,.5],[.5,1]], observation noise .25I,
observations y=(1,−1), zero prior mean, test covariance k*=(.5,.5), and
k**=1. Solve (K+.25I)α=y rather than forming an inverse in production code:
α=(4/3,−4/3). Thus the predictive mean is k*ᵀα=0. Solving the same matrix
against k* gives (2/7,2/7), so latent variance is 1−2/7=5/7.
Predicting a noisy observation adds .25: variance27/28. The posterior over
POD coefficients is propagated through the modes; truncation and simulator
discrepancy are additional omissions, not automatically GP variance.

**Checkpoint:** label the units of K (coefficient²), α (1/coefficient), and
k*ᵀα (coefficient). If u=mean+Φa, field covariance is Φ Cov(a) Φᵀ; retaining
only its diagonal discards spatial correlations. A fieldwise simultaneous
interval is not established by pointwise marginal coverage.

## Worked scoring example: the variance term cannot be dropped
For one Gaussian prediction, NLL=.5 log(2πσ²)+(y−μ)²/(2σ²).
For y=μ, σ=.5,1,2 give NLL≈.2258,.9189,1.6121. If y−μ=1, these
become 2.2258,1.4189,1.7371: confidently wrong predictions are penalized.
At y=μ, Gaussian CRPS=σ(√2−1)/√π≈.233695σ, in target units.
General CRPS integrates [F(z)−1{y≤z}]² over z; the observed step CDF is not
a second learned distribution. Narrower is better only when justified by
accuracy and coverage; widening all intervals is not a complete solution.

**Exit task:** calculate the posterior by hand, then identify observation,
parameter, truncation and discrepancy contributions before running the field
example. Preserve its original under-coverage rather than tuning on test.

---

## Worked POD and hyper-reduction example
Take centered snapshots S=[[-1,0],[0,0],[1,0]]. Then SᵀS=diag(2,0), the first
singular value is √2 and a valid first spatial mode is φ=(1,0)ᵀ (its sign is
arbitrary). For a query q=(2,.5), the exact coefficient is φᵀq=2 and the rank-one
reconstruction is (2,0). The norm of the representation residual is .5;
no more accurate coefficient regressor can recover its missing second component.
If a learned coefficient is 1.8, reconstruction error has both .2 coefficient
error and .5 representation error: Euclidean norm √(.04+.25)=√.29.

Galerkin evolves coefficients by projecting the equations, rather than fitting
a static Re-to-coefficient map. Computing Φᵀf(Φa) may still require the entire
spatial nonlinear vector; a low-dimensional state alone is not low online cost.
For a nonlinear basis ψ=(1,2)ᵀ/√5, DEIM selects row2. If f=(2,4), recover
f=ψ f2/ψ2=(2,4). For f=(2,5), the same approximation gives (2.5,5): the
unobserved component error is .5. The inverse selected entry controls sensitivity.

**Exercise:** choose row1 instead and perturb the measured value by .1.
Its coefficient error is .1√5, versus .1√5/2 at row2. Explain why choosing a
large independent pivot reduces amplification, but cannot remove basis error.

---

## Worked two-sensor conditioning example (after Week 7 prerequisites)
For centered q=Φa with Φ=[[1,0],[0,1],[0,0]], the first two sensors give
CΦ=I and a=(y1,y2); measurement noise (.01,−.01) has the same coefficient
norm √2×.01. To isolate conditioning from basis error, consider a local sensor
system A=[[1,0],[1,.01]]. With true a=(1,2), y=(1,1.02). A noise increment
(0,.01) changes recovered a to (1,3): a 0.01 observation error becomes a unit
coefficient error. This A is an algebraic local example, not an orthonormal full
POD basis. Its nearly parallel rows explain the amplification.

**Exercise:** replace .01 by .1; the same noise gives coefficient error .1.
Then add a component outside the retained POD span. Even exact, well-conditioned
sensors cannot remove that representation residual. In the real wake experiment,
compare sensor layouts at the same budget and use paired noise as implemented.
Study this lab after Week 7's modal dynamics; its directory does not impose an
additional mandatory pre-project dynamics unit.

---

## Readiness example: an Ornstein–Uhlenbeck moment balance
Before the 16-input/9-output closure, derive a one-component relaxation:
∂tf=γ∂v[(v−U)f]+D∂vvf, with constant U,γ,D and decaying velocity-space tails.
Integrating by parts gives d⟨v−U⟩/dt=−γ⟨v−U⟩ and
 d⟨(v−U)²⟩/dt=−2γ⟨(v−U)²⟩+2D.
Thus stationary variance is D/γ. For γ=2 s⁻¹ and D=8 m²s⁻³,
variance is 4 m²s⁻². Starting at variance9 gives V(t)=4+5exp(−4t).
At t=.5 s, V≈4.6767. Do not confuse this raw velocity variance with
pressure ρV or temperature mV/kB; density and units matter.

**Exercise:** a time step .01 s with forward Euler from V=9 gives
Vnew=9+.01(−4×9+16)=8.8. The exact value is about8.80395.
This scalar example explains relaxation and moment constraints, not the full
nonlinear cavity closure. The C tensor and Γ vector in the actual 9×9 problem
must be interpreted using their stated moment equations and normalization.
Before admission to 6A, explain a second- and third-moment balance, solve this
example, and distinguish coefficient regression from closed-loop field fidelity.
This instructor-assigned GPU research track is not a mandatory beginner week6 task.

## Week 3: finite-particle temperature control
For two three-dimensional velocities (−1,0,0) and (1,0,0) m/s, the sample
mean is zero and the squared-deviation sum is 2 m²/s². Set m/kB =
1 K s²/m². The instantaneous estimator T_N = m Σ|v_i−v̄|²/(3 kB N)
gives 1/3 K. Dividing by N−1 instead gives 2/3 K. These are two estimators
from the same sample, not two known physical equilibrium temperatures.

For a fixed number N of independent, identically distributed samples with
finite second moment and a mean estimated from those same samples,
E[T_N] = (N−1)T/N: factors .875 at N=8 and .95833 at N=24. Correlated
particles, fluctuating occupancy and temporal averaging require additional
analysis; the factors alone do not establish a measured bias in a particular
DSMC run. The historical Week 3 code is retained as a labelled biased-estimator
control. P5 pools raw moments over its sampling window and has a different
estimator contract; seed averaging does not by itself remove common bias.

**Exercise:** explain why N=0 has no velocity mean and N=1 has no sample
variance with denominator N−1. Mask these cells when interpreting an
instantaneous map; a numerical floor is not a temperature observation.
The original Week 3 deck retains its 34 slides and appends this worked control
and its conditions on slides 35–37.
