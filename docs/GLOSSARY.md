# Glossary

Short definitions of the terms and abbreviations that the course pages use
without expanding them. Each entry says where the term first matters.

## Flow solvers and data

- **CFD** (computational fluid dynamics): numerical solution of the continuum
  flow equations; the OpenFOAM step flows of Week 15 and the cavity solvers of
  Weeks 1 and 4 are CFD.
- **DSMC** (direct simulation Monte Carlo): particle method for rarefied gases,
  used for the Week 3, 9, 10 and 12 data. **dsmcFoam** is the OpenFOAM DSMC solver.
- **LBM** (lattice Boltzmann method): the D2Q9 two-relaxation-time (TRT)
  solver that produces the Week 7 cylinder wakes.
- **RANS** (Reynolds-averaged Navier-Stokes): the averaged turbulence model
  framework of Week 14; a **closure** is the model that supplies the unresolved
  stresses.
- **Re** (Reynolds number): the ratio of inertial to viscous effects; each
  Week 15 geometry is solved at Re = 25, 50 and 100.
- **Geometry IDs g001 to g051**: the 51 Week 15 step-flow masks. **g005** is the
  one whose floor drops in two steps and then rises again; it shares the
  two-descending-step motif of the test family without belonging to it and is
  therefore **quarantined** (used neither for training, validation nor testing).
- **H44 and H67**: the two held-out cases of the Week 9 DSMC micro-step study,
  named by their step height (the numbers are not percentages).
- **Case-wise split**: training, validation and test sets separated by whole
  cases (a geometry with all its Reynolds numbers, a trajectory with all its
  frames), never by random samples from one case.
- **Retrospective test**: a test set that was inspected during the research
  programme before the final protocol was frozen, so its results are reported
  as retrospective rather than prospectively blind (Week 15 double-step family).

## Models

- **POD** (proper orthogonal decomposition): the data-driven modal basis of
  Weeks 4, 5 and 7; **POD-Galerkin** projects the equations onto it, **POD-DEIM**
  adds a sparse interpolation of the nonlinear term, **gappy POD** fills missing
  measurements with it.
- **DeepONet**: an operator network with a **branch** (encodes the input
  function or parameters) and a **trunk** (encodes the query coordinates); their
  product gives the field value. **Geom-DeepONet** adds the geometry (SDF raster)
  to the branch; the historical Week 15 notebook labels its geometry-conditioned
  DeepONet runs (`geom`) **Geo-DeepONet**.
- **FNO** (Fourier neural operator): learns the operator in Fourier space on a
  regular grid; **Geo-FNO** maps an irregular geometry to that grid first;
  **U-FNO** adds a U-Net path.
- **SMART**, **GeoTransolver**, **DoMINO**: the three published geometry-aware
  architectures compared in Week 15, run from their authors' pinned code
  (Hagnberger and Niepert; NVIDIA PhysicsNeMo). DoMINO is used in a planar
  adaptation of its three-dimensional design.
- **Proxy** (PhysicsX-inspired, LIFT-inspired): a small model written for the
  course that imitates a described layer or feature of a company system; it is
  not that system's code and its numbers say nothing about the company product.
- **PINN** (physics-informed neural network): a network trained on the residual
  of the governing equations (Weeks 13 and 14).
- **MAPA**: the pretraining protocol of Tang, Spalding and Cogan (2026,
  arXiv:2609.13507) that Weeks 7.3 and 7.4 follow in spirit; no MAPA code or
  data is used.
- **SDF** (signed distance field): for each grid point, the distance to the
  nearest wall, with opposite signs on the fluid and solid sides; the geometry
  input of Geom-DeepONet.
- **Seed**: the random initialisation of a training run; **three-seed mean**
  averages three such runs. A **warm start** continues training from an earlier
  checkpoint; a **checkpoint** is a saved model state (**BEST** = lowest
  validation score, **LAST** = final epoch).

## Metrics

- **Relative L2 error**: the norm of the prediction error divided by the norm
  of the reference field, in percent; the main velocity, pressure and vorticity
  score of the course.
- **NRMSE** (normalised root-mean-square error): root-mean-square error divided
  by the range or scale of the reference signal.
- **IoU** (intersection over union): overlap of two regions divided by their
  union. The Week 15 **filtered reverse-flow IoU** compares the regions where
  u < -0.01 after keeping only four-connected components of at least eight
  cells; the historical notebook's **raw** version uses u < 0 without the filter.
- **Dice coefficient**: twice the overlap divided by the sum of the two region
  sizes (Week 11 cavitation masks).
- **GCI** (grid convergence index): Roache's estimate of discretisation
  uncertainty from three grid levels (Week 7 cylinder refinement).
- **NLL** (negative log-likelihood) and **CRPS** (continuous ranked probability
  score): proper scores for probabilistic predictions (Week 2.1).
- **Validation score**: the balanced validation objective used to select
  checkpoints and learning rates in Week 15; lower is better.
