# Annotated course references

The list below is a learning route, not a requirement to read every source before starting. The accompanying `course_references.bib` contains BibTeX records used by the manuscript.

## Start here

1. **Barba & Forsyth (2018), CFD Python.** A model for teaching numerical PDEs through transparent notebooks. Use it before Week 1 if finite differences are new. [DOI](https://doi.org/10.21105/jose.00021)
2. **Brunton & Kutz (2019), Data-Driven Science and Engineering.** The broadest bridge from dynamical systems and linear algebra to modern data-driven modeling. [DOI](https://doi.org/10.1017/9781108380690)
3. **Goodfellow, Bengio & Courville (2016), Deep Learning.** Background on optimization, regularization, and neural networks. The online edition is useful for terminology and derivations.
4. **Brunton, Noack & Koumoutsakos (2020), Machine Learning for Fluid Mechanics.** A compact map of ML tasks in fluid mechanics and the role of physical structure. [DOI](https://doi.org/10.1146/annurev-fluid-010719-060214)

## Probabilistic prediction and calibration

- **Rasmussen & Williams (2006), Gaussian Processes for Machine Learning.** Canonical treatment of GP regression, predictive distributions, kernels, and decision theory. [Open book site](https://gaussianprocess.org/gpml/)
- **Gneiting & Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation.** Foundation for evaluating full predictive distributions with scores such as the logarithmic score and CRPS. [DOI](https://doi.org/10.1198/016214506000001437)
- **Kennedy & O'Hagan (2001), Bayesian Calibration of Computer Models.** Separates uncertain model inputs from discrepancy between a simulator and physical observations. [DOI](https://doi.org/10.1111/1467-9868.00294)
- **Lakshminarayanan, Pritzel & Blundell (2017), Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles.** Practical ensemble approach and probabilistic evaluation; useful for understanding what FlowMLLab's repeated-seed spread does and does not measure. [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847fffa5a85bce38-Abstract.html)

The incremental Week-2.1 probabilistic-UQ notebook uses these public sources and
FlowMLLab-owned data. Its single-field interval rescaling is retained as a
descriptive calibration experiment, not a coverage guarantee for correlated
spatial nodes or shifted physical regimes.

## Numerical benchmark and validation

- **Ghia, Ghia & Shin (1982).** Classical centerline data for the lid-driven cavity; used to distinguish solver convergence from validation. [DOI](https://doi.org/10.1016/0021-9991(82)90058-4)
- **Wilson et al. (2014).** Practical scientific-computing habits for readable, testable, reusable work. [DOI](https://doi.org/10.1371/journal.pbio.1001745)
- **Sandve et al. (2013).** Ten concise rules for preserving the evidence chain of a computational study. [DOI](https://doi.org/10.1371/journal.pcbi.1003285)
- **Kluyver et al. (2016).** Jupyter as a reproducible computational narrative rather than a collection of disconnected cells. [DOI](https://doi.org/10.3233/978-1-61499-649-1-87)

### Week 7: lattice Boltzmann and cylinder wakes

- **He & Luo (1997).** Incompressible lattice-Boltzmann formulation and the hydrodynamic limit used to interpret D2Q9. [DOI](https://doi.org/10.1023/B:JOSS.0000015179.12689.e4)
- **Bouzidi, Firdaouss & Lallemand (2001).** Interpolated bounce-back for curved boundaries; the refinement target beyond the stair-step classroom boundary. [DOI](https://doi.org/10.1063/1.1399290)
- **Jackson (1987).** Global stability analysis locating the two-dimensional cylinder-wake Hopf bifurcation. [DOI](https://doi.org/10.1017/S0022112087002234)
- **Gautier, Biau & Lamballais (2013).** High-accuracy steady-cylinder reference at `Re=40`, including domain-sensitivity evidence. [Preprint](https://arxiv.org/abs/1310.6641)
- **Qu et al. (2013).** Two-dimensional cylinder force, Strouhal, and recirculation data across the laminar shedding regime. [DOI](https://doi.org/10.1016/j.jfluidstructs.2013.02.007)
- **Barkley & Henderson (1996).** Three-dimensional Mode-A threshold, used here to explain why the 2-D lesson stops at `Re=180`. [DOI](https://doi.org/10.1017/S0022112096004317)
- **Lee & You (2019).** Multi-frame prediction, Reynolds-case holdout, physical losses, and recursive-rollout diagnostics that motivate—but are not reproduced by—the compact Week-7 neural exercise. [DOI](https://doi.org/10.1017/jfm.2019.700)

## Physics-informed and operator learning

- **Raissi, Perdikaris & Karniadakis (2019), PINNs.** Foundational formulation for incorporating PDE residuals into neural training. [DOI](https://doi.org/10.1016/j.jcp.2018.10.045)
- **Karniadakis et al. (2021), Physics-informed machine learning.** Review of data–physics integration, uncertainty, and hybrid modeling. [DOI](https://doi.org/10.1038/s42254-021-00314-5)
- **Lu et al. (2021), DeepONet.** Branch/trunk operator-learning formulation used in Week 4. [DOI](https://doi.org/10.1038/s42256-021-00302-5)
- **Li et al. (2021), Fourier Neural Operator.** Spectral operator-learning alternative and useful comparison with coordinate networks. [Paper](https://openreview.net/forum?id=c8P9NQVtmnO)
- **Cai et al. (2021), PINNs for fluid mechanics.** Fluid-specific review and examples. [DOI](https://doi.org/10.1007/s10409-021-01148-1)

## POD and reduced-order modeling

- **Sirovich (1987).** Method of snapshots and coherent structures.
- **Berkooz, Holmes & Lumley (1993).** Classical review of POD in turbulent flows. [DOI](https://doi.org/10.1146/annurev.fl.25.010193.002543)
- **Rowley, Colonius & Murray (2004).** POD/Galerkin model reduction for compressible flows. [DOI](https://doi.org/10.1016/j.physd.2003.03.001)
- **Chaturantabut & Sorensen (2010).** Discrete empirical interpolation for reducing the online cost of nonlinear POD models. [DOI](https://doi.org/10.1137/090766498)
- **Taira et al. (2017).** Broad overview of modal analysis and the physical meaning of modes. [DOI](https://doi.org/10.2514/1.J056060)
- **Hesthaven & Ubbiali (2018).** Non-intrusive neural prediction of reduced coefficients. [DOI](https://doi.org/10.1016/j.jcp.2018.02.037)

## Rarefied gas dynamics and particle methods

- **Bird (1994), Molecular Gas Dynamics and the Direct Simulation of Gas Flows.** Standard DSMC reference; use for physical interpretation beyond the teaching solver.
- **Cercignani (1988), The Boltzmann Equation and Its Applications.** Kinetic-theory foundations. [DOI](https://doi.org/10.1007/978-1-4612-1039-9)
- **Roohi, Akhlaghi & Stefanov (2025), Advances in Direct Simulation Monte Carlo.** Contemporary rarefied-flow and DSMC background used by the Week 2–3 lectures.

## Research-to-classroom extensions

- **Roohi et al. (2026), Neural networks for rarefied gas dynamics.** Physics-enforced learning from relaxation to hypersonic flow. [DOI](https://doi.org/10.1063/5.0334590)
- **Roohi & Mahdavi (2026), Physics-guided DeepONet for a rarefied micro-step.** A zonal-loss example relevant to Track 2. [DOI](https://doi.org/10.1007/s10404-026-02899-8)
- **Roohi & Mahdavi (2026), Shock-centered rarefied micro-nozzle modeling.** Basis for the Week-9 moving-shock POD and neural-operator lab. [DOI](https://doi.org/10.1063/5.0343101) · [Public DSMC/POD data](https://github.com/Ehsan-Roohi/roohi-nozzle-pod-reproducibility)
- **Roohi et al. (2026), Physics-constrained neural collision operators.** Learned stochastic collision modeling with physical constraints. [DOI](https://doi.org/10.1063/5.0328463)
- **Roohi (2026), GPU-native neural surrogate for Fokker–Planck closure.** Research basis for Track 6 and its closed-loop validation requirement. [DOI](https://doi.org/10.1016/j.jcp.2026.115261)

## How to read a paper for this course

For every paper, record five items:

1. the physical problem and nondimensional regime;
2. how “truth” data were generated and validated;
3. the development/validation/test unit;
4. the simplest credible baseline; and
5. one physical quantity that could fail while global error remains small.
