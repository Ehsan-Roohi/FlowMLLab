# Neural networks across the Ghia cavity cases: primary-source check

Checked 15 November 2026 by an independent research agent. These are verified
reports of what papers contain, not independently reproduced numerical results.
The exact target set is Re = 100, 400, 1000, 3200, 5000, 7500, 10000 with the
standard square cavity and constant moving lid.

## Verified exact coverage

**Wei et al., FFV-PINN (2025)** explicitly covers **all seven**. See
[full text, Section 3.1 and Table 3](https://arxiv.org/html/2603.24114v1#S3.SS1),
printed PDF p.26; Ghia centreline/vortex comparisons are Figures 8-9, pp.25-26.
The network inputs are x,y; case-specific residual optimization solves each Re.
It outputs full u,v,p fields, rather than fitting only centreline samples, and
uses no interior CFD training labels. CFD from Chiu (2018) is the evaluation
reference; its grid resolution was not verified in this review. The architecture
uses frequency mapping, shared and variable-specific layers, SiLU and Adam.
There is no reported held-out-Re parametric-surrogate split.

The publication is CMAME 444 (2025), article 118139,
[DOI 10.1016/j.cma.2025.118139](https://doi.org/10.1016/j.cma.2025.118139).
The [author arXiv version](https://arxiv.org/abs/2603.24114) was uploaded in March
2026; that upload date does not change the journal publication year.

## Data-driven and parametric comparisons

| Primary study | Actual coverage and information used | Difference from the exact seven-case test |
| --- | --- | --- |
| [Lu et al., fair neural-operator comparison (2022)](https://doi.org/10.1016/j.cma.2022.114778); [full text](https://arxiv.org/html/2111.05512v1) | Section 5.7/Table 10 and Appendices S2/S3/S5: 100 training cases Re=100:20:2080, ten random unseen test Re in that interval. Supervised full u,v fields; POD-DeepONet uses six modes. MRT-LBM reference on 256 squared. | Spatially varying regularized lid; not the standard constant-lid high-Re Ghia set. Exact ten test values and a separate validation-Re list were not identified. |
| [Rocha et al., FNO cavity benchmark (2023)](https://www.mdpi.com/2076-3417/13/5/3165) | Section 2.1, Figures 2-5: Re=1000,2000,4000,6000,10000,14000,16000,18000,20000; 10000 held out, other eight training. FNO/CNN-LSTM/ConvLSTM forecast full transient velocity fields. OpenFOAM pisoFoam, k-epsilon RANS, 128 squared; continuity penalty supplements labels. | Different transient/RANS task; only 1000 and 10000 overlap the requested seven. Re=8000,12000 deliberately excluded. No separate validation-Re set verified. |
| [Paksoy and Aradag, POD/ANN cavity dynamics (2015)](https://dergipark.org.tr/en/download/article-file/400595) | Tables 1/3 and Figure 2; ANN discussion pp.9-11: Re=100,500,1000,5000,10000, sinusoidally moving lid. POD/ANN temporal amplitudes, in-house streamfunction/vorticity CFD, 101 squared. | Temporal splitting/rescaling and different lid; not a common steady-field model with all seven withheld by whole Re. |
| [Vorticity-enhanced PINN with log-Re embedding (2026)](https://www.mdpi.com/2311-5521/11/4/93) | Sections 3.2-3.5 and Figures 12-13: one parametric coordinate network, CFD velocity labels plus PDE/vorticity losses. Training Re=1000:1000:50000; demonstrations at 100 and 100000. CFDBench/Fluent labels interpolated to 64 squared. | Does not establish 400,3200,7500 coverage. Within-range separation is spatial sampling rather than withheld training-range Re. |
| [Chen et al., physics-informed neural-operator training (2026)](https://arxiv.org/html/2606.06164v1) | Section 4.5, Appendices C.3/D: Re sampled log-uniformly in [50,1000], 100 unseen test Re; PI-DeepONet, PI-FNO, PI-CViT with SOAP. Main experiment uses PDE training; separate supervised ablation. Full u,v,p. | Regularized lid and low-Re range; not the seven high-Re Ghia cases. |

## Other important high-Re PINN results

- [Jiang et al., FD-PINN (2023)](https://doi.org/10.1002/fld.5217), Section 3.1.1/Figure 3
  and Section 3.1.3/Figures 7-12: **100,1000,5000,10000**. Case-specific u,v,p
  solutions; difficult cases use 10-100 interior CFD labels. Reference is
  OpenFOAM simpleFoam on a 201-squared nonuniform mesh; PINN grids 51/101 squared.
  Four/eight layers of 50 units, Adam then L-BFGS-B. It is not fully data-free.
- [Scale-PINN (2026)](https://arxiv.org/html/2602.19475v1), Figure 3(a), PDF p.7:
  **400,3200,7500,10000,20000**. Case-specific data-free sequential correction.
  Section 4.4.1, p.17 describes a 512-squared IDFC-squared CFD reference,
  downsampled for evaluation. Uses Ghia through 10000 and Erturk at 20000.
  Re=5000 mentioned in comparisons is not part of its own five-case result.
- [Bai and Bian, hybrid DeepONet/CFD (2025)](https://arxiv.org/html/2506.03016v2),
  Section 3.1, Tables 2-3 and Figures 15-17: **1000,3200,10000**. The DeepONet
  accelerates pressure-Poisson iterations inside CFD. It is not a standalone
  Re-to-steady-field neural surrogate.

An additional [physics-data finite-volume ROM paper](https://doi.org/10.1063/5.0308144)
remains an unverified lead: sufficient primary full text was unavailable to
confirm its exact Re list and data split.

## Implication for FlowMLLab

Do not claim the first neural solution of all seven Ghia cases: FFV-PINN already
reports them. Full-field parametric cavity surrogates also exist. In this search,
an exact all-seven, constant-lid supervised POD/operator study with a clearly
withheld-Re protocol was not verified; this is not proof that none exists.
Describe the actual information budget, validated CFD grids, training-only POD,
fixed whole-Re split, uncertainty and baseline wins/losses. Our new field model
must be distinguished from the earlier sparse-Ghia-profile MLP and from solving
a PINN anew for each Re.

## FFV-PINN versus the Roohi-McDevitt deep-cavity assessment

The exact CMAME PDF has 27 pages: Table 3 is on page 12, Ghia profiles
on page 14, vortex centres on page 15, and conclusions on page 23.
These differ from the longer arXiv layout cited above.

FFV-PINN evaluates velocities at control-volume faces and adds a concurrent
L1 residual-correction loss from current/previous epoch fields (Eqs.18-27).
This is an approximate SIMPLE-inspired update constraint, not another network
or supervised CFD target. It uses Adam and an LSA numerical boundary treatment.
Its unsteady example retains AD for temporal derivatives.

[Roohi and McDevitt v2](https://arxiv.org/html/2610.38814v2) assess D/W=5,
Re=100/250/500/1000, with a Gaussian-regularized lid. An inherited analytical
streamfunction-pressure representation enforces continuity and wall velocities;
AD momentum residuals are depth-scaled. Prescribed regional sampling and
transferred initialization feed SOAP followed by self-scaled Broyden.
The contribution is a matched regional assessment against spectral/hp CFD,
not invention of all the inherited formulation components.

FFV-PINN explicitly misses the third small lower-left vortex at Re=10000.
This motivates weak-flow regional diagnostics, but proves no matched-method
superiority. Its Table 3 V error concerns speed magnitude, not velocity-vector
discrepancy. Its 3.03% cannot be ranked against the deep-cavity refined 2.658%
interior vector error: geometry, lid, Re, variable, masks, reference and
budgets differ. Neither experiment establishes one network predicting all
seven unseen Reynolds cases.
