> **Week 16 scope (24 September 2026):** the release covers the verified teaching-body CFD, neural model, design checks and Taylor–Maccoll study. NASA SEEB-ALR CFD failed convergence/physical checks and remains a deferred research extension; no successful NASA validation is claimed.

# Week 16: fixed-volume body of revolution at Mach 1.8

Status: development; release requires `results/week16_lowboom/release_check.json` to pass.

The physical body is three-dimensional and axisymmetric at zero incidence. SU2 solves the axisymmetric compressible Euler equations on a Gmsh meridional mesh with the radial source terms enabled. This is not planar two-dimensional flow. It has no wings, lift constraint, propulsion or viscous drag.

The two shape parameters modify a smooth radius profile. Length is 1 m; volume is pi*0.06^2/2 m^3. Reference area for pressure drag is L^2, not frontal area. Freestream pressure and temperature are 101325 Pa and 288.15 K. These values define a nondimensional teaching case, not a cruise-altitude sonic-boom flight prediction.

## Reproduction

Install the pinned Python requirements and checksum-verified official SU2 8.0.1 executable. Gmsh creates the meshes; SU2 solves the actual axisymmetric Euler flow. On Ubuntu install `libglu1-mesa`, `libxft2`, `libxinerama1` and `libxcursor1` if needed. No display is used.

```bash
python -m pip install -r qa/week16/requirements.txt
python qa/week16/install_su2_801.py
export SU2_CFD="$PWD/.tools/week16_su2_801/bin/SU2_CFD"
# In a fresh working copy/output directory:
for batch in 0 1 2 3 4 5 6 7 8 9 10; do
  python qa/week16/clean_campaign_v801.py --batch "$batch"
done
python qa/week16/clean_campaign_v801.py --assemble
# Explicit one-time training; refuses to overwrite retained weights.
python qa/week16/clean_model_v801.py --train
python qa/week16/clean_model_v801.py --check-only
```

The clean campaign preserves the original 24 training, six validation, eight test and six extrapolation geometry identities. It uses unique `clean_v801_*` run folders and separate `reference/clean_dataset_v801.npz`. The historical dataset and model remain unchanged. To retrain in a release checkout, first preserve the distributed clean-model files elsewhere and declare the new fit identity; never silently replace released evidence.

The model comparison also requires the retained eight-case finer reference (`weakwall_checkpoint_test.npz` and its audit). Complete raw evidence is produced by the linked GitHub Actions workflows. The `weakwall_cfd.py` replay additionally requires each archived original mesh/configuration, so it can verify that the solver-version comparison kept these files identical. See the reference guides for exact scope and checks.

Every new label must have finite positive pressure and density, final log10 density residual at most -9, a residual drop of at least five orders, drag variation below 1e-4 over its last 100 iterations, maximum total-enthalpy deviation at most 10%, and density below 110% of the isentropic stagnation bound. All exported fluid nodes are checked; tip nodes are not removed. These are rejection criteria, not physical uncertainty bounds. The limiter freezes after iteration 300 and convergence starts after 500.

Independent solver verification uses the Taylor-Maccoll cone and a three-mesh study. Retained candidate geometries have ten new design-point, alternative and off-design CFD checks. The NASA SEEB-ALR experimental benchmark has its own mesh family and acceptance report; the complete release remains blocked until it passes.

Pressure signatures are extracted at r/L=0.25, 0.5 and 0.75. The design target is the maximum Cp at r/L=0.5, subject to a pressure-drag constraint. Ground noise, PLdB and acoustic certification are outside this module's computed results. A smaller near-field peak does not establish a smaller ground sonic boom.

## Provenance

All geometry, Gmsh meshes and SU2 data are generated independently for FlowMLLab. The Zheng et al. 2026 paper motivates the question; no figures, dataset or implementation were copied from it. No restricted course material was used. The independent solver benchmark is a Taylor-Maccoll cone pressure solution integrated by a separate ODE implementation. Dataset train/validation/test/extrapolation identities are frozen in `design_plan.json` before learning. All splits share one two-parameter shape family; new-family generalization is not claimed.

### Install the pinned Linux solver

```bash
python -m pip install -r qa/week16/requirements.txt
python qa/week16/install_su2_801.py
export SU2_CFD="$PWD/.tools/week16_su2_801/bin/SU2_CFD"
```

The helper verifies the official binary SHA-256 recorded during this experiment. On Ubuntu, install Gmsh runtime libraries if absent (`libxft2`, `libglu1-mesa`, `libxinerama1`, `libxcursor1`). The helper's `.tools` directory is a local runtime and must not be committed. It supports Linux x86-64 only; use the corresponding official SU2 release for other systems.

The repository's `numerical_evidence.zip` preserves configurations, histories, pressure profiles and per-case checks. Extract it into `results/week16_lowboom` to inspect these records under `runs/`. Complete meshes and volume fields belong to the separately retained raw-CFD archive. They can also be regenerated from the case parameters. Figure regeneration that reads `flow.vtu`, `mesh.msh` or `restart_flow.csv` needs that raw archive or a new solver run.

## Historical evidence

SU2 8.5.0 results remain as diagnostic evidence. The eight historical refined test fields failed the added full-field enthalpy allowance despite tiny residuals. `POINTED_BODY_PHYSICS_AUDIT.md` explains the failure and controlled version comparison. Historical domain studies and plots are explicitly labelled; they do not establish new-version domain independence. New clean training labels and a new model identity are used for the main student exercise.
