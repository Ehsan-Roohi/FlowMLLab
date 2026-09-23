# Week 16: fixed-volume body of revolution at Mach 1.8

Status: development; release requires `results/week16_lowboom/release_check.json` to pass.

The physical body is three-dimensional and axisymmetric at zero incidence. SU2 solves the axisymmetric compressible Euler equations on a Gmsh meridional mesh with the radial source terms enabled. This is not planar two-dimensional flow. It has no wings, lift constraint, propulsion or viscous drag.

The two shape parameters modify a smooth radius profile. Length is 1 m; volume is pi*0.06^2/2 m^3. Reference area for pressure drag is L^2, not frontal area. Freestream pressure and temperature are 101325 Pa and 288.15 K. These values define a nondimensional teaching case, not a cruise-altitude sonic-boom flight prediction.

## Reproduction

Install Gmsh 4.15.2, NumPy, SciPy, scikit-learn, Matplotlib, pandas, meshio, nbformat and nbclient. Install official SU2 8.5.0 separately. Set `SU2_CFD` to its executable's absolute path. On Linux Gmsh may require libXft and other graphical shared libraries even in batch mode. No display is used.

From the repository root:

```bash
python qa/week16/benchmark.py
python qa/week16/cfd.py --name baseline_fine_stable --level 1.5
python qa/week16/cfd.py --name baseline_finer --level 2
python qa/week16/cfd.py --name baseline_large_domain --level 1.5 --height 2 --end 4
python qa/week16/analyze.py baseline_fine_stable baseline_finer baseline_large_domain
python qa/week16/campaign.py
python qa/week16/learning.py
python qa/week16/verify_designs.py
python qa/week16/report.py
python qa/week16/build_materials.py
```

The campaign uses three independent single-threaded SU2 processes. Reduce the worker count on a small laptop. A passing executable return code alone does not accept a label: the pressure and density must remain physical, the density residual must fall by at least five orders, and the drag range over the last 100 iterations must be below 1e-4 relative to its mean. The limiter is frozen after iteration 300 and convergence is checked only after iteration 500.

Pressure signatures are extracted at r/L=0.25, 0.5 and 0.75. The design target is the maximum Cp at r/L=0.5, subject to a pressure-drag constraint. Ground noise, PLdB and acoustic certification are outside this module's computed results. A smaller near-field peak does not establish a smaller ground sonic boom.

## Provenance

All geometry, Gmsh meshes and SU2 data are generated independently for FlowMLLab. The Zheng et al. 2026 paper motivates the question; no figures, dataset or implementation were copied from it. No restricted course material was used. The independent solver benchmark is a Taylor-Maccoll cone pressure solution integrated by a separate ODE implementation. Dataset train/validation/test/extrapolation identities are frozen in `design_plan.json` before learning. All splits share one two-parameter shape family; new-family generalization is not claimed.

### Install the pinned Linux solver

```bash
python -m pip install -r qa/week16/requirements.txt
python qa/week16/install_su2.py
export SU2_CFD="$PWD/.tools/week16_su2/bin/SU2_CFD"
```

The helper verifies the official binary SHA-256 recorded during this experiment. On Ubuntu, install Gmsh runtime libraries if absent (`libxft2`, `libglu1-mesa`, `libxinerama1`, `libxcursor1`). The helper's `.tools` directory is a local runtime and must not be committed. It supports Linux x86-64 only; use the corresponding official SU2 release for other systems.

The repository's `numerical_evidence.zip` preserves configurations, histories, pressure profiles and per-case checks. Extract it into `results/week16_lowboom` to inspect these records under `runs/`. Complete meshes and volume fields belong to the separately retained raw-CFD archive. They can also be regenerated from the case parameters. Figure regeneration that reads `flow.vtu`, `mesh.msh` or `restart_flow.csv` needs that raw archive or a new solver run.
