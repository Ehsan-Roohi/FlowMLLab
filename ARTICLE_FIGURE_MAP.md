# Manuscript-to-notebook reproducibility map

The course release and manuscript use the same retained numerical/model outputs.  The routines in `common/article_validation.py` are the authoritative builders for the continuum and DSMC validation figures.  Running a notebook must not silently substitute digitized curves, screenshots, or synthetic data for a solver result.

| Manuscript evidence | Owning notebook or command | Retained inputs | Reproduced outputs |
| --- | --- | --- | --- |
| Cavity speed, streamlines, and Ghia velocity centerlines | `notebooks/week01/03_cavity_ghia.ipynb`; `python common/article_validation.py ghia` | `data/cavity_data.npz`, Ghia tables in `common/w4utils.py` | `results/article_figures/fig02_cavity_benchmark.{png,pdf}` and metrics JSON |
| Recovered cavity pressure and Botella--Peyret centerlines | same Week-1 notebook; `python common/article_validation.py pressure` | executed `results/article_validation/re1000_n65.npz`, `re1000_n129.npz`, and reference CSV | `results/article_figures/fig08_pressure_recovery.{png,pdf}` and metrics JSON |
| Direct validation of our DSMC solver | `notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb`; `python common/article_validation.py dsmc` | four executed HS--NTC runs and vector-extracted Mohammadzadeh Fig. 3 markers | `results/article_figures/fig10a_mohammadzadeh_validation.{png,pdf}` and metrics JSON |
| Cavity neural-surrogate blind case | Week-4 Lab 2 and `notebooks/week05_06/P1_Re_Generalization.ipynb` | full-case cavity data, frozen development split, retained ensemble predictions | manuscript Figure 9 and neural metrics CSV |
| POD-DeepONet validation and speed/fidelity tradeoff | Week-4 Lab 3 and `notebooks/week05_06/P3_POD_Study.ipynb` | `results/pod_deeponet/` | executed validation PNG/PDF/SVG, predictions, selection, timing, and metrics |
| Standard versus hybrid neural--DSMC fields/profiles | `notebooks/week05_06/P5_Rarefied_Cavity.ipynb` | retained research fields, centerlines, and published quantitative errors | manuscript Figures 10--12; blue standard DSMC and red dashed hybrid neural--DSMC |
| Fokker--Planck closure | `notebooks/week05_06/P6_FP_Cavity_Closure.ipynb` and `advanced/fp_closure/` | exact/reference and learned-closure outputs | Francis project result fields and comparison profiles |

## Exact reruns

- Pressure CFD: set `RECOMPUTE_PRESSURE_RUNS=True` in the Week-1 notebook, or run `python common/run_cavity_pressure_validation.py`.  The two grids are deliberately expensive enough to show refinement, but simple enough for students to inspect.
- DSMC: set `RUN_FULL_DSMC_VALIDATION=True` in the Week-3 notebook.  It runs the three `40^2` seeds and the reported `60^2` case with the exact paper budgets before rebuilding the figure.
- All three solver-validation figures: `python common/article_validation.py all`.

The release QA checks the required inputs, notebook synchronization markers, numerical thresholds, and output creation.
