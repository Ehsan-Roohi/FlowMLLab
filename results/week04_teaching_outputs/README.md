# Week 4: physical outputs and verification

These figures are produced from the numerical arrays used in the Week 4
notebooks. The comparisons show physical fields, signed profiles, and full
error ranges. They do not change the retained CFD or trained-model evidence.

| Notebook | What Run All does | Figure outputs |
|---|---|---:|
| [Lab 1: CFD production](../../notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) | Solves the reproduction and assigned cavity cases; plots flow, Ghia discrepancies, residuals and divergence | 4 |
| [Lab 3: POD-DeepONet](../../notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) | Recomputes retained field scores and shows all three public held-out cases, including pressure and signed errors | 12 |
| [Lab 4: Stokes correction](../../notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb) | Shows all seven historical pilot methods and native comparisons of the original POD and correction stages | 24 |
| [Lab 4 grid companion](../../notebooks/week04/W4_Lab4_Grid51_Validation.ipynb) | Audits retained 25-to-51-node refinement evidence | 3 |
| [Lab 6: Ghia profiles](../../notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) | Compares retained one-dimensional profile models and interpolation at all seven Ghia Reynolds numbers | 7 |
| [Lab 7: high-Re CFD](../../notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) | Audits native CFD grids and adds signed mesh differences and an upper-left corner comparison | 7 |
| [Lab 8: high-Re fields](../../notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) | Inspects frozen two-dimensional field predictions and their physical discrepancies | 10 |
| [Lab 9: label-free PINN audit](../../notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb) | Audits retained PINN outcomes and illustrates the role of importance weights in adaptive sampling | 30 |
| [Classical ROM companion](../../notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) | Runs time-matched FOM, POD-Galerkin and POD-DEIM trajectories and plots both reduced methods | 4 |

Lab 2 and Lab 5 have their own current result packages and verification records.
The table counts rendered notebook outputs, including previously retained
evidence. It is not a count of new simulation campaigns.

## Reading the comparisons

- **Lab 3:** Re = 175, 275 and 375; 65-by-65 nodes. The default mode reads
  retained predictions. The fresh-fit mode uses the current in-memory model.
- **Lab 4 pilot:** the default is the first stored constant-lid test case,
  Re = 378.4 on 25-by-25 nodes. Family and case controls expose every retained
  test case. The original pilot used 16 training and 4 validation cases per
  family, a limited neural training budget, and a single declared neural seed.
  Its poor predictions remain visible; this is not a universal architecture
  ranking. Error-map ranges differ between figures. Compare numerical maxima,
  relative norms and signed profiles rather than colors alone.
- **Lab 4 correction:** the original POD uses one seed; the two correction
  stages use three-seed ensembles. Representation, optimization and data change
  across stages. Their differences do not isolate an optimizer effect.
- **Pressure:** the plots compare recovered CFD pressure labels after matching
  their zero-mean gauges. Ghia's velocity tables do not independently validate
  pressure. Pressure fields use a labelled symmetric-log scale; signed pressure
  errors remain linear and retain their extrema.
- **Lab 6:** a centreline regressor is not a two-dimensional vortex model. The
  additional `lab6_optional_*` files verify one optional student trial at
  validation Re = 400. It used 1200 joint objective/gradient evaluations and
  no test targets. The retained model selection and frozen results are unchanged.
- **Lab 7:** the new default figure compares 128, 256 and 512-cell CFD grids at
  Re = 7500. This is mesh sensitivity, not error against an exact solution.
- **ROM:** Re = 275, 33-by-33 nodes and final time t* = 5. Both reduced methods
  are compared at the same physical times against the current full-order run;
  these are not newly claimed steady Ghia solutions.

The default notebooks run on CPU. Loading completed high-Re or PINN evidence
does not repeat the expensive original simulations or training. Optional fitting
and regeneration switches are explicit. The optional Lab 8 fitting branch was
source-reviewed but was not executed in this verification.

## Reproduce the notebook output check

From a compatible Python environment at the repository root:

```text
python qa/week04_output_audit.py --update --complete-lab4-gallery
python qa/week04_output_audit.py --execute lab3
```

The selected keys are `lab1`, `lab3`, `lab4`, `grid51`, `lab6`, `lab7`, `lab8`,
`lab9` and `rom`. Repeated application of the byte-idempotent updater preserves
the already revised notebooks and their executed outputs.
The Lab 3, ROM, Stokes and grid51 source generators invoke the same additive
updater for their own notebook, so rebuilding their source retains these
physical comparisons. Re-execute a rebuilt notebook before claiming fresh
outputs. Default student-generated files go to `student_outputs/week04_current`;
the verifier writes here.

[Execution receipt](execution.json) records actual local CPU Run All results,
output counts and notebook/code hashes. [Verification checks](checks.json)
separate executed paths from source-only checks. Local elapsed times during this
review were affected by other laptop work and are not algorithm benchmarks.
Hosted Google Colab execution is not claimed.
