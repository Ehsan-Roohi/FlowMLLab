# Weeks 17-22: active Transformer laboratories

This sequence extends the masked pretraining of Week 7.3, the diverse-wake
pretraining of Week 7.4 and the operator architectures of Week 15. It does not
restart the course as if those modules did not exist. The language-model lesson
is required, and the final week audits transfer claims rather than repeating a
smaller pretraining demonstration.

| Week | Distinct investigation | Required coding work |
| --- | --- | --- |
| 17 | Attention as a learned kernel | Q/K/V, normalization, permutation, temperature, counterexample |
| 18 | Next-character and next-state training | Shifted targets, cross-entropy, causal prefix, all-position loss |
| 19 | Information and observation design | Patch inversion, POD floor, gappy solve, token memory |
| 20 | Learning with missing sensors | Token dropout, capacity, budget diagnosis, validation selection |
| 21 | Autonomous dynamics | History update, error decomposition, frequency, phase drift |
| 22 | Transfer claim audit | Total labels, compute accounting, representation ceiling, signed gain |

Each week has a student notebook at `notebooks/weekXX/WXX_CFD_Transformer.ipynb`,
a separate instructor solution at `instructor/weekXX/WXX_Solutions.ipynb`, an
editable source in `lectures/source/weekXX_transformer_course.md` and a PDF in
`lectures/weekXX_cfd_transformer.pdf`. Student worked examples run immediately.
Uncompleted exercises are explicitly reported as NOT SUBMITTED. Students fill
the functions, set `RUN_EXERCISES=True`, and run the accompanying checks.

## Installation and execution

Use Python 3.12 in an isolated environment. Install the repository and the single
optional extra; the core package's numerical constraints still apply:

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your operating system.
python -m pip install -e ".[transformer,test]" -c qa/transformer-environment.lock.txt
python -m unittest discover -s tests -p test_transformer_course.py -v
python qa/verify_transformer_course.py
```

The lock records the actual authoring environment. Cross-platform numerical
reproduction needs the explicit tolerance check below; a package version pin
alone does not promise bitwise agreement across BLAS libraries and processors.

Open a notebook from inside this checkout and use a kernel from that environment.
Plain Run All uses temporary output space and does not update retained evidence.
Each instructor notebook must pass all four coding checks in an independent
kernel. There is no hosted LLM API, GPU requirement or paid service.

## Three different operations

1. **Read and rescore evidence:** the notebook or verifier reconstructs saved
   predictions and compares them with the original checksummed fields.
2. **Fresh fitting:** `python qa/run_transformer_course.py --output NEW_EMPTY_DIR`
   trains the complete protocol. `--quick` is a smoke protocol, not release evidence.
3. **Render teaching materials:** `python qa/build_transformer_course.py` writes
   only under `build/transformer-course`. `--pdf-only` renders PDFs only; `--check`
   compares notebook inputs with canonical lesson specifications without writing.
   Updating this checkout's derived outputs requires explicit `--output . --release`.

The compatibility fluid builder delegates to the same renderer. Neither builder
edits source notes, lesson specifications, guides or retained results. Canonical
notebook authoring lives in `course/lessons/weekXX.json`; editing a PDF source
means editing the corresponding Markdown file, not a dictionary inside a builder.

## Data roles across the course

All four modal wake cases have been inspected in earlier course work. Re105 is
validation in Weeks 7.3/7.4 and retained evaluation in the new sensing comparison.
It is never described as a new blind case. Inter-lesson reuse is explicit and
precludes turning these demonstrations into an unbiased research benchmark.

| Experiment | Fitting information | Selection information | Retained evaluation |
| --- | --- | --- | --- |
| W17/19 mechanism exercises | Explicit development examples | No architecture selection | No new performance claim |
| W18 character model | Modal-data contract, results guide, Week 7.3 text | Separate Week 7.4 document; training-only vocabulary | No broad language claim |
| W20 sensing | Re90/Re110 simulated v, stride 2 | Re100 | Re105, all / fixed half-sensor |
| W21 dynamics | Re110 omega, frames 0:160 | Autonomous 160:210 | Same uninterrupted rollout, 210:281 |
| W22 transfer | Source Re90 0:160; selected target blocks | Source 160:210; disjoint target blocks before 150 | Targets Re100/105/110, 210:281 |

W22 counts 44, 84 and 154 target frames INCLUDING validation and four observed
initialization frames. Forecast initialization is always 156:160. Each arm
predicts 160:281 without further assimilation. Training windows never cross
unobserved gaps. The source-POD scratch arm is not source-free. The target-POD
MLP is a diagnostic representation control, not an architecture-matched ablation.

## Evidence boundaries

Fields come from the same coarse author-generated LBM archive as the existing
modal labs. The 32x78 fluid ROI excludes the cylinder. No grid independence,
force accuracy, no-slip enforcement, full-domain conservation or foundation-model
reproduction is established. A local vorticity probe is not lift-derived Strouhal.

All neural comparisons report fitting history, selected and executed steps,
parameter count and time. Budget-boundary selections remain flagged rather than
being called converged. Equal update counts do not equal FLOPs. Three-seed ranges
are descriptive, not confidence intervals. Classical methods are allowed to win.

The separate `results/transformer_budget_audit` repeats forecasting with a
2400-step ceiling and validation-only selection. Five of six selected checkpoints
precede that ceiling; History-MLP seed 29 still selects at 2400. Validation losses
improve while field errors remain close to the POD floor. Inspect the full traces:
selection before the ceiling does not imply that every run exhausted its patience
or that global convergence was established. This audit does not silently replace
the fixed 1200-step primary comparison.

Checkpoints include the exact training representation. Prediction storage factors
are separately named and are only a compact serialization of computed outputs.
The verifier rescores every retained sensor, forecast and transfer prediction.
Text source hashes normalize LF/CRLF; binary artifacts retain exact byte hashes.

## Assessment and release gates

For each week: executable coding checks 40%; derivation and tensor/data reasoning
25%; evidence interpretation and limitations 25%; reproducibility 10%. The final
capstone must include labels, source access, compute, representation and signed
transfer effects. No grade depends on a neural model beating a classical method.

Release checks include causal perturbation, actual-experiment evaluation poisoning,
checkpoint round trips, error decomposition, all-row rescoring, independent notebook
kernels, and builder source preservation. See `docs/TRANSFORMER_REVIEW_RESPONSE.md`
and `qa/transformer_validation.json` for measured outcomes and any remaining limits.

## Colab without publishing the unpublished branch

This local redesign has no public notebook URL. To run it in a fresh Colab session,
upload the supplied portable course ZIP, extract it under `/content/course`, change
to its root and run `pip install -e ".[transformer]"` before opening/executing the
chosen notebook. The package and data are included; no access token is needed.
`qa/colab_transformer_bootstrap.py` checks archive paths before extraction.
This upload path is documented; an interactive Colab session is not claimed tested.
