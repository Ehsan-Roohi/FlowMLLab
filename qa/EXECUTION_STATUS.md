# Execution audit — 2026-09-05

Development branch only. No new release, DOI, front-page research claim, or
change to the frozen scientific acceptance criteria. Upload reviewed code to
GitHub first; Unity runs must retrieve a full immutable commit, not a moving
branch or an edited browser-only script.

## What needs computation

| Work | Verified state | Next action |
|---|---|---|
| DMD, sparse sensors, SINDy companions | Both notebooks and all retained metrics reproduced locally | Research forecasts require qualified CFD first, not another fit to the old coarse labels |
| Cylinder CFD qualification | Unity CPU job **64023967** running from `6f463ac`; one CPU, 8 GiB, 24-hour limit | Finish D40 and inspect the fixed grid/statistics gates before any domain/Mach/trajectory campaign |
| Micro-step architecture V5 | Existing A40 job **64004321** completed 18 controlled fits plus two anchors in 7:05 | [Report-level audit](../results/step_architecture_v5/README.md); no duplicate training or promotion of poor vortex results |
| Weeks 11 and 12 | Both notebooks executed locally; Week 12 freshly refitted | Execution/reproduction only; not a new independent research validation |
| SPARTA GPU benchmark | Original **64023942** failed on unresolved `libcudart.so.12`; retry **64024425** also failed (see below) | Preserve the failed preflight; no further allocation or speedup claim from this stabilization audit |
| Nozzle correction | All 15 published exports recovered in two local ZIPs after line-ending normalization; exported-field inconsistency remains, cause unresolved | Recover the producing source/input/raw moments and follow the [versioned data-note procedure](../docs/NOZZLE_DATA_NOTE.md); more ML training cannot repair reference data |
| Week 7.1 target scaling | All 44,500 retained rows matched to raw archived MA/TOV/P; no freestream division in the course import | Preserve NPZ/model bytes and compatibility aliases; recover producing input before asserting reference constants or nondimensional fields |

Read-only Unity accounting at **2026-09-06 00:20 UTC** (September 5 locally)
still showed D40 **64023967 RUNNING**, elapsed 3:27:29, with no final production
assessment. The log's 40-step smoke result is not the 80,000-step D40 result.
No running environment, frozen acceptance criterion or original output was
changed, and no duplicate D40 was submitted.

Retry **64024425 FAILED**, exit 1, after 13:38. Its CPU fresh preflight passed;
the Kokkos fresh preflight reached SPARTA but stopped on
`ERROR: Illegal package kokkos command`, with last command
`package kokkos gpu/aware off`. This is a command-level preflight failure, not
an accuracy comparison or a performance result. Guard **64024426 COMPLETED**
in 3 seconds; that scheduler state does not make the failed solver successful.
No new GPU benchmark or 39-run campaign is authorized by this audit.

SPARTA's original CPU guard **64023943** correctly printed `NO_AUTOMATIC_RETRY`
after the software failure; its exit 0 was not a successful solver continuation.
The explicit retry followed five passing Linux path-resolution tests, a passing
mocked scheduler/retry regression, and an actual `ldd` check of the saved CUDA
binary with no unresolved dependencies. That preflight did not execute a solver
or a GPU kernel. The exact failed-job ownership, run directory and accounting
state were checked, and no active duplicate benchmark existed before submission.
The failed retry requested one A40, 16 CPU ranks for the paired reference and
48 GiB host RAM,
with a four-hour limit per allocation. It retains the existing at-most-eight
allocation recovery policy for preemption/node failure/timeouts, not software
failures. Old logs and outputs are preserved in their original run directory.
No CPU/GPU speedup or statistical equivalence result is available yet.
The archived 39-run refinement matrix is not approved by this audit. The saved
benchmark remains at the existing pilot's 1000x200 grid, PPC20 and approximately
5.1 million particles, not a new production-data campaign.

D40 is serial NumPy, not a CUDA program. Its measured local 100-step full-grid
timing was 53.90 seconds; the full 80,000-step run is genuinely long. Do not
duplicate it locally or request GPUs for unchanged serial code. Its Unity
environment has newer NumPy/pandas/SciPy/scikit-learn versions than the declared
package ranges. Preserve that running environment and its provenance; any
acceptance needs explicit cross-environment checks, not a silent package change.
See the [research qualification gates](MODAL_RESEARCH_PROTOCOL.md).

The archive checks and [term stabilization policy](../PUBLISHING.md) introduce
no new physical data, model-selection experiments, releases or DOIs. Recovered
import semantics
are in the [cylinder source audit](../data/hypersonic_cylinder/README.md);
the original producing decks/reference conditions remain a separate gate.

## Local verification

For this source-lineage follow-up: **145 core tests collected**, 3 optional
TensorFlow skips, no failures. The revised Week 7.1 executable cells completed
end-to-end on the local QA environment; all **359 data/results files**
were byte-identical before/after execution. Its HTML was regenerated with both
embedded figures, unchanged displayed percentage metrics, and no private local
paths. Executable changes are limited to wrapping overlapping plot titles;
data processing, training, splitting and metric code are unchanged. The HTML
verifier now forces the inline backend and fails if figures are lost rather than
publishing an empty plot area. Re-running the complete 20-case archive comparison
reproduced the retained
source-audit JSON values exactly. Notebook execution repeats its frozen MLP
fit; no new hyperparameter selection, saved model or source field was introduced.

The earlier, broader local verification below is retained with its own scope
and counts; it is not a claim that every notebook was rerun in this follow-up.

Command from the repository root, with the project dependencies and Jupyter
notebook execution tools installed:

```bash
python qa/run_local_followup.py --output tmp/followup-new
```

The runner refuses an existing output directory or an output inside retained
data/results/notebooks/lectures. It uses a private Jupyter kernel/configuration
and writes executed notebooks, fit results, logs, versions and source hashes to
scratch. On Windows it needs normal local-process permissions for secure
Jupyter connection files and loopback sockets; do not disable Jupyter security.

Verified on Windows/Python 3.12.14: 137 core tests collected, 3 optional skips,
no failures; 13 V5 protocol tests collected, 2 Linux-only skips, no failures.
All 13 V5 tests also passed separately on Unity/Linux (mocked job submission,
not additional allocations). Four notebooks executed end-to-end: Week 5 sparse
sensing, Week 7 modal forecasting, Week 11 identification, and Week 12 DSMC
reconstruction. All modal metrics and all 28 Week 12 metric rows reproduced
within the pre-existing 2% relative / 1e-5 absolute cross-library tolerance.
All **437 retained files remained byte-identical** during the final audit.

Local environment: NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3,
scikit-learn 1.6.1, matplotlib 3.10.9, nbclient 0.11.0. These scientific package
versions satisfy the repository constraints. The complete audit was repeated
successfully on published commit `369ecf6e0ffaf3e86b49156eaab8559a50f0d733`,
after integrating the already-published SPARTA recovery code. The scratch
summary records individual source hashes; runner SHA256 is
`c852b98a7948f8f48226f278888ad5a46fc262bd945e960ab295136ecbaaa749`.

Week 12 still retains its finite-iteration optimizer warning and the baseline
advantage for qx. No test-guided epoch sweep, new accuracy claim or selective
replacement of those results was made. The V5 CSV was checked byte-for-byte
against an extraction of the original report (SHA256
`afd5b847326981a8a49a79f6a38c83bada7f0bb893965a734546d28f641f6c82`).
