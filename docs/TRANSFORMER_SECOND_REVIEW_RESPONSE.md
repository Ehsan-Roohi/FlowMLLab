# Second review corrections - 6 October 2026

This public update addresses the independent review of the Weeks 17-22 redesign.
It publishes six Transformer/flow laboratories with one required small language
model lab. It does not claim completion of the separate proposed UMass semester
course on LLMs for mechanical engineering.

| Finding | Correction | Acceptance check |
| --- | --- | --- |
| N1: stale fourth tasks in worksheets/PDFs/guide | All four task titles and prompts synchronized; PDFs rebuilt | `qa/verify_transformer_curriculum.py`, 24 tasks across six editions/sources |
| N2: undeclared interfaces | Return keys, shapes, masks, errors and callable arguments in every prompt and starter docstring; W17 permutation takes an explicit callable | Reference solution audit and independently varied inputs; mask True means allowed |
| N3: five wrong implementations pass | Added off-subspace truth, independent field norms, near-ceiling steps, squared-threshold discriminator and unequal-arm flags | Five named semantic mutations rejected directly and in notebook check cells |
| N4: audit import/overwrite | Direct and module entry points work; default is read-only; explicit report uses exclusive creation | Report safety tests; historical report retained |
| N5: target-POD step-zero flag missing | Fresh-initialization flag includes target-POD-MLP; pretrained zero means no target adaptation | W22 table and plot distinguish 25 fresh and 9 pretrained step-zero fits at 44 labels |
| N6: maintenance prose in student notes | Removed review appendix; integrated scientific explanations and matching worksheet | No maintenance section in student source; phase equation uses t-t0 |
| N7: source selection interpretation | All three source fits selected 300/300; 50-step source and six-step target validation still differ | No attribution of gain to a selection effect or to one factor without ablation |
| N8: rescoring provenance | Distinguish saved main trajectories from freshly inferred auxiliary observed resets; record runtime environment | Prediction/checkpoint bytes unchanged; both evidence manifests record operation types |

Additional earlier findings are addressed: sensor coordinates use grid mean and
full span (symmetric grid endpoints are +/-0.5); W19 prints the representation
floor; W18 reports training/validation character counts, window counts and unknown
token fraction; the retained branch width is 208; poisoning covers the actual
protocol; frequency tests use off-grid frequencies. Sinusoidal goodness of fit is
reported: reference R-squared is about 0.698, with constant-signal R-squared and
frequency undefined. A local vorticity probe is not a lift/drag validation.

## Numerical acceptance

The original strict checkpoint test (`rtol=1e-5`, `atol=1e-6`) remains unchanged.
Its independent Linux failures are not overwritten by Windows passes. A separately
named scientific cross-platform criterion was introduced after the review:
normalized field discrepancy and relative-L2 score drift each <=1e-5. This is a
declared numerical acceptance rule, not a preregistered threshold. Every audit
reports both criteria and its environment; strict remains the default.

The original prediction and checkpoint archives are unchanged. Main scores are
float64 rescoring of those stored predictions. Twelve additional observed-reset
trajectories per evidence directory are fresh checkpoint inference, not stored
trajectory rescoring. No training or POD refit occurs during rescoring.

The separate matched sensor augmentation ablation uses seeds 17, 29 and 43,
identical paired initial states, identical observations/noise, basis, optimizer,
validation objective, ceilings and early-stop rules. Executed and selected budgets
can differ. It isolates a training recipe within SensorSet; it does not establish
an architecture ranking. Full traces, predictions and checkpoints are retained
under `results/transformer_sensor_ablation`.

Paired initialization is verified within the original training run, across both
checkpoint bundles, all condition rows, and checksummed metadata. Recreating a
new initialization on another platform can produce a different byte digest;
the verifier reports that digest and both environments as a diagnostic. It still
requires the original training pair to match and retains both prediction criteria.

## Release scope and teaching acceptance

The published repository contains corrections, runnable examples, student and
instructor notebooks, PDFs and supporting evidence. GitHub Actions checks the
published revision on Ubuntu and uploads both numerical audit statuses and fresh
kernel timings. Local outcomes are recorded in
`qa/transformer_second_review_validation.json`; older QA files describe their
historical versions and are not current release certificates.

The intermediate `qa/transformer-exercise-audit-second-review-pre-fix.json`
records a failed local check before the seeded attention fixture was corrected.
The final passing exercise audit is
`qa/transformer-exercise-audit-second-review-validated.json`, with hashes of the
six canonical lesson files. Both are retained with their distinct status.

Repository-wide CI also exposed an optional-dependency integration error. The
new test modules now explicitly skip when PyTorch is absent from a base install;
the dedicated Transformer job installs the extra and runs all 21 tests. Missing
dependencies inside an installed PyTorch distribution still raise errors.

The repository notebook validator now recognizes the exact portable setup of
these six laboratories and checks it against canonical lesson prompts/starters,
student metadata, temporary experiment output and incomplete-task reporting.
The separate documented Colab setup remains an open interactive acceptance check.
Week 16's existing custom Colab bootstrap is validated by its actual setup
contract rather than requiring the older bootstrap marker used by other weeks.

Two acceptance activities require actual users/sessions: an interactive Colab
run and a workload/grading pilot with representative students. They are recorded
as open classroom checks, not silently treated as completed. The new engineering
LLM course still needs its proposed pretrained inference, RAG, numerical-tool,
engineering benchmark and adaptation labs. Public publication of these six labs
does not certify that larger syllabus as ready to teach.
