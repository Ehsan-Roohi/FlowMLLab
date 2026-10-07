# Post review corrections on 6 October 2026

Historical first correction record. The subsequent eight findings and current
release checks are addressed in [the second-review response](TRANSFORMER_SECOND_REVIEW_RESPONSE.md).

This update follows the independent Linux audit of the first redesign. It is a bounded correction release and supporting material for a proposed UMass course, not a claim that the complete new LLM syllabus is ready to teach.

## Corrected numerical behavior

Phase is now measured relative to the beginning of the rollout. Reference and prediction arrays are promoted to float64 before error subtraction and norm reduction. Constant float32 probes report no identifiable frequency. Transfer initialization follows the configured context. The checkpoint audit writes a report only when an explicit non-overwriting output path is supplied.

The saved primary predictions (160 rows) and budget-audit predictions (9 rows) were rescored without training or POD refitting. Original checkpoints, predictions and training provenance are preserved. Revised numerical behavior passed 17 tests; all 126 checkpoint reconstructions and the independent 160-row evidence verifier passed on Windows using unchanged audit tolerances. See qa/transformer-numerical-corrections.json.

## Exercises and interpretation

Each week retains four assessed tasks, with the fourth now an integrated implementation: attention investigation, per-position language evaluation, sensor reconstruction audit, validation-only training, autonomous rollout, or matched transfer audit. Students also change a protocol and explain the resulting evidence. The implementation must work on varied inputs rather than reproduce a fixed example.

All 24 reference solutions passed the focused exercise audit; all 24 frozen-first-result mutations were rejected. This is a defined mutation suite, not a guarantee against every tailored hardcoded solution. Partial submissions retain a denominator of four; failed checks and unsubmitted implementations are separate states. See qa/transformer-exercise-audit.json.

Week 22 exposes selected step and budget-boundary status and identifies step-zero scratch/matched controls as untrained. The source/target validation-horizon asymmetry is disclosed. Week 20 explicitly separates training recipe from architecture claims. The forecast figure separates in-subspace differences from the much larger persistence error.

## Evidence still required before teaching

- Run this corrected version on Linux and interactive Colab. The independent review's earlier Linux failures are not erased by Windows checks.
- Add the matched, multi-seed sensor augmentation ablation. The review's one-seed diagnostic is a useful hypothesis, not a replacement for that control.
- Pilot the integrated exercises with representative students and calibrate workload and grading. More lines of code alone do not establish educational quality.
- Build the new pretrained inference, RAG, numerical tools, engineering benchmark and adaptation labs specified in the separate UMass proposal.

The original qa/transformer_validation.json, reproduction report and first review response describe the prior release. Current notebook and package checks are recorded in qa/transformer_post_review_validation.json. Cross-platform tolerances were not widened to hide failures. The 2400-step selection pattern is an observation of the original Windows training environment, not a platform-invariant result.

## Review interpretation

The review's claim that 27 state pairs necessarily underdetermine an 8-by-8 DMD operator is not accepted on sample count alone: each output has eight coefficients and 27 observations; matrix rank and conditioning determine identifiability. No regularization change was made solely on that premise. This does not establish that the small-budget DMD estimate is well-conditioned or accurate.
