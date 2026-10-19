# Reviewed Weeks 1.2–6: build and provenance

The review keeps every original lecture topic and scientific result. New worked
examples add prerequisites and hand-computable checkpoints. The main Week 2.1
PDF is now a searchable self-study article built from
`lectures/source/week02_1_probabilistic_uq_notes.tex`; the fourteen-slide MJS/PPTX
is a companion, not its PDF source. No version number was changed.

Use `python qa/build_week01_06_review.py --notebooks` for the reviewed notebook
source overlay. `qa/notebook_review_patches.json` records exact original cell
sources, replacements and inserted examples. Existing builders for pressure,
POD-DeepONet, classical ROM, UQ, grid51 and modal sensing invoke the same overlay
after construction. An original-source mismatch fails closed so generator drift
cannot silently remove a correction. New and changed code cells have empty
execution outputs; historical Markdown tables explicitly identify retained
results. This is a source update, not a claim of a fresh numerical run.

Use `python qa/build_week01_06_review.py --pdf --build-dir /tmp/flowmllab-pdf-build`
with XeLaTeX and the documented Python dependencies. Exact licensed Times New
Roman is required by `qa/LECTURE_TYPOGRAPHY.md`; code alignment and specialized
mathematical glyphs have documented font exceptions. This entry point renders
only lecture PDFs from retained inputs. It deliberately avoids the older mixed
`main()` functions that also regenerate figures, result summaries or simulations.
It includes the shared Week7 modal typography render without rerunning that lab.
The Week3 deck remains an independent editable PPTX and is exported with the
presentation workflow; its original scientific panels are not recreated here.

Fresh student DeepONet results go to a unique `student_outputs` subdirectory,
with a run ID, selected settings, dataset fingerprint, current predictions,
metrics and inference timing. Public held-out cases remain public regression
tests, not newly blind student evidence. Retained CFD timing is never combined
with fresh timing to claim a new measured speedup.

P3 and P5 require `common/student_experiment_protocols.py`. A flat package must
include that helper and a manifest matching the revised notebook. Historical
archives are not silently replaced. P5 cache identity includes every simulation
setting, source hash and case order; legacy archives lacking this contract are
rejected rather than guessed compatible.

The optional Week3 reference-comparison cell requires student-digitized,
provenance-labelled data for the correct Kn=0.05 curves. It does not supply
invented reference points or initiate a GPU solve. Original temperature maps
remain labelled biased-estimator controls; pooled production sampling is taught
separately in P5.
