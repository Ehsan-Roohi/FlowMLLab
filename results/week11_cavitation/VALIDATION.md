# Week 11 cavitation integration validation

Verified locally on 2026-09-16, using Python 3.12 and CPU PyTorch.

- All **158 frames / seven cases** reproduce the original archived neural argmax
  arrays exactly. Pooled TP/FP/FN and Dice match the original evidence.
- The alpha/pressure comparison reproduces **16 shared moving-case frames**
  with all three pressure seeds: 192 archived class/binary array checks pass.
  This includes the original alpha model, 3×3 model, pressure U-Net and pressure
  topology U-Net. The fixed pressure threshold is evaluated on the same fields.
- Fresh Jupyter-kernel **Run All** passed: ten code cells, four comparison
  figures, about 32 seconds in the tested environment. All 51 retained input,
  model and result files remained unchanged. Full optional training stayed off.
- Five detector contracts passed: inclusive vapor threshold and connectivity,
  ignored/empty reference conventions, archive replay, zero/noisy inputs with
  frozen weights, and a two-update adaptation check that preserves the parent
  and needs only TRAIN cases.
- Four existing course-navigation tests and five existing Week 11/12 research
  evidence tests passed. The original shock/vortex gallery and reconstruction
  evidence remain intact.
- The Week 11/12 material verifier passed. The expanded Week 11 lecture has
  16 pages; all seven new pages were rendered and visually inspected. The
  existing Week 12 lecture remains 12 pages.
- The notebook verifier checked 242 relative links across the selected public
  entry points. The Colab launcher follows the repository's existing bootstrap
  and installs the optional PyTorch dependency.

This verifies local execution and evidence preservation. A hosted Google Colab
runtime was not launched. The short adaptation contract is not a new complete
training reproduction. These checks do not establish independent physical
accuracy, new-case generalization or temporal cloud tracking.
