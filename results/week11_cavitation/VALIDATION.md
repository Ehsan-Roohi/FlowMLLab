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

## Contour readability and solid-error clarification

The author's screenshot review identified faint green cloud outlines and
unexplained contours around the hydrofoil. The original Plunging3 masks confirm
that the main reference cloud has 653 pixels: the pressure threshold detects
571 of them and the pressure U-Net detects 651. The cloud is not wholly missed;
the original thin green outline lacked contrast against the blue field.
The pressure U-Net and pressure topology U-Net also predict 951 and 863 solid
pixels as vapor, respectively, out of 1087 solid pixels. These are model errors.

Contours now use thicker lines with white contrast halos. Arrows mark the
unchanged solid errors, and panel scores explicitly say fluid-only. The same
16-frame/three-seed replay still passes all 192 archived-array checks; numerical
metrics are unchanged. No threshold, weights, masks or geometry veto changed.
The updated notebook passed a fresh local Run All (10 code cells, four figures,
244 relative links, 51 retained files unchanged). Both updated figures and
lecture page 15 were visually checked; the lecture remains 16 pages.
