# Week 11 — Shock and vortex identification

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 10.1](../week10_1/README.md) · [Week 12 →](../week12/README.md)

Shock/vortex identification; alpha and pressure methods for vapor clouds.

![Week 11 real airfoil field and learned shock and vortex masks](../../results/week11_research/airfoil_2.png)

## Lecture and notebooks

**Lecture:** [Lecture 11](../../lectures/week11_shock_vortex_identification.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Shock and vortex identification | [Open notebook](../../notebooks/week11/W11_Shock_Vortex_Identification.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Shock_Vortex_Identification.ipynb) |
| Lab 2: reconstruction followed by identification | [Open notebook](../../notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb) |
| Hydrofoil vapor-cloud detection | [Open notebook](../../notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb) |

**Module guides:** [Week 11 lab](../../notebooks/week11/README.md)

## What you will work on

- Shock/core identification; shear versus rotation; hydrofoil vapor-cloud detection

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Identify shocks/vortex cores and test reconstruction before detection.<br>
**CFD / data:** Archived ShockVortexML compressible fields for the lead figure; FlowMLLab D2Q9–TRT LBM for the wake extension. The lead archive's exact producing-solver revision is not established here.<br>
**Learning method:** Harmonized Joint (HJ), a custom shared-encoder, multi-branch
encoder-decoder with specialist shock, vortex-core, wake/shear and expansion
paths. The displayed masks use the frozen task-preserving HJ shock-repair
checkpoint: the shared encoder and complete vortex pathway remain fixed while
the shock-specific path is adapted. HJ-joint and a capacity-matched U-Net are
comparison models, not the network shown here. A separate lab uses U-Net for
velocity-field reconstruction before physical vortex identification.



Six fresh fixed-checkpoint forward passes on existing airfoil and cylinder fields from
Roohi's [ShockVortexML research](https://github.com/Ehsan-Roohi/ShockVortexML).
ML-only outputs; previously inspected development-test cases, not human-validated accuracy.
[All six full-size figures and provenance](../../results/week11_research/README.md) ·
[Notebook and lecture](../../notebooks/week11/README.md).
Synthetic controls remain as the warm-up.

**Watch the extended research pipeline — airfoil and cylinder**

[![Watch Movie S9: numerical schlieren on the left and learned airfoil structures on the right](https://huggingface.co/spaces/ehsanroohi/ShockVortexML-Demo/resolve/e54de0a6f8665ef3cb098061d0704b4d96e0c414/airfoil_movie_v2.png)](https://www.youtube.com/watch?v=ULA8x2jUEvA)

[▶ Watch the airfoil video (S9)](https://www.youtube.com/watch?v=ULA8x2jUEvA)
· [▶ Watch the cylinder video (S10)](https://www.youtube.com/watch?v=opVMf1OVdM4)
· [Explore both examples on Hugging Face](https://huggingface.co/spaces/ehsanroohi/ShockVortexML-Demo)
· [Download original-quality movies and provenance](https://github.com/Ehsan-Roohi/ShockVortexML/releases/tag/movies-localfront-v2-20260909)

Click the preview to open the video on YouTube. The left panel shows numerical
density schlieren; the right shows localized learned shock fronts, vortex-core
candidates, wake/shear and expanding-flow regions. These movies use the extended
local-front model with retained core/wake branches and the PM-v4 expansion-region
branch, a different configuration from the HJ control checkpoint described above.
Blue denotes expanding flow, not a validated centred Prandtl–Meyer fan.
These inspected research examples retain detection errors; their visual coverage
is not an independent accuracy measurement.

**Real-field extension:** [U-Net reconstruction followed by vortex identification](../../results/week11_reconstruction/README.md)
compares interpolation, reconstructed velocity plus swirling strength, and direct
mask prediction on the same retained LBM wake cases. Three training seeds, loss
histories and saved-checkpoint audits are included. These are weak-reference
vortex scores on coarse incompressible data, not shock accuracy or a blind test.

#### Hydrofoil vapor-cloud detection

**Problem:** Detect attached cavities and disconnected vapor clouds around a hydrofoil.<br>
**CFD / data:** The author's existing Fluent fields: 158 original alpha-input
snapshots, plus 16 shared moving-case snapshots for comparing the methods.<br>
**Methods:** Alpha-input context U-Net, fixed pressure threshold, pressure-only
3×3 model, pressure U-Net and pressure topology U-Net. The notebook reruns the
original saved models, including all three pressure-model seeds.

![Same-frame hydrofoil vapor-cloud detection with alpha and pressure methods](../../results/week11_cavitation/methods_Plunging3.png)

All panels show the same CFD field, geometry and time. Orange/magenta denote
attached/disconnected classes; green denotes total cavity for methods without
a topology output. Pressure models receive no alpha input. Colored contours
have white contrast halos; arrows mark erroneous vapor predictions
inside the solid hydrofoil. The displayed Dice excludes solid/uncertain support.
The illustrated frame has the largest valid CFD cavity area in this retained trajectory and
uses the first seed (11); the notebook includes all 16 frames and three seeds.
These inspected nontraining cases use algorithmic weak references and different
input information, so this is not a blind or matched-input accuracy ranking.

[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb)
· [Notebook](../../notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb)
· [Detection and training code](../../flowmllab/cavitation_detection.py)
· [Alpha/pressure comparison code](../../flowmllab/cavitation_methods.py)
· [Results and provenance](../../results/week11_cavitation/README.md)
· [Expanded Lecture 11](../../lectures/week11_shock_vortex_identification.pdf)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 10.1](../week10_1/README.md) · [Week 12 →](../week12/README.md)
