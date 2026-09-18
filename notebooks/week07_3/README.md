# Week 7.3 - Self-supervised pretraining and label efficiency on the cylinder wake

[Notebook](W7_3_Masked_Pretraining_Label_Efficiency.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_3/W7_3_Masked_Pretraining_Label_Efficiency.ipynb) ·
[Lecture](../../lectures/week07_3_masked_pretraining.pdf) ·
[Retained evidence](../../results/week07_3_pretraining/README.md)

Prerequisites: Weeks 5 and 7 (POD, the wake data) and Week 7.2 (validation-only
selection). Allow 90 minutes. CPU only, PyTorch required (`pip install -e
.[reconstruction]` or the Colab bootstrap). About ten minutes of compute on two
cores (1 min pretraining, 4 min label-efficiency loop, 3 min budget control);
`QUICK = True` in the setup cell cuts it to about three.

## The question

Many wake fields are available but only a few are labelled for the case we care
about. Does pretraining a network on the unlabelled fields reduce the number of
labels the new case needs, and does the pretrained network beat a classical
method that receives exactly the same information? The protocol is that of MAPA
(Tang, Spalding and Cogan, 2026; masked autoencoding on intracranial
recordings, then a label-efficiency curve for a new subject), transplanted to
the FlowMLLab lattice-Boltzmann wake.

## What the notebook does

| Section | Content | What is fitted, on what |
| --- | --- | --- |
| 1 to 2 | Definitions (label, mask, MAE, zero-shot, probe, fine-tuning, gappy POD, matched baseline, label saving) and the frozen information contract | nothing |
| 3 | Gappy POD with a basis transferred from the unlabelled Re = 90 and 100 frames | rank selected on Re = 105 |
| 4 | A 90k-parameter masked autoencoder (4 x 6 patches, 75% masked, encoder sees visible patches only) pretrained on the 562 unlabelled frames with fresh masks every step | early stopping on Re = 105 |
| 5 | Zero-shot completion of the Re = 110 test window by both methods, field and error maps | nothing |
| 6 to 7 | For k = 1 to 128 labelled Re = 110 frames and three mask seeds: target-only POD, pooled POD, frozen-encoder ridge probe, the architecture from scratch, and the pretrained model fine-tuned; the label-efficiency curve and label savings | every choice on Re = 105 |
| 8 | Budget control: the architecture from scratch for the full pretraining budget at k = 2, 8 and 128 | early stopping on Re = 105 |
| 9 to 10 | Reading the result, limits, exercises, references | |

Roles are frozen before any number is looked at: Re = 90 and 100 are unlabelled
pretraining data; Re = 105 is the validation trajectory for every selection;
Re = 110 frames `[0, 160)` are the labelled pool and `[210, 281)` the test window,
scored with fixed random masks and never fitted.

## What the evidence supports

- With identical labelled frames and identical downstream compute, the
  fine-tuned pretrained model improves on its zero-shot start from k = 4
  (about 10% hidden-pixel error) and settles near 8% from k = 8 upwards, while
  the same architecture from scratch stays at the mean-field plateau; with the full pretraining budget the from-scratch model
  still fails at k = 2, comes within about a percentage point at k = 8 and at
  k = 128 is at least as good as the 300-step fine-tuned model. Pretraining
  pays most where labels are fewest.
- Gappy POD with a transferred basis and zero labels is an order of magnitude
  more accurate than the fine-tuned network, and the pooled basis improves
  with every label. On this periodic, low-rank wake the classical method with
  matched information wins; the notebook says why and when that would change.
- The frozen linear probe is weaker than the pretrained decoder for small k.

The numbers are in `results/week07_3_pretraining/metrics.json` and in the
lecture's table. They are three mask seeds on one educational LBM trajectory
whose target case was inspected in earlier weeks: a protocol demonstration,
not a blind benchmark.

## Rebuilding

- `python qa/build_week07_3_materials.py --execute` rewrites the notebook,
  executes it (PyTorch, about ten minutes) so that `results/week07_3_pretraining/`
  is refreshed, and rebuilds the lecture PDF from the retained metrics.
- `python qa/build_week07_3_materials.py --pdf-only` rebuilds the PDF alone.
- A plain Run All writes to a temporary folder and compares its numbers with
  the retained ones; the tracked data and evidence are never modified.
- `pytest -q tests/test_masked_pretraining.py` tests `flowmllab/masked_pretraining.py`.
