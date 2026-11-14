# FlowMLLab v1.11.0 — Optimizers for scientific machine learning

Week 1.3 follows the TensorFlow and Keras lab from Week 1. It adds an
[executed notebook](https://github.com/Ehsan-Roohi/FlowMLLab/blob/v1.11.0/notebooks/week01_3/W1_3_Keras_Optimizers.ipynb),
an [18-page lecture](https://github.com/Ehsan-Roohi/FlowMLLab/blob/v1.11.0/lectures/week01_3_keras_optimizers.pdf),
and six exercises with explicit deliverables.

- Compare Momentum, Adam, L-BFGS and Adam followed by L-BFGS on four targets with matched starting weights and a shared evaluation budget.
- Diagnose a loss jump, vary sample density and model capacity, and compare validation errors in original units.
- Compare the deep-cavity SOAP and self-scaled Broyden sequence against matched controls on a fifth target with rising frequency. The lab uses a small supervised regression and a dense CPU implementation. The retained example uses one paired seed; the exercises ask students to repeat it.
- The release also includes the existing updates to the Week 1 TensorFlow guide and Week 1.2 pressure-velocity coupling lab, and a linked set of matched circle and ellipse DSMC contour comparisons.

The Week 1.3 notebook was executed on CPU with Python 3.12, TensorFlow 2.20.0 and SciPy 1.15.3. It retains eight figures, and holds test scores behind an explicit switch. The archive series uses the [Zenodo concept DOI](https://doi.org/10.5281/zenodo.22074169).
