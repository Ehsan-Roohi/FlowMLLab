# Week 1.3 — Optimizers for Scientific Machine Learning with TensorFlow and Keras

[Course home](../../README.md) · [All weeks](../README.md) · [Week 1 foundations](../week01/README.md) · [Week 2 →](../week02/README.md)

[Lecture PDF](../../lectures/week01_3_keras_optimizers.pdf) · [Open the notebook](../../notebooks/week01_3/W1_3_Keras_Optimizers.ipynb) · [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_3/W1_3_Keras_Optimizers.ipynb) · [Local CPU setup](../../notebooks/week01_3/README.md)

Start after [TensorFlow for AI in Fluids](../../notebooks/week01/02_tensorflow_for_ai_fluids.ipynb). The lab returns to the same sine function, then uses harder analytic targets to make differences in training behavior easier to see.

1. Compare optimizers with the same network, starting weights, and samples.
2. Check how scaling affects the loss and the errors after transforming predictions back to the original units.
3. Compare the fitted functions, error across the interval, validation history, and computational work.
4. Change one model or training setting and explain the resulting tradeoff.

The chirp extension compares SOAP followed by self-scaled Broyden with matched controls. It connects the supervised lab to the staged methods in the [Week 13 cavity PINN lab](../week13/README.md).
