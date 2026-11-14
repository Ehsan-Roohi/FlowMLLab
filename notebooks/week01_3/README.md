# Week 1.3 notebook setup

[Week 1.3 guide](../../weeks/week01_3/README.md) · [Lecture PDF](../../lectures/week01_3_keras_optimizers.pdf) · [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_3/W1_3_Keras_Optimizers.ipynb)

Use a normal CPU kernel. The notebook generates its own targets and uses TensorFlow/Keras for the network and automatic differentiation, SciPy for L-BFGS-B, and NumPy and Matplotlib for arrays and figures. No FlowMLLab package, external dataset, or GPU is required.

In a clean Python environment, install the dependencies and open the notebook:

~~~sh
python -m pip install "tensorflow>=2.16,<2.22" "numpy>=2.0,<2.3" "scipy>=1.13,<1.16" matplotlib jupyter
jupyter notebook W1_3_Keras_Optimizers.ipynb
~~~

Run the notebook from the top in a fresh kernel. Complete the baseline first, then change a single setting per experiment. Keep the held-out test set for your final comparison.
