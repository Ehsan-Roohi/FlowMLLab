# Week 2: learn, train and inspect a fluid surrogate

[Open the student notebook in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb) | [Lecture PDF](../../lectures/week02_supervised_learning_rarefaction.pdf)

Run the notebook from top to bottom with its default settings first. The opening
route identifies each stage, its expected output and the evidence to submit.
The target is a synthetic teaching relation, not a measured or validated CFD field.

The lesson connects a hand-calculated neuron update to a repeated learning-rate
experiment, then checks feature/target shapes and training-only standardization.
The DNN controls expose learning rate, batch size, epoch budget, patience,
hidden-layer widths and seed. Parameter and update counts come from the actual
array shapes and controls.

Compare polynomial regression, nearest neighbors and the DNN using both errors
and fixed-parameter flow profiles against the noise-free synthetic reference.
The final investigation separates validation-based rate selection, final test
evaluation and the out-of-range experiment. Trying three rates with three seeds
means nine network fits and can take much longer than the single default fit.

## Reproduce the educational checks

```bash
python qa/improve_week02_student_path.py
python qa/validate_week02_student_path.py
```

The validator executes the NumPy/scikit-learn exercises and baselines, checks
the split, scaling, metric arithmetic and physical-sweep inputs, and audits the
TensorFlow cell syntax and settings. It does not claim to train the DNN without
a TensorFlow runtime. Run the full notebook in Colab to obtain its DNN results.
