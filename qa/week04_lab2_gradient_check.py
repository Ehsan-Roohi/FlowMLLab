"""Small numerical checks of the actual TensorFlow boundary construction."""
from pathlib import Path
import os, sys, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import tensorflow as tf
from flowmllab.cavity_lab2 import make_network, NetworkConfig

raw = make_network(3, 2, NetworkConfig(widths=(4, 4), seed=41))
aware = make_network(3, 2, NetworkConfig(widths=(4, 4), boundary_aware=True, boundary_lifting="corner", seed=41))
initial_match = max(float(np.max(np.abs(a-b))) for a, b in zip(raw.get_weights(), aware.get_weights()))
assert initial_match == 0
features = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 1.],
                     [0., .5, 1.], [0., 0., .5], [0., 1., .5], [0., .5, 0.]])
expected = np.zeros((len(features), 2)); expected[4, 0] = 1
wall_error = float(np.max(np.abs(np.asarray(aware(features))-expected)))
assert wall_error < 1e-14
inside = tf.constant([[.2, .3, .7], [-.2, .6, .4]], tf.float64)
target = tf.constant([[.12, -.02], [.04, .03]], tf.float64)
with tf.GradientTape() as tape:
    value = tf.reduce_mean((aware(inside)-target)**2)
gradient = tape.gradient(value, aware.trainable_variables)[0].numpy()
variable = aware.trainable_variables[0]
original = variable.numpy()
worst = 0.
for i, j in [(0, 0), (1, 2), (2, 3)]:
    plus = original.copy(); plus[i, j] += 1e-6
    variable.assign(plus); fp = float(tf.reduce_mean((aware(inside)-target)**2))
    minus = original.copy(); minus[i, j] -= 1e-6
    variable.assign(minus); fm = float(tf.reduce_mean((aware(inside)-target)**2))
    numerical = (fp-fm)/2e-6
    worst = max(worst, abs(numerical-gradient[i, j]))
variable.assign(original)
assert worst < 1e-8
receipt = {"tensorflow": tf.__version__, "boundary_lifting": "corner", "same_initial_trainable_weights_max_difference": initial_match,
           "exact_boundary_max_error": wall_error, "three_weight_central_difference_max_error": worst,
           "step": 1e-6, "passed": True}
(ROOT/"qa/week04_lab2_gradient_check.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
