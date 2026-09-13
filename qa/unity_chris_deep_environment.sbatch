#!/bin/bash
#SBATCH --job-name=chris-deep-tf-check
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=chris-deep-env-%j.log
set -euo pipefail
# Isolated dependency qualification only: this does not train a square cavity.
: "${CHRIS_BASE_PYTHON:?Set a Python with TensorFlow 2.15.1 installed}"
: "${CHRIS_ENV_ROOT:?Set an absolute private output directory}"
base="$CHRIS_BASE_PYTHON"
root="$CHRIS_ENV_ROOT"
[[ "$root" = /* ]] || { echo 'CHRIS_ENV_ROOT must be absolute'; exit 2; }
mkdir -p "$root"
test -x "$base"
if [[ ! -x "$root/venv/bin/python" ]]; then
    "$base" -m venv --system-site-packages "$root/venv"
fi
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$root/venv/bin/python" -m pip install 'numpy==1.26.4' 'scipy==1.13.1' 'deepxde==1.11.0' 'tensorflow[and-cuda]==2.15.1'
# Make the pip-installed CUDA shared libraries visible before importing TF.
cuda_libs=$("$root/venv/bin/python" - <<'PY'
import glob
import site
print(':'.join(path for base in site.getsitepackages()
               for path in glob.glob(base + '/nvidia/*/lib')))
PY
)
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvidia-smi
"$root/venv/bin/python" - <<'PY'
import json
import os
import numpy as np
import scipy
import tensorflow as tf
import deepxde as dde
from pathlib import Path
assert np.__version__ == '1.26.4'
assert scipy.__version__ == '1.13.1'
assert dde.__version__ == '1.11.0'
assert tf.__version__ == '2.15.1', tf.__version__
assert dde.backend.backend_name == 'tensorflow.compat.v1'
gpus = tf.config.list_physical_devices('GPU')
assert gpus, 'No TensorFlow GPU detected; do not start training.'
dde.config.set_default_float('float64')
v1 = dde.backend.tf
with v1.device('/GPU:0'):
    x = v1.constant([[1., 2.], [3., 4.]], dtype=v1.float64)
    y = v1.matmul(x, x)
with v1.Session(config=v1.ConfigProto(allow_soft_placement=False)) as sess:
    value = sess.run(y)
np.testing.assert_allclose(value, [[7., 10.], [15., 22.]], rtol=0, atol=0)
report = dict(tensorflow=tf.__version__, numpy=np.__version__, scipy=scipy.__version__,
              deepxde=dde.__version__, gpu=[str(g) for g in gpus], fp64_matmul='PASS',
              status='ENVIRONMENT_ONLY', training_started=False,
              remaining='Verify original SSBroyden2 dependency and deep source before training')
Path(os.environ['CHRIS_ENV_ROOT'], 'environment.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
