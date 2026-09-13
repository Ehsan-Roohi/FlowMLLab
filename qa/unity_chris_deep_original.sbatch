#!/bin/bash
#SBATCH --job-name=chris-deep-original
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=chris-deep-original-%j.log
set -euo pipefail
: "${CHRIS_ENV_ROOT:?}"
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python="$CHRIS_ENV_ROOT/venv/bin/python"
cuda_libs=$("$python" - <<'PY'
import glob, site
print(':'.join(p for b in site.getsitepackages() for p in glob.glob(b + '/nvidia/*/lib')))
PY
)
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cuda_data=$("$python" - <<'PY'
import pathlib, site
for base in site.getsitepackages():
    p = pathlib.Path(base) / 'nvidia/cuda_nvcc'
    if (p / 'nvvm/libdevice/libdevice.10.bc').is_file():
        print(p)
        break
else:
    raise SystemExit('CUDA libdevice missing; install TensorFlow CUDA extra')
PY
)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_data"
child=""
trap '[[ -n "$child" ]] && kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
"$python" qa/run_chris_deep_original.py --archive "$CHRIS_ENV_ROOT/chris-deep-original-inputs.zip" --output "$CHRIS_ENV_ROOT/training" &
child=$!
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; status=$?; fi
set -e
if [[ "$status" == 99 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
