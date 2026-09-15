#!/bin/bash
#SBATCH --job-name=chris-Re100-depth-cont
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=chris-depth-cont-%j.log
set -euo pipefail
: "${CHRIS_ENV_ROOT:?}"
: "${CHRIS_RUN_OUTPUT:?}"
python="$CHRIS_ENV_ROOT/venv/bin/python"
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
site_packages=$("$python" -c 'import site; print(site.getsitepackages()[0])')
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$site_packages/nvidia/cuda_nvcc"
cuda_libs=$(find "$site_packages/nvidia" -type d -name lib -printf '%p:' | sed 's/:$//')
export LD_LIBRARY_PATH="$cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
child=""
trap '[[ -n "$child" ]] && kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
"$python" qa/run_chris_depth_continuation.py --archive "$CHRIS_ENV_ROOT/chris-deep-original-inputs.zip" --output "$CHRIS_RUN_OUTPUT" &
child=$!
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; status=$?; fi
set -e
if [[ "$status" == 99 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
