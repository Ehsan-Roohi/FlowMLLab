#!/usr/bin/env bash
#SBATCH --job-name=chris-deep-eval
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=chris-deep-eval-%j.log

set -euo pipefail
: "${CHRIS_ENV_ROOT:?Set CHRIS_ENV_ROOT to the private environment/result root}"
: "${CHRIS_CHECKPOINT:?Set CHRIS_CHECKPOINT to a complete checkpoint prefix}"

python="$CHRIS_ENV_ROOT/venv/bin/python"
site_packages="$($python -c 'import site; print(site.getsitepackages()[0])')"
cuda_root="$site_packages/nvidia/cuda_nvcc"
export DDE_BACKEND=tensorflow.compat.v1
export MPLBACKEND=Agg
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_root"
export LD_LIBRARY_PATH="$(find "$site_packages/nvidia" -type d -name lib -printf '%p:' | sed 's/:$//')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$python" qa/evaluate_chris_deep_original.py \
  --source "$CHRIS_ENV_ROOT/training/source/CavityTrapREDepthSSB20.py" \
  --checkpoint "$CHRIS_CHECKPOINT" \
  --output "$CHRIS_ENV_ROOT/evaluation/re1000-d2p2-tri0" \
  --re 1000 --depth 2.2 --tri 0 --nx 301 --ny 661
