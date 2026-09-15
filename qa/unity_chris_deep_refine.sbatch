#!/usr/bin/env bash
#SBATCH --job-name=chris-re100-d5-refine
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --requeue
#SBATCH --mail-type=NONE
#SBATCH --signal=B:USR1@180
#SBATCH --output=chris-re100-d5-refine-%j.log
set -euo pipefail
: "${CHRIS_ENV_ROOT:?}"
: "${CHRIS_BASE_CHECKPOINT:?}"
: "${CHRIS_REFINE_OUTPUT:?}"
anchors="${CHRIS_LOWER_ANCHORS:-40000}"
adam_lr="${CHRIS_ADAM_LR:-1e-4}"
adam_steps="${CHRIS_ADAM_STEPS:-5000}"
ssb_steps="${CHRIS_SSB_STEPS:-18000}"
python="$CHRIS_ENV_ROOT/venv/bin/python"
site_packages="$($python -c 'import site; print(site.getsitepackages()[0])')"
cuda_root="$site_packages/nvidia/cuda_nvcc"
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_root"
export LD_LIBRARY_PATH="$(find "$site_packages/nvidia" -type d -name lib -printf '%p:' | sed 's/:$//')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
refine_args=(
  --archive "$CHRIS_ENV_ROOT/chris-deep-original-inputs.zip"
  --output "$CHRIS_REFINE_OUTPUT" --case-index 0
  --refine-from "$CHRIS_BASE_CHECKPOINT"
  --refine-adam-lr "$adam_lr"
  --refine-adam-steps "$adam_steps"
  --refine-ssb-steps "$ssb_steps"
)
if (( anchors > 0 )); then
  refine_args+=(--lower-anchors "$anchors")
fi
set +e
"$python" qa/run_chris_deep_original.py "${refine_args[@]}"
status=$?
set -e
if (( status == 99 )); then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
