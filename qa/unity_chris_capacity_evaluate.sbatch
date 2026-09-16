#!/usr/bin/env bash
# Evaluate the retained Re=100, D/W=5, width-64 Chris-source checkpoint.
# The author-supplied source stays private on Unity.
#SBATCH --job-name=chris-d5-w64-eval
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=chris-d5-w64-eval-%j.log
set -euo pipefail
: "${CHRIS_ENV_ROOT:?Set the existing private environment root}"
: "${CHRIS_RUN_OUTPUT:?Set the retained training output directory}"
cd "${SLURM_SUBMIT_DIR:?}"
python="$CHRIS_ENV_ROOT/venv/bin/python"
run="$CHRIS_RUN_OUTPUT"
site_packages="$($python -c 'import site; print(site.getsitepackages()[0])')"
cuda_root="$site_packages/nvidia/cuda_nvcc"
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_root"
export LD_LIBRARY_PATH="$(find "$site_packages/nvidia" -type d -name lib -printf '%p:' | sed 's/:$//')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$python" qa/evaluate_chris_deep_original.py \
  --source "$run/source/CavityTrapREDepthSSB20.py" \
  --checkpoint "$run/model/restart-112.ckpt" \
  --parameters "$run/case-parameters.json" \
  --provenance "$run/provenance.json" \
  --hidden-width 64 --re 100 --depth 5 --tri 0 \
  --nx 201 --ny 1001 --output "$run/evaluation"

