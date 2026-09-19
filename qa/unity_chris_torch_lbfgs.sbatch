#!/usr/bin/env bash
# GPU-native L-BFGS refinement of the retained Re=100, D/W=5 checkpoint.
#SBATCH --job-name=chris-d5-torch-lbfgs
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=chris-d5-torch-lbfgs-%j.log

set -euo pipefail
: "${CHRIS_ENV_ROOT:?Set the existing private environment root}"
: "${CHRIS_CHECKPOINT:?Set the complete TensorFlow checkpoint prefix}"
: "${CHRIS_OUTPUT:?Set a new output directory}"

cd "${SLURM_SUBMIT_DIR:?}"
python="$CHRIS_ENV_ROOT/venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for suffix in .index .meta .data-00000-of-00001; do
  test -f "$CHRIS_CHECKPOINT$suffix"
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$python" qa/run_chris_torch_lbfgs.py \
  --checkpoint "$CHRIS_CHECKPOINT" --output "$CHRIS_OUTPUT" \
  --width 64 --re 100 --depth 5 --taper 0 \
  --points "${CHRIS_POINTS:-25000}" \
  --iterations "${CHRIS_ITERATIONS:-400}" \
  ${CHRIS_PREFLIGHT:+--preflight}
