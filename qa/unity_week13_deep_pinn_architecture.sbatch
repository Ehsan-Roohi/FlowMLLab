#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=flowml-pinn-arch
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=pinn-arch-%A_%a.log

set -euo pipefail
: "${FLOWML_PINN_PYTHON:?Set FLOWML_PINN_PYTHON}"
: "${DEEPPLASMA_LDC:?Set DEEPPLASMA_LDC}"
: "${FLOWML_PINN_DEPS:?Set FLOWML_PINN_DEPS}"
: "${FLOWML_PINN_OUTPUT:?Set FLOWML_PINN_OUTPUT}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="$FLOWML_PINN_DEPS${PYTHONPATH:+:$PYTHONPATH}"

activations=(tanh silu)
activation="${activations[${SLURM_ARRAY_TASK_ID:?}]}"
run_root="$FLOWML_PINN_OUTPUT/re500-d5-w128-${activation}-seed1234"

set +e
"$FLOWML_PINN_PYTHON" qa/run_week13_rectangular_pinn.py \
  --source "$DEEPPLASMA_LDC" --output "$run_root" \
  --re 500 --aspect-ratio 5 --seed 1234 --points 65535 \
  --hidden-width 128 --hidden-layers 3 --activation "$activation" \
  --adam-steps 5000 --ssb-steps 12000 --checkpoint-every 100 --resume
status=$?
set -e
if [[ "$status" -eq 99 ]]; then
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
exit "$status"
