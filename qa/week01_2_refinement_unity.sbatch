#!/bin/bash
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
set -euo pipefail
: "${TASK_ROOT:?actual allocated Workspace task path required}"
: "${PYTHON_BIN:?verified Python environment required}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -u "$TASK_ROOT/source/qa/run_week01_2_refinement.py" \
  --case "$SLURM_ARRAY_TASK_ID" --output "$TASK_ROOT/results"
