#!/bin/bash
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
: "${TASK_ROOT:?actual Workspace benchmark path required}"
: "${PYTHON_BIN:?verified Python environment required}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${1:?phase required} == pilot ]]; then
  exec "$PYTHON_BIN" -u "$TASK_ROOT/source/qa/run_cavity_solver_matrix.py" --phase pilot --cells 128 --root "$TASK_ROOT"
else
  CELLS=(256 512)
  exec "$PYTHON_BIN" -u "$TASK_ROOT/source/qa/run_cavity_solver_matrix.py" --phase production --cells "${CELLS[$SLURM_ARRAY_TASK_ID]}" --root "$TASK_ROOT" --selection "$TASK_ROOT/selection.json"
fi
