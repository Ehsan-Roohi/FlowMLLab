#!/bin/bash
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${TASK_ROOT:?actual Workspace benchmark path required}"
: "${PYTHON_BIN:?verified Python environment required}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
pids=()
for cells in 256 512; do
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --cpu-bind=verbose,cores \
    "$PYTHON_BIN" -u "$TASK_ROOT/source/qa/run_cavity_solver_matrix.py" \
    --phase production --cells "$cells" --root "$TASK_ROOT" --selection "$TASK_ROOT/selection.json" \
    > "$TASK_ROOT/logs/production-${SLURM_JOB_ID}-n${cells}.log" 2>&1 &
  pids+=("$!")
done
status=0
for child in "${pids[@]}"; do
  if wait "$child"; then :; else status=1; fi
done
exit "$status"