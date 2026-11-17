#!/bin/bash
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
set -euo pipefail
: "${TASK_ROOT:?allocated Workspace task directory required}"
: "${PYTHON_BIN:?verified environment required}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -u "$TASK_ROOT/source/qa/run_week01_2_literature_studies.py" --output "$TASK_ROOT/results"
