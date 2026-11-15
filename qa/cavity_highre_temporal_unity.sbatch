#!/bin/bash
#SBATCH --job-name=FlowML-Ghia-timeaudit
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --requeue
#SBATCH --chdir=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/source
#SBATCH --output=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/logs/time-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TASK_ROOT=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115
REYNOLDS=(3200 3200 5000 5000 7500 7500 10000 10000)
STEPS=(0.002 0.001 0.002 0.001 0.002 0.001 0.002 0.001)
exec "$TASK_ROOT/venv/bin/python" -u qa/run_cavity_highre_temporal.py --root "$TASK_ROOT" --re "${REYNOLDS[$SLURM_ARRAY_TASK_ID]}" --dt "${STEPS[$SLURM_ARRAY_TASK_ID]}"
