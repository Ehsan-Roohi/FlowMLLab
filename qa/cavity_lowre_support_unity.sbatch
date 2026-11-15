#!/bin/bash
#SBATCH --job-name=FlowML-Ghia-lowSupport
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --chdir=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/source
#SBATCH --output=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/logs/lowsupport-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TASK_ROOT=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115
REYNOLDS=(500 750)
exec "$TASK_ROOT/venv/bin/python" -u qa/run_cavity_lowre_support.py --root "$TASK_ROOT" --re "${REYNOLDS[$SLURM_ARRAY_TASK_ID]}"
