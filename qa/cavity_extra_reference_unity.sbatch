#!/bin/bash
#SBATCH --job-name=FlowML-Ghia-MLcases
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --requeue
#SBATCH --chdir=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/source
#SBATCH --output=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/logs/mlcases-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TASK_ROOT=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115
REYNOLDS=(250 2500 4000 6000 6500 9000)
exec "$TASK_ROOT/venv/bin/python" -u qa/run_cavity_extra_reference.py --root "$TASK_ROOT" --re "${REYNOLDS[$SLURM_ARRAY_TASK_ID]}"
