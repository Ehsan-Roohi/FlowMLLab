#!/bin/bash
#SBATCH --job-name=FlowML-Ghia-lowref
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --chdir=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/source
#SBATCH --output=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/logs/lowref-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TASK_ROOT=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115
REYNOLDS=(100 400 1000)
SEEDS=(re100-n64.npz re400-n128.npz re1000-n128.npz)
exec "$TASK_ROOT/venv/bin/python" -u qa/run_cavity_ml_reference.py --re "${REYNOLDS[$SLURM_ARRAY_TASK_ID]}" --seed "$TASK_ROOT/input/${SEEDS[$SLURM_ARRAY_TASK_ID]}" --output "$TASK_ROOT/results"
