#!/bin/bash
#SBATCH --job-name=FlowML-Ghia-highRe
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --chdir=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/source
#SBATCH --output=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/logs/steady-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TASK_ROOT=/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115
CELLS=(128 256 512)
exec "$TASK_ROOT/venv/bin/python" -u qa/run_cavity_highre.py --cells "${CELLS[$SLURM_ARRAY_TASK_ID]}" --seed "$TASK_ROOT/input/re1000-n128.npz" --output "$TASK_ROOT/results"
