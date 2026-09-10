#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=w13-cfd-continue
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --constraint=avx512
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --array=34,35,37,38,40,41,43,44,46,47%4
#SBATCH --output=week13-continue-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${FLOWML_CFD_INITIAL:?Set initial run directory}"
: "${FLOWML_CFD_RUNS:?Set distinct continuation directory}"
python3 qa/prepare_week13_continuation.py \
 --source "$FLOWML_CFD_INITIAL/production-${SLURM_ARRAY_TASK_ID:?}" \
 --destination "$FLOWML_CFD_RUNS/production-${SLURM_ARRAY_TASK_ID:?}"
export FLOWML_CFD_END_TIME=120000 FLOWML_CFD_SMOKE=0
# exec keeps the tested signal/checkpoint handler in the batch-shell PID.
exec bash qa/unity_week13_cfd.sbatch
