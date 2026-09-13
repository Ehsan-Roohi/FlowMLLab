#!/bin/bash
#SBATCH --job-name=chris-original-r1000
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=chris-original-%j.log
set -euo pipefail
: "${CHRIS_PYTHON:?}" "${CHRIS_SOURCE:?}" "${CHRIS_OUTPUT:?}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
child=""
trap '[[ -n "$child" ]] && kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
"$CHRIS_PYTHON" qa/run_chris_original.py --source "$CHRIS_SOURCE" --output "$CHRIS_OUTPUT" &
child=$!
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; status=$?; fi
set -e
if [[ "$status" == 99 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
