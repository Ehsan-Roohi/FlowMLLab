#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=pinn-r500-ar2p2-v2
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=pinn-r500-ar2p2-v2-%j.log

set -euo pipefail
: "${FLOWML_PINN_PYTHON:?}"
: "${DEEPPLASMA_LDC:?}"
: "${FLOWML_PINN_DEPS:?}"
: "${FLOWML_PINN_OUTPUT:?}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="$FLOWML_PINN_DEPS${PYTHONPATH:+:$PYTHONPATH}"
run_root="$FLOWML_PINN_OUTPUT/re500-ar2p2-seed1234-validation-v2"
child=""
trap '[[ -n "$child" ]] && kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
"$FLOWML_PINN_PYTHON" qa/run_week13_rectangular_pinn.py \
  --source "$DEEPPLASMA_LDC" --output "$run_root" --re 500 --aspect-ratio 2.2 \
  --seed 1234 --points 65536 --hidden-width 80 --hidden-layers 4 --activation tanh \
  --adam-steps 10000 --ssb-steps 20000 --checkpoint-every 100 --resume &
child=$!; wait "$child"; status=$?
set -e
if [[ "$status" == 99 ]]; then scontrol requeue "${SLURM_JOB_ID:?}"; exit 0; fi
exit "$status"
