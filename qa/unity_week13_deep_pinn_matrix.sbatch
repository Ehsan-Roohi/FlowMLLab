#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=flowml-deep-pinn
#SBATCH --partition=gpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --array=0-4%2
#SBATCH --output=deep-pinn-%A_%a.log

set -euo pipefail
: "${FLOWML_PINN_PYTHON:?Set FLOWML_PINN_PYTHON}"
: "${DEEPPLASMA_LDC:?Set DEEPPLASMA_LDC}"
: "${FLOWML_PINN_DEPS:?Set FLOWML_PINN_DEPS}"
: "${FLOWML_PINN_REFERENCE:?Set FLOWML_PINN_REFERENCE}"
: "${FLOWML_PINN_OUTPUT:?Set FLOWML_PINN_OUTPUT}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="$FLOWML_PINN_DEPS${PYTHONPATH:+:$PYTHONPATH}"

# Exactly the deep geometries/Reynolds numbers for which matched CFD campaigns
# exist.  This first array uses one declared seed; additional seeds are a
# separate replication stage and must never overwrite these directories.
cases=("100 5" "500 5" "1000 5" "500 7" "1000 7")
read -r re aspect <<< "${cases[${SLURM_ARRAY_TASK_ID:?}]}"
seed="${FLOWML_PINN_SEED:-1234}"
run_root="$FLOWML_PINN_OUTPUT/re${re}-d${aspect}-seed${seed}"

"$FLOWML_PINN_PYTHON" qa/check_week42_environment.py
set +e
"$FLOWML_PINN_PYTHON" qa/run_week13_rectangular_pinn.py \
  --source "$DEEPPLASMA_LDC" --reference-npz "$FLOWML_PINN_REFERENCE" \
  --output "$run_root" --re "$re" --aspect-ratio "$aspect" --seed "$seed" \
  --points "${FLOWML_PINN_POINTS:-32768}" \
  --adam-steps "${FLOWML_PINN_ADAM_STEPS:-5000}" \
  --ssb-steps "${FLOWML_PINN_SSB_STEPS:-10000}" --resume
status=$?
set -e
if [[ "$status" -eq 99 ]]; then
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
[[ "$status" -eq 0 ]] || exit "$status"
"$FLOWML_PINN_PYTHON" -m pip freeze > "$run_root/environment-lock.txt"
