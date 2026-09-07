#!/bin/bash
#SBATCH --job-name=flowml-w13-pinn
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
#SBATCH --array=0-3
#SBATCH --output=week13-pinn-%A_%a.log

set -euo pipefail
: "${FLOWML_PINN_PYTHON:?Set FLOWML_PINN_PYTHON}"
: "${DEEPPLASMA_LDC:?Set DEEPPLASMA_LDC}"
: "${FLOWML_PINN_DEPS:?Set FLOWML_PINN_DEPS}"
: "${FLOWML_PINN_REFERENCE:?Set FLOWML_PINN_REFERENCE}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="$FLOWML_PINN_DEPS${PYTHONPATH:+:$PYTHONPATH}"

cases=("100 1" "400 1" "100 2" "400 2")
read -r re aspect <<< "${cases[${SLURM_ARRAY_TASK_ID:?}]}"
run_root="${SLURM_SUBMIT_DIR}/tmp/week13-pinn-re${re}-d${aspect}"

"$FLOWML_PINN_PYTHON" qa/check_week42_environment.py
set +e
"$FLOWML_PINN_PYTHON" qa/run_week13_rectangular_pinn.py \
  --source "$DEEPPLASMA_LDC" --reference-npz "$FLOWML_PINN_REFERENCE" \
  --output "$run_root" --re "$re" --aspect-ratio "$aspect" \
  --points "${FLOWML_PINN_POINTS:-16384}" \
  --adam-steps "${FLOWML_PINN_ADAM_STEPS:-1000}" \
  --ssb-steps "${FLOWML_PINN_SSB_STEPS:-1000}" --resume
status=$?
set -e
if [[ "$status" -eq 99 ]]; then
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
[[ "$status" -eq 0 ]] || exit "$status"
"$FLOWML_PINN_PYTHON" -m pip freeze > "$run_root/environment-lock.txt"
