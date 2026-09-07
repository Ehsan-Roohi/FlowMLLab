#!/bin/bash
#SBATCH --job-name=flowml-w42-pinn
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
#SBATCH --output=week42-pinn-%j.log

set -euo pipefail
: "${FLOWML_PINN_PYTHON:?Set FLOWML_PINN_PYTHON to the audited Python executable}"
: "${DEEPPLASMA_LDC:?Set DEEPPLASMA_LDC to the pinned LDC_module_square.py}"
: "${FLOWML_PINN_DEPS:?Set FLOWML_PINN_DEPS to the task-specific dependency directory}"
[[ "$FLOWML_PINN_PYTHON" = /* && -x "$FLOWML_PINN_PYTHON" ]]
[[ "$DEEPPLASMA_LDC" = /* && -f "$DEEPPLASMA_LDC" ]]
cd "${SLURM_SUBMIT_DIR:?}"
test -f qa/run_week42_deepplasma.py
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="$FLOWML_PINN_DEPS${PYTHONPATH:+:$PYTHONPATH}"
: "${FLOWML_PINN_RUN_ID:?Set a stable run identifier shared by requeues}"
: "${FLOWML_PINN_REFERENCE:?Set the absolute retained CFD NPZ path}"
RUN_ROOT="${SLURM_SUBMIT_DIR}/tmp/week42-pinn-${FLOWML_PINN_RUN_ID}"

# Dependency and hardware gate: failure stops before training.
"$FLOWML_PINN_PYTHON" qa/check_week42_environment.py
set +e
extra_args=()
[[ -z "${FLOWML_PINN_STEPS:-}" ]] || extra_args+=(--steps "$FLOWML_PINN_STEPS")
[[ "${FLOWML_PINN_RENDER_ONLY:-0}" != "1" ]] || extra_args+=(--render-only)
"$FLOWML_PINN_PYTHON" qa/run_week42_deepplasma.py \
  --mode "${FLOWML_PINN_MODE:-smoke}" \
  --source "$DEEPPLASMA_LDC" \
  --reference-npz "$FLOWML_PINN_REFERENCE" \
  --output "$RUN_ROOT" --resume "${extra_args[@]}"
runner_status=$?
set -e
if [[ "$runner_status" -eq 99 ]]; then
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
[[ "$runner_status" -eq 0 ]] || exit "$runner_status"
"$FLOWML_PINN_PYTHON" -m pip freeze > "$RUN_ROOT/environment-lock.txt"
