#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=pinn-r1000-ar2p2
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
#SBATCH --output=pinn-r1000-ar2p2-%j.log

set -euo pipefail
: "${FLOWML_PINN_PYTHON:?Set FLOWML_PINN_PYTHON}"
: "${DEEPPLASMA_LDC:?Set DEEPPLASMA_LDC}"
: "${FLOWML_PINN_DEPS:?Set FLOWML_PINN_DEPS}"
: "${FLOWML_PINN_OUTPUT:?Set FLOWML_PINN_OUTPUT}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH="$FLOWML_PINN_DEPS${PYTHONPATH:+:$PYTHONPATH}"

child_pid=""
forward_checkpoint() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -USR1 "$child_pid"
  fi
}
trap forward_checkpoint USR1 TERM

run_stage() {
  local re="$1" init_checkpoint="$2"
  local run_root="$FLOWML_PINN_OUTPUT/re${re}-ar2p2-seed1234"
  if [[ -f "$run_root/COMPLETE" ]]; then
    echo "Re=$re already complete; skipping"
    return 0
  fi
  local init_args=()
  if [[ -n "$init_checkpoint" && ! -f "$run_root/checkpoint.pt" ]]; then
    init_args=(--init-checkpoint "$init_checkpoint")
  fi
  set +e
  "$FLOWML_PINN_PYTHON" qa/run_week13_rectangular_pinn.py \
    --source "$DEEPPLASMA_LDC" --output "$run_root" \
    --re "$re" --aspect-ratio 2.2 --seed 1234 --points 32768 \
    --hidden-width 50 --hidden-layers 3 --activation tanh \
    --adam-steps 5000 --ssb-steps 12000 --checkpoint-every 100 --resume \
    "${init_args[@]}" &
  child_pid=$!
  local status
  while true; do
    wait "$child_pid"
    status=$?
    if kill -0 "$child_pid" 2>/dev/null; then
      continue
    fi
    break
  done
  child_pid=""
  # Leave errexit disabled until the caller has captured this return status.
  # Re-enabling it here would terminate the batch shell on return 99 before
  # the outer block can request a Slurm requeue.
  return "$status"
}

previous=""
for re in 100 500 1000; do
  set +e
  run_stage "$re" "$previous"
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    if [[ "$status" -eq 99 ]]; then
      scontrol requeue "${SLURM_JOB_ID:?}"
      exit 0
    fi
    exit "$status"
  fi
  # Initialize the next Reynolds number from the fixed-holdout-selected model,
  # not the terminal quasi-Newton iterate.
  previous="$FLOWML_PINN_OUTPUT/re${re}-ar2p2-seed1234/selected-model.pt"
done

"$FLOWML_PINN_PYTHON" -m pip freeze > "$FLOWML_PINN_OUTPUT/environment-lock.txt"
