#!/usr/bin/env bash
# Chris-source formulation, Re=100 and D/W=5; capacity-only experiment.
# The private author archive stays on Unity and is never published here.
#SBATCH --job-name=chris-d5-w64-n200k
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:55:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --mail-type=NONE
#SBATCH --output=chris-d5-w64-n200k-%j.log
set -euo pipefail
: "${CHRIS_ENV_ROOT:?Set the existing private environment root}"
: "${CHRIS_RUN_OUTPUT:?Set a new, dedicated output directory}"
cd "${SLURM_SUBMIT_DIR:?}"
python="$CHRIS_ENV_ROOT/venv/bin/python"
archive="$CHRIS_ENV_ROOT/chris-deep-original-inputs.zip"
test -f "$archive"
test -x "$python"
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
site_packages="$($python -c 'import site; print(site.getsitepackages()[0])')"
cuda_root="$site_packages/nvidia/cuda_nvcc"
test -f "$cuda_root/nvvm/libdevice/libdevice.10.bc"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_root"
export LD_LIBRARY_PATH="$(find "$site_packages/nvidia" -type d -name lib -printf '%p:' | sed 's/:$//')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
child=""
trap '[[ -n "$child" ]] && kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
set +e
"$python" qa/run_chris_deep_original.py \
  --archive "$archive" --output "$CHRIS_RUN_OUTPUT" \
  --case-index 0 --hidden-width 64 --num-domain 200000 &
child=$!
wait "$child"
status=$?
set -e
if (( status == 99 )); then
  scontrol requeue "${SLURM_JOB_ID:?}"
  exit 0
fi
exit "$status"

