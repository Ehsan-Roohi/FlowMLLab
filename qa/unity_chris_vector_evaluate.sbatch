#!/usr/bin/env bash
# Read-only snapshot evaluation of the running Re100 D/W5 width-64 PINN.
# The training job and private author source are not modified or published.
#SBATCH --job-name=chris-vector-eval
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=chris-vector-eval-%j.log
set -euo pipefail
: "${CHRIS_ENV_ROOT:?Set the existing private environment root}"
: "${CHRIS_RUN_OUTPUT:?Set the running training output directory}"
: "${CHRIS_VECTOR_EVAL:?Set a new, dedicated evaluation root}"
cd "${SLURM_SUBMIT_DIR:?}"
python="$CHRIS_ENV_ROOT/venv/bin/python"
test -x "$python"
test -f "$CHRIS_RUN_OUTPUT/phase-1-vector.npy"
test ! -e "$CHRIS_VECTOR_EVAL"
mkdir -p "$CHRIS_VECTOR_EVAL"
cp -- "$CHRIS_RUN_OUTPUT/phase-1-vector.npy" "$CHRIS_VECTOR_EVAL/phase-vector.npy"
sha256sum "$CHRIS_VECTOR_EVAL/phase-vector.npy" > "$CHRIS_VECTOR_EVAL/vector.sha256"
site_packages="$($python -c 'import site; print(site.getsitepackages()[0])')"
cuda_root="$site_packages/nvidia/cuda_nvcc"
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_root"
export LD_LIBRARY_PATH="$(find "$site_packages/nvidia" -type d -name lib -printf '%p:' | sed 's/:$//')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$python" qa/chris_vector_to_checkpoint.py \
  --source "$CHRIS_RUN_OUTPUT/source/CavityTrapREDepthSSB20.py" \
  --vector "$CHRIS_VECTOR_EVAL/phase-vector.npy" \
  --parameters "$CHRIS_RUN_OUTPUT/case-parameters.json" \
  --provenance "$CHRIS_RUN_OUTPUT/provenance.json" \
  --hidden-width 64 --re 100 --depth 5 --tri 0 \
  --output "$CHRIS_VECTOR_EVAL/evaluation"
