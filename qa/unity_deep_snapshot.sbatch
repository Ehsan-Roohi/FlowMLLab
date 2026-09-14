#!/bin/bash
#SBATCH --job-name=deep-pinn-cfd-audit
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --mail-type=NONE
#SBATCH --output=deep-snapshot-%j.log
set -euo pipefail
: "${CHRIS_ENV_ROOT:?}" "${CHRIS_CASE_ROOT:?}" "${CHRIS_CHECKPOINT:?}"
: "${FLOWML_SOURCE_CHUNK:?}" "${FLOWML_MESH_ROOT:?}" "${FLOWML_AUDIT_OUTPUT:?}"
export DDE_BACKEND=tensorflow.compat.v1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
python="$CHRIS_ENV_ROOT/venv/bin/python"
"$python" qa/evaluate_chris_deep_original.py \
 --source "$CHRIS_CASE_ROOT/source/CavityTrapREDepthSSB20.py" \
 --parameters "$CHRIS_CASE_ROOT/case-parameters.json" \
 --checkpoint "$CHRIS_CHECKPOINT" --output "$FLOWML_AUDIT_OUTPUT/pinn" \
 --re 100 --depth 5 --nx 161 --ny 801
"$python" qa/compare_deep_snapshot.py --pinn "$FLOWML_AUDIT_OUTPUT/pinn/field.npz" \
 --chunk "$FLOWML_SOURCE_CHUNK" --output "$FLOWML_AUDIT_OUTPUT/comparison" \
 --foam "$FLOWML_MESH_ROOT/openfoam-continue/production-34" \
 --foam "$FLOWML_MESH_ROOT/openfoam-continue/production-35" \
 --foam "$FLOWML_MESH_ROOT/openfoam-270/production-0"
