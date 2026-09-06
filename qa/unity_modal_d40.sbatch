#!/bin/bash
# Submit from the isolated, verified FlowMLLab checkout on Unity.
#SBATCH --job-name=flowml-d40
#SBATCH --partition=cpu
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=flowml-d40-%j.log

set -euo pipefail
: "${FLOWML_PYTHON:?Set FLOWML_PYTHON to the verified absolute Python executable}"
[[ "$FLOWML_PYTHON" = /* && -x "$FLOWML_PYTHON" ]]
cd "${SLURM_SUBMIT_DIR:?}"
test -f qa/qualify_modal_cfd.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
RUN_ROOT="${SLURM_SUBMIT_DIR}/tmp/unity-d40-${SLURM_JOB_ID:?}"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
"$FLOWML_PYTHON" -c 'import sys,numpy,pandas,matplotlib; print(sys.executable); print(numpy.__version__,pandas.__version__)'
"$FLOWML_PYTHON" qa/qualify_modal_cfd.py plan > "$RUN_ROOT/declared_protocol.json"
"$FLOWML_PYTHON" qa/qualify_modal_cfd.py smoke --output "$RUN_ROOT/smoke"
# Exit 2 means the completed scientific evidence failed the grid gate.
# Do not automatically resubmit, loosen thresholds, or publish results.
"$FLOWML_PYTHON" qa/qualify_modal_cfd.py run --output "$RUN_ROOT/production"
"$FLOWML_PYTHON" qa/qualify_modal_cfd.py inspect --output "$RUN_ROOT/production"
