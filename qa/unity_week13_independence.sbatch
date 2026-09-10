#!/bin/bash
#SBATCH --job-name=w13-re500-indep
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-7%3
#SBATCH --output=independence-%A_%a.log
set -euo pipefail
: "${FLOWML_INDEP_ROOT:?New campaign root required}"
: "${FLOWML_INDEP_COMMIT:?Pinned published commit required}"
repo="${SLURM_SUBMIT_DIR:?}"
test "$(git -C "$repo" rev-parse HEAD)" = "$FLOWML_INDEP_COMMIT"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
runtime=/project/pi_roohie_umass_edu/FlowMLLab_nektar_qualification_20260909
python=/project/pi_roohie_umass_edu/FlowMLLab_modal_qualification_20260905/.venv/bin/python
source_chunk=/project/pi_roohie_umass_edu/FlowMLLab_nektar_post200_20260910/re500-d5-p6-n16x80/chunk-0219
test -x "$python"
test -s "$runtime/nektar-v5.9.0.sif"
command -v apptainer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" -c 'import numpy, scipy; print("DEPENDENCIES", numpy.__version__, scipy.__version__)'
set +e
"$python" "$repo/qa/run_week13_independence.py" \
  --case "$SLURM_ARRAY_TASK_ID" --output "$FLOWML_INDEP_ROOT" \
  --source-chunk "$source_chunk" --image "$runtime/nektar-v5.9.0.sif" \
  --commit "$FLOWML_INDEP_COMMIT" --seconds 6600
status=$?
set -e
if (( status == 75 )); then
  if (( ${SLURM_RESTART_COUNT:-0} >= 24 )); then
    echo 'Requeue budget exhausted; retained checkpoint needs review' >&2
    exit 1
  fi
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit "$status"
