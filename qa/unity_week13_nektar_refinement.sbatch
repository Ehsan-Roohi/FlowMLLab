#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=w13-nektar-refine
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-1%2
#SBATCH --output=nektar-refinement-%A_%a.log
set -euo pipefail
: "${FLOWML_REFINE_ROOT:?Fresh campaign root required}"
: "${FLOWML_REFINE_COMMIT:?Pinned published commit required}"
repo="${SLURM_SUBMIT_DIR:?}"
test "$(git -C "$repo" rev-parse HEAD)" = "$FLOWML_REFINE_COMMIT"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
runtime=/project/pi_roohie_umass_edu/FlowMLLab_nektar_qualification_20260909
source_root=/project/pi_roohie_umass_edu/FlowMLLab_nektar_post200_20260910
python=/project/pi_roohie_umass_edu/FlowMLLab_modal_qualification_20260905/.venv/bin/python
test -x "$python"
test -s "$runtime/nektar-v5.9.0.sif"
command -v apptainer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$python" -c 'import numpy, scipy'
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) re=1000; nx=32; ny=160; chunk=0259; end=360; label=re1000-fine ;;
  1) re=1000; nx=16; ny=80; chunk=0259; end=360; label=re1000-coarse-control ;;
  *) echo 'Unsupported array task; only the Re1000 pair is authorized' >&2; exit 2 ;;
esac
set +e
"$python" "$repo/qa/run_week13_nektar_refinement.py" \
  --output "$FLOWML_REFINE_ROOT/$label" \
  --source-chunk "$source_root/re${re}-d5-p6-n16x80/chunk-$chunk" \
  --image "$runtime/nektar-v5.9.0.sif" \
  --gate "$runtime/restart-gate-64127917.json" \
  --commit "$FLOWML_REFINE_COMMIT" --re "$re" --nx "$nx" --ny "$ny" \
  --end-time "$end" --seconds 6600
status=$?
set -e
if (( status == 75 )); then
  if (( ${SLURM_RESTART_COUNT:-0} >= 30 )); then
    echo 'Requeue budget exhausted; review required' >&2
    exit 1
  fi
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit "$status"
