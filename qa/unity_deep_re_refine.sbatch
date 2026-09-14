#!/bin/bash
#SBATCH --job-name=deep-re-refine
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-type=NONE
#SBATCH --output=deep-re-refine-%j.log
set -euo pipefail
: "${FLOWML_SOURCE_CHUNK:?}" "${FLOWML_REFINE_ROOT:?}" "${FLOWML_REFINE_COMMIT:?}"
: "${FLOWML_RE:?}" "${FLOWML_NX:?}" "${FLOWML_END:?}"
test "$(git rev-parse HEAD)" = "$FLOWML_REFINE_COMMIT"
runtime=/project/pi_roohie_umass_edu/FlowMLLab_nektar_qualification_20260909
python=/project/pi_roohie_umass_edu/FlowMLLab_modal_qualification_20260905/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +e
"$python" qa/run_week13_nektar_refinement.py --output "$FLOWML_REFINE_ROOT" \
 --source-chunk "$FLOWML_SOURCE_CHUNK" --image "$runtime/nektar-v5.9.0.sif" \
 --gate "$runtime/restart-gate-64127917.json" --commit "$FLOWML_REFINE_COMMIT" \
 --re "$FLOWML_RE" --nx "$FLOWML_NX" --ny "$((5*FLOWML_NX))" \
 --end-time "$FLOWML_END" --seconds 6600
status=$?
set -e
if (( status == 75 )); then
 (( ${SLURM_RESTART_COUNT:-0} < 30 )) || exit 1
 scontrol requeue "$SLURM_JOB_ID"
 exit 0
fi
exit "$status"
