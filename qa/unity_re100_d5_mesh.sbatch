#!/bin/bash
#SBATCH --job-name=re100-d5-nektar-mesh
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --mail-type=NONE
#SBATCH --array=0-2%2
#SBATCH --output=re100-d5-nektar-%A_%a.log
set -euo pipefail
: "${FLOWML_REFINE_ROOT:?}"
: "${FLOWML_REFINE_COMMIT:?}"
: "${FLOWML_SOURCE_CHUNK:?}"
test "$(git rev-parse HEAD)" = "$FLOWML_REFINE_COMMIT"
runtime=/project/pi_roohie_umass_edu/FlowMLLab_nektar_qualification_20260909
python=/project/pi_roohie_umass_edu/FlowMLLab_modal_qualification_20260905/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
meshes=(16 32 64)
nx=${meshes[${SLURM_ARRAY_TASK_ID:?}]}
set +e
"$python" qa/run_week13_nektar_refinement.py --output "$FLOWML_REFINE_ROOT/n${nx}" \
 --source-chunk "$FLOWML_SOURCE_CHUNK" --image "$runtime/nektar-v5.9.0.sif" \
 --gate "$runtime/restart-gate-64127917.json" --commit "$FLOWML_REFINE_COMMIT" \
 --re 100 --nx "$nx" --ny "$((5*nx))" --end-time 120 --seconds 6600
status=$?
set -e
if (( status == 75 )); then
  (( ${SLURM_RESTART_COUNT:-0} < 30 )) || exit 1
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit "$status"
