#!/bin/bash
#SBATCH --job-name=w13-cfd-reference
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --constraint=avx512
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --array=0-47%4
#SBATCH --output=week13-cfd-%A_%a.log

set -eo pipefail
module load uri/main OpenFOAM/v2406-foss-2023a
source "${FOAM_BASH:?}"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?}"
: "${FLOWML_CFD_RUNS:?Set an absolute output directory}"
[[ "$FLOWML_CFD_RUNS" = /* ]] || exit 2
task="${SLURM_ARRAY_TASK_ID:?}"
suffix="production"
extra=()
if [[ "${FLOWML_CFD_SENSITIVITY:-0}" == 1 ]]; then
    extra+=(--sensitivity)
fi
if [[ "${FLOWML_CFD_SMOKE:-0}" == 1 ]]; then
    suffix="smoke"
    extra+=(--smoke)
fi
case_dir="$FLOWML_CFD_RUNS/${suffix}-${task}"
python3 qa/prepare_week13_cfd.py --output "$case_dir" --task "$task" \
    --ranks "$SLURM_NTASKS" "${extra[@]}"
cd "$case_dir"
echo "OpenFOAM ${WM_PROJECT_VERSION:?} precision=${WM_PRECISION_OPTION:?}"
git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD
command -v simpleFoam
if [[ ! -f mesh-ready ]]; then
    blockMesh > log.blockMesh 2>&1
    checkMesh > log.checkMesh 2>&1
    grep -q 'Mesh OK' log.checkMesh
    decomposePar > log.decomposePar 2>&1
    touch mesh-ready
fi
foamDictionary system/controlDict -entry stopAt -set endTime
if [[ -n "${FLOWML_CFD_END_TIME:-}" ]]; then
    [[ "$FLOWML_CFD_END_TIME" =~ ^[0-9]+$ ]] || exit 2
    (( FLOWML_CFD_END_TIME >= 20000 && FLOWML_CFD_END_TIME <= 120000 )) || exit 2
    foamDictionary system/controlDict -entry endTime -set "$FLOWML_CFD_END_TIME"
fi
stopping=0
checkpoint() {
    stopping=1
    foamDictionary system/controlDict -entry stopAt -set writeNow
}
trap checkpoint USR1 TERM
runlog="log.simpleFoam.${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
# Single-node jobs do not need the OFI provider (libfabric init crashed on URI).
mpirun --mca pml ob1 --mca btl self,vader,tcp -np "$SLURM_NTASKS" simpleFoam -parallel > "$runlog" 2>&1 &
child=$!
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then
    wait "$child"
    status=$?
fi
set -e
[[ "$status" == 0 ]] || exit "$status"
grep -q '^End' "$runlog"
if [[ "$stopping" == 1 ]]; then
    if (( ${SLURM_RESTART_COUNT:-0} >= 20 )); then
        echo 'Checkpoint retained; automatic restart budget exhausted.'
        exit 3
    fi
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
reconstructPar -latestTime > log.reconstructPar 2>&1
echo 'Solver finished. This is NOT a validated CFD reference; convergence, grid and vortex audits remain.'
touch solver-finished-unvalidated
