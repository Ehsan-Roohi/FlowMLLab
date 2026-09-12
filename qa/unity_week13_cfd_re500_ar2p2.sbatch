#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=cfd-r500-ar2p2
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --constraint=avx512
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --array=0-2%3
#SBATCH --output=cfd-r500-ar2p2-%A_%a.log

set -euo pipefail
module load uri/main OpenFOAM/v2406-foss-2023a
source "${FOAM_BASH:?}"
: "${FLOWML_CFD_RUNS:?Set an absolute output directory}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The appended Re=500,D/W=2.2 case preserves all historical task mappings.
task=$((48 + ${SLURM_ARRAY_TASK_ID:?}))
grids=(80 120 180)
case_dir="$FLOWML_CFD_RUNS/re500-ar2p2-n${grids[${SLURM_ARRAY_TASK_ID}]}"
python3 qa/prepare_week13_cfd.py --output "$case_dir" --task "$task" --ranks "$SLURM_NTASKS"
cd "$case_dir"
if [[ ! -f mesh-ready ]]; then
  blockMesh > log.blockMesh 2>&1
  checkMesh > log.checkMesh 2>&1
  grep -q 'Mesh OK' log.checkMesh
  decomposePar > log.decomposePar 2>&1
  touch mesh-ready
fi
foamDictionary system/controlDict -entry endTime -set "${FLOWML_CFD_END_TIME:-120000}"
stopping=0
checkpoint(){ stopping=1; foamDictionary system/controlDict -entry stopAt -set writeNow; }
trap checkpoint USR1 TERM
set +e
mpirun --mca pml ob1 --mca btl self,vader,tcp -np "$SLURM_NTASKS" simpleFoam -parallel > "log.simpleFoam.${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}" 2>&1 &
child=$!; wait "$child"; status=$?
set -e
[[ "$status" == 0 ]] || exit "$status"
if [[ "$stopping" == 1 ]]; then
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
reconstructPar -latestTime > log.reconstructPar 2>&1
touch solver-finished-unvalidated
