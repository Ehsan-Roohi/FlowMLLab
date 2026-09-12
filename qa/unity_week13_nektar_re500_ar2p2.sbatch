#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=nektar-r500-ar2p2
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=nektar-r500-ar2p2-%j.log

set -euo pipefail
: "${FLOWML_NEKTAR_ROOT:?Nektar runtime root required}"
: "${FLOWML_NEKTAR_RUNS:?Fresh campaign root required}"
: "${FLOWML_NEKTAR_EQUIVALENCE_GATE:?Reviewed restart gate required}"
repo="${SLURM_SUBMIT_DIR:?}"
test -s "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif"
test -s "$FLOWML_NEKTAR_EQUIVALENCE_GATE"
commit="$(git -C "$repo" rev-parse HEAD)"
snapshot="$FLOWML_NEKTAR_RUNS/code"
mkdir -p "$FLOWML_NEKTAR_RUNS"
if [[ ! -d "$snapshot" ]]; then
  mkdir "$snapshot"
  cp "$repo/qa/prepare_week13_nektar.py" "$snapshot/"
  cp "$repo/qa/run_week13_nektar_reynolds.py" "$snapshot/"
  cp "$FLOWML_NEKTAR_EQUIVALENCE_GATE" "$snapshot/restart-gate.json"
  printf '%s\n' "$commit" > "$snapshot/code-commit.txt"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +e
python3 "$snapshot/run_week13_nektar_reynolds.py" \
  --output "$FLOWML_NEKTAR_RUNS/re500-ar2p2-p6-n24x54" \
  --image "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" \
  --gate "$snapshot/restart-gate.json" --commit "$commit" \
  --re 500 --depth 2.2 --corner-convention stationary_endpoints \
  --order 6 --dt 0.00025 --end-time "${FLOWML_NEKTAR_END_TIME:-120}" \
  --nx 24 --ny 54 --seconds "${FLOWML_NEKTAR_SECONDS:-6600}"
status=$?
set -e
if (( status == 75 )); then
  if (( ${SLURM_RESTART_COUNT:-0} >= 40 )); then
    echo 'Requeue budget exhausted; review required' >&2
    exit 1
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
