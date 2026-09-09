#!/bin/bash
#SBATCH --job-name=w13-nektar-Re
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-2%3
#SBATCH --output=nektar-reynolds-%A_%a.log
set -euo pipefail

: "${FLOWML_NEKTAR_ROOT:?Nektar image/runtime root required}"
: "${FLOWML_NEKTAR_PRODUCTION_ROOT:?Fresh campaign root required}"
: "${FLOWML_NEKTAR_EQUIVALENCE_GATE:?Reviewed restart gate required}"

repo="${SLURM_SUBMIT_DIR:?}"
task="${SLURM_ARRAY_TASK_ID:?}"
reynolds=(100 500 1000)
re="${reynolds[$task]}"
commit="$(git -C "$repo" rev-parse HEAD)"

# Fail closed unless this exact commit is present on a published origin branch.
git -C "$repo" fetch --quiet origin
git -C "$repo" branch -r --contains "$commit" | grep -q 'origin/'
test -s "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif"
test -s "$FLOWML_NEKTAR_EQUIVALENCE_GATE"

snapshot="$FLOWML_NEKTAR_PRODUCTION_ROOT/code-re${re}"
mkdir -p "$FLOWML_NEKTAR_PRODUCTION_ROOT"
if [[ ! -d "$snapshot" ]]; then
  mkdir "$snapshot"
  cp "$repo/qa/prepare_week13_nektar.py" "$snapshot/"
  cp "$repo/qa/run_week13_nektar_reynolds.py" "$snapshot/"
  cp "$FLOWML_NEKTAR_EQUIVALENCE_GATE" "$snapshot/restart-gate.json"
  printf '%s\n' "$commit" > "$snapshot/code-commit.txt"
  touch "$snapshot/snapshot-complete"
fi
test -f "$snapshot/snapshot-complete"
test "$(cat "$snapshot/code-commit.txt")" = "$commit"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +e
python3 "$snapshot/run_week13_nektar_reynolds.py" \
  --output "$FLOWML_NEKTAR_PRODUCTION_ROOT/re${re}-d5-p6-n16x80" \
  --image "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" \
  --gate "$snapshot/restart-gate.json" --commit "$commit" \
  --re "$re" --corner-convention stationary_endpoints \
  --order 6 --dt 0.00025 --end-time 80 --nx 16 --ny 80 --seconds 6600
status=$?
set -e
if (( status == 75 )); then
  if (( ${SLURM_RESTART_COUNT:-0} >= 30 )); then
    echo 'Requeue budget exhausted; review required' >&2
    exit 1
  fi
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${task}"
  exit 0
fi
exit "$status"
