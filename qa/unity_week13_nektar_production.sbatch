#!/bin/bash
#SBATCH --job-name=w13-nektar-pilot
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-3%2
#SBATCH --output=nektar-pilot-%A_%a.log
set -euo pipefail
: "${FLOWML_NEKTAR_ROOT:?}"
: "${FLOWML_NEKTAR_PRODUCTION_ROOT:?}"
: "${FLOWML_NEKTAR_EQUIVALENCE_GATE:?Path to reviewed successful restart gate}"
test -s "$FLOWML_NEKTAR_EQUIVALENCE_GATE"
python3 - "$FLOWML_NEKTAR_EQUIVALENCE_GATE" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    gate = json.load(stream)
assert gate.get('gate_passed') is True, 'Restart agreement gate did not pass'
assert gate.get('time_match_verified') is True, 'Restart time equality not verified'
assert gate.get('Re') == 100, 'Wrong Reynolds number gate'
assert {4, 6, 8}.issubset(gate.get('orders', [])), 'Missing order checks'
assert {0.0005, 0.00025}.issubset(gate.get('dt', [])), 'Missing time-step checks'
assert gate.get('scope') == 'short_time_restart_operability_and_agreement_not_steady_accuracy'
PY
repo="${SLURM_SUBMIT_DIR:?}"
task="${SLURM_ARRAY_TASK_ID:?}"
orders=(4 6 8 6)
dt=0.0005
if (( task == 3 )); then dt=0.00025; fi
test -f "$FLOWML_NEKTAR_ROOT/setup-complete"
cd "$FLOWML_NEKTAR_ROOT"
sha256sum -c container.sha256
commit="$(git -C "$repo" rev-parse HEAD)"
snapshot="$FLOWML_NEKTAR_PRODUCTION_ROOT/code-task-${task}"
mkdir -p "$FLOWML_NEKTAR_PRODUCTION_ROOT"
if [[ ! -d "$snapshot" ]]; then
  mkdir "$snapshot"
  cp "$repo/qa/prepare_week13_nektar.py" "$repo/qa/run_week13_nektar_production.py" "$snapshot/"
  printf '%s\n' "$commit" > "$snapshot/code-commit.txt"
  cp "$FLOWML_NEKTAR_EQUIVALENCE_GATE" "$snapshot/restart-gate.json"
  touch "$snapshot/snapshot-complete"
fi
test -f "$snapshot/snapshot-complete"
commit="$(cat "$snapshot/code-commit.txt")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +e
python3 "$snapshot/run_week13_nektar_production.py" \
  --output "$FLOWML_NEKTAR_PRODUCTION_ROOT/task-${task}" \
  --image "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" \
  --order "${orders[$task]}" --dt "$dt" --commit "$commit" \
  --end-time 20 --seconds 6000
status=$?
set -e
if (( status == 75 )); then
  if (( ${SLURM_RESTART_COUNT:-0} >= 20 )); then
    echo 'Requeue budget exhausted; review required' >&2
    exit 1
  fi
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${task}"
  exit 0
fi
exit "$status"
