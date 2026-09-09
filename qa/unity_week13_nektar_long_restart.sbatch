#!/bin/bash
#SBATCH --job-name=w13-nektar-restart-bias
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-1%2
#SBATCH --output=nektar-long-restart-%A_%a.log
# Bounded Re100,p4 test: t=2 uninterrupted vs four t=0.5 segments.
# Task1 halves dt. Three startup events expose accumulation; this is NOT proof
# that t=20 production or weak lower vortices are insensitive to restarting.
# A wall-time-limited run is incomplete, never a failed numerical comparison.
set -euo pipefail
: "${FLOWML_NEKTAR_ROOT:?}"
: "${FLOWML_NEKTAR_CORNER:?Explicit convention required}"
repo="${SLURM_SUBMIT_DIR:?}"
task="${SLURM_ARRAY_TASK_ID:?}"
dt=0.0005
segment_steps=1000
if (( task == 1 )); then dt=0.00025; segment_steps=2000; fi
if [[ "${FLOWML_NEKTAR_FINE_DT:-0}" == 1 ]]; then
  dt=0.00025; segment_steps=2000
  if (( task == 1 )); then dt=0.000125; segment_steps=4000; fi
fi
if (( task < 0 || task > 1 )); then exit 2; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$FLOWML_NEKTAR_ROOT"
sha256sum -c container.sha256
run="$FLOWML_NEKTAR_ROOT/long-restart-check-${SLURM_ARRAY_JOB_ID}-${task}"
test ! -e "$run"
mkdir "$run"
git -C "$repo" rev-parse HEAD > "$run/code-commit.txt"
cp "$repo/qa/prepare_week13_nektar.py" "$run/prepare_week13_nektar.py"
run_segment() {
  local name="$1" steps="$2"
  shift 2
  python3 "$run/prepare_week13_nektar.py" --output "$run/$name" --re 100 \
    --order 4 --dt "$dt" --steps "$steps" --check-steps "$steps" \
    --corner-convention "$FLOWML_NEKTAR_CORNER" \
    --purpose accumulated_restart_bias_test_not_steady_evidence "$@"
  cd "$run/$name"
  apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" \
    IncNavierStokesSolver cavity.xml > solver.log 2>&1
  test -s cavity.fld
  apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" \
    FieldConvert cavity.xml cavity.fld cavity.vtu:vtu:uncompress > convert.log 2>&1
  test -s cavity.vtu
  sha256sum cavity.fld cavity.vtu > output.sha256
}
run_segment continuous "$((4*segment_steps))"
run_segment chunk0 "$segment_steps"
for index in 1 2 3; do
  previous="$((index-1))"
  run_segment "chunk${index}" "$segment_steps" \
    --restart "$run/chunk${previous}/cavity.fld"
done
# Compare continuous/cavity.vtu vs chunk3/cavity.vtu at equal metadata time2;
# remove pressure gauge, retain absolute and scaled errors, and compare task0/1.
# Do not silently relax the already-declared short-test tolerance on failure.
touch "$run/execution-complete-needs-comparison"
