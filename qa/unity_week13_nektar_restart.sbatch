#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=w13-nektar-restart
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%2
#SBATCH --output=nektar-restart-%A_%a.log
set -euo pipefail
: "${FLOWML_NEKTAR_ROOT:?}"
repo="${SLURM_SUBMIT_DIR:?}"
task="${SLURM_ARRAY_TASK_ID:?}"
orders=(4 6 8)
order="${orders[$((task % 3))]}"
dt=0.0005
half=100
if (( task >= 3 )); then dt=0.00025; half=200; fi
corner="${FLOWML_NEKTAR_CORNER:-legacy_conflicting}"
if [[ "${FLOWML_NEKTAR_FINE_DT:-0}" == 1 ]]; then
    dt=0.00025; half=200
    if (( task >= 3 )); then dt=0.000125; half=400; fi
fi
cd "$FLOWML_NEKTAR_ROOT"
sha256sum -c container.sha256
run="$FLOWML_NEKTAR_ROOT/restart-check-${SLURM_ARRAY_JOB_ID}-${task}"
test ! -e "$run"
mkdir "$run"
git -C "$repo" rev-parse HEAD > "$run/code-commit.txt"
cp "$repo/qa/prepare_week13_nektar.py" "$run/prepare_week13_nektar.py"
run_segment() {
    local name="$1" steps="$2"
    shift 2
    python3 "$run/prepare_week13_nektar.py" --output "$run/$name" --re 100 --order "$order" --dt "$dt" --steps "$steps" --corner-convention "$corner" "$@"
    cd "$run/$name"
    apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" IncNavierStokesSolver cavity.xml > solver.log 2>&1
    test -s cavity.fld
    apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" FieldConvert cavity.xml cavity.fld cavity.vtu:vtu:uncompress > convert.log 2>&1
    test -s cavity.vtu
}
run_segment continuous "$((2*half))"
run_segment first "$half"
run_segment restarted "$half" --restart "$run/first/cavity.fld"
touch "$run/execution-complete-needs-comparison"
