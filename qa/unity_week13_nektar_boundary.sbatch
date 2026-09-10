#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --job-name=w13-nektar-boundary
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --array=0-5%2
#SBATCH --output=nektar-boundary-%A_%a.log
# Diagnostic only: no production or restart gate can be inferred from this.
set -euo pipefail
: "${FLOWML_NEKTAR_ROOT:?}"
repo="${SLURM_SUBMIT_DIR:?}"
task="${SLURM_ARRAY_TASK_ID:?}"
if (( task < 0 || task > 5 )); then exit 2; fi
orders=(4 6 8)
convention=legacy_conflicting
if (( task >= 3 )); then convention=stationary_endpoints; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$FLOWML_NEKTAR_ROOT"
sha256sum -c container.sha256
run="$FLOWML_NEKTAR_ROOT/boundary-check-${SLURM_ARRAY_JOB_ID}-${task}"
test ! -e "$run"
mkdir "$run"
git -C "$repo" rev-parse HEAD > "$run/code-commit.txt"
cp "$repo/qa/prepare_week13_nektar.py" "$run/prepare_week13_nektar.py"
python3 "$run/prepare_week13_nektar.py" --output "$run/case" --re 100 \
  --order "${orders[$((task%3))]}" --dt 0.0005 --steps 100 --check-steps 100 \
  --corner-convention "$convention" --purpose boundary_diagnostic_not_steady_evidence
cd "$run/case"
apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" \
  IncNavierStokesSolver cavity.xml > solver.log 2>&1
test -s cavity.fld
apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" \
  FieldConvert cavity.xml cavity.fld cavity.vtu:vtu:uncompress > convert.log 2>&1
test -s cavity.vtu
sha256sum cavity.fld cavity.vtu > output.sha256
# Subsequent audit must require full side/bottom speed <=1e-10; separately report
# central lid x[0.125,0.875] error and near-corner lid deviation/overshoot.
# The finite-order CG trace cannot represent an endpoint jump pointwise exactly.
touch "$run/execution-complete-needs-boundary-review"
