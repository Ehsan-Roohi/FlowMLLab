#!/bin/bash
#SBATCH --job-name=w13-nektar-qualify
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%2
#SBATCH --output=nektar-qualify-%A_%a.log
set -euo pipefail
: "${FLOWML_NEKTAR_ROOT:?}"
repo="${SLURM_SUBMIT_DIR:?}"
task="${SLURM_ARRAY_TASK_ID:?}"
orders=(4 6 8)
re=100
if (( task >= 3 )); then re=500; fi
order="${orders[$((task % 3))]}"
test -f "$FLOWML_NEKTAR_ROOT/setup-complete"
cd "$FLOWML_NEKTAR_ROOT"
sha256sum -c container.sha256
case_dir="$FLOWML_NEKTAR_ROOT/qualification-${SLURM_ARRAY_JOB_ID}-${task}"
test ! -e "$case_dir"
mkdir "$case_dir"
git -C "$repo" rev-parse HEAD > "$case_dir/code-commit.txt"
python3 "$repo/qa/prepare_week13_nektar.py" --output "$case_dir/initial" --re "$re" --order "$order"
cd "$case_dir/initial"
apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" IncNavierStokesSolver cavity.xml > solver.log 2>&1
test -s cavity.fld
python3 "$repo/qa/prepare_week13_nektar.py" --output "$case_dir/restart" --re "$re" --order "$order" --restart "$case_dir/initial/cavity.fld"
cd "$case_dir/restart"
apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" IncNavierStokesSolver cavity.xml > solver.log 2>&1
test -s cavity.fld
apptainer exec --cleanenv --bind /project "$FLOWML_NEKTAR_ROOT/nektar-v5.9.0.sif" FieldConvert cavity.xml cavity.fld cavity.vtu > convert.log 2>&1
test -s cavity.vtu
sha256sum cavity.fld cavity.vtu > output.sha256
touch "$case_dir/execution-complete-needs-numerical-review"
# This checks restart operability, NOT restart equivalence or CFD accuracy.
# Production is deliberately gated on review of finite fields, time advancement,
# corner treatment, and comparison with an uninterrupted trajectory.
