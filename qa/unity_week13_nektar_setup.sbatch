#!/bin/bash
#SBATCH --job-name=w13-nektar-setup
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=nektar-setup-%j.log
set -euo pipefail
: "${FLOWML_NEKTAR_ROOT:?Set dedicated absolute installation/run root}"
[[ "$FLOWML_NEKTAR_ROOT" = /* ]] || exit 2
mkdir -p "$FLOWML_NEKTAR_ROOT"
cd "$FLOWML_NEKTAR_ROOT"
export APPTAINER_CACHEDIR="$FLOWML_NEKTAR_ROOT/cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}"
# Official Nektar++ release container; record the resulting immutable SIF hash.
if [[ ! -f nektar-v5.9.0.sif ]]; then
    apptainer pull nektar-v5.9.0.sif docker://nektarpp/nektar:v5.9.0
fi
sha256sum nektar-v5.9.0.sif > container.sha256
apptainer exec --cleanenv nektar-v5.9.0.sif IncNavierStokesSolver --help
touch setup-complete
