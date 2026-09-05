#!/usr/bin/env bash
# Batch process only. Never source this script into a login shell.
set -euo pipefail
PHASE="${1:?Missing phase}"
OUT="${SPARTA_CAMPAIGN_OUT:?Missing campaign directory}"
CODE="$OUT/code"
trap 'rc=$?; printf "SPARTA_CAMPAIGN_JOB_FAILED phase=%s rc=%s line=%s\n" "$PHASE" "$rc" "$LINENO" >&2; exit "$rc"' ERR
cd "$OUT"
sha256sum -c code.sha256
export SPARTA_STEP_OUT="$OUT" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
if [[ "$PHASE" = build ]]; then
  if [[ -e "$OUT/sparta-source" || -e "$OUT/mpi-smoke" ]]; then
    PREVIOUS="$(mktemp -d "$OUT/previous-build-XXXXXXXX")"
    for ITEM in sparta-source mpi-smoke binary.sha256 binary-libraries.txt mpi-library.sha256 mpi-library-path.txt mpi-launcher.txt mpi-compiler.txt; do
      if [[ -e "$OUT/$ITEM" ]]; then mv "$OUT/$ITEM" "$PREVIOUS/"; fi
    done
  fi
  # Reuse the known-good pinned build/MPI checks from job 64013620.
  bash "$CODE/unity_job.sh" build
fi
# Load exactly the same module inside each allocation; preserve HOME.
module purge
module load "${SPARTA_STEP_MPI_MODULE:-openmpi/5.0.3}"
export TMPDIR="$(mktemp -d "/tmp/stepc-${SLURM_JOB_ID}-XXXXXXXX")"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR" PRTE_MCA_prte_tmpdir_base="$TMPDIR"
PYTHON="$(command -v python3)"
MPIRUN="$(command -v mpirun)"
BIN="$OUT/sparta-source/src/spa_mpi"
printf 'PHASE=%s\nHOST=%s\nJOB=%s\nMPI=%s\n' "$PHASE" "$(hostname)" "$SLURM_JOB_ID" "$MPIRUN"
sha256sum -c "$OUT/binary.sha256"
sha256sum -c "$OUT/mpi-library.sha256"
test "$(cat "$OUT/mpi-launcher.txt")" = "$MPIRUN"
MPI_LIB="$(ldd "$BIN" | awk '$1 ~ /^libmpi\.so/ {print $3; exit}')"
test "$(cat "$OUT/mpi-library-path.txt")" = "$MPI_LIB"
case "$PHASE" in
  build)
    "$PYTHON" -I "$CODE/verify_campaign.py" --binary "$BIN" --launcher "$MPIRUN" --ranks 2
    touch "$OUT/PREFLIGHT_PASS"
    echo SPARTA_CAMPAIGN_PREFLIGHT_PASS
    ;;
  coarse|medium|fine|geometry)
    if [[ "$PHASE" = geometry ]]; then test -s "$OUT/VALIDATION_PASS"; fi
    "$PYTHON" -I "$CODE/campaign.py" run-case --out "$OUT" --index "${SLURM_ARRAY_TASK_ID:?}" --binary "$BIN" --launcher "$MPIRUN"
    ;;
  gate) "$PYTHON" -I "$CODE/campaign.py" gate --out "$OUT" ;;
  collect) "$PYTHON" -I "$CODE/campaign.py" collect --out "$OUT" ;;
  *) echo "UNKNOWN_CAMPAIGN_PHASE=$PHASE" >&2; exit 2 ;;
esac
