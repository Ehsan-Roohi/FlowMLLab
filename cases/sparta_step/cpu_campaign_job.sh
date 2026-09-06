#!/usr/bin/env bash
set -euo pipefail
OUT="${SPARTA_CPU_OUT:?}"
PHASE="${1:?}"
cd "$OUT"
sha256sum -c code.sha256
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME OMPI_CXX MPICH_CXX
module purge
module load openmpi/5.0.3
export TMPDIR="$(mktemp -d "/tmp/stepcpu-${SLURM_JOB_ID}-XXXXXXXX")"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR" PRTE_MCA_prte_tmpdir_base="$TMPDIR"
SOURCE="$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$OUT/manifest.json")"
BIN="$SOURCE/sparta-source/src/spa_mpi"
sha256sum -c "$SOURCE/binary.sha256"
sha256sum -c "$SOURCE/mpi-library.sha256"
test "$(cat "$SOURCE/mpi-launcher.txt")" = "$(command -v mpirun)"
MPI_LIB="$(ldd "$BIN" | awk '$1 ~ /^libmpi\.so/ {print $3; exit}')"
test "$(cat "$SOURCE/mpi-library-path.txt")" = "$MPI_LIB"
python3 -I -c 'import hashlib,json,sys; m=json.load(open(sys.argv[1])); assert hashlib.sha256(open(m["binary"],"rb").read()).hexdigest()==m["binary_sha256"]' "$OUT/manifest.json"
printf 'PHASE=%s HOST=%s JOB=%s MPI=%s\n' "$PHASE" "$(hostname)" "$SLURM_JOB_ID" "$(command -v mpirun)"
case "$PHASE" in
  preflight) python3 -I "$OUT/code/cpu_campaign.py" preflight --out "$OUT" --binary "$BIN" ;;
  cases) python3 -I "$OUT/code/cpu_campaign.py" run --out "$OUT" --binary "$BIN" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  collect) python3 -I "$OUT/code/cpu_campaign.py" collect --out "$OUT" ;;
  *) exit 2 ;;
esac
