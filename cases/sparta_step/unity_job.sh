#!/usr/bin/env bash
# Executed by sbatch, never sourced in a login shell.
set -euo pipefail
PHASE="${1:?Missing phase}"
OUT="${SPARTA_STEP_OUT:?Missing SPARTA_STEP_OUT}"
trap 'rc=$?; printf "SPARTA_STEP_JOB_FAILED phase=%s rc=%s line=%s\n" "$PHASE" "$rc" "$LINENO" >&2; exit "$rc"' ERR
CODE="$OUT/code"
SPARTA_REF=95b9abaa8bd548991cc3c3f1c58b34722f7ade74
SOURCE="$OUT/sparta-source"
BIN="$SOURCE/src/spa_mpi"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
# Preserve HOME. Do not use --export=NONE or mix Conda MPI with module MPI.
# Short node-local path also avoids Unix-domain-socket path-length failures.
export TMPDIR="$(mktemp -d "/tmp/step-${SLURM_JOB_ID}-XXXXXXXX")"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR" PRTE_MCA_prte_tmpdir_base="$TMPDIR"
module purge
module load "${SPARTA_STEP_MPI_MODULE:-openmpi/5.0.3}"
module list
PYTHON="$(command -v python3)"
test -n "$PYTHON"
MPIRUN="$(command -v mpirun)"
MPICXX="$(command -v mpicxx)"
test "$(dirname "$MPIRUN")" = "$(dirname "$MPICXX")"
printf 'PHASE=%s\nHOST=%s\nJOB=%s\nMPI=%s\n' "$PHASE" "$(hostname)" "$SLURM_JOB_ID" "$MPIRUN"
"$MPIRUN" --version
if [[ "$PHASE" = build ]]; then
  git init "$SOURCE"
  git -C "$SOURCE" remote add origin https://github.com/sparta/sparta.git
  git -C "$SOURCE" fetch --depth 1 origin "$SPARTA_REF"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
  test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SPARTA_REF"
  make -C "$SOURCE/src" -j8 mpi CC="$MPICXX" LINK="$MPICXX"
  test -x "$BIN"
  ldd "$BIN" > "$OUT/binary-libraries.txt"
  if grep -q 'not found' "$OUT/binary-libraries.txt"; then
    echo MISSING_BINARY_LIBRARY >&2
    exit 2
  fi
  "$MPICXX" --showme:command > "$OUT/mpi-compiler.txt"
  printf '%s\n' "$MPIRUN" > "$OUT/mpi-launcher.txt"
  MPI_LIB="$(awk '$1 ~ /^libmpi\.so/ {print $3; exit}' "$OUT/binary-libraries.txt")"
  test -f "$MPI_LIB"
  printf '%s\n' "$MPI_LIB" > "$OUT/mpi-library-path.txt"
  sha256sum "$MPI_LIB" > "$OUT/mpi-library.sha256"
  sha256sum "$BIN" > "$OUT/binary.sha256"
  "$PYTHON" -I "$CODE/pilot.py" generate --smoke --out "$OUT/mpi-smoke"
  cd "$OUT/mpi-smoke"
  # stdin prevents MPI/PRRTE interpreting SPARTA arguments as launcher options.
  "$MPIRUN" -np 2 "$BIN" < in.step
  "$PYTHON" -I "$CODE/pilot.py" report --out "$OUT/mpi-smoke"
  echo SPARTA_STEP_MPI_PREFLIGHT_PASS
elif [[ "$PHASE" = pilot ]]; then
  sha256sum -c "$OUT/binary.sha256"
  sha256sum -c "$OUT/mpi-library.sha256"
  test "$(cat "$OUT/mpi-launcher.txt")" = "$MPIRUN"
  MPI_LIB="$(ldd "$BIN" | awk '$1 ~ /^libmpi\.so/ {print $3; exit}')"
  test "$(cat "$OUT/mpi-library-path.txt")" = "$MPI_LIB"
  "$PYTHON" -I "$CODE/pilot.py" generate --out "$OUT/pilot"
  cd "$OUT/pilot"
  "$MPIRUN" -np "${SLURM_NTASKS:?}" "$BIN" < in.step
  test -s restart.final
  test -s grid.final.gz
  echo SPARTA_STEP_PILOT_SOLVER_JOB_COMPLETE
elif [[ "$PHASE" = collect ]]; then
  "$PYTHON" -I "$CODE/pilot.py" report --out "$OUT/pilot"
  touch "$OUT/PIPELINE_COMPLETE"
  echo SPARTA_STEP_PIPELINE_COMPLETE
else
  echo "Unknown phase: $PHASE" >&2
  exit 2
fi
