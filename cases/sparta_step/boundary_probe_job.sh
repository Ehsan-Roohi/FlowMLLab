#!/usr/bin/env bash
set -euo pipefail
OUT=${SPARTA_BOUNDARY_OUT:?}
cd "$OUT"
sha256sum -c code.sha256
module purge
module load openmpi/5.0.3
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME OMPI_CXX MPICH_CXX
export TMPDIR=$(mktemp -d "/tmp/stepbc-${SLURM_JOB_ID}-XXXXXXXX")
export PRTE_MCA_prte_tmpdir_base="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR"
SOURCE=$(python3 -I -c 'import json; print(json.load(open("probe.json"))["source_pilot"])')
sha256sum -c "$SOURCE/mpi-library.sha256"
test "$(cat "$SOURCE/mpi-launcher.txt")" = "$(command -v mpirun)"
BIN="$OUT/source/src/spa_mpi"
case "$1" in
build)
  sha256sum -c "$SOURCE/binary.sha256"
  if ! test -f "$OUT/PATCH_APPLIED"; then
    # Only source is copied; the production binary and runs are untouched.
    mkdir -p "$OUT/source"
    cp -a "$SOURCE/sparta-source/src" "$OUT/source/"
    python3 -I "$OUT/code/patch_face_window.py" "$OUT/source/src"
    touch "$OUT/PATCH_APPLIED"
  fi
  make -C "$OUT/source/src" -j8 mpi CC="$(command -v mpicxx)" LINK="$(command -v mpicxx)"
  sha256sum "$BIN" > "$OUT/patched-binary.sha256"
  python3 -I "$OUT/code/boundary_probe.py" preflight --out "$OUT/preflight" --binary "$BIN" --original "$SOURCE/sparta-source/src/spa_mpi"
  ;;
arms)
  sha256sum -c "$OUT/patched-binary.sha256"
  python3 -I "$OUT/code/boundary_probe.py" run --out "$OUT" --binary "$BIN" --index "${SLURM_ARRAY_TASK_ID:?}"
  ;;
review)
  python3 -I "$OUT/code/boundary_probe.py" review --out "$OUT"
  ;;
*) exit 2 ;;
esac
