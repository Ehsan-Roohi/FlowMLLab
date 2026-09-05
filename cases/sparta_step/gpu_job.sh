#!/usr/bin/env bash
# Batch child process only. Preserve HOME; never source into a login shell.
set -euo pipefail
OUT="${SPARTA_GPU_OUT:?Missing GPU benchmark output directory}"
CODE="$OUT/code"
trap 'rc=$?; printf "SPARTA_GPU_JOB_FAILED rc=%s line=%s\n" "$rc" "$LINENO" >&2; exit "$rc"' ERR
cd "$OUT"
sha256sum -c code.sha256
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME OMPI_CXX MPICH_CXX
module purge
# These module names are published in Unity's module usage documentation.
module load cuda/12.6 openmpi/5.0.3-cuda12.6
module list
export TMPDIR="$(mktemp -d "/tmp/stepgpu-${SLURM_JOB_ID}-XXXXXXXX")"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR" PRTE_MCA_prte_tmpdir_base="$TMPDIR"
PYTHON="$(command -v python3)"
MPIRUN="$(command -v mpirun)"
MPICXX="$(command -v mpicxx)"
test "$(dirname "$MPIRUN")" = "$(dirname "$MPICXX")"
printf 'HOST=%s\nJOB=%s\n' "$(hostname)" "$SLURM_JOB_ID"
nvcc --version
g++ --version
"$MPIRUN" --version
nvidia-smi
"$PYTHON" -I - "$OUT/manifest.json" "$MPIRUN" "$MPICXX" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p));d.update(mpi_launcher=sys.argv[2],mpi_compiler=sys.argv[3])
with open(p,'w') as f: json.dump(d,f,indent=2)
PY
# Use system CMake when sufficient; otherwise a private tool-only venv, never ~/.local.
if command -v cmake >/dev/null && "$PYTHON" -I -c 'import re,subprocess; v=re.search(r"(\d+)\.(\d+)",subprocess.check_output(["cmake","--version"],text=True)); raise SystemExit(tuple(map(int,v.groups())) < (3,22))'; then
  CMAKE="$(command -v cmake)"
else
  "$PYTHON" -I -m venv "$OUT/cmake-tools"
  "$OUT/cmake-tools/bin/python" -I -m pip --isolated install 'cmake==3.31.6'
  CMAKE="$OUT/cmake-tools/bin/cmake"
fi
"$CMAKE" --version
nvcc -std=c++20 "$CODE/cuda_probe.cu" -o "$OUT/cuda-probe"
"$OUT/cuda-probe" > "$OUT/cuda-device.json"
ARCH="$("$PYTHON" -I - "$OUT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);h=json.loads((p/'cuda-device.json').read_text());c=json.loads((p/'manifest.json').read_text())
if h['compute_capability']!=c['expected_compute_capability'] or not h['kernel_pass']:
    raise SystemExit('CUDA_DEVICE_DOES_NOT_MATCH_REQUEST')
print(c['kokkos_arch'])
PY
)"
cat "$OUT/cuda-device.json"
echo CUDA_KERNEL_PREFLIGHT_PASS
SOURCE="$OUT/source"
git init "$SOURCE"
git -C "$SOURCE" remote add origin https://github.com/sparta/sparta.git
git -C "$SOURCE" fetch --depth 1 origin 95b9abaa8bd548991cc3c3f1c58b34722f7ade74
git -C "$SOURCE" checkout --detach FETCH_HEAD
test "$(git -C "$SOURCE" rev-parse HEAD)" = 95b9abaa8bd548991cc3c3f1c58b34722f7ade74
# Separate out-of-source builds; same revision, MPI ABI and optimization level.
"$CMAKE" -S "$SOURCE/cmake" -B "$OUT/build-cpu" \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="$MPICXX" \
  -D BUILD_MPI=ON -D PKG_KOKKOS=OFF -D SPARTA_MACHINE=mpi
"$CMAKE" --build "$OUT/build-cpu" -j8
export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
# The upstream preset defaults to Hopper. Disable it before selecting A40/A100.
"$CMAKE" -C "$SOURCE/cmake/presets/kokkos_cuda.cmake" \
  -S "$SOURCE/cmake" -B "$OUT/build-gpu" \
  -D Kokkos_ARCH_HOPPER90=OFF "-DKokkos_ARCH_${ARCH}=ON" \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF
"$CMAKE" --build "$OUT/build-gpu" -j8
sha256sum "$OUT/build-cpu/src/spa_mpi" "$OUT/build-gpu/src/spa_kokkos_cuda" > "$OUT/binary.sha256"
ldd "$OUT/build-cpu/src/spa_mpi" > "$OUT/cpu-libraries.txt"
ldd "$OUT/build-gpu/src/spa_kokkos_cuda" > "$OUT/gpu-libraries.txt"
if grep -q 'not found' "$OUT/cpu-libraries.txt" "$OUT/gpu-libraries.txt"; then
  echo UNRESOLVED_BINARY_LIBRARY >&2; exit 1
fi
CPU_MPI="$(awk '$1 ~ /^libmpi\.so/ {print $3; exit}' "$OUT/cpu-libraries.txt")"
GPU_MPI="$(awk '$1 ~ /^libmpi\.so/ {print $3; exit}' "$OUT/gpu-libraries.txt")"
test -n "$CPU_MPI"
test "$CPU_MPI" = "$GPU_MPI"
sha256sum "$CPU_MPI" > "$OUT/mpi-library.sha256"
"$PYTHON" -I "$CODE/gpu_benchmark.py" run --out "$OUT"
"$PYTHON" -I "$CODE/gpu_benchmark.py" pack --out "$OUT"
