#!/usr/bin/env bash
# Tiny afterany CPU job: survives GPU preemption even if there is no TERM grace.
set -euo pipefail
OUT="${SPARTA_GPU_OUT:?}"
cd "$OUT"
sha256sum -c code.sha256
python3 -I "$OUT/code/gpu_preempt.py" guard --out "$OUT" --job "${SPARTA_WATCH_JOB:?}"
