#!/usr/bin/env bash
# Child script: no login-shell module or option changes.
set -euo pipefail
REF="${1:?Pass the full FlowMLLab commit SHA}"
shift
[[ "$REF" =~ ^[0-9a-f]{40}$ ]] || { echo FULL_COMMIT_SHA_REQUIRED >&2; exit 2; }
BASE="${FLOW_SPARTA_BASE:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-sparta-step}"
mkdir -p "$BASE/downloads"
CODE="$(mktemp -d "$BASE/downloads/gpu-${REF:0:12}-XXXXXXXX")"
URL="https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/$REF/cases/sparta_step"
for FILE in gpu_benchmark.py gpu_job.sh submit_gpu_benchmark.sh cuda_probe.cu verify_gpu.py GPU.md campaign.py pilot.py; do
  curl --fail --location --retry 3 "$URL/$FILE" --output "$CODE/$FILE"
done
python3 -I "$CODE/gpu_benchmark.py" submit --base "$BASE" --root "$CODE" --ref "$REF" "$@"
echo GPU_BENCHMARK_SUBMITTED_TERMINAL_REMAINS_OPEN
