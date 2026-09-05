#!/usr/bin/env bash
# Download an immutable small case snapshot; run this as a child bash, not source.
set -euo pipefail
REF="${1:?Pass a full FlowMLLab commit SHA}"
shift
if [[ ! "$REF" =~ ^[0-9a-f]{40}$ ]]; then
  echo FULL_COMMIT_SHA_REQUIRED >&2
  exit 2
fi
BASE="${FLOW_SPARTA_BASE:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-sparta-step}"
mkdir -p "$BASE/downloads"
CODE="$(mktemp -d "$BASE/downloads/code-${REF:0:12}-XXXXXXXX")"
URL="https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/$REF/cases/sparta_step"
for FILE in pilot.py unity_job.sh README.md VALIDATION.md verify_local.py; do
  curl --fail --location --retry 3 "$URL/$FILE" --output "$CODE/$FILE"
done
# Isolated stdlib Python: no TensorFlow/Conda/user-site dependency for submission.
python3 -I "$CODE/pilot.py" submit --base "$BASE" --root "$CODE" --ref "$REF" "$@"
echo SUBMISSION_COMPLETE_TERMINAL_REMAINS_OPEN
