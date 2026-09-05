#!/usr/bin/env bash
# Child bash only: immutable small download, no login-node module changes.
set -euo pipefail
REF="${1:?Pass the full FlowMLLab commit SHA}"
shift
[[ "$REF" =~ ^[0-9a-f]{40}$ ]] || { echo FULL_COMMIT_SHA_REQUIRED >&2; exit 2; }
BASE="${FLOW_SPARTA_BASE:-/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-sparta-step}"
mkdir -p "$BASE/downloads"
CODE="$(mktemp -d "$BASE/downloads/campaign-${REF:0:12}-XXXXXXXX")"
URL="https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/$REF/cases/sparta_step"
for FILE in campaign.py campaign_job.sh submit_campaign.sh campaign_matrix.csv verify_campaign.py CAMPAIGN.md pilot.py unity_job.sh verify_local.py README.md VALIDATION.md; do
  curl --fail --location --retry 3 "$URL/$FILE" --output "$CODE/$FILE"
done
python3 -I "$CODE/campaign.py" submit --base "$BASE" --root "$CODE" --ref "$REF" "$@"
echo CAMPAIGN_SUBMISSION_COMPLETE_TERMINAL_REMAINS_OPEN
