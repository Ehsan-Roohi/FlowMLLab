#!/usr/bin/env bash
# Retrieve immutable run artifacts; verification happens in validate_reference.py.
set -euo pipefail
mkdir -p raw_artifacts report_artifacts results/week16_lowboom/reference
fetch_raw() {
  local run_id="$1" pattern="$2" group="$3"
  gh run download "$run_id" --pattern "$pattern" --dir "raw_artifacts/$group"
}
fetch_report() {
  local run_id="$1" artifact_name="$2"
  gh run download "$run_id" --name "$artifact_name" --dir "report_artifacts/$artifact_name"
  cp "report_artifacts/$artifact_name/"* results/week16_lowboom/reference/
}
fetch_raw 35936740632 'neural-case-*' historical_neural
fetch_raw 35942843724 'weakwall-case-*' accepted_neural
fetch_raw 35942843724 'cone-v801' cone_coarse
fetch_raw 35943668572 'cone-level-*' cone_refinement
fetch_raw 35943668623 'design-v801-*' accepted_designs
fetch_raw 35944299803 'clean-batch-*' clean_campaign
fetch_raw 35935655513 'nasa-level-1' rejected_nasa_original
fetch_raw 35941629706 'nasa-resolved-level-1' rejected_nasa_resolved
fetch_raw 35944409924 'nasa-resolved-v801-level-1' nasa_coarse
fetch_raw 35946027208 'nasa-resolved-v801-level-*' nasa_refinement
while IFS= read -r -d '' archive; do tar -xzf "$archive"; done < <(find raw_artifacts -name '*.tar.gz' -print0)
fetch_report 35942843724 weakwall-checkpoint-report
fetch_report 35943668623 weakwall-design-report
fetch_report 35943668572 cone-refinement-report
fetch_report 35944888983 clean-model-report
