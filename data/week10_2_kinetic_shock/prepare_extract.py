"""Reproduce this small teaching extract from immutable, public data bytes.

Original FlowMLLab packaging utility. No upstream research code is executed.
Only files enumerated in source_files.json are read/downloaded, all with size,
Git blob SHA1 and SHA256 checks before use. No checkpoint is permitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
PIN = "add5388f7da5f7eace95fff97d34d6ed58b90169"
BASE = "code/supervised_shocktube/supervised_base_eight_moment/"
FIELDS = ["rho", "u", "T", "q", "sigma", "m", "R", "Delta", "S"]
UNITS = "nondimensional_R1_gamma5over3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(raw, rec):
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if (len(raw) != rec["bytes"] or digest(raw) != rec["sha256"] or blob != rec["git_blob_sha1"]):
        raise ValueError("upstream identity mismatch: " + rec["upstream_path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-release", type=Path)
    args = parser.parse_args()
    sources = json.loads((ROOT / "source_files.json").read_text())
    assert sources["upstream_commit"] == PIN
    receipts = []
    records = {r["upstream_path"]: r for r in sources["files"]}
    for path, rec in records.items():
        if Path(path).suffix not in {".csv", ".npz", ".json", ".txt"}:
            raise ValueError("extract excludes code/checkpoints: " + path)
        destination = ROOT / rec["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = "immutable_public_raw"
        if destination.exists():
            raw = destination.read_bytes()
            verify(raw, rec)
            source = "existing_verified_extract"
        elif args.local_release is not None and (args.local_release / path).is_file():
            candidate = (args.local_release / path).read_bytes()
            try:
                verify(candidate, rec)
                raw = candidate
                source = "local_bytes_matched_public_git_blob_and_sha256"
            except ValueError:
                with urllib.request.urlopen(rec["url"], timeout=45) as response:
                    raw = response.read(rec["bytes"] + 1)
        else:
            with urllib.request.urlopen(rec["url"], timeout=45) as response:
                raw = response.read(rec["bytes"] + 1)
        verify(raw, rec)
        if not destination.exists():
            destination.write_bytes(raw)
        receipts.append({"path": rec["path"], "source": source, "sha256": rec["sha256"],
                         "git_blob_sha1": rec["git_blob_sha1"], "bytes": len(raw)})

    def path(public):
        return ROOT / records[public]["path"]

    def identity(file, x):
        raw = file.read_bytes()
        return {"path": file.relative_to(ROOT).as_posix(), "sha256": digest(raw), "bytes": len(raw),
                "rows": len(x), "x_endpoints": [float(x[0]), float(x[-1])],
                "x_sha256": digest(np.asarray(x, dtype="<f8").tobytes())}

    cases = {}
    for family in ("bgk", "shakhov"):
        public = f"data/shocktube/{family}_matched_fields.csv"
        file = path(public)
        data = np.genfromtxt(file, delimiter=",", names=True)
        cid = "tube_" + family
        cases[cid] = {**identity(file, data["x"]), "case_id": cid, "format": "matched_csv",
                      "units": UNITS, "problem": "transient_shock_tube",
                      "collision_model": "BGK" if family == "bgk" else "Shakhov",
                      "Pr": 1.0 if family == "bgk" else 2 / 3, "tau_law": "constant_1", "time": 0.02,
                      "columns": FIELDS, "methods": ["parent", "final", "reference"],
                      "method_roles": {"parent": "equation-trained 1000 Adam + 100 SSB endpoint",
                                       "final": "same run with 200 further SSB updates",
                                       "reference": "collision-matched 3200-cell DVM"},
                      "reference_role": "post-solve comparison; no field-target training",
                      "convergence_certified": False, "clipping_applied": False,
                      "upstream_path": public, "upstream_git_blob": records[public]["git_blob_sha1"],
                      "alignment": "upstream supplied common comparison grid; no additional interpolation, shift, scaling or smoothing"}
    for seed in (141001, 141002, 141003):
        for stage in ("adam20000", "ssb500", "fixed500"):
            public = f"data/independent_m2/evaluation/seed{seed}_{stage}_core.npz"
            file = path(public)
            with np.load(file, allow_pickle=False) as data:
                x = data["x"].copy()
                keys = data.files
            cid = f"m2_seed{seed}_{stage}"
            cases[cid] = {**identity(file, x), "case_id": cid, "format": "m2_core_npz",
                          "npz_keys": keys, "profile_keys": {"prediction": "fields", "reference": "reference"},
                          "units": UNITS, "problem": "steady_normal_shock", "collision_model": "Shakhov",
                          "Pr": 2 / 3, "tau_law": "1/rho", "mach": 2, "seed": seed, "stage": stage,
                          "columns": FIELDS[:5], "methods": ["prediction", "reference"],
                          "method_roles": {"prediction": "independent equation-trained initialization, sequential optimizer stage",
                                           "reference": "saved Shakhov DVM interpolation at the unchanged 81 core nodes"},
                          "reference_role": "post-solve comparison; no field-target training",
                          "window": "core [-10,10]", "convergence_certified": False,
                          "all_acceptance_tolerances_pass": False, "clipping_applied": False,
                          "stage_warning": "sequential Adam, SSB, split-half quadrature SSB; not an equal-budget optimizer benchmark",
                          "upstream_path": public, "upstream_git_blob": records[public]["git_blob_sha1"],
                          "alignment": "exact supplied common grid/reference; no further phase or amplitude adjustment"}

    with np.load(path(BASE + "accepted_state/evidence/ASSESSMENT_FIELDS.npz"), allow_pickle=False) as a:
        xa = a["x"].copy()
        names = list(a["columns"])
        indices = [names.index("S_raw" if c == "S" else c) for c in FIELDS]
        stage_fields = {name: a[key][:, indices].copy() for name, key in
                        (("original_envelope", "model170"), ("common_tail_initial", "model170T"),
                         ("data_assisted", "stageB"), ("reference", "reference"))}
    with np.load(path(BASE + "comparison_inputs/EQUATION_TRAINED_BGK_FIELDS.npz"), allow_pickle=False) as a:
        xe = a["x"].copy()
        names = list(a["columns"])
        equation = a["fields"][:, [names.index(c) for c in FIELDS]].copy()
    with np.load(path(BASE + "manuscript_checks/stage0/STAGE0_FIELDS.npz"), allow_pickle=False) as a:
        x0 = a["x"].copy()
        names = list(a["columns"])
        indices = [names.index("S_raw" if c == "S" else c) for c in FIELDS]
        five_field = a["five_field"][:, indices].copy()
        reference0 = a["reference"][:, indices].copy()
    if not np.array_equal(x0, xa) or not np.array_equal(reference0, stage_fields["reference"]):
        raise ValueError("five-field S0 and eight-moment assessments must share the exact reference/grid")
    stage_fields["five_field_S0"] = five_field
    # Restrict to actual common coverage. Never extrapolate into boundary cells.
    overlap = (xe >= xa[0]) & (xe <= xa[-1])
    x = xe[overlap]
    interpolated = {name: np.column_stack([np.interp(x, xa, y[:, j]) for j in range(len(FIELDS))])
                    for name, y in stage_fields.items()}
    interpolated["equation_trained"] = equation[overlap]
    methods = ["five_field_S0", "original_envelope", "common_tail_initial", "data_assisted", "equation_trained", "reference"]
    cid = "supervised_s0_current"
    output = ROOT / "supervised_s0_current.npz"
    np.savez_compressed(output, x=x, columns=np.array(FIELDS), case_id=np.array(cid),
                        **{name: interpolated[name] for name in methods})
    with np.load(output, allow_pickle=False) as a:
        keys = a.files
    cases[cid] = {**identity(output, x), "case_id": cid, "format": "supervised_comparison_npz",
                  "npz_keys": keys, "profile_keys": {name: name for name in methods}, "units": UNITS,
                  "problem": "transient_shock_tube", "collision_model": "BGK", "Pr": 1.0,
                  "tau_law": "constant_1", "time": 0.02, "columns": FIELDS, "methods": methods,
                  "method_roles": {"five_field_S0": "actual five-field S0 base state, checkpoint a0734fccb6cc0766bf046c8a23f8f98bf15e2e03b39d6e96ed70e1a170d09003",
                                   "original_envelope": "eight-moment model after 170 accepted updates; not the five-field S0 model",
                                   "common_tail_initial": "same state with common-tail envelope replacement before 35 updates",
                                   "data_assisted": "common-tail eight-moment final state after 35 accepted output-layer updates",
                                   "equation_trained": "independently equation-trained BGK characteristic-PINN endpoint",
                                   "reference": "6400-cell DVM supplied in assessment arrays"},
                  "reference_role": "observations, reference-derived scales, acceptance checks, and comparison for data-assisted lineage",
                  "method_reference_roles": {
                      "five_field_S0": {"training_reference_cells": 400, "comparison_reference_cells": 6400,
                                        "comparison_role": "post-solve reevaluation on 48x20x20 velocity grid",
                                        "trained_on_comparison_reference": False},
                      "data_assisted": {"training_reference_cells": 6400, "comparison_reference_cells": 6400,
                                        "comparison_role": "same reference supplies observations, scales and some acceptance guards",
                                        "trained_on_comparison_reference": True},
                      "equation_trained": {"training_reference_cells": None, "native_comparison_reference_cells": 3200,
                                           "comparison_reference_cells": 6400, "comparison_role": "post-solve only",
                                           "trained_on_comparison_reference": False}},
                  "paper_admission": False, "all_criteria_pass": False, "convergence_certified": False,
                  "admission_flag_scope": "final common-tail 35-update data-assisted state; not a shared admission flag for all methods",
                  "clipping_applied": False, "observed_reference_count": 690,
                  "observation_sets": records[BASE + "accepted_state/inputs/OBSERVATION_LOCATIONS_FROZEN.npz"],
                  "independent_test_claim": False,
                  "contamination": {"reference_used_for_training": True, "reference_derived_scales": True,
                                    "reference_used_in_some_acceptance_guards": True,
                                    "excluding_observation_neighbors_is_not_independent_test": True},
                  "contamination_scope": "final data-assisted lineage; consult method_reference_roles for each comparator",
                  "failed_criteria": {"E4": "64 separate physics points, predeclared 0.1% tolerance failed",
                                      "E8": "64 points unused by any stage; criterion added after first assessment; 0.1% tolerance failed"},
                  "alignment": "linear interpolation of assessment fields to original equation-trained x nodes within common coverage; no shift, rescaling, smoothing or extrapolation",
                  "dropped_equation_nodes_outside_assessment_coverage": int((~overlap).sum()),
                  "original_equation_node_count": len(xe),
                  "five_field_reference_grid_and_values_exactly_match_eight_moment_assessment": True,
                  "source_reference_x_endpoints": [float(xa[0]), float(xa[-1])],
                  "metric_difference_from_paper": "teaching common-overlap spatial trapezoidal norm; not paper's unweighted full401-node score",
                  "source_files": [records[BASE + p] for p in
                                   ("accepted_state/evidence/ASSESSMENT_FIELDS.npz", "comparison_inputs/EQUATION_TRAINED_BGK_FIELDS.npz",
                                    "manuscript_checks/stage0/STAGE0_FIELDS.npz")]}
    manifest = {"schema_version": 1, "upstream_repository": sources["upstream_repository"],
                "upstream_commit": PIN, "paper_url": sources["paper"], "paper_license": "CC BY 4.0",
                "upstream_software_license": "none assigned; no upstream code copied",
                "units": UNITS, "gas_constant": 1, "gamma": 5 / 3, "velocity_dimensions": 3,
                "stress_convention": "integral (cx^2-c^2/3) f dv",
                "field_conventions": {"q": "0.5 integral c^2 cx f dv", "m": "integral (cx^3-3cx c^2/5) f dv",
                                      "R": "integral (c^2-7T)(cx^2-c^2/3) f dv",
                                      "Delta": "integral c^4 f dv - 15 rho T^2", "S": "R+Delta/3"},
                "cases": cases, "source_receipts": receipts,
                "created_utc": datetime.now(timezone.utc).isoformat(), "no_upstream_code_executed": True,
                "no_checkpoints_included": True, "new_training_or_shock_solver_run": False}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "source_files": len(receipts), "source_bytes": sum(r["bytes"] for r in receipts),
                      "supervised_rows": len(x), "dropped_outside_coverage": int((~overlap).sum())}))


if __name__ == "__main__":
    main()
