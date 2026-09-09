"""Read-only same-grid restart comparison; NOT a CFD accuracy certification.

Export with FieldConvert session.xml field.fld output.vtu:vtu:uncompress.
Only ASCII VTU is accepted, intentionally failing closed on other encodings.
RMS norms are sample norms, NOT quadrature-weighted physical L2 norms.
Supply tolerances chosen before inspecting the comparison. Check equal physical
times from solver logs/field metadata separately: VTU need not contain time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def read_vtu(path):
    root = ET.parse(path).getroot()
    pieces = root.findall("./UnstructuredGrid/Piece")
    if not pieces:
        raise ValueError("No VTU UnstructuredGrid/Piece")

    def array(node):
        if node is None or node.get("format", "ascii") != "ascii":
            raise ValueError("Require ASCII VTU: export with :vtu:uncompress")
        # split/float rejects malformed tokens instead of silently truncating.
        values = np.asarray([float(x) for x in (node.text or "").split()])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite VTU data")
        return values

    result = []
    for piece in pieces:
        n = int(piece.attrib["NumberOfPoints"])
        points = array(piece.find("./Points/DataArray")).reshape(n, 3)
        fields = {}
        for name in ("u", "v", "p"):
            node = piece.find(f"./PointData/DataArray[@Name='{name}']")
            fields[name] = array(node).reshape(n)
        topology = {name: array(piece.find(f"./Cells/DataArray[@Name='{name}']"))
                    for name in ("connectivity", "offsets", "types")}
        result.append((points, fields, topology))
    return result


def compare(reference, candidate, atol, rtol, coordinate_atol=1e-12):
    if min(atol, rtol, coordinate_atol) < 0 or not np.isfinite([atol, rtol, coordinate_atol]).all():
        raise ValueError("Tolerances must be finite and nonnegative")
    a, b = read_vtu(reference), read_vtu(candidate)
    if len(a) != len(b):
        raise ValueError("Piece count differs")
    for (pa, _, ta), (pb, _, tb) in zip(a, b):
        if pa.shape != pb.shape or not np.allclose(pa, pb, atol=coordinate_atol, rtol=0):
            raise ValueError("Coordinates/order differ; interpolation is not restart equivalence")
        if any(not np.array_equal(ta[k], tb[k]) for k in ta):
            raise ValueError("VTU topology differs")
    points = np.concatenate([x[0] for x in a])
    report = {"purpose": "same_grid_restart_comparison_not_physics_validation",
              "time_match_verified": False,
              "norm": "unweighted VTU point-sample RMS (duplicates retained)",
              "points": len(points), "atol": atol, "rtol": rtol,
              "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
              "candidate_sha256": hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),
              "fields": {}}
    for name in ("u", "v", "p"):
        av = np.concatenate([x[1][name] for x in a])
        bv = np.concatenate([x[1][name] for x in b])
        gauge_offset = float(np.mean(bv-av)) if name == "p" else 0.0
        difference = bv-av-gauge_offset
        scale = av-np.mean(av) if name == "p" else av
        rms = float(np.sqrt(np.mean(difference**2)))
        ref_rms = float(np.sqrt(np.mean(scale**2)))
        maximum = float(np.max(np.abs(difference)))
        threshold = atol+rtol*float(np.max(np.abs(scale)))
        report["fields"][name] = dict(sample_rms_difference=rms,
            sample_relative_rms=rms/ref_rms if ref_rms else None,
            max_absolute_difference=maximum, max_threshold=threshold,
            removed_pressure_constant=gauge_offset, passed=maximum <= threshold)
    # Observational BC report, excluding incompatible lid corners from wall norms.
    x, y = points[:, 0], points[:, 1]
    corner = np.isclose(y, 5, atol=coordinate_atol, rtol=0) & (
        np.isclose(x, 0, atol=coordinate_atol, rtol=0) |
        np.isclose(x, 1, atol=coordinate_atol, rtol=0))
    lid = np.isclose(y, 5, atol=coordinate_atol, rtol=0) & ~corner
    walls = (np.isclose(x, 0, atol=coordinate_atol, rtol=0) |
             np.isclose(x, 1, atol=coordinate_atol, rtol=0) |
             np.isclose(y, 0, atol=coordinate_atol, rtol=0)) & ~corner
    u = np.concatenate([part[1]["u"] for part in b])
    v = np.concatenate([part[1]["v"] for part in b])
    side = (np.isclose(x, 0, atol=coordinate_atol, rtol=0) |
            np.isclose(x, 1, atol=coordinate_atol, rtol=0)) & ~corner
    side_ids = np.flatnonzero(side)
    worst_side = int(side_ids[np.argmax(np.hypot(u[side], v[side]))]) if side.any() else None
    top_side_ids = np.flatnonzero(side & (y >= 4.75))
    # Preserve duplicate values if element traces disagree at the same coordinate.
    top_samples = sorted(set((float(x[i]), float(y[i]), float(u[i]), float(v[i]))
                             for i in top_side_ids))
    central_lid = lid & (x >= .125-coordinate_atol) & (x <= .875+coordinate_atol)
    nearcorner_lid = lid & ~central_lid

    def lid_region(mask):
        ids = np.flatnonzero(mask)
        if not len(ids):
            return dict(point_count=0, max_velocity_error=None, max_error_location=None)
        worst = int(ids[np.argmax(np.hypot(u[mask]-1, v[mask]))])
        return dict(point_count=len(ids),
            max_velocity_error=float(np.hypot(u[worst]-1, v[worst])),
            max_error_location=dict(x=float(x[worst]), y=float(y[worst]),
                                    u=float(u[worst]), v=float(v[worst])))

    report["candidate_boundary_observations"] = {
        "assumed_domain": "[0,1] x [0,5]; moving lid u=1",
        "lid_max_velocity_error": float(np.max(np.hypot(u[lid]-1, v[lid]))) if lid.any() else None,
        "wall_max_speed": float(np.max(np.hypot(u[walls], v[walls]))) if walls.any() else None,
        "side_max_speed_location": dict(x=float(x[worst_side]), y=float(y[worst_side]),
            u=float(u[worst_side]), v=float(v[worst_side])) if worst_side is not None else None,
        "side_top_samples_columns": ["x", "y", "u", "v"],
        "side_top_samples_y_ge_4_75": top_samples,
        "central_lid_x_0_125_to_0_875": lid_region(central_lid),
        "nearcorner_lid_excluding_endpoints": lid_region(nearcorner_lid),
        "nearcorner_lid_samples_columns": ["x", "y", "u", "v"],
        "nearcorner_lid_samples": sorted(set((float(x[i]), float(y[i]), float(u[i]), float(v[i]))
            for i in np.flatnonzero(nearcorner_lid))),
        "boundary_partition_note": "Fixed first/last 0.125 of unit lid reported separately, not removed from full-lid error; central success does not certify classical lid BC",
        "corner_u_values": sorted(set(float(z) for z in u[corner])),
        "corner_note": "Lid/wall Dirichlet values conflict; reported, not excused by smoothing"}
    report["numeric_comparison_passed"] = all(v["passed"] for v in report["fields"].values())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.atol, args.rtol)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["numeric_comparison_passed"] else 2)
