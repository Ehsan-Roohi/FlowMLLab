"""Pre-fit checks for paired DeepONet rows; metadata must come from source records."""
import numpy as np


def audit_pairs(branch_case_ids, target_case_ids, query_coordinates,
                target_coordinates, *, atol=1e-12):
    """Reject mismatched case IDs, coordinates, shape, and nonfinite coordinates.

    target_coordinates must be preserved from the solver/export, independently
    of query construction. Matching arrays copied from the same bad source
    cannot establish physical correctness. Coordinate units must already agree.
    Accept coordinates of shape (N,) or (N, d), including 2D cavity queries.
    """
    b, t = np.asarray(branch_case_ids), np.asarray(target_case_ids)
    q, x = np.asarray(query_coordinates), np.asarray(target_coordinates)
    if b.ndim != 1 or t.shape != b.shape or b.size == 0:
        raise ValueError("Case IDs must be nonempty one-dimensional arrays of equal shape")
    if q.ndim not in (1, 2) or q.shape != x.shape or q.shape[0] != b.size:
        raise ValueError("Query and source coordinates must have matching sample dimensions")
    if not np.isfinite(q).all() or not np.isfinite(x).all():
        raise ValueError("Coordinates must be finite")
    if not np.array_equal(b, t):
        raise ValueError("Branch and target case IDs are misaligned")
    if not np.allclose(q, x, rtol=0, atol=atol):
        raise ValueError("Query coordinates do not match the source target coordinates")
    return {"rows": int(b.size), "max_coordinate_error": float(np.max(np.abs(q-x)))}


def audit_fixed_sensors(sensor_coordinates_by_case, *, atol=1e-12):
    """Check identical sensor positions/order, not identical geometry values.

    Coordinates have shape (cases, sensors) or (cases, sensors, dimensions).
    This contract applies to the fixed-sensor branch used in these lessons.
    """
    x = np.asarray(sensor_coordinates_by_case)
    if x.ndim not in (2, 3) or any(n == 0 for n in x.shape):
        raise ValueError("Expected nonempty case-by-sensor coordinates")
    if not np.isfinite(x).all() or not np.allclose(x, x[0], rtol=0, atol=atol):
        raise ValueError("Branch sensor locations/order must be fixed across cases")
    return True


def audit_case_splits(train, validation, test):
    """Use globally unique physical case keys, not row IDs or RNG seeds."""
    groups = [set(v) for v in (train, validation, test)]
    if any(not g for g in groups):
        raise ValueError("Each split must contain physical cases")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Physical cases overlap across splits")
    return True
