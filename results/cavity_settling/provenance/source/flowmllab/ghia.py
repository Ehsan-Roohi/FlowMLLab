"""All seven square-cavity centreline benchmarks from Ghia et al. (1982).

Tables I/II, JCP 48, 387--411, doi:10.1016/0021-9991(82)90058-4.
Numbers are published samples, not full fields or mesh-converged truth.
Questionable printed entries are retained; screened and unfiltered scores
are separate. No replacement values are invented. See references/ghia/.
"""
from __future__ import annotations

import numpy as np

REYNOLDS = (100, 400, 1000, 3200, 5000, 7500, 10000)
Y = np.array([1., .9766, .9688, .9609, .9531, .8516, .7344, .6172,
              .5, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.])
X = np.array([1., .9688, .9609, .9531, .9453, .9063, .8594, .8047,
              .5, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0.])
_U = np.array([
 [1,1,1,1,1,1,1],
 [.84123,.75837,.65928,.53236,.48223,.47244,.47221],
 [.78871,.68439,.57492,.48296,.46120,.47048,.47783],
 [.73722,.61756,.51117,.46547,.45992,.47323,.48070],
 [.68717,.55892,.46604,.46101,.46036,.47167,.47804],
 [.23151,.29093,.33304,.34682,.33556,.34228,.34635],
 [.00332,.16256,.18719,.19791,.20087,.20591,.20673],
 [-.13641,.02135,.05702,.07156,.08183,.08342,.08344],
 [-.20581,-.11477,-.06080,-.04272,-.03039,-.03800,.03111],
 [-.21090,-.17119,-.10648,-.86636,-.07404,-.07503,-.07540],
 [-.15662,-.32726,-.27805,-.24427,-.22855,-.23176,-.23186],
 [-.10150,-.24299,-.38289,-.34323,-.33050,-.32393,-.32709],
 [-.06434,-.14612,-.29730,-.41933,-.40435,-.38324,-.38000],
 [-.04775,-.10338,-.22220,-.37827,-.43643,-.43025,-.41657],
 [-.04192,-.09266,-.20196,-.35344,-.42901,-.43590,-.42537],
 [-.03717,-.08186,-.18109,-.32407,-.41165,-.43154,-.42735],
 [0,0,0,0,0,0,0],
])
_V = np.array([
 [0,0,0,0,0,0,0],
 [-.05906,-.12146,-.21388,-.39017,-.49774,-.53858,-.54302],
 [-.07391,-.15663,-.27669,-.47425,-.55069,-.55216,-.52987],
 [-.08864,-.19254,-.33714,-.52357,-.55408,-.52347,-.49099],
 [-.10313,-.22847,-.39188,-.54053,-.52876,-.48590,-.45863],
 [-.16914,-.23827,-.51550,-.44307,-.41442,-.41050,-.41496],
 [-.22445,-.44993,-.42665,-.37401,-.36214,-.36213,-.36737],
 [-.24533,-.38598,-.31966,-.31184,-.30018,-.30448,-.30719],
 [.05454,.05186,.02526,.00999,.00945,.00824,.00831],
 [.17527,.30174,.32235,.28188,.27280,.27348,.27224],
 [.17507,.30203,.33075,.29030,.28066,.28117,.28003],
 [.16077,.28124,.37095,.37119,.35368,.35060,.35070],
 [.12317,.22965,.32627,.42768,.42951,.41824,.41487],
 [.10890,.20920,.30353,.41906,.43648,.43564,.43124],
 [.10091,.19713,.29012,.40917,.43329,.44030,.43733],
 [.09233,.18360,.27485,.39560,.42447,.43979,.43983],
 [0,0,0,0,0,0,0],
])
GHIA = {re: {"x": X.copy(), "y": Y.copy(), "u": _U[:, i].copy(),
             "v": _V[:, i].copy()} for i, re in enumerate(REYNOLDS)}

# A predeclared transcription/benchmark sensitivity policy, not data cleaning
# based on the model under test. Always also report the published-data score.
FLAGGED = {(400, "v"): (.9063, "isolated tabulation anomaly"),
           (3200, "u"): (.4531, "printed -0.86636; suspected decimal error"),
           (10000, "u"): (.5, "printed +0.03111; suspected missing minus sign")}


def reference(reynolds):
    """Return a copy; reject off-grid Re instead of rounding it to a case."""
    if reynolds not in REYNOLDS:
        raise ValueError(f"Ghia reference requires one of {REYNOLDS}, got {reynolds}")
    return {k: v.copy() for k, v in GHIA[reynolds].items()}


def valid_mask(reynolds, component, positions=None):
    ref = reference(reynolds)
    if component not in ("u", "v"):
        raise ValueError("component must be u or v")
    coord = np.asarray(positions if positions is not None else ref['y' if component == 'u' else 'x'])
    flag = FLAGGED.get((reynolds, component))
    return np.ones(coord.shape, bool) if flag is None else ~np.isclose(coord, flag[0], atol=1e-5, rtol=0)


def profile_errors(reynolds, component, positions, values):
    """Interpolate a simulated profile to the original tabulated positions.

    Both norms use a single reference norm; never divide by pointwise velocity.
    Coordinates must span [0,1]. Extrapolation is an error, not a wall value.
    """
    ref = reference(reynolds)
    coord = ref['y' if component == 'u' else 'x']
    positions, values = np.asarray(positions), np.asarray(values)
    if positions.ndim != 1 or positions.shape != values.shape or len(positions) < 2:
        raise ValueError('Expected equal one-dimensional coordinate/profile arrays')
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite profile')
    if not np.all(np.diff(positions) > 0) or positions[0] > 0 or positions[-1] < 1:
        raise ValueError('Profile coordinates must increase and span [0,1]')
    prediction = np.interp(coord, positions, values)
    err, truth = prediction-ref[component], ref[component]
    keep = valid_mask(reynolds, component)
    return {'screened_relative_l2': float(np.linalg.norm(err[keep])/np.linalg.norm(truth[keep])),
            'published_relative_l2': float(np.linalg.norm(err)/np.linalg.norm(truth)),
            'screened_rmse_U': float(np.sqrt(np.mean(err[keep]**2))),
            'screened_linf_U': float(np.max(np.abs(err[keep]))),
            'n_screened': int(keep.sum()), 'n_published': len(keep)}


def case_roles():
    """Frozen whole-Re split for the separate centreline-surrogate exercise."""
    return {re: ('validation' if re == 400 else 'test' if re in (3200, 7500) else 'train')
            for re in REYNOLDS}
