"""Experimental higher-Re wrapper; the retained Re=100 solver is unchanged.

No higher-Re accuracy is implied by accepting a configuration. Ghia velocity
tables I/II (1982), DOI 10.1016/0021-9991(82)90058-4, supply comparison data.
The published Re=400 v value at x=.9063 is preserved for provenance, but
excluded from qualified error norms as a known anomalous table entry. Full
unfiltered norms remain available as explicitly named diagnostics.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .pressure_velocity import CavityConfig, run_cavity as _run, centerlines, GHIA_Y, GHIA_X

REFERENCE = {
    400: {
        'u': [1,.75837,.68439,.61756,.55892,.29093,.16256,.02135,-.11477,-.17119,-.32726,-.24299,-.14612,-.10338,-.09266,-.08186,0],
        'v': [0,-.12146,-.15663,-.19254,-.22847,-.23827,-.44993,-.38598,.05186,.30174,.30203,.28124,.22965,.20920,.19713,.18360,0]},
    1000: {
        'u': [1,.65928,.57492,.51117,.46604,.33304,.18719,.05702,-.06080,-.10648,-.27805,-.38289,-.29730,-.22220,-.20196,-.18109,0],
        'v': [0,-.21388,-.27669,-.33714,-.39188,-.51550,-.42665,-.31966,.02526,.32235,.33075,.37095,.32627,.30353,.29012,.27485,0]}}


GHIA_RE400_V_ANOMALOUS_X = 0.9063


def benchmark_valid_mask(reynolds, component, positions):
    """Validity mask for published Ghia benchmark coordinates.

    Ghia et al. (1982), Table II, reports v/U=-0.23827 at Re=400,
    x/L=0.9063. This isolated entry is a known tabulation anomaly.
    Preserve the reported value; do not invent a replacement.
    """
    positions = np.asarray(positions)
    if int(reynolds) == 400 and component == 'v':
        return ~np.isclose(positions, GHIA_RE400_V_ANOMALOUS_X, rtol=0, atol=1e-5)
    return np.ones(positions.shape, dtype=bool)


@dataclass(frozen=True)
class ExperimentalConfig(CavityConfig):
    reynolds: float = 400.

    def __post_init__(self):
        if not np.isfinite(self.reynolds) or self.reynolds not in REFERENCE:
            raise ValueError('Experimental branch currently supports only Re=400 or 1000')
        # Validate every other parameter through the existing qualified config.
        CavityConfig(**{**asdict(self), 'reynolds': 100.})


def benchmark_errors(result):
    re=int(result['config']['reynolds']); ref=REFERENCE[re]
    y,u,x,v=centerlines(result)
    errors={}
    for name,positions,axis,values in [('u',GHIA_Y,y,u),('v',GHIA_X,x,v)]:
        exact=np.asarray(ref[name]); error=np.interp(positions,axis,values)-exact
        mask=benchmark_valid_mask(re,name,positions)
        errors[f'ghia_{name}_relative_l2']=float(np.linalg.norm(error[mask])/np.linalg.norm(exact[mask]))
        errors[f'ghia_{name}_linf']=float(np.max(np.abs(error[mask])))
        if re==400 and name=='v':
            errors['ghia_v_relative_l2_including_x09063']=float(np.linalg.norm(error)/np.linalg.norm(exact))
            errors['ghia_v_linf_including_x09063']=float(np.max(np.abs(error)))
            # Backward-compatible sensitivity metric; now also the qualified norm.
            errors['ghia_v_relative_l2_without_x09063']=errors['ghia_v_relative_l2']
    return errors


def run_experiment(config=ExperimentalConfig(), progress=False):
    # Reuse precisely the original momentum and correction loop. Its legacy
    # Re=100 diagnostics are replaced before any result is returned or saved.
    result=_run(config,progress=progress)
    result.update(benchmark_errors(result))
    result['qualification']='Experimental 2D steady discrete-equation study; inspect mesh and benchmark errors. Not turbulent-flow or transient-accuracy validation.'
    result['nominal_Re_over_N']=config.reynolds/config.cells
    return result
