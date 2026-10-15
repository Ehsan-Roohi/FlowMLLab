"""Experimental higher-Re wrapper; the retained Re=100 solver is unchanged.

No higher-Re accuracy is implied by accepting a configuration. Ghia velocity
tables I/II (1982), DOI 10.1016/0021-9991(82)90058-4, supply comparison data.
The published Re=400 v value at x=.9063 is retained, with a separately named
sensitivity diagnostic that omits that isolated anomalous table entry.
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
        errors[f'ghia_{name}_relative_l2']=float(np.linalg.norm(error)/np.linalg.norm(exact))
        errors[f'ghia_{name}_linf']=float(np.max(np.abs(error)))
        if re==400 and name=='v':
            mask=np.arange(len(exact))!=5
            errors['ghia_v_relative_l2_without_x09063']=float(np.linalg.norm(error[mask])/np.linalg.norm(exact[mask]))
    return errors


def run_experiment(config=ExperimentalConfig(), progress=False):
    # Reuse precisely the original momentum and correction loop. Its legacy
    # Re=100 diagnostics are replaced before any result is returned or saved.
    result=_run(config,progress=progress)
    result.update(benchmark_errors(result))
    result['qualification']='Experimental 2D steady discrete-equation study; inspect mesh and benchmark errors. Not turbulent-flow or transient-accuracy validation.'
    result['nominal_Re_over_N']=config.reynolds/config.cells
    return result
