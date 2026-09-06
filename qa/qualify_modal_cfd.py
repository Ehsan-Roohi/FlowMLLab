"""Fail-closed research qualification: grid continuation, never automatic promotion.

The existing source solver and retained evidence are not modified. Run a case
into a NEW directory; inspect returns nonzero until the grid gate passes. Even
a passing grid gate cannot set research_ready without the remaining evidence.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'qa'))
from flowmllab.cylinder_lbm import simulate_cylinder,grid_convergence_diagnostics
from run_cylinder_grid_independence import _settings,_metrics,_save_case,_json_ready,QUANTITY_GATES

SOURCE=ROOT/'results/cylinder_grid_convergence'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return {
        'schema':1,'stage':'solution_verification_only','research_ready':False,
        'case':'Re100_D40_fixed_domain_continuation','settings':_settings(40),
        'solver_sha256':digest(ROOT/'flowmllab/cylinder_lbm.py'),
        'qualification_code_sha256':digest(__file__),
        'grid_metric_code_sha256':digest(ROOT/'qa/run_cylinder_grid_independence.py'),
        'coarse_evidence_sha256':{p:digest(SOURCE/p) for p in ('grid_metrics.csv','re100_D018.npz','re100_D027.npz')},
        'gci_sequence':[18,27,40],'gci_method':'generalized unequal-ratio Richardson, positive monotone order required',
        'quantity_gates':copy.deepcopy(QUANTITY_GATES),
        'next_gate_policy':'No automatic promotion to model training, main, release or front page.',
        'remaining_required':['sampling convergence','domain/blockage sensitivity','Mach sensitivity',
            'physically matched independent validation','independent multi-case long-horizon ML tests'],
    }


def assess(rows):
    rows=sorted(rows,key=lambda r:r['nodes_per_diameter'])
    if [int(r['nodes_per_diameter']) for r in rows]!=[18,27,40]:
        raise ValueError('Require exactly the declared 18,27,40 grid sequence')
    for row in rows:
        # Prevent comparing a larger-domain or differently timed run as grid-only.
        d=int(row['nodes_per_diameter'])
        if (int(row['Re'])!=100 or int(row['nx'])!=20*d or int(row['ny'])!=8*d
            or not np.isclose(row['Mach'],np.sqrt(3)*.05)
            or row['observation_time_D_over_U']!=100 or row['statistics_start_D_over_U']!=45):
            raise ValueError('Incompatible physical/grid-study settings')
    quantities={}
    for q,g in QUANTITY_GATES.items():
        result=grid_convergence_diagnostics([18,27,40],[r[q] for r in rows])
        result['pass']=bool(result['valid_asymptotic_sequence']
            and result['fine_pair_relative_change_percent']<=g['fine_pair_percent']
            and result['fine_grid_gci_percent']<=g['gci_percent'])
        quantities[q]=result
    statistics=all(r['statistical_convergence_pass'] is True or isinstance(r['statistical_convergence_pass'],np.bool_) and bool(r['statistical_convergence_pass']) for r in rows)
    return _json_ready({'grid_gate_pass':statistics and all(r['pass'] for r in quantities.values()),
        'statistics_pass':statistics,'quantities':quantities,'research_ready':False,
        'reason':'Fixed-domain grid gate only; domain, Mach, external validation and independent ML gates remain pending.'})


def write_json(path,value):
    path.write_text(json.dumps(_json_ready(value),indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def run(output,smoke=False):
    if output.exists():raise FileExistsError('Refuse to overwrite evidence: choose a fresh directory')
    output.mkdir(parents=True);p=protocol();write_json(output/'protocol.json',p)
    settings=dict(p['settings'])
    if smoke:
        # Validate the I/O path cheaply, explicitly NOT the 40-node experiment.
        settings.update(nx=80,ny=32,diameter=4,center=(20,15.5),steps=40,
            history_stride=2,statistics_start=20,startup_ramp_steps=20)
    started=time.perf_counter();result=simulate_cylinder(100,**settings)
    result.update(elapsed_seconds=time.perf_counter()-started,evidence_source='smoke_test' if smoke else 'fresh_D40_continuation')
    if smoke:
        np.savez_compressed(output/'smoke.npz',u=result['u'],v=result['v'],rho=result['rho'],
            metadata=np.asarray(json.dumps({'mode':'smoke_only','research_ready':False,'settings':settings})))
        summary={'mode':'smoke_only','actual_settings':settings,'research_ready':False,'grid_gate_pass':False,
            'elapsed_seconds':result['elapsed_seconds'],'finite_fields':bool(np.isfinite(result['u']).all())}
    else:
        _save_case(result,output/'re100_D040.npz')
        row=_metrics(result);write_json(output/'new_grid_metrics.json',row)
        old=pd.read_csv(SOURCE/'grid_metrics.csv')
        rows=old[old.nodes_per_diameter.isin([18,27])].to_dict('records')+[row]
        pd.DataFrame(rows).to_csv(output/'grid_metrics.csv',index=False,lineterminator='\n')
        summary=assess(rows)
        write_json(output/'evidence_hashes.json',{p:digest(output/p) for p in ('re100_D040.npz','new_grid_metrics.json','grid_metrics.csv')})
    write_json(output/'assessment.json',summary)
    print(json.dumps(summary,indent=2))
    return summary


def inspect(output):
    p=json.loads((output/'protocol.json').read_text())
    if p!=json.loads(json.dumps(protocol())):
        raise ValueError('Protocol/source hashes changed; do not silently reassess with different code')
    if not (output/'re100_D040.npz').is_file():raise ValueError('No completed D40 production evidence')
    hashes=json.loads((output/'evidence_hashes.json').read_text())
    if set(hashes)!={'re100_D040.npz','new_grid_metrics.json','grid_metrics.csv'}:
        raise ValueError('Incomplete output inventory')
    for path,expected in hashes.items():
        if digest(output/path)!=expected:raise ValueError('Output checksum mismatch: '+path)
    row=json.loads((output/'new_grid_metrics.json').read_text())
    old=pd.read_csv(SOURCE/'grid_metrics.csv')
    return assess(old[old.nodes_per_diameter.isin([18,27])].to_dict('records')+[row])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','smoke','run','inspect'])
    parser.add_argument('--output',type=Path,default=ROOT/'tmp/modal_cfd_qualification')
    a=parser.parse_args()
    if a.mode=='plan':print(json.dumps(protocol(),indent=2));return 0
    if a.mode=='inspect':s=inspect(a.output);print(json.dumps(s,indent=2));return 0 if s['grid_gate_pass'] else 2
    s=run(a.output,smoke=a.mode=='smoke')
    return 0 if a.mode=='smoke' else (0 if s['grid_gate_pass'] else 2)


if __name__=='__main__':raise SystemExit(main())
