"""Execute both companions, reproduce metrics, and prove retained-file isolation."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab.modal_experiments import load_cases,forecast_experiment,sensor_experiment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--in-process',action='store_true');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh verification folder')
    args.output.mkdir(parents=True)
    files=[p for folder in ('results','data','notebooks','lectures') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    retained=json.loads((ROOT/'results/modal_labs/metrics.json').read_text())
    for path,digest in retained['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    with threadpool_limits(limits=1):
        cases=load_cases(ROOT/'data/modal_labs');forecast,_=forecast_experiment(cases);sensors,_=sensor_experiment(cases)
    assert forecast['selected_dmd']==retained['forecast']['selected_dmd']
    assert forecast['selected_sindy']==retained['forecast']['selected_sindy']
    assert sensors['selected_budget']==retained['sensors']['selected_budget']
    # Compare all reported numeric scores, not merely the winning test result.
    def compare(actual,expected,path=''):
        if isinstance(expected,dict):
            assert actual.keys()==expected.keys(),path
            for k,v in expected.items():compare(actual[k],v,path+'/'+str(k))
        elif isinstance(expected,list):
            assert len(actual)==len(expected),path
            for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(i))
        elif isinstance(expected,(int,float)):
            if path.endswith('/phase_error_at_reference_peak'):
                # Phase lives on a circle; relative error is ill-conditioned near zero.
                delta = np.arctan2(np.sin(actual-expected), np.cos(actual-expected))
                assert abs(delta) <= 1e-3, (path,actual,expected)
                return
            assert np.isclose(actual,expected,rtol=.02,atol=1e-5),(path,actual,expected)
        else:assert actual==expected,(path,actual,expected)
    # JSON object keys are strings (the in-memory budget keys are integers).
    compare(json.loads(json.dumps(forecast)),retained['forecast'],'forecast')
    compare(json.loads(json.dumps(sensors)),retained['sensors'],'sensors')
    durations={}
    for relative in ('notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb','notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb'):
        path=ROOT/relative;nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        assert len({c.id for c in nb.cells})==len(nb.cells)
        start=time.perf_counter()
        if args.in_process:
            from check_stabilization_notebooks import execute_in_process
            nb = execute_in_process(nb)
        else:
            NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
        durations[path.name]=time.perf_counter()-start
        assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
        nbformat.write(nb,args.output/path.name)
        body,_=HTMLExporter().from_notebook_node(nb)
        (args.output/(path.stem+'.html')).write_text(body,encoding='utf-8',newline='\n')
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert before==after,'Retained files changed during verification'
    summary={'all_metric_reproduction':'PASS, rtol=2%, atol=1e-5 (cross-library tolerance)',
        'phase_portability':'wrapped absolute tolerance 1e-3 rad; all non-phase gates unchanged',
        'retained_files_unchanged':len(files),'notebook_seconds':durations}
    (args.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n',newline='\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
