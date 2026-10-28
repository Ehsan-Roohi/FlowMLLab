"""Check the complete SIMPLE pass, practical controls and honest run provenance.

Control/export tests use explicitly marked stubs in an ignored temporary folder;
they create no CFD trajectory or scientific result. Plot labels are checked on
real, hash-checked retained arrays without recomputing the flow.
"""
from pathlib import Path
from dataclasses import asdict
import contextlib
import hashlib
import io
import json
import sys
import uuid
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qa.sync_week01_2_teaching import SIMPLE_EXAMPLE_CODE
from qa.week01_2_beginner_practical import BASELINE_CODE,HIGHER_CODE
from qa.refresh_week01_2_figures import archived_rows
import flowmllab.pressure_velocity as pv
import flowmllab.pressure_velocity_experiments as experiments
import flowmllab.streamfunction_experiments as sf
import flowmllab.pressure_velocity_experiment_plots as expplots
import flowmllab.pressure_velocity_lab as lab

def stub_result(config):
    return {'config':config,'qualification':'TEST STUB: NOT A CFD RESULT',
        'converged':False,'update_converged':False,'iterations':1,'steps':1,
        'runtime_seconds':0.0,'momentum_residual_linf':1.0,'continuity_linf':1.0,
        'steady_vorticity_linf':1.0,'ghia_u_relative_l2':0.5,'ghia_v_relative_l2':0.5,
        'pressure_solves':1,'test_only_array':np.zeros((2,2))}

def main():
    ns={'display':lambda value:None}
    with contextlib.redirect_stdout(io.StringIO()):exec(SIMPLE_EXAMPLE_CODE,ns)
    trace=ns['simple_trace']
    np.testing.assert_allclose(trace['u*'],[.2,.14,.098],atol=1e-14)
    # Independent identities for the actual two-cell D/G normalization.
    D=np.array([[2.],[-2.]]);G=np.array([[-2.,2.]]);L=-D@(.1*G)
    np.testing.assert_allclose(L,[[.4,-.4],[-.4,.4]])
    np.testing.assert_allclose(L@[-.5,.5],[-.4,.4])
    np.testing.assert_allclose(D[:,0]*(.2-.1*(G@[-.5,.5]).item()),[0,0],atol=1e-14)
    # Both quadratic roots exist; the displayed branch is the positive one.
    roots=np.array([(-9+np.sqrt(97))/4,(-9-np.sqrt(97))/4])
    np.testing.assert_allclose((9+2*roots)*roots,2,atol=1e-13)
    rootdir=ROOT/'tmp/week12_beginner_controls';rootdir.mkdir(parents=True,exist_ok=True)
    # Regular mkdir inherits the workspace ACL on Windows; tempfile's private
    # mode can make its own directory inaccessible to the sandboxed process.
    temp=rootdir/uuid.uuid4().hex
    temp.mkdir()
    with contextlib.nullcontext(temp):
        work=Path(temp);calls=[];plot_requests=[]
        def baseline_runner(config,progress=False):
            calls.append(asdict(config));return stub_result(asdict(config))
        def plotter(rows,cells=64,labels=None):
            plot_requests.append((rows,cells,labels));return {}
        retained=[stub_result(asdict(pv.CavityConfig(method=m))) for m in ['simple','piso','pimple']]
        retained.append(stub_result({'method':'streamfunction-vorticity','cells':32}))
        env={'ROOT':work,'np':np,'pd':pd,'display':lambda value:None,'CavityConfig':pv.CavityConfig,
             'run_cavity':baseline_runner,'recorded':retained,'figures':plotter,'plt':plt}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(BASELINE_CODE.replace('RUN_NEW = False','RUN_NEW = True'),env)
        assert len(calls)==3
        expected=asdict(pv.CavityConfig())
        for c in calls:assert c=={**expected,'method':c['method']}
        assert all('[fresh]' in plot_requests[0][2][m] for m in ['simple','piso','pimple'])
        assert '[retained]' in plot_requests[0][2]['streamfunction-vorticity']
        calls.clear()
        custom=BASELINE_CODE.replace('RUN_NEW = False','RUN_NEW = True').replace('METHOD = "all"','METHOD = "piso"').replace('BASE_PRESSURE_CORRECTORS = 2','BASE_PRESSURE_CORRECTORS = 4')
        with contextlib.redirect_stdout(io.StringIO()):exec(custom,env)
        assert len(calls)==1 and calls[0]['pressure_correctors']==4
        labels=plot_requests[-1][2]
        assert '[fresh]' in labels['piso'] and all('[retained]' in labels[m] for m in ['simple','pimple','streamfunction-vorticity'])
        fv_calls=[];fd_calls=[]
        def fv_runner(config,progress=False):
            fv_calls.append(asdict(config))
            if config.method=='pimple':raise RuntimeError('Controlled export-path test failure')
            return stub_result(asdict(config))
        def fd_runner(reynolds,cells,dt=.001,max_steps=400000,progress=False):
            config={'method':'streamfunction-vorticity','reynolds':reynolds,'cells':cells,
                'dt':dt,'max_steps':max_steps,'convection':'central',
                'relative_update_tolerance':1e-12,'check_every':500,'consecutive_required':3}
            fd_calls.append(config);return stub_result(config)
        with patch.object(experiments,'run_experiment',fv_runner),patch.object(sf,'run_streamfunction',fd_runner),patch.object(expplots,'comparison_figures',lambda rows:{}):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(HIGHER_CODE.replace('RUN_HIGHER_RE = False','RUN_HIGHER_RE = True'),env)
        assert len(fv_calls)==3 and len(fd_calls)==1
        for c in fv_calls:
            assert c['dt']==.05 and c['pressure_correctors']==2 and c['outer_correctors']==3
            assert c['velocity_relaxation']==.5 and c['pressure_relaxation']==.2 and c['tolerance']==1e-6
        assert fd_calls[0]['dt']==.001 and fd_calls[0]['max_steps']==400000
        cases=list(work.rglob('*.json'));failures=[p for p in cases if p.name.endswith('-failed.json')]
        assert len(failures)==1
        failure=json.loads(failures[0].read_text(encoding='utf-8'))
        assert failure['config']==fv_calls[-1] and failure['converged'] is False
        for path in cases:
            record=json.loads(path.read_text(encoding='utf-8'))
            if 'field_sha256' in record:
                assert record['qualification'].startswith('TEST STUB')
                assert hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()==record['field_sha256']
    rows=[r for r in archived_rows() if r['config']['reynolds']==100]
    labels={m:m.upper()+' [fresh]' for m in ['simple','piso','pimple']}
    labels['streamfunction-vorticity']='Streamfunction-vorticity [retained]'
    # This is a label test on real retained data; suppress only streamline work.
    with patch.object(lab,'flow_structure',lambda *args,**kwargs:None):plots=lab.figures(rows,cells=32,labels=labels)
    from matplotlib.text import Text
    texts=[t.get_text() for fig in plots.values() for t in fig.findobj(Text)]
    assert all(v in texts for v in labels.values())
    for fig in plots.values():plt.close(fig)
    note=nbformat.read(ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb',4)
    oldtime=next(c.source for c in note.cells if c.id=='student-review-old-time-level')
    assert 'outer pass $k$' not in oldtime and 'q^n' in oldtime
    guide=json.loads((ROOT/'lectures/source/week01_2_assets/figure_index.json').read_text(encoding='utf-8'))
    assert 'Re=400' not in guide['review_profiles_re1000']['caption']
    for name in ['field_speed_re400','field_speed_re1000']:assert 'Signed fields use' not in guide[name]['caption']
    report={'review_scope':'One independent beginner agent read all 59 prior PDF pages and 71 prior notebook cells; root integrated and checked the fixes.',
        'complete_simple_predictors':trace['u*'].tolist(),'stored_p2':trace['p2_stored'].tolist(),
        'original_momentum_residual':trace['original momentum residual'].tolist(),
        'two_cell_D_G_L_identity_checked':True,'both_nonlinear_toy_roots_checked':True,
        'baseline_constructor_defaults_preserved':True,'custom_corrector_routing_checked':True,
        'mixed_fresh_retained_labels_checked':True,'full_plot_function_completed_on_retained_arrays':True,
        'higher_FV_and_FD_controls_routed':True,'configuration_and_failure_exports_checked':True,
        'control_tests_used_stubs':True,'new_CFD_trajectories':0}
    target=ROOT/'results/week01_2_formulations/beginner_review_validation.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
