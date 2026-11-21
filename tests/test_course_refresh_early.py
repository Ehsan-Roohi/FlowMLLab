"""Regression checks for the early-course scientific teaching controls."""
from pathlib import Path
import ast,json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
from flowmllab.early_sensing import sensor_experiment

def synthetic_cases():
    rng=np.random.default_rng(37);basis=np.linalg.qr(rng.normal(size=(20,2)))[0]
    return {r:{'x':np.linspace(0,1,5),'y':np.linspace(0,1,4),
        'v':(rng.normal(size=(9,2))@basis.T).reshape(9,4,5)} for r in (90,100,105,110)}

def test_sensor_rank_and_budget_control_reconstruct_known_span():
    cases=synthetic_cases()
    result,_=sensor_experiment(cases,rank=2,budgets=(2,4),noise_fraction=0,seeds=(10,),validation_target=1e-12)
    assert result['selected_budget']==2
    assert result['settings']['rank']==2
    assert max(r['relative_l2'] for r in result['records']if r['method']=='QR')<1e-12
    assert len(result['positions']['QR-2'])==2
    assert len(result['positions']['QR-4'])==4

def test_assessment_fields_cannot_select_sensor_budget_or_positions():
    cases=synthetic_cases()
    first,_=sensor_experiment(cases,rank=2,budgets=(2,4),noise_fraction=.01,seeds=(10,11))
    cases[105]['v']=np.random.default_rng(9).normal(size=cases[105]['v'].shape)
    second,_=sensor_experiment(cases,rank=2,budgets=(2,4),noise_fraction=.01,seeds=(10,11))
    assert first['validation_means']==second['validation_means']
    assert first['selected_budget']==second['selected_budget']
    assert first['positions']==second['positions']

@pytest.mark.parametrize('kwargs',[{'rank':3,'budgets':(2,)},{'rank':2,'budgets':(2,2)},{'rank':2,'noise_fraction':-1},{'rank':2,'seeds':()}])
def test_invalid_sensor_design_rejected(kwargs):
    with pytest.raises(ValueError):sensor_experiment(synthetic_cases(),**kwargs)

def test_project_figures_preserve_pressure_gauge_inputs_and_full_error():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from w5_common import plot_case_evidence
    x=np.linspace(0,1,9);X,Y=np.meshgrid(x,x)
    u=X*(1-X)*Y;v=-Y*(1-Y)*X;p=X**2-Y**2
    data={'Re':np.array([100]),'x':x,'y':x,'u':u[None],'v':v[None],'p':p[None]}
    pred=(u+.5,v,p+17)
    saved=[a.copy()for a in pred]
    figure=plot_case_evidence(data,100,{'known offset':pred})
    assert np.isclose(figure.axes[2].collections[0].norm.vmax,.5)
    for a,b in zip(pred,saved):np.testing.assert_array_equal(a,b)
    assert 'Re = 100' in figure._suptitle.get_text()
    plt.close(figure)

def test_week2_depth_control_constructs_every_hidden_layer():
    tf=pytest.importorskip('tensorflow')
    path=ROOT/'notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    source=next(''.join(c['source'])for c in nb['cells']if 'def build_surrogate' in ''.join(c['source']))
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='build_surrogate')
    scope={'keras':tf.keras,'layers':tf.keras.layers}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    for widths in [(32,),(32,32),(16,8,4)]:
        model=scope['build_surrogate'](widths)
        sizes=[4,*widths,1]
        assert model.count_params()==sum((i+1)*o for i,o in zip(sizes,sizes[1:]))
        assert len(model.layers)==len(widths)+1

def test_halved_timestep_matches_every_post_move_sample_time():
    # Sampling uses (step-start) % stride == 0 after the move at (step+1)*dt.
    for start,steps,stride,dt in [(300,650,10,.003),(700,2200,3,.0025)]:
        old=(np.arange(start,steps,stride)+1)*dt
        matched=(np.arange(2*start+1,2*steps,2*stride)+1)*(dt/2)
        np.testing.assert_allclose(matched,old,rtol=0,atol=1e-14)
        shifted=(np.arange(2*start,2*steps,2*stride)+1)*(dt/2)
        assert not np.allclose(shifted,old,rtol=0,atol=1e-14)
