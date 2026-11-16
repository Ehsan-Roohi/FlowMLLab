"""CPU admission for the separately staged, hash-pinned author D5 implementation.

Set FLOWMLLAB_D5_SOURCE to the verified author source directory. These tests are
not a claim of trained scientific accuracy; they admit derivatives and recovery.
"""
import copy
import importlib.util
import os
from pathlib import Path
import random

import numpy as np
import pytest
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('d5_admission', ROOT/'flowmllab/deep_cavity_training.py')
d5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d5)


@pytest.fixture(scope="module", autouse=True)
def preserve_process_torch_settings():
    """Admission settings must not leak into the rest of the package suite."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # Do not initialize CUDA merely to run CPU admission tests.
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@pytest.fixture(scope='module')
def source():
    path = os.getenv('FLOWMLLAB_D5_SOURCE')
    if not path:
        pytest.skip('Authentic author source is separately staged; set FLOWMLLAB_D5_SOURCE')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    return d5.load_author_source(path)


@pytest.fixture
def problem(source):
    return d5.make_problem(source, 12345, width=8, depth=2)


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            assert_tree_equal(x,y)
    else:
        assert a == b


def test_full_paper_architecture(source):
    net, geom = d5.make_problem(source, 12345)
    assert net.mlp.n_params() == 21122
    assert (geom.W,geom.D,geom.Re,geom.y0,geom.kappa,geom.y0_env,geom.beta) == (1.,5.,1000.,-4.,4.212,.25,10.)
    assert geom.dx == pytest.approx(np.sqrt(.001))
    assert geom.dy == pytest.approx(.1)
    assert geom.mask_radius == .01


def test_full_product_rule_and_continuity(source, problem):
    net,g = problem
    x = torch.tensor([[.3],[.71],[.12],[.6]],requires_grad=True)
    y = torch.tensor([[-3.8],[-1.4],[.75],[.98]],requires_grad=True)
    u,v,p,psi = source.cavity._uvp(net,x,y,g)
    px,py = torch.autograd.grad(psi,(x,y),torch.ones_like(psi),create_graph=True)
    torch.testing.assert_close(u,py,rtol=3e-14,atol=3e-16)
    torch.testing.assert_close(v,-px,rtol=3e-14,atol=3e-16)
    ux = torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]
    vy = torch.autograd.grad(v,y,torch.ones_like(v),create_graph=True)[0]
    assert float((ux+vy).detach().abs().max()) < 2e-14
    assert p.dtype == torch.float64


def test_hard_walls_and_gaussian_lid(source, problem):
    net,g = problem
    result = d5.wall_audit(source,net,g,n=513)
    assert max(v for wall in result.values() for v in wall.values()) < 3e-14


def test_envelope_transition_derivatives(source, problem):
    _,g = problem
    y=torch.tensor([[-4.],[-1.],[.75],[1.]],requires_grad=True)
    E,Ey=source.cavity.envelope(y,g)
    ey=torch.autograd.grad(E,y,torch.ones_like(E),create_graph=True)[0]
    torch.testing.assert_close(Ey,ey,rtol=3e-14,atol=1e-16)
    sigmoid=torch.sigmoid(g.beta*((1-y)-g.y0_env))
    eyy_expected=(g.kappa**2*sigmoid.square()-g.kappa*g.beta*sigmoid*(1-sigmoid))*E
    eyy=torch.autograd.grad(ey,y,torch.ones_like(ey),create_graph=True)[0]
    torch.testing.assert_close(eyy,eyy_expected,rtol=2e-13,atol=2e-15)
    eyyy=torch.autograd.grad(eyy,y,torch.ones_like(eyy))[0]
    assert torch.isfinite(eyyy).all()
    assert E[-1].item() == pytest.approx(1.,abs=1e-15)


def test_raw_and_weighted_equations(source, problem):
    net,g=problem
    points=torch.tensor([[.2,.94],[.4,.5],[.7,-2.],[.2,-3.9],[.004,.995]])
    raw=d5.raw_residuals(source,net,points,g)
    r1,r2=source.cavity.residual(net,points,g)
    x,y=points[:,:1],points[:,1:2]
    E,_=source.cavity.envelope(y,g)
    M=(1-torch.exp(-(x.square()+(1-y).square())/.0001))*(1-torch.exp(-((x-1).square()+(1-y).square())/.0001))
    torch.testing.assert_close(r1,raw['rx']*M/E,rtol=2e-13,atol=2e-13)
    torch.testing.assert_close(r2,raw['ry']*M/E,rtol=2e-13,atol=2e-13)


def test_frozen_pair_and_lower_corner_coverage(source):
    a,g=d5.make_problem(source,12345,width=8,depth=2)
    b,_=d5.make_problem(source,12345,width=8,depth=2)
    assert d5.model_digest(a)==d5.model_digest(b)
    X=d5.frozen_points(source,1024,500009)
    assert torch.equal(X,d5.frozen_points(source,1024,500009))
    x,y=X[:,0],X[:,1]
    assert ((x*x+(1-y)**2)>.02**2).all()
    assert (((x-1)**2+(1-y)**2)>.02**2).all()
    assert min(float((x*x+(y+4)**2).min()),float(((x-1)**2+(y+4)**2).min())) < .001**2


def test_chunk_loss_and_gradient_equivalence(source, problem):
    net,g=problem
    X=d5.frozen_points(source,31,555)
    net.zero_grad(set_to_none=True)
    full=d5.loss_backward(source,net,X,g,31)
    grad=[p.grad.clone() for p in net.parameters()]
    net.zero_grad(set_to_none=True)
    parts=d5.loss_backward(source,net,X,g,7)
    torch.testing.assert_close(full,parts,rtol=1e-13,atol=1e-13)
    for before,p in zip(grad,net.parameters()):
        torch.testing.assert_close(before,p.grad,rtol=2e-12,atol=2e-12)


def quadratic_step(net,opt):
    def closure():
        opt.zero_grad(set_to_none=True)
        v=torch.cat([p.flatten() for p in net.parameters()])
        weights=torch.linspace(1.,9.,len(v))
        loss=((v-.13).square()*weights).mean()
        loss.backward()
        return loss
    if isinstance(opt,(torch.optim.Adam,)) or type(opt).__name__=='SOAP':
        closure();opt.step()
    else:
        opt.step(closure)


@pytest.mark.parametrize('method,phase',[('adam_lbfgs','warm'),('soap_ssb','warm'),('adam_lbfgs','qn'),('soap_ssb','qn')])
def test_optimizer_roundtrip_exact(source,method,phase,tmp_path):
    d5.seed_everything(792)
    a=torch.nn.Linear(3,2,dtype=torch.float64)
    oa=d5.make_optimizer(source,a,method,phase,20)
    for _ in range(3):
        quadratic_step(a,oa)
    payload={'model':d5.cpu_copy(a.state_dict()),'optimizer':d5.pack_optimizer(oa),'rng':d5.rng_state()}
    d5.atomic_torch_save(payload,tmp_path/'restart.pt')
    loaded=torch.load(tmp_path/'restart.pt',weights_only=False)
    b=torch.nn.Linear(3,2,dtype=torch.float64)
    b.load_state_dict(loaded['model'])
    ob=d5.make_optimizer(source,b,method,phase,20)
    d5.unpack_optimizer(ob,loaded['optimizer'])
    d5.restore_rng(loaded['rng'])
    for _ in range(3):
        quadratic_step(a,oa)
        quadratic_step(b,ob)
    assert_tree_equal(a.state_dict(),b.state_dict())
    assert_tree_equal(d5.pack_optimizer(oa),d5.pack_optimizer(ob))
    if method=='soap_ssb' and phase=='qn':
        assert oa.H.n > 0
        assert oa.H.dropped == 0


@pytest.mark.parametrize('method',['adam_lbfgs','soap_ssb'])
def test_budget_mid_line_search_rolls_back(source,problem,method):
    net,g=problem
    points=d5.frozen_points(source,8,123)
    opt=d5.make_optimizer(source,net,method,'qn',10)
    before=d5.model_digest(net)
    counts={'qn_evaluations':0,'function_evaluations':0,'gradient_evaluations':0}
    history=[]
    closure=d5.CountedClosure(source,net,points,g,8,opt,1,counts,history,'qn',0,lambda:0.)
    with pytest.raises(d5.EvaluationBudgetReached):
        d5.controlled_step(closure)
    assert d5.model_digest(net)==before
    assert counts['function_evaluations']==counts['gradient_evaluations']==1
    assert len(history)==1
    if hasattr(opt,'H'):
        assert opt.H.n==0 and opt.H.dropped==0


def test_rng_roundtrip():
    d5.seed_everything(159)
    state=d5.rng_state()
    expected=(random.random(),np.random.random(),torch.rand(3))
    d5.restore_rng(state)
    actual=(random.random(),np.random.random(),torch.rand(3))
    assert expected[:2]==actual[:2]
    assert torch.equal(expected[2],actual[2])


def test_cli_resume_freeze_is_immutable(source,tmp_path):
    rspec=importlib.util.spec_from_file_location('d5_runner_test',ROOT/'qa/run_deep_cavity_ablation.py')
    runner=importlib.util.module_from_spec(rspec)
    rspec.loader.exec_module(runner)
    common=['--source-dir',os.environ['FLOWMLLAB_D5_SOURCE'],
            '--method','adam_lbfgs','--warm-evaluations','2','--qn-evaluations','3',
            '--points','8','--chunk-size','8','--audit-points','2',
            '--grid-nx','5','--grid-nh','9','--checkpoint-every','1']
    direct=tmp_path/'direct'
    resumed=tmp_path/'resumed'
    assert runner.main(common+['--output',str(direct)])==0
    assert runner.main(common+['--output',str(resumed),'--max-steps','1'])==0
    assert runner.main(common+['--output',str(resumed),'--resume'])==0
    a=torch.load(direct/'checkpoint.pt',weights_only=False)
    b=torch.load(resumed/'checkpoint.pt',weights_only=False)
    assert_tree_equal(a['model'],b['model'])
    assert_tree_equal(a['optimizer'],b['optimizer'])
    assert a['counters']==b['counters']
    assert a['sample_sha256']==b['sample_sha256']
    before={p.name:d5.sha256(p) for p in resumed.iterdir() if p.is_file()}
    assert runner.main(common+['--output',str(resumed),'--resume'])==0
    after={p.name:d5.sha256(p) for p in resumed.iterdir() if p.is_file()}
    assert before==after
    # Recover an interrupted export from the immutable completed checkpoint.
    import json
    stamp=json.loads((resumed/'manifest.json').read_text())['frozen_at_utc']
    ckhash=d5.sha256(resumed/'checkpoint.pt')
    (resumed/'manifest.json').unlink()
    (resumed/'history.csv').unlink()
    assert runner.main(common+['--output',str(resumed),'--resume'])==0
    assert json.loads((resumed/'manifest.json').read_text())['frozen_at_utc']==stamp
    assert d5.sha256(resumed/'checkpoint.pt')==ckhash
