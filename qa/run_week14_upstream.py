"""Run locally downloaded Davidson scripts; keep upstream code outside publication.

The assembled solver follows global + setup_case body + modify_case + core.
Reassembly is important: the supplied NN executable predates the README fixes.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['baseline', 'pinn', 'nn10000', 'nn5200', 'train_ck', 'balance', 'inverse', 'audit'])
args = parser.parse_args()
source = ROOT / 'tmp/w14'
out = ROOT / 'results/week14_validation'
out.mkdir(parents=True, exist_ok=True)
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
torch = None
if args.mode != 'baseline':
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
np.random.seed(42)
if args.mode != 'baseline':
    import matplotlib
    matplotlib.use('Agg')
started = time.perf_counter()
if args.mode == 'audit':
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    nb = nbformat.read(Path.home() / 'Downloads/W14_pyCALC_RANS_ML_Closure_Audit.ipynb', as_version=4)
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(nb, timeout=300, km=km, resources={'metadata': {'path': str(out)}}).execute()
    nbformat.write(nb, out / 'supplied_notebook_executed.ipynb')
    print('Supplied notebook executed')
    sys.exit(0)

if args.mode in ['train_ck', 'balance', 'inverse']:
    directory = source / ('PINN-NN-' + args.mode)
    if not directory.exists():
        shutil.copytree(source / 'PINN-NN', directory)
    name = {'train_ck':'neural-k-omega-c_k-vist-over-y-and-uv_tot.py',
            'balance':'compute-c_k-and-c_omega_2-from-balance-of-k-and-omega-eqns.py',
            'inverse':'vist-diffusion-pinn-5200-half-channel-load.py'}[args.mode]
    script = (directory / name).read_text()
else:
    case = {'baseline': 'channel-5200-half-channel-yfac1.1',
            'pinn': 'channel-5200-half-channel-PINN-vist-over-y-uv_tot-2nd-submission',
            'nn10000': 'nn10000','nn5200':'nn5200'}[args.mode]
    directory = source / case
    if args.mode=='nn5200' and not directory.exists():
        # Explicit case assembly, NOT an unchanged archived 5200 NN case.
        parent=source/'channel-5200-half-channel-PINN-vist-over-y-uv_tot-2nd-submission'
        shutil.copytree(parent,directory)
        y=np.loadtxt(parent/'y_u_k_om_uv_5200-RANS-half-channel.txt')[:,0]
        coeffs=[np.loadtxt(source/'PINN-NN'/f) for f in
          ['prand_k_5200-plus-units-from-balance-smooth.txt',
           'c_k_pred_5200-plus-units-from-balance.txt',
           'c_omega_2_pred_5200-plus-units-from-balance.txt']]
        np.savetxt(directory/'y-prand_k-c_k-c_omega_2-CFD.txt',np.c_[y,coeffs[0],coeffs[1][:,0],coeffs[2]])
    setup = (directory / 'setup_case.py').read_text().split('def setup_case():', 1)[1]
    modification=(directory / 'modify_case.py').read_text()
    core=(directory / 'pyCALC-RANS.py' if (directory / 'pyCALC-RANS.py').exists()
          else source / 'pyCALC-RANS.py').read_text()
    if args.mode=='nn5200':
        modification=(source/'nn10000/modify_case.py').read_text()
        assert modification.count('t_int = 3000')==1
        modification=modification.replace('t_int = 3000','t_int = 500')
        core=(source/'pyCALC-RANS.py').read_text()
    script = ((source / 'global').read_text() + setup + '\n' +
              modification + '\n' + core)
    (directory / 'flowmllab-assembled.py').write_text(script)
os.chdir(directory)
# __main__ is required by the original full-module PyTorch checkpoint format.
namespace = globals()
with (out / (args.mode + '.log')).open('w', encoding='utf-8') as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        exec(compile(script, str(directory / 'flowmllab-run.py'), 'exec'), namespace)
record = {'mode': args.mode, 'seconds': time.perf_counter()-started,
          'executed_source_sha256': hashlib.sha256(script.encode()).hexdigest(),
          'torch': torch.__version__ if torch else None, 'numpy': np.__version__, 'seed': 42}
if args.mode in ['baseline', 'pinn', 'nn10000','nn5200']:
    record.update(iteration=int(namespace['iter']), residual=float(namespace['resmax']),
                  threshold=float(namespace['sormax']),
                  converged=bool(namespace['resmax'] < namespace['sormax']),
                  restart=True, grid=[int(namespace['ni']), int(namespace['nj'])])
    y = namespace['yp2d'][0]
    u,k,om = [namespace[n][0] for n in ['u2d','k2d','om2d']]
    nu = namespace['viscos']
    record['wall_shear'] = float(nu*u[0]/y[0])
    record['finite'] = bool(np.isfinite([u,k,om]).all())
    if args.mode=='nn5200':
        record['adaptation']='5200 PINN case grid/setup/restart + original NN deployment, m=500 per paper; initialized with released spatial coefficients'
    np.savez_compressed(out / (args.mode+'.npz'), y=y,u=u,k=k,omega=om,
                        viscosity=nu, nut=namespace['vis2d'][0]-nu)
elif args.mode == 'train_ck':
    record.update(test_mse=float(namespace['test_loss']), epochs=int(namespace['N_epochs']))
    np.savez_compressed(out / 'train_ck.npz', y=namespace['y_L'],
                        target=namespace['c_k'], test_indices=namespace['index_test'],
                        prediction=namespace['c_k_NN'], loss=namespace['loss_v'])
(out / (args.mode+'.json')).write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
