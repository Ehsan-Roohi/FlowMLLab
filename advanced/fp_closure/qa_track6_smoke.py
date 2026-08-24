#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Instructor smoke test for Track 6 without an NVIDIA GPU.

A tiny NumPy-backed CuPy compatibility stub is created in a temporary folder.
This verifies configuration, exact-data generation, deployment-array loading,
closed-loop wrappers, output shapes, and plotting. It does not validate GPU
performance or TensorFlow training.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


CUPY_STUB = r'''
import numpy as _np
from numpy import *
ndarray=_np.ndarray
newaxis=_np.newaxis
float64=_np.float64
float32=_np.float32
int32=_np.int32
int64=_np.int64
random=_np.random
add=_np.add
class _Linalg:
    LinAlgError=_np.linalg.LinAlgError
    @staticmethod
    def solve(a,b):
        a=_np.asarray(a); b=_np.asarray(b)
        if a.ndim>=3 and b.ndim==a.ndim-1:
            return _np.linalg.solve(a,b[...,None])[...,0]
        return _np.linalg.solve(a,b)
linalg=_Linalg()
def asnumpy(x): return _np.asarray(x)
class _Null:
    def synchronize(self): return None
class _Stream: null=_Null()
class _Runtime:
    @staticmethod
    def getDeviceProperties(i): return {'name': b'NumPy-CuPy-stub'}
    @staticmethod
    def getDeviceCount(): return 0
class _Cuda: Stream=_Stream; runtime=_Runtime
cuda=_Cuda()
def __getattr__(name): return getattr(_np,name)
'''


def run(cmd, cwd, env):
    print('RUN:', ' '.join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main():
    here = Path(__file__).resolve().parent
    scripts = [
        'fp_project_utils.py', 'generate_fp_cavity_dataset.py',
        'train_fp_closure.py', 'evaluate_fp_closure.py',
        'run_fp_cavity_test.py', 'analyze_fp_cavity_results.py',
    ]
    for name in scripts:
        compile((here / name).read_text(encoding='utf-8'), name, 'exec')

    with tempfile.TemporaryDirectory(prefix='fp_track6_qa_') as td:
        td = Path(td)
        stub = td / 'stub'
        stub.mkdir()
        (stub / 'cupy.py').write_text(CUPY_STUB, encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(stub) + os.pathsep + env.get('PYTHONPATH', '')

        data = td / 'tiny.npz'
        run([
            sys.executable, 'generate_fp_cavity_dataset.py',
            '--out', str(data), '--role', 'train',
            '--lid-speeds', '100', '--density-scales', '1',
            '--nx', '3', '--ny', '3', '--ppc', '30',
            '--steps', '8', '--sample-start', '2', '--sample-stride', '2',
            '--seed', '1', '--max-abs-coeff', '1e20',
            '--min-particles-per-cell', '1', '--progress-every', '0',
        ], here, env)
        d = np.load(data)
        assert d['inputs'].shape[1] == 16
        assert d['targets'].shape[1] == 9

        X = d['inputs']; y = d['targets']
        rng = np.random.default_rng(0)
        dims = [16, 8, 8, 8, 8, 9]
        model = {
            'X_mean': X.mean(0),
            'X_scale': np.where(X.std(0)>0, X.std(0), 1),
            'y_mean': y.mean(0),
            'y_scale': np.where(y.std(0)>0, y.std(0), 1),
        }
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:]),1):
            model[f'W{i}'] = rng.normal(scale=.01, size=(a,b))
            model[f'b{i}'] = np.zeros(b)
        model_path = td / 'model.npz'
        np.savez(model_path, **model)

        run([sys.executable, 'evaluate_fp_closure.py', '--model', str(model_path),
             '--data', str(data), '--label', 'tiny', '--outdir', str(td/'eval')], here, env)
        run([sys.executable, 'run_fp_cavity_test.py', '--model', str(model_path),
             '--outdir', str(td/'closed'), '--u-lid', '100', '--nx', '3', '--ny', '3',
             '--ppc', '30', '--steps', '8', '--sample-start', '2', '--sample-stride', '2',
             '--seed', '2', '--progress-every', '0'], here, env)
        run([sys.executable, 'analyze_fp_cavity_results.py',
             '--physics', str(td/'closed/fp_cavity_PHYSICS.npz'),
             '--ml', str(td/'closed/fp_cavity_ML.npz'), '--outdir', str(td/'analysis')], here, env)
        assert (td/'analysis/fp_cavity_metrics.csv').exists()
        assert (td/'analysis/fp_centerlines.png').exists()

    print('Track-6 non-GPU smoke test: PASS')
    print('TensorFlow training and real CuPy/GPU timing must be checked in the target environment.')


if __name__ == '__main__':
    main()
