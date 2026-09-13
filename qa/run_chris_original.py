"""Execute the hash-verified original attachment; add SSB checkpoint I/O only.

Scientific source is supplied privately, not redistributed here. SOAP is run
unchanged on a fresh start. An interruption before SSB restarts SOAP; after SSB
starts, model, full optimizer state, points, histories and RNG are restored.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import signal
import sys

SHA256 = '68af02f0ca96caadc95dc8be725c7bec85a590868ea1e6b6b1a9a3c933315efa'
HIST = ('loss_hist', 'lp_hist', 'lb_hist', 'steps_hist', 'test_loss_hist',
        'test_lp_hist', 'test_lb_hist', 'test_steps', 'global_step')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Not the original attachment; refusing substitute source')
    source = raw.decode('utf-8')
    tree = ast.parse(source)
    split = next(i for i, n in enumerate(tree.body)
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id == 'X_test' for t in n.targets))
    stage = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == 'USE_SSBROYDEN2')
    loop = next(n for n in stage.body if isinstance(n, ast.For))
    boundary = loop.body[0].lineno
    filename = str(a.source.resolve())
    a.output.mkdir(parents=True, exist_ok=True)
    os.chdir(a.output)
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; refusing CPU fallback')
    if not torch.__version__.startswith('2.10.'):
        raise RuntimeError('Original email requires torch 2.10')
    print('ORIGINAL_SHA256', SHA256, 'GPU', torch.cuda.get_device_name(), flush=True)
    ns = {'__name__': '__chris_original__', '__file__': filename}
    checkpoint = Path('resume.pt')
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    def save(next_k):
        torch.cuda.synchronize()
        data = {k: ns[k] for k in HIST}
        data.update(source_sha256=SHA256, next_k=next_k,
                    model=ns['model'].state_dict(), optimizer=ns['opt_ssb'].state_dict(),
                    X_test=ns['X_test'], Xpde_qn_base=ns['Xpde_qn_base'],
                    numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(data, 'resume.tmp')
        os.replace('resume.tmp', checkpoint)
        print('CHECKPOINT next_ssb_iteration', next_k, flush=True)
    def trace(frame, event, arg):
        if frame.f_code.co_filename != filename or frame.f_code.co_name != '<module>':
            return None
        if event == 'line' and frame.f_lineno == boundary:
            k = ns['k']
            if k % 100 == 0 or stopping:
                save(k)
            if stopping:
                raise SystemExit(99)
        return trace
    if checkpoint.exists():
        exec(compile(ast.Module(body=tree.body[:split], type_ignores=[]), filename, 'exec'), ns)
        d = torch.load(checkpoint, weights_only=False, map_location='cuda')
        if d['source_sha256'] != SHA256:
            raise ValueError('Checkpoint provenance mismatch')
        ns['model'].load_state_dict(d['model'])
        ns.update({k: d[k] for k in HIST})
        for k in ('X_test', 'Xpde_qn_base'):
            ns[k] = d[k]
        # Constructor expression is taken verbatim from the original AST.
        init = next(n for n in stage.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'opt_ssb' for t in n.targets))
        exec(compile(ast.Module(body=[init], type_ignores=[]), filename, 'exec'), ns)
        ns['opt_ssb'].load_state_dict(d['optimizer'])
        np.random.set_state(d['numpy_rng'])
        torch.set_rng_state(d['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in d['cuda_rng']])
        loop.iter.args.insert(0, ast.Constant(d['next_k']))
        tail = tree.body[tree.body.index(stage)+1:]
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop]+tail, type_ignores=[])), filename, 'exec')
    else:
        code = compile(source, filename, 'exec')
    sys.settrace(trace)
    try:
        exec(code, ns)
    finally:
        sys.settrace(None)
    save(ns['SSB_STEPS'])
    Path('COMPLETE').write_text('Original scientific setup completed; accuracy requires separate audit.\n')

if __name__ == '__main__':
    main()
