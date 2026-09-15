"""Experimental depth continuation using the hash-verified private author source.

Each stage is a separate process and directory. Transfer is a warm start of
normalized-coordinate network weights, not an exact physical-field mapping.
CFD fields are reserved for independent evaluation, never used as labels.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(reynolds=100, depths=[2.2, 3.0, 4.0, 5.0],
                    adam_steps=5000, ssb_steps=3000, ssb_phases=2,
                    seed=1234, fixed_sampling=True, lower_anchors=0,
                    stage_loss_gate=1e-5,
                    transfer='normalized-coordinate weight warm start; not exact field mapping')
    protocol_file = root / 'continuation.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('Existing continuation has different parameters')
    protocol_file.write_text(json.dumps(protocol, indent=2))
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, stop)
    checkpoint = None
    for i, depth in enumerate(protocol['depths']):
        if stopping:
            return 99
        stage = root / f'stage-{i}-d{depth:g}'
        done = stage / 'stage-complete.json'
        if done.exists():
            checkpoint = json.loads(done.read_text())['checkpoint']
            if not Path(checkpoint + '.index').is_file():
                raise FileNotFoundError(checkpoint)
            continue
        command = [sys.executable, str(Path(__file__).with_name('run_chris_deep_original.py')),
                   '--archive', str(args.archive.resolve()), '--output', str(stage),
                   '--depth', str(depth), '--reynolds', '100', '--fixed-sampling',
                   '--seed', '1234', '--refine-adam-steps', '5000',
                   '--refine-ssb-steps', '3000', '--ssb-phases', '2']
        if checkpoint:
            command += ['--refine-from', checkpoint, '--refine-adam-lr', '1e-4']
        print(f'STAGE {i}: Re=100 depth={depth}; warm_start={checkpoint}', flush=True)
        child = subprocess.Popen(command)
        status = child.wait()
        if status or stopping:
            return 99 if stopping else status
        state = json.loads((stage / 'resume.json').read_text())
        loss = state.get('last_external_loss', float('nan'))
        if not math.isfinite(loss) or loss > protocol['stage_loss_gate']:
            raise RuntimeError(f'Stage {i} loss {loss} failed progression gate; inspect before continuing')
        checkpoint = state['checkpoint']
        if not os.path.isabs(checkpoint):
            checkpoint = str(stage / checkpoint)
        for suffix in ('.index', '.meta', '.data-00000-of-00001'):
            if not Path(checkpoint + suffix).is_file():
                raise FileNotFoundError(checkpoint + suffix)
        done.write_text(json.dumps(dict(checkpoint=checkpoint, loss=loss, depth=depth), indent=2))
    print('CONTINUATION COMPLETE; independent CFD comparison required', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
