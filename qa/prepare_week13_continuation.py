"""Copy completed 20k CFD evidence to isolated continuation directories."""
import argparse
import json
import shutil
from pathlib import Path


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be disjoint')
    record = dict(source=str(source), initial_iteration=20000, target_iteration=120000,
                  status='continuation-candidate-not-validated')
    marker=destination/'continuation.json'
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()) != record:
            raise ValueError('Existing destination is not a complete prepared continuation')
        return
    if not (source/'solver-finished-unvalidated').is_file():
        raise ValueError('Source solver has not finished')
    for proc in range(8):
        for field in ('U','p','phi'):
            p=source/f'processor{proc}'/'20000'/field
            if not p.is_file() or not p.stat().st_size:
                raise ValueError(f'Missing checkpoint: {p}')
    shutil.copytree(source,destination)
    (destination/'solver-finished-unvalidated').rename(destination/'initial-solver-finished-unvalidated')
    marker.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args(); prepare(args.source,args.destination)
