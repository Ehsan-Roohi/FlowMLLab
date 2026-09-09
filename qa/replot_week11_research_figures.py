"""Replot retained Week 11 arrays without rerunning the research network.

The retained NPZ files store density and coordinates already nondimensionalized
by the source workflow. This utility changes presentation only; it verifies the
recorded NPZ hashes and never claims a new inference run.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from run_week11_research_figures import digest, figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        default=Path('results/week11_research/research_manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.arrays.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; preserve completed plots')
    output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    records = {Path(row['figure']).stem: row for row in manifest['runs']}
    hashes = {}
    for name, row in records.items():
        array_path = source / f'{name}_arrays.npz'
        if digest(array_path) != row['array_sha256']:
            raise ValueError(f'Retained array hash mismatch: {array_path}')
        with np.load(array_path) as archive:
            fields = {key: archive[key] for key in
                      ('x', 'y', 'rho', 'pressure', 'u', 'v',
                       'geometry', 'observation_mask')}
            probability = archive['probabilities']
            masks = [archive['shock'].astype(bool),
                     archive['vortex_core'].astype(bool)]
        plot_row = {
            'case': row['case'],
            'time': row['time'],
            'reference': {'reference_length': 1.0, 'rho_inf': 1.0},
        }
        figure(fields, probability, masks, plot_row, output, name)
        hashes[name] = {
            'source_array_sha256': row['array_sha256'],
            'png_sha256': digest(output / f'{name}.png'),
            'pdf_sha256': digest(output / f'{name}.pdf'),
        }
    report = {
        'operation': 'presentation-only replot; no inference or threshold change',
        'model_label': 'task-preserving Harmonized Joint (HJ)',
        'coordinate_contract': 'retained arrays already use reference-length coordinates',
        'density_contract': 'retained density is normalized by freestream density',
        'items': hashes,
    }
    (output / 'REPLOT_MANIFEST.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'COMPLETE {len(records)} presentation-only replots')


if __name__ == '__main__':
    main()
