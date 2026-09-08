"""Promote a completed local run into a NEW evidence directory, with checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def retain(source):
    protocol=json.loads((source/'protocol.json').read_text())
    rows=json.loads((source/'metrics.json').read_text())
    if protocol['epochs']!=60 or protocol['seeds']!=[17,29,43] or len(rows)!=7:
        raise ValueError('Not the frozen six-fit comparison')
    destination=ROOT/'results/week11_reconstruction'
    if destination.exists(): raise FileExistsError(destination)
    required=['protocol.json','metrics.json','comparison.png','loss.png','plot_provenance.json']
    required += [f'{task}_{seed}{suffix}' for task in ('reconstruction','segmentation')
                 for seed in (17,29,43) for suffix in ('.pt','_loss.npy')]
    for name in required:
        if not (source/name).is_file(): raise FileNotFoundError(name)
    destination.mkdir()
    for name in required:
        if name.endswith('.json'):
            (destination/name).write_bytes((source/name).read_bytes().replace(b'\r\n',b'\n'))
        else:
            shutil.copy2(source/name,destination/name)
    lines=['# Week 11: retained reconstruction / identification comparison','',
           'Three CPU training seeds, complete-case splitting, validation-selected checkpoints.',
           'These are previously generated coarse author LBM wakes, not new CFD or a blind research benchmark.',
           'The mask reference is native-ROI swirling strength, not human annotation. No shock accuracy is claimed.','',
           '| Method | Seed | Velocity L2 | Vorticity L2 | Dice | IoU |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        fmt=lambda key: f"{row[key]:.4f}" if key in row else 'N/A'
        lines.append(f"| {row['method']} | {row['seed'] if row['seed'] is not None else '-'} | {fmt('velocity_relative_l2')} | {fmt('vorticity_relative_l2')} | {fmt('dice')} | {fmt('iou')} |")
    lines += ['', 'L2 values are fractions, not percentages. Dice/IoU are frame-macro scores excluding a two-cell ROI margin.',
              'No best-test model or best-looking frame is selected. Seeds quantify optimizer variability only.','',
              '![Fixed midpoint comparison](comparison.png)','', 'Black contours: each velocity field\'s diagnostic mask. In the direct-mask panel, orange outlines are the native weak reference.',
              'Color limits come from training v; all velocity panels share them. Physical aspect ratio is preserved.',
              'Filled contours interpolate level crossings for display only; all scores use the original 32 x 78 arrays.','',
              '![Training and validation loss](loss.png)','',
              '[Full protocol, provenance and limitations](../../notebooks/week11/RECONSTRUCTION_PROTOCOL.md) · ',
              '[Executable companion](../../notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb)','',
              'Reproduce into a fresh directory: `python qa/run_week11_reconstruction.py --output output/new_w11_run`.',
              'The supplied Ricardo-course code/checkpoint is not redistributed; this is an independent adaptation of the reconstruction-first idea.']
    (destination/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    hashes={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in required+['README.md']}
    (destination/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8',newline='\n')
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    retain(parser.parse_args().source)
