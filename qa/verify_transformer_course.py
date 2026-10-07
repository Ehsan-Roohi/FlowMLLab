"""Verify source lineage, all retained predictions and independent notebook runs."""
from pathlib import Path
import argparse
import json
import numpy as np
import nbformat
from flowmllab import transformer_course as lab

ROOT=Path(__file__).resolve().parents[1]
def verify(evidence=None,notebooks=True):
    out=Path(evidence) if evidence else ROOT/'results/transformer_course_v3'
    manifest=json.loads((out/'manifest.json').read_text())
    for name,digest in manifest['source_hashes'].items():
        assert lab.canonical_digest(ROOT/name)==digest,('source',name)
    for name,digest in manifest['files'].items():
        assert lab.digest(out/name)==digest,('artifact',name)
    cases,_=lab.load_data(ROOT);archive=np.load(out/'predictions.npz',allow_pickle=False)
    checked=0
    for experiment in ('sensors','forecast','transfer'):
        record=json.loads((out/f'{experiment}.json').read_text())
        for row in record['rows']:
            pred=lab.prediction(archive,experiment+'__'+row['key'])
            if experiment=='sensors':truth=cases[105]['v'].reshape(281,-1)
            else:
                truth=cases[row.get('target_re',110)]['omega'][210:].reshape(71,-1);pred=pred[50:]
            # Independent scoring must promote before reduction, as in the metric definition.
            pred=np.asarray(pred,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
            actual=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth))
            assert np.isclose(actual,row['metrics']['field_relative_l2'],rtol=1e-6,atol=1e-8),(experiment,row['key'])
            if row.get('method')!='Persistence':assert abs(row['metrics']['pythagorean_residual'])<1e-7
            if experiment=='transfer':
                labels=row['labels'];union=set(labels['training_frames'])|set(labels['validation_frames'])|set(labels['initialization_frames'])
                assert len(union)==labels['total_target_labels'] and max(union)<160
            checked+=1
    if notebooks:
        for w in range(17,23):
            for p in (ROOT/f'notebooks/week{w}/W{w}_CFD_Transformer.ipynb',ROOT/f'instructor/week{w}/W{w}_Solutions.ipynb'):
                nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
                cells=[c for c in nb.cells if c.cell_type=='code']
                assert [c.execution_count for c in cells]==list(range(1,len(cells)+1)),p
                assert not any(o.output_type=='error' for c in cells for o in c.outputs),p
                serialized=json.dumps([c.outputs for c in cells])
                assert r'C:\\Users\\' not in serialized and 'C:/Users/' not in serialized,p
                if 'instructor' in p.parts:
                    assert any('Coding tasks passed: 4 / 4' in o.get('text','') for c in cells for o in c.outputs),p
            assert (ROOT/f'lectures/week{w}_cfd_transformer.pdf').read_bytes().startswith(b'%PDF')
    print(f'PASS: lineage, hashes and {checked} independently rescored prediction rows; notebooks={notebooks}')
    return checked

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path);parser.add_argument('--evidence-only',action='store_true');args=parser.parse_args()
    verify(args.evidence,not args.evidence_only)
