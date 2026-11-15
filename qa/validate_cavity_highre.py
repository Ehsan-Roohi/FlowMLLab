"""Recheck native CFD and frozen full-field prediction evidence for release."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab import cavity_highre as cfd,cavity_field_operator as op,cavity_field_evidence as plots

def main():
    cases=cfd.load_cases(ROOT,reynolds=sorted(set(op.TRAIN+op.VALIDATION+op.TEST)))
    assert len(cases)==45
    time_rows,pairs=cfd.load_temporal(ROOT)
    assert len(pairs)==4 and all(row['finite_horizon_time_check_pass'] for row in time_rows)
    fields,info=plots.load(ROOT)
    assert set(info['train_re'])==set(op.TRAIN) and set(info['test_re'])==set(op.TEST)
    assert set(map(int,info['data_read_during_training']))==set(op.TRAIN+op.VALIDATION)
    assert not set(map(int,info['data_read_during_training']))&set(op.TEST)
    for name,expected in info['source_sha256_lf'].items():
        import hashlib
        assert hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,name
    for name,expected in info['model_sha256'].items():assert cfd.file_hash(ROOT/name)==expected
    out=ROOT/'results/cavity_field_deeponet'
    with (out/'test_scores.csv').open() as stream:rows=list(csv.DictReader(stream))
    for re,predictions in fields.items():
        f=predictions['Selected NN'];truth=predictions['CFD']
        error=op.velocity_error(np.r_[f['u'].ravel(),f['v'].ravel()],np.r_[truth['u'].ravel(),truth['v'].ravel()])
        row=next(s for s in rows if int(s['Re'])==re and s['method']=='selected_three_seed_mean')
        np.testing.assert_allclose(error['relative_l2'],float(row['relative_l2']),rtol=1e-12)
    record=dict(native_fields=45,grids=[128,256,512],temporal_pairs=4,
                frozen_models=len(info['model_sha256']),target_leakage_detected=False,
                heldout_scores_recomputed=True,source_hashes_match=True,
                scientific_scope='specified discrete steady branch; mesh independence not established')
    (ROOT/'qa/cavity_highre_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
