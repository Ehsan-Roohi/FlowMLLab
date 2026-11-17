"""Publication gates for the generated browser artifact (no training)."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab import ghia

def check(site):
    site=Path(site);data=site/'assets/cavity'
    manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['reynolds']==list(ghia.REYNOLDS)
    split=manifest['split'];groups=[set(split[k]) for k in ('train_re','validation_re','test_re')]
    assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
    assert groups[2]=={3200,7500}
    for path,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    total=0
    for re in ghia.REYNOLDS:
        case=json.loads((data/f're{re}.json').read_text())
        assert case['re']==re
        role='training' if re in groups[0] else 'validation' if re in groups[1] else 'held-out test'
        assert case['role']==role
        source=np.load(ROOT/f'results/cavity_highre_ghia/n512/re{re}.npz')
        np.testing.assert_array_equal(case['coordinate'],source['x'])
        for component,array in [('u',source['u'][:,256]),('v',source['v'][256,:])]:
            np.testing.assert_array_equal(case['profiles']['CFD'][component],array)
            np.testing.assert_array_equal(case['ghia'][component],ghia.reference(re)[component])
            for model in ('CFD','POD-DeepONet'):
                assert len(case['profiles'][model][component])==513
                assert np.isfinite(case['profiles'][model][component]).all()
        for component in ('u','v'):
            assert (component in case['flags'])==((re,component) in ghia.FLAGGED)
        for field,meta in case['fields'].items():
            shape=[513,513] if field=='speed' else [511,511]
            assert meta['native_shape']==shape
            for kind in ('cfd','nn','error'):
                with Image.open(data/meta[kind]) as image:
                    assert image.mode=='RGB' and list(image.size)==shape
                    image.verify();total+=1
            scale=manifest['scales'][field]
            for low,high in meta['ranges'].values():
                assert scale['min']<=low<=high<=scale['max']
        assert (data/f're{re}-profiles.csv').is_file()
    page=(site/'index.html').read_text(encoding='utf-8')
    inv=json.loads((site/'course_inventory.json').read_text())
    assert page.count('<article class="module"')==len(inv['modules'])
    assert 'https://doi.org/10.1016/j.aitf.2026.100064' in page
    assert 'https://github.com/Ehsan-Roohi/FlowMLLab' in page
    assert (site/'build.json').is_file()
    print(f'PASS: {len(inv["modules"])} units; 7 matched cases; {total} native field images; exact CFD/Ghia profiles; frozen split and source hashes.')

if __name__=='__main__':check(Path(sys.argv[1]))
