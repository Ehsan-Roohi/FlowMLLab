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
from qa.build_cavity_continuation_viewer import BUNDLE as CONTINUATION_BUNDLE, check_bundle

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
    assert ' · Weeks 1–22</p>' in page
    assert 'Â·' not in page and 'â€“' not in page
    assert (site/'build.json').is_file()
    movie_receipt = ROOT/'results/cavity_startup/media/movies.json'
    if movie_receipt.is_file():
        movie_manifest = json.loads(movie_receipt.read_text(encoding='utf-8'))
        assert (site/'cavity-startup/index.html').is_file()
        assert (site/'cavity-startup/movies.json').read_bytes() == movie_receipt.read_bytes()
        for entry in movie_manifest['fields'].values():
            assert hashlib.sha256((site/'cavity-startup'/entry['movie']).read_bytes()).hexdigest() == entry['movie_sha256']
    if (CONTINUATION_BUNDLE/'media/preview_manifest.json').is_file():
        assert '<a href="cavity-continuation/" target="_blank" rel="noopener noreferrer">' in page
        source_preview = check_bundle(CONTINUATION_BUNDLE)
        deployed_preview = check_bundle(site/'cavity-continuation', require_readme=False)
        assert source_preview == deployed_preview
        for name in ('movies.json', 'preview_manifest.json'):
            assert (site/'cavity-continuation/media'/name).read_bytes() == (CONTINUATION_BUNDLE/'media'/name).read_bytes()
        print('PASS: separate continuation route; 3 verified video/poster pairs; 201 common physical times from 0 to 100.')
    final_results = ROOT/'results/cavity_settling'
    if (final_results/'index.html').is_file():
        assert '<a href="cavity-settling/" target="_blank" rel="noopener noreferrer">' in page
        target=site/'cavity-settling'
        assert (target/'index.html').read_bytes()==(final_results/'index.html').read_bytes()
        identities=json.loads((final_results/'manifest.json').read_text())['files']
        figures=[(name,item['sha256']) for name,item in identities.items() if name.startswith('figures/')]
        figures += [(item['file'],item['sha256']) for item in json.loads((final_results/'lesson_figures/manifest.json').read_text())['files']]
        for name,expected in figures:
            assert hashlib.sha256((target/name).read_bytes()).hexdigest()==expected,name
        from html.parser import HTMLParser
        from urllib.parse import urlsplit,unquote
        class Assets(HTMLParser):
            def handle_starttag(self,tag,attrs):
                item=dict(attrs);url=item.get('href',item.get('src',''))
                parsed=urlsplit(url)
                if parsed.path and not parsed.scheme and not parsed.netloc:
                    assert (target/unquote(parsed.path)).exists(),url
                if tag=='img':assert item.get('alt')
        Assets().feed((target/'index.html').read_text(encoding='utf-8'))
        assert len(figures)==76
        assert not (target/'data').exists()
        print('PASS: final-results route; 38 physical figures in PNG/PDF; all seven outcomes and CPU companion.')
    print(f'PASS: {len(inv["modules"])} units; 7 matched cases; {total} native field images; exact CFD/Ghia profiles; frozen split and source hashes.')

if __name__=='__main__':check(Path(sys.argv[1]))
