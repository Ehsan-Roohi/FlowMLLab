"""Final teaching gate: complete evidence, bound visual reviews and fresh Run All.

This gate does not train models or certify scientific accuracy. Sequential mode
executes plain Python cells in a fresh child process; Jupyter mode uses a real
nbclient kernel. Source notebooks are never overwritten.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
BOOKS = {
    'ghia': 'notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb',
    'd5': 'notebooks/week13/W13_D5_Controlled_Optimizer_Study.ipynb',
}
PDFS = {'week04_cavity_surrogates_deeponet': 34, 'week13_rectangular_cavity_pinn': 18}
MARKER = 'FLOWMLLAB_CURRICULUM_PROBE='


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def entry(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Teaching artifact leaves checkout')
    return dict(path=path.relative_to(ROOT.resolve()).as_posix(), sha256=sha(path))


def checked(item):
    path = (ROOT/item['path']).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path) == item['sha256'],
            'Reviewed teaching artifact changed: '+item['path'])
    return path


def image_entries(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value and str(value['path']).lower().endswith('.png'):
            yield value
        for child in value.values(): yield from image_entries(child)
    elif isinstance(value, list):
        for child in value: yield from image_entries(child)


def figure_pdf_entries(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value and str(value['path']).lower().endswith('.pdf'):
            yield value
        for child in value.values(): yield from figure_pdf_entries(child)
    elif isinstance(value, list):
        for child in value: yield from figure_pdf_entries(child)


def verify_figure_pdf(item):
    from pypdf import PdfReader
    path = checked(item)
    pages = len(PdfReader(path).pages)
    require(item['pages'] == pages and pages > 0, 'Reviewed standalone figure PDF page count changed')
    rows = item['individual_page_reviews']
    require(len(rows) == pages and {row['page'] for row in rows} == set(range(1,pages+1)),
            'Every standalone figure PDF page requires an individual review')
    require(all(re.fullmatch('[0-9a-f]{64}', row['png_sha256'])
                and (row.get('status') or row.get('visual_review')) for row in rows),
            'Standalone figure PDF page lacks a rendered-review identity')
    require(not item.get('out_of_bounds_text',[]), 'Unresolved standalone figure PDF layout defect')
    return dict(**entry(path), pages=pages, individually_reviewed_pages=len(rows))


def family(path):
    text = str(path).replace('\\', '/')
    if '/cavity_pinn_stratified/' in text: return 'stratified'
    if '/deep_cavity_ablation/' in text: return 'd5'
    if '/cavity_pinn_campaign/' in text: return 'ghia'
    return 'other'



def verify_week04_replacement(legacy, evidence):
    """Bind the deliberately rebuilt guide without claiming old-page identity.

    Render caches are expendable, unlike the compact review record. Present
    caches are checked byte for byte; absent caches do not erase a recorded
    individual visual inspection or imply that it was repeated by this gate.
    """
    from pypdf import PdfReader
    review_path = ROOT/'qa/week04_lecture_review.json'
    review = read_json(review_path)
    require(review.get('schema_version') == 1
            and review.get('review_type') == 'week04_full_guide_replacement',
            'Unsupported Week4 replacement review version')
    require(review.get('status') == 'complete',
            'Week4 replacement review is pending or incomplete')
    require(review.get('pdf') == 'lectures/week04_cavity_surrogates_deeponet.pdf'
            and review.get('source') == 'lectures/source/week04_lab_guide.py'
            and review.get('evidence_source') == evidence['path'],
            'Week4 replacement review names an unexpected artifact')
    require(review.get('pre_revision') == dict(pages=legacy['total_pages'], sha256=legacy['sha256']),
            'Week4 replacement is not bound to the reviewed historical PDF')
    require((review.get('guide_pages'), review.get('retained_evidence_pages'), review.get('total_pages')) == (49,54,103),
            'Week4 replacement requires 49 guide and 54 retained evidence pages')
    require(review.get('evidence_source_sha256') == evidence['sha256'],
            'Week4 scientific evidence source differs from the historical review')
    pdf_item = dict(path=review['pdf'], sha256=review['pdf_sha256'])
    source_item = dict(path=review['source'], sha256=review['source_sha256'])
    evidence_item = dict(path=review['evidence_source'], sha256=review['evidence_source_sha256'])
    pdf_path = checked(pdf_item)
    checked(source_item); checked(evidence_item)
    require(len(PdfReader(pdf_path).pages) == 103,
            'Week4 replacement PDF page count differs from its review')
    guide_map = review.get('page_map', [])
    appendix_map = review.get('appendix_map', [])
    require([row.get('page') for row in guide_map] == list(range(1,50))
            and all(isinstance(row.get('title'),str) and row['title'].strip() for row in guide_map),
            'Week4 guide page map is incomplete')
    require([row.get('page') for row in appendix_map] == list(range(50,104))
            and [row.get('id') for row in appendix_map] == [f'E{i:02d}' for i in range(1,55)]
            and all(isinstance(row.get('title'),str) and row['title'].strip() for row in appendix_map),
            'Week4 retained evidence page map is incomplete')
    visual = review.get('visual_review', {})
    require(visual.get('pdf_sha256') == review['pdf_sha256'],
            'Week4 visual review is not bound to the current PDF')
    require(isinstance(visual.get('dpi'), (int,float)) and visual['dpi'] >= 125,
            'Week4 individual page review resolution is insufficient')
    require(visual.get('out_of_bounds_text') == [],
            'Week4 replacement has an unresolved or unchecked layout defect')
    rows = visual.get('pages', [])
    require(len(rows) == 103 and {row.get('page') for row in rows} == set(range(1,104)),
            'Every Week4 replacement page requires an individual review')
    require(all(isinstance(row.get('image_sha256'),str)
                and re.fullmatch('[0-9a-f]{64}',row['image_sha256'])
                and row.get('status') == 'passed'
                and isinstance(row.get('reviewer'),str) and row['reviewer'].strip()
                and isinstance(row.get('review_basis'),str) and row['review_basis'].strip()
                for row in rows),
            'Week4 replacement page lacks a passed rendered-review identity')
    rendered_files_checked = 0
    for row in rows:
        if not row.get('image_path'): continue
        image_path = (ROOT/row['image_path']).resolve()
        require(image_path.is_relative_to(ROOT.resolve()) and image_path.suffix.lower() == '.png',
                'Week4 page render leaves the checkout or is not a PNG')
        if image_path.exists():
            checked(dict(path=row['image_path'],sha256=row['image_sha256']))
            rendered_files_checked += 1
    return dict(pdf=dict(**pdf_item, pages=103, guide_pages=49, retained_evidence_pages=54,
                         preserved_pages=0, individually_reviewed_pages=103,
                         review_mode='fully_reviewed_replacement; no old-page identity claim',
                         available_render_files_checked=rendered_files_checked),
                review=entry(review_path), source=source_item, evidence=evidence_item)

def verify_reviews():
    from pypdf import PdfReader
    from PIL import Image
    study_path = ROOT/'qa/cavity_pinn_studies_validation.json'
    studies = read_json(study_path)
    require(studies['validator_sha256'] == sha(ROOT/'qa/validate_cavity_pinn_studies.py'), 'Scientific validation receipt is stale')
    require(studies['all_studies_present'] is True
            and studies['release_evidence_check'] == 'passed_for_available_evidence'
            and all(studies[name]['integrity'] == 'passed' for name in ('ghia', 'd5', 'stratified')),
            'All three complete scientific evidence families are required')
    lecture_path = ROOT/'qa/cavity_pinn_lecture_review.json'
    lecture = read_json(lecture_path)
    require(lecture['status'] == 'complete', 'Final lecture review is pending or incomplete')
    require(lecture['evidence_status'] == dict(ghia='complete_failed_baseline', d5='complete', stratified='complete'),
            'Lecture review does not cover all final evidence')
    science_manifests = []
    bindings = (
        ('results/cavity_pinn_campaign/evaluation_manifest.json', studies['ghia']['manifest_sha256']),
        ('results/cavity_pinn_campaign/sampling_coverage.json', studies['ghia']['sampling_coverage_sha256']),
        ('results/deep_cavity_ablation/evaluation_manifest.json', studies['d5']['manifest_sha256']),
        ('results/cavity_pinn_stratified/experiment_manifest.json', studies['stratified']['experiment_manifest_sha256']),
        ('results/cavity_pinn_stratified/comparison/comparison_manifest.json', studies['stratified']['comparison_manifest_sha256']),
        ('results/cavity_pinn_stratified/evaluation_manifest.json', studies['stratified']['native_integrity']['manifest_sha256']),
    )
    for name, digest in bindings:
        item = dict(path=name, sha256=digest)
        checked(item)
        science_manifests.append(item)
    lecture_sources = {name: dict(lecture[name]) for name in ('source','preparer')}
    for item in lecture_sources.values(): checked(item)
    pdfs = {}; appended = 0; replacement_reviews = {}
    for name, original in PDFS.items():
        reviewed = lecture['pdfs'][name]; path = ROOT/'lectures'/(name+'.pdf')
        replacement = None
        if name == 'week04_cavity_surrogates_deeponet' and sha(path) != reviewed['sha256']:
            replacement = verify_week04_replacement(reviewed, lecture_sources['source'])
            # Validate the historical receipt on its own terms below. Its
            # preserved-page assertion is not carried onto the rebuilt guide.
            pages = reviewed['total_pages']
        else:
            require(sha(path) == reviewed['sha256'], 'PDF changed after individual visual review: '+name)
            pages = len(PdfReader(path).pages)
        require(reviewed['original_pages'] == original and reviewed['total_pages'] == pages and pages > original,
                'PDF page counts/preserved extent differ')
        require(not reviewed['old_page_text_changes'] and not reviewed['old_page_rgb72_changes']
                and not reviewed['out_of_bounds'], 'Unresolved page-preservation/layout defect')
        review_rows = reviewed['added_page_visual_reviews']
        require(len(review_rows) == pages-original and {r['page'] for r in review_rows} == set(range(original+1,pages+1)),
                'Every appended PDF page requires an individual review')
        require(all(re.fullmatch('[0-9a-f]{64}', r['png_sha256']) and r.get('visual_review') for r in review_rows),
                'Missing individual rendered-page review identity')
        if replacement is None:
            pdfs[name] = dict(**entry(path), pages=pages, preserved_pages=original, individually_reviewed_pages=pages-original)
        else:
            pdfs[name] = replacement['pdf']
            replacement_reviews[name] = replacement['review']
            lecture_sources['week04_replacement_source'] = replacement['source']
        appended += pages-original
    require(lecture['preserved_pages'] == sum(PDFS.values()) == 52
            and lecture['individually_reviewed_added_pages'] == appended, 'Lecture review totals differ')
    figure_path = ROOT/'qa/cavity_pinn_figure_review.json'
    figures = read_json(figure_path)
    require(figures['status'] == 'complete', 'Final figure review is pending or incomplete')
    images = {}; hashes = {name:set() for name in ('ghia','d5','stratified')}
    for item in image_entries(figures):
        path = checked(item)
        if path in images: continue
        with Image.open(path) as image:
            require(image.mode == 'RGB', 'Reviewed scientific figure is not RGB')
            if 'mode' in item: require(item['mode'] == image.mode, 'Image review mode changed')
            if 'dimensions' in item: require(item['dimensions'] == list(image.size), 'Image review dimensions changed')
        require(bool(item.get('visual_review') or item.get('status')), 'Figure lacks an explicit visual-review record')
        images[path] = entry(path)
        role = family(path)
        if role in hashes: hashes[role].add(item['sha256'])
    require(all(hashes.values()), 'Reviewed physical figures for Ghia, D5 and stratified studies are all required')
    figure_pdfs = {}; pdf_families = set()
    for item in figure_pdf_entries(figures):
        verified = verify_figure_pdf(item)
        figure_pdfs[verified['path']] = verified
        pdf_families.add(family(verified['path']))
    require({'ghia','d5','stratified'}.issubset(pdf_families),
            'Reviewed standalone figure PDFs for all three studies are required')
    return dict(studies=entry(study_path), lecture_review=entry(lecture_path), figure_review=entry(figure_path),
                scientific_manifests=science_manifests, lecture_sources=lecture_sources,
                replacement_lecture_reviews=replacement_reviews, pdfs=pdfs, figure_pdfs=list(figure_pdfs.values()), images=list(images.values()),
                image_hashes_by_family={key:sorted(value) for key,value in hashes.items()})


def local_links(book, path):
    missing = []
    for cell in book.cells:
        if cell.cell_type != 'markdown': continue
        for text in re.findall(r'\]\(([^)]+)\)', cell.source):
            text = text.strip().split(' "')[0].strip('<>')
            if not text or text.startswith('#') or urlparse(text).scheme: continue
            target = (path.parent/unquote(text.split('#',1)[0].split('?',1)[0])).resolve()
            if not target.exists(): missing.append(text)
    require(not missing, 'Missing local notebook links: '+repr(missing))
    return []


def probe_source(kind):
    common = """import json as _probe_json
assert isinstance(CAMPAIGN, dict)
_probe_rows = CAMPAIGN['records']
"""
    if kind == 'ghia':
        check = """assert len(_probe_rows) == 42
assert all(row['outcome'] == 'scored' for row in _probe_rows)
assert {(row['Re'],row['seed'],row['collar']) for row in _probe_rows} == {(re,seed,eps) for re in (100,400,1000,3200,5000,7500,10000) for seed in (7,23,61) for eps in (.01,.02)}
assert isinstance(STRATIFIED, dict) and len(STRATIFIED['records']) == 12
assert {(row['Re'],row['seed'],row['collar']) for row in STRATIFIED['records']} == {(re,seed,eps) for re in (100,7500) for seed in (7,23,61) for eps in (.01,.02)}
_probe = dict(ghia_cases=42,stratified_pairs=12)
"""
    else:
        check = """assert len(_probe_rows) == 6
assert all(row['outcome'] == 'controlled_pilot_scored' for row in _probe_rows)
assert {(row['seed'],row['method']) for row in _probe_rows} == {(seed,method) for seed in (12345,23456,34567) for method in ('adam_lbfgs','soap_ssb')}
_probe = dict(d5_controlled_arms=6)
"""
    return common+check+"print('"+MARKER+"'+_probe_json.dumps(_probe,sort_keys=True))"


def execute_one(kind, mode, destination):
    import nbformat
    path = ROOT/BOOKS[kind]; before = sha(path)
    book = nbformat.read(path, as_version=4); nbformat.validate(book)
    require(book.metadata.get('flowmllab',{}).get('launches_training') is False, 'Notebook must declare evidence-only Run All')
    local_links(book, path)
    count = sum(cell.cell_type == 'code' for cell in book.cells)
    book.cells.append(nbformat.v4.new_code_cell(probe_source(kind)))
    initialization_cells = 0
    if mode == 'jupyter':
        from nbclient import NotebookClient
        book.cells.insert(0, nbformat.v4.new_code_cell("get_ipython().run_line_magic('matplotlib', 'inline')"))
        initialization_cells = 1
        NotebookClient(book, timeout=240, kernel_name='python3', allow_errors=False,
                       resources={'metadata': {'path': str(ROOT)}}).execute()
        method = 'nbclient with a real Jupyter kernel'
    else:
        spec = importlib.util.spec_from_file_location('curriculum_sequential_runner', ROOT/'qa/execute_cavity_lesson.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        book = runner.execute(book, ROOT)
        method = 'Sequential plain-Python code cells in a fresh child process; no Jupyter-kernel certification'
    require(sha(path) == before, 'Run All changed the source notebook')
    outputs = [output for cell in book.cells if cell.cell_type == 'code' for output in cell.get('outputs',[])]
    require(not any(output.output_type == 'error' for output in outputs), 'Notebook execution retained an error')
    image_hashes = set(); probes = []
    for output in outputs:
        payload = output.get('data',{}).get('image/png')
        if payload:
            if isinstance(payload,list): payload=''.join(payload)
            image_hashes.add(hashlib.sha256(base64.b64decode(payload)).hexdigest())
        if output.output_type == 'stream':
            for line in output.text.splitlines():
                if line.startswith(MARKER): probes.append(json.loads(line[len(MARKER):]))
    require(len(probes) == 1, 'Completed notebook evidence probe is missing or repeated')
    executed = [cell for cell in book.cells if cell.cell_type == 'code']
    require(len(executed) == count+1+initialization_cells and all(cell.execution_count is not None for cell in executed),
            'Not every original code cell executed')
    result = dict(**entry(path), execution_method=method, jupyter_kernel_executed=mode=='jupyter',
                  original_code_cells=count, completed_original_code_cells=count, validation_initialization_cells=initialization_cells, post_execution_probe=probes[0],
                  output_count=len(outputs), displayed_png_sha256=sorted(image_hashes), missing_local_links=[],
                  source_notebook_unchanged=True)
    Path(destination).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


def validate(mode):
    reviews = verify_reviews()
    result = dict(schema_version=1, status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  validator=entry(__file__), execution_mode=mode, reviewed_evidence=reviews, notebooks={},
                  visual_QA='Hash-binding of recorded individual visual inspections; this machine gate does not itself certify visual appearance',
                  scientific_scope='Integrity and teaching delivery only; no model fitting, accuracy promotion or new endpoint selection')
    runtime_parent = (ROOT/'qa/.curriculum_runtime').resolve()
    require(runtime_parent.is_relative_to(ROOT.resolve()), 'Temporary notebook runtime leaves checkout')
    runtime_parent.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='validated_',dir=runtime_parent) as temporary:
            work=Path(temporary).resolve()
            require(work.is_relative_to(runtime_parent), 'Temporary notebook execution directory leaves its owned runtime')
            for kind in BOOKS:
                destination=work/(kind+'_execution.json')
                env=os.environ.copy()
                for name in ('JUPYTER_RUNTIME_DIR','IPYTHONDIR','MPLCONFIGDIR','XDG_CACHE_HOME'):
                    folder=work/name.lower();folder.mkdir(exist_ok=True);env[name]=str(folder)
                env.update(PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--execute-one',kind,
                    '--execution-mode',mode,'--child-output',str(destination)],cwd=ROOT,env=env,
                    capture_output=True,text=True,timeout=1200)
                require(completed.returncode == 0, kind+' Run All failed:\n'+completed.stdout[-4000:]+completed.stderr[-6000:])
                item=read_json(destination)
                expected=('ghia','stratified') if kind=='ghia' else ('d5',)
                item['displayed_current_gallery_families']=[]
                for role in expected:
                    require(set(item['displayed_png_sha256']) & set(reviews['image_hashes_by_family'][role]),
                            kind+' notebook did not display a current reviewed '+role+' physical figure')
                    item['displayed_current_gallery_families'].append(role)
                result['notebooks'][kind]=item
    finally:
        if runtime_parent.exists() and not any(runtime_parent.iterdir()): runtime_parent.rmdir()
    for name in ('studies','lecture_review','figure_review'): checked(reviews[name])
    for item in reviews['replacement_lecture_reviews'].values(): checked(item)
    for item in reviews['scientific_manifests']: checked(item)
    for item in reviews['lecture_sources'].values(): checked(item)
    for item in reviews['pdfs'].values(): checked(item)
    for item in reviews['images']: checked(item)
    for item in reviews['figure_pdfs']: checked(item)
    for item in result['notebooks'].values(): checked(item)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-mode',choices=('sequential','jupyter'),default='sequential')
    parser.add_argument('--output',type=Path,default=ROOT/'qa/cavity_pinn_curriculum_validation.json')
    parser.add_argument('--execute-one',choices=tuple(BOOKS),help=argparse.SUPPRESS)
    parser.add_argument('--child-output',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.execute_one:
        require(args.child_output is not None,'Internal output is required')
        execute_one(args.execute_one,args.execution_mode,args.child_output);return
    result=validate(args.execution_mode)
    output=args.output.resolve();require(output.is_relative_to(ROOT.resolve()),'Receipt must stay in checkout')
    temporary=output.with_name(output.name+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(output)
    print(json.dumps(dict(receipt=str(output),status='passed',execution_mode=args.execution_mode,
        notebooks={kind:item['completed_original_code_cells'] for kind,item in result['notebooks'].items()})))


if __name__=='__main__':main()

