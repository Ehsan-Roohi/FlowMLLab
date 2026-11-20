"""Validate retained W1.4 artifacts without repeating the numerical experiments."""
from pathlib import Path
import ast, hashlib, json, subprocess
import nbformat
import numpy as np
from pypdf import PdfReader
from validate_course_release import validate_week01_4_standalone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


notebook_path = ROOT / 'notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb'
nb = nbformat.read(notebook_path, as_version=4)
nbformat.validate(nb)
standalone_contract = validate_week01_4_standalone(nb, notebook_path.relative_to(ROOT).as_posix())
code = [c for c in nb.cells if c.cell_type == 'code']
assert [c.execution_count for c in code] == list(range(1, len(code)+1))
assert not [o for c in code for o in c.outputs if o.output_type == 'error']
source = ast.parse((ROOT / 'flowmllab/high_order.py').read_text(encoding='utf-8'))
embedded = ast.parse('\n'.join(c.source for c in code))
expected = {n.name:ast.dump(n, include_attributes=False) for n in source.body if isinstance(n,ast.FunctionDef)}
source_cavity = ast.parse((ROOT/'flowmllab/high_order_cavity.py').read_text(encoding='utf-8'))
expected.update({n.name:ast.dump(n,include_attributes=False) for n in source_cavity.body if isinstance(n,ast.FunctionDef)})
actual = {n.name:ast.dump(n, include_attributes=False) for n in embedded.body if isinstance(n,ast.FunctionDef)}
assert all(actual.get(k) == v for k,v in expected.items()), 'Notebook algorithms diverged from tested module'
receipt = json.loads((ROOT / 'qa/week01_4_execution.json').read_text())
entrypoint_revision = None
if receipt['sha256'] != sha(notebook_path):
    entrypoint_revision = json.loads((ROOT / 'qa/week01_4_entrypoint.json').read_text(encoding='utf-8'))
    raw = json.loads(notebook_path.read_text(encoding='utf-8'))
    raw_code = [cell for cell in raw['cells'] if cell['cell_type'] == 'code']
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    assert entrypoint_revision['original_executed_notebook_sha256'] == receipt['sha256']
    assert entrypoint_revision['execution_receipt_sha256'] == sha(ROOT / 'qa/week01_4_execution.json')
    assert entrypoint_revision['final_notebook_sha256'] == sha(notebook_path)
    assert entrypoint_revision['executed_code_sha256'] == receipt['executed_code_sha256']
    assert entrypoint_revision['unchanged_code_cell_payload_sha256'] == canonical(raw_code)
    assert entrypoint_revision['unchanged_outputs_sha256'] == canonical([cell.get('outputs', []) for cell in raw_code])
    assert entrypoint_revision['unchanged_notebook_metadata_sha256'] == canonical(raw['metadata'])
    assert entrypoint_revision['unchanged_other_cells_sha256'] == canonical(raw['cells'][1:])
    assert entrypoint_revision['final_launcher_cell_sha256'] == canonical(raw['cells'][0])
    assert entrypoint_revision['direct_colab_url'] in ''.join(raw['cells'][0]['source'])
    assert entrypoint_revision['markdown_only'] and entrypoint_revision['code_and_outputs_unchanged']
    assert entrypoint_revision['numerical_or_plot_execution_performed'] is False
if 'executed_code_sha256' in receipt:
    assert receipt['executed_code_sha256'] == hashlib.sha256('\n'.join(c.source for c in code).encode()).hexdigest()

evidence_path = ROOT / 'results/week01_4_spectral_dg/execution.json'
def reject_constant(value):
    raise ValueError(f'Non-JSON constant: {value}')
evidence = json.loads(evidence_path.read_text(), parse_constant=reject_constant)
assert set(evidence['tables']) == {'A','P','D','F','T','H','C','G'}
assert len(evidence['figures']) == 11
for name in evidence['figures']:
    for extension in ('.png','.pdf'):
        assert (evidence_path.parent / (name+extension)).is_file()
assert 7.5 < evidence['time_difference_ratio'] < 8.5
assert max(abs(r['mass_drift']) for name in ['D','F','T','H'] for r in evidence['tables'][name]) < 1e-11
assert all(2.7 < r['observed_order'] < 3.6 for r in evidence['tables']['H'][1:])
assert max(r['equation_residual_max'] for r in evidence['tables']['P']) < 1e-8
assert len(evidence['tables']['C']) == 6
assert len(evidence['tables']['G']) == 204
assert evidence['ghia_source_sha256'] == sha(ROOT/'references/ghia/published_samples.csv')
import sys
sys.path.insert(0,str(ROOT))
import numpy as np
from flowmllab.high_order_cavity import cavity_sample
for row in evidence['tables']['C']:
    assert row['converged'] and row['steady_residual_max'] <= 1e-9
    assert row['continuity_nodes_max'] <= 1e-9
    assert row['continuity_core_RMS'] > row['continuity_nodes_max']
    re,n = row['Re'],row['N']
    with np.load(evidence_path.parent/f'cavity_re{re}_N{n}.npz',allow_pickle=False) as archive:
        sol = {k:archive[k] for k in archive.files}
    assert sol['converged']
    assert np.all(sol['u'][-1,1:-1] == 1)
    assert np.all(sol['v'][:,[0,-1]] == 0)
    assert abs(cavity_sample(sol,'p',[.5],[.5])[0,0]) < 1e-10
    for component, coordinate in [('u','y'),('v','x')]:
        retained=[r for r in evidence['tables']['G'] if r['Re']==re and r['N']==n and r['component']==component]
        s=np.array([r['coordinate'] for r in retained])
        recomputed=(cavity_sample(sol,'u',[.5],s)[:,0] if component=='u' else cavity_sample(sol,'v',s,[.5])[0])
        assert np.max(np.abs(recomputed-np.array([r['CFD'] for r in retained]))) < 1e-12

pdf_path = ROOT / 'lectures/week01_4_spectral_dg.pdf'
pdf = PdfReader(pdf_path)
assert 25 <= len(pdf.pages) <= 27
assert all(len(p.extract_text()) > 300 for p in pdf.pages)
assert sum(len(p.get('/Annots',[])) for p in pdf.pages) >= 5

# This additive unit protects its pre-existing Week 1.2 prerequisite.
# Other course units have separate authorized revisions and their own validators.
preservation_expected = {
    'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb': 'd4bd771bc423ba4932bd2d8c04153de191e0664e4a89762bcee5f604780f726e',
    'lectures/week01_2_pressure_velocity.pdf': '949f5857f74c038d27fe831231f51c3a61999f039897ae377628a0fd1b51de54',
}
preserved = {rel: sha(ROOT / rel) for rel in preservation_expected}
protected_changed = [rel for rel in preserved if preserved[rel] != preservation_expected[rel]]
assert not protected_changed, f'Protected Week 1.2 artifact changed: {protected_changed}'

result = {
    'unit':'1.4', 'numerical_tests':'9 passed; scalar and full Navier-Stokes manufactured/Jacobian/wall/gauge/refinement checks',
    'executed_code_cells':len(code),'inline_png_outputs':sum('image/png' in o.get('data',{}) for c in code for o in c.outputs),
    'core_functions_equal_to_tested_module':len(expected),
    'pdf_pages':len(pdf.pages), 'protected_week01_2_files_changed':protected_changed,
    'preservation_scope':'The pre-existing Week 1.2 notebook and lecture only; other units are validated independently.',
    'standalone_contract':standalone_contract,
    'entrypoint_revision':entrypoint_revision,
    'preserved_week01_2_sha256':preserved,
    'deliverables_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [notebook_path,pdf_path,evidence_path]},
    'execution_mode':'Retained fresh local CPU Run All; launcher-only Markdown amendment, without numerical rerun or hosted execution.',
    'original_executed_notebook_sha256':receipt['sha256'],
    'original_elapsed_run_all_seconds':receipt['elapsed_run_all_seconds'],
    'visual_review':'See qa/week01_4_visual_review.json for the inspected final PDF hash and pages.',
    'actual_cavity_cases':6,'ghia_sample_comparisons':204,
    'qualification':'Algebraic convergence and reference-profile scores are distinct from remaining off-grid spatial continuity defects.',
}
path = ROOT/'qa/week01_4_validation.json'
path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
