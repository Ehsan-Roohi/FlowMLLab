"""Re-render six figures from frozen arrays after a presentation-only adjustment.

Never fits a model, touches a prediction, or changes a notebook code cell.
The original Run All receipt remains the source for compute timing/provenance.
"""
from pathlib import Path
import ast, base64, hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/week04_lab2'
TMP = ROOT / 'tmp/week04_lab2'
os.environ['MPLCONFIGDIR'] = str(TMP / 'mpl')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

helper = ROOT / 'flowmllab/cavity_lab2.py'
notebook = ROOT / 'notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb'
frozen = json.loads((OUT / 'frozen_models.json').read_text())
original_source = (OUT / 'training_helper_source.py').read_bytes()
assert hashlib.sha256(original_source).hexdigest() == frozen['helper_source_sha256']

def scientific_ast(source):
    tree = ast.parse(source)
    tree.body = [item for item in tree.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or item.name not in {'plot_profiles', 'plot_wall_diagnostics'}]
    return ast.dump(tree, include_attributes=False)
assert scientific_ast(original_source) == scientific_ast(helper.read_bytes()), 'Non-presentation source changed'

protected_names = ['native_test_predictions.npz', 'scalar_predictions.npz', 'frozen_weights.npz',
                   'frozen_models.json', 'field_metrics.csv', 'week4_scalar_results.csv', 'training_histories.json']
protected = {name: sha(OUT / name) for name in protected_names}
nb = json.loads(notebook.read_text(encoding='utf-8'))
def code_hash(book):
    return hashlib.sha256('\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code').encode()).hexdigest()
original_code_hash = code_hash(nb)
execution = json.loads((ROOT / 'qa/week04_lab2_execution.json').read_text())
assert original_code_hash == execution['executed_code_sha256']

spec = importlib.util.spec_from_file_location('week04_lab2_plot_helper', helper)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.figure_style()
arrays = np.load(OUT / 'native_test_predictions.npz')
re_values = arrays['Re']
data = {'x': arrays['x'], 'y': arrays['y'], 'Re': re_values,
        'u': np.stack([arrays[f'Re{re:g}_CFD_u'] for re in re_values]),
        'v': np.stack([arrays[f'Re{re:g}_CFD_v'] for re in re_values])}
methods = ['Interpolation', 'Coordinate DNN', 'Boundary-aware DNN']
revised = []
for function_name, stem in [('plot_profiles', 'profiles'), ('plot_wall_diagnostics', 'wall_diagnostics')]:
    cells = [c for c in nb['cells'] if c['cell_type']=='code' and f'fig = {function_name}(' in ''.join(c['source'])]
    assert len(cells)==1
    image_outputs = [o for o in cells[0]['outputs'] if 'image/png' in o.get('data',{})]
    assert len(image_outputs)==len(re_values)
    for index, re in enumerate(re_values):
        predictions = {name: (arrays[f'Re{re:g}_{name.replace(" ","_")}_u'],
                              arrays[f'Re{re:g}_{name.replace(" ","_")}_v']) for name in methods}
        figure = getattr(module, function_name)(data, index, predictions)
        filename = f'{stem}_re{re:g}'
        module.save_figure(figure, OUT, filename)
        plt.close(figure)
        image_outputs[index]['data']['image/png'] = base64.b64encode((OUT / (filename+'.png')).read_bytes()).decode()
        revised.extend([filename+'.png', filename+'.pdf'])
assert code_hash(nb) == original_code_hash
notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
assert protected == {name: sha(OUT / name) for name in protected_names}
receipt = {
    'verified_utc': datetime.now(timezone.utc).isoformat(),
    'change': 'Reduced signed-error tick counts and added Reynolds labels to wall diagnostic panels.',
    'training_rerun': False,
    'model_arrays_weights_metrics_histories_unchanged': True,
    'scientific_source_ast_unchanged': True,
    'notebook_code_unchanged': True,
    'training_helper_sha256': frozen['helper_source_sha256'],
    'presentation_helper_sha256': sha(helper),
    'original_executed_notebook_sha256': execution['notebook_sha256'],
    'presented_notebook_sha256': sha(notebook),
    'executed_code_sha256': original_code_hash,
    'protected_scientific_artifacts': protected,
    'revised_artifacts': {name: sha(OUT / name) for name in revised},
    'replot_script_sha256': sha(Path(__file__)),
    'retained_training_helper_source': 'results/week04_lab2/training_helper_source.py',
    'reproduction_source_note': 'The original training helper was byte-verified and retained separately so expendable transfer archives can be removed.',
    'claim': 'Only six saved PNG/PDF plots and their inline images were regenerated from frozen native arrays. Run All timing is unchanged.'}
(ROOT / 'qa/week04_lab2_presentation.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: receipt[k] for k in ['training_rerun','model_arrays_weights_metrics_histories_unchanged','scientific_source_ast_unchanged','notebook_code_unchanged','presented_notebook_sha256']},indent=2))
