"""Build the real-field companion; Run All audits retained evidence read-only."""
from pathlib import Path
import argparse
import nbformat as nbf
from add_colab_entrypoints import badge, bootstrap

ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[md('''# Week 11 Lab 2 - Reconstruct velocity, then identify vortices

<!-- FlowMLLab retained-LBM reconstruction audit v1 -->

Read [the complete experimental protocol](RECONSTRUCTION_PROTOCOL.md) first.
This is an independent compact U-Net implementation inspired by the supplied
super-resolution course material, not a reproduction of its code or results.
The real LBM wake has no shocks. Masks are velocity-derived weak references.
Run All audits the retained run without modifying it. Full retraining is an
explicit opt-in below; it requires CPU PyTorch and takes longer than this audit.
'''),code('''from pathlib import Path
import hashlib, json, sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'flowmllab/modal_experiments.py').is_file())
sys.path.insert(0, str(ROOT))
from flowmllab.modal_experiments import load_cases
cases = load_cases(ROOT / 'data/modal_labs')
evidence = ROOT / 'results/week11_reconstruction'
manifest = json.loads((evidence / 'manifest.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest, name
protocol = json.loads((evidence / 'protocol.json').read_text())
assert protocol['train'] == [90, 110]
assert protocol['validation'] == 100 and protocol['retained_test'] == 105
assert protocol['data_manifest_sha256'] == hashlib.sha256((ROOT / 'data/modal_labs/manifest.json').read_bytes()).hexdigest()
print(protocol['scope'])
'''),md('''## Separate field accuracy from structure agreement

Velocity MSE does not optimize derivatives or binary masks. Compare vorticity
error as well as velocity error. Dice and IoU quantify agreement with the same
native-ROI swirling-strength rule, not human-validated vortex accuracy.
Direct segmentation has no reconstructed field, hence its field metrics are N/A.
The three seeds measure optimizer variation, not uncertainty over new flows.
'''),code('''import pandas as pd
metrics = pd.DataFrame(json.loads((evidence / 'metrics.json').read_text()))
display(metrics)
display(metrics.groupby('method')[['velocity_relative_l2','vorticity_relative_l2','dice','iou']].agg(['mean','std']))
'''),code('''from IPython.display import Image, display
display(Image(filename=str(evidence / 'comparison.png'), width=1100))
display(Image(filename=str(evidence / 'loss.png'), width=1100))
'''),md('''## Fresh training (optional)

Set `RETRAIN=True` to train all six models into a new scratch folder. Install
PyTorch for your platform first. Do not lower the budget and label the result
as the retained experiment. Both neural paths see identical degraded inputs.
The original course network had more encoder levels and a different dataset;
this compact model is an adaptation, not a claim of exact replication.
'''),code('''RETRAIN = False
if RETRAIN:
    import subprocess, tempfile
    scratch = Path(tempfile.mkdtemp(prefix='flowmllab_w11_')) / 'run'
    subprocess.run([sys.executable, str(ROOT / 'qa/run_week11_reconstruction.py'),
                    '--output', str(scratch), '--epochs', '60'], check=True)
    print('Fresh evidence:', scratch)
'''),md('''## Research questions

1. Does the lowest velocity error necessarily yield the highest mask Dice?
2. Why can the divergence RMS differ from zero for the native coarse ROI?
3. What changes if the diagnostic threshold is varied using training/validation
   cases only? Freeze that experiment before evaluating Re105 again.
4. Which additional reference data would be needed to claim shock detection,
   rather than incompressible vortex recovery?

Do not select a seed, frame or probability threshold using the retained test.
''')]
cells[0].source += badge('notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb')
cells[1].source = bootstrap('notebooks/week11') + cells[1].source
for i,c in enumerate(cells): c.id=f'w11-reconstruction-{i:02d}'
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbf.validate(nb)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--execute',action='store_true')
if parser.parse_args().execute:
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT/'notebooks/week11')}}).execute()
nbf.write(nb,ROOT/'notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb')
