"""Package the existing author detector; build and execute its Week 11 lab.

Only --research-root accesses the private research checkout. Public replay and
student Run All use the bundled numeric subset and never need that checkout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import textwrap

import numpy as np
import nbformat as nbf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'qa'))
from flowmllab.cavitation_detection import load_bundle, load_model, predict, scores, comparison_figure
from add_colab_entrypoints import badge, bootstrap

DATA = ROOT/'data/week11_cavitation'
RESULTS = ROOT/'results/week11_cavitation'
NOTEBOOK = 'notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def export(research):
    research = Path(research)
    base = research/'results/cavitation_20260912'
    run = base/'native_alpha20_v6'
    if (DATA/'manifest.json').exists():
        raise FileExistsError('Already packaged; omit --research-root to replay')
    DATA.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run/'PROTOCOL.json').read_text())
    evaluation = json.loads((run/'EVALUATION.json').read_text())
    names = protocol['training_cases']+[protocol['validation_case'], 'Case_24', 'Case_23']
    manifest = dict(version=1, author='Ehsan Roohi',
        source='Author hydrofoil computer-vision experiment, native_alpha20_v6, 2026-09-12',
        scope='Retained CFD vapor rasters and native weak labels; not photographs or human ground truth',
        reference_threshold=.20, classes={'0':'background', '1':'attached_2d', '2':'disconnected_2d', '255':'ignore'},
        inference='3 input channels, neural argmax only; no topology repair',
        source_checkpoint_sha256=sha(run/'selected.pt'),
        parent_checkpoint_sha256=sha(base/'native_v4/selected.pt'),
        protocol=protocol, cases=[], files={},
        prior_work='https://doi.org/10.1063/5.0345365')
    for name in names:
        source = base/'data_audit'/f'{name}_raster.npz'
        labels = base/'native_topology_alpha20'/f'{name}_native_labels.npz'
        with np.load(source, allow_pickle=False) as z:
            arrays = {k:z[k] for k in ['alpha_v', 'wall', 'times', 'x', 'y']}
        arrays['alpha'] = arrays.pop('alpha_v').astype('float32')
        with np.load(labels, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['times'], arrays['times'])
            arrays['reference'] = z['target']
        with np.load(run/f'{name}_prediction.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['weak_reference'], arrays['reference'])
            arrays['archived_prediction'] = z['prediction']
        role = 'TRAIN' if name in protocol['training_cases'] else ('validation (inspected)' if name == protocol['validation_case'] else 'nontraining (previously inspected)')
        filename = name+'.npz'
        np.savez_compressed(DATA/filename, **arrays)
        assert sha(source) == evaluation[name]['source_sha256']
        assert sha(labels) == evaluation[name]['reference_sha256']
        manifest['cases'].append(dict(name=name, file=filename, role=role,
            frames=len(arrays['times']), shape=list(arrays['alpha'].shape),
            source_raster_sha256=sha(source), native_reference_sha256=sha(labels),
            archived_metrics=evaluation[name]['metrics']))
    for source, target in [(run/'selected.pt', 'detector_weights.npz'),
                           (base/'native_v4/selected.pt', 'parent_weights.npz')]:
        # Trusted local author archive only. Published files are numeric NPZ.
        state = torch.load(source, map_location='cpu', weights_only=False)['state_dict']
        np.savez_compressed(DATA/target, **{k:v.detach().numpy() for k,v in state.items()})
    for p in sorted(DATA.glob('*.npz')):
        manifest['files'][p.name] = sha(p)
    manifest['source_code_sha256'] = {p.name:sha(p) for p in [
        research/'tools/cavitation_context_v3_20260912.py',
        research/'tools/cavitation_native_train_20260912.py',
        research/'tools/cavitation_alpha20_20260912.py',
        research/'tools/cavitation_alpha20_transfer_20260912.py']}
    write_json(DATA/'manifest.json', manifest)


def replay():
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    manifest, cases = load_bundle(DATA)
    model = load_model(DATA/'detector_weights.npz')
    RESULTS.mkdir(parents=True, exist_ok=True)
    report, predictions = {}, {}
    for entry in manifest['cases']:
        case = cases[entry['name']]
        p = predict(model, case['alpha'], case['wall'])
        predictions[entry['name']] = p
        np.testing.assert_array_equal(p, case['archived_prediction'])
        measured = scores(p, case['reference'])
        assert measured == entry['archived_metrics']
        zero = predict(model, np.zeros_like(case['alpha'][0]), case['wall'])
        report[entry['name']] = dict(role=case['role'], frames=len(p), metrics=measured,
            exact_archived_replay=True, wall_positive_pixels=int(((p>0)&case['wall']).sum()),
            zero_vapor_positive_pixels=int((zero>0).sum()))
    write_json(RESULTS/'replay_metrics.json', report)
    figure_hashes = {}
    specifications = [
        ('cloud_detection', 'Case1LES', [0,22], 'Hydrofoil vapor-cloud detection | Case1LES (TRAIN)'),
        ('les_sequence', 'Case1LES', [0,17,22,31], 'Retained LES times | training trajectory'),
        ('transfer_and_failure', 'Case_24', [0,17,24,26], 'Case 24 | inspected nontraining trajectory; t = 2.00 s is the retained worst-error frame'),
        ('les_worst_error', 'Case1LES', [26], 'Case1LES | retained worst-error frame (training trajectory)')]
    for filename, name, indices, title in specifications:
        fig = comparison_figure(cases[name], predictions[name], indices, title)
        fig.savefig(RESULTS/(filename+'.png'), dpi=160)
        plt.close(fig)
        figure_hashes[filename+'.png'] = dict(sha256=sha(RESULTS/(filename+'.png')),
            case=name, indices=indices, times=cases[name]['times'][indices].tolist())
    write_json(RESULTS/'figure_provenance.json', dict(figures=figure_hashes,
        model_sha256=sha(DATA/'detector_weights.npz'), manifest_sha256=sha(DATA/'manifest.json'),
        code_sha256=sha(ROOT/'flowmllab/cavitation_detection.py'),
        selection='Original retained review times plus original worst-error frames; no best-test selection'))
    print(f'Exact replay: {sum(v["frames"] for v in report.values())} frames, {len(report)} cases', flush=True)


def build_notebook():
    md = lambda x: nbf.v4.new_markdown_cell(textwrap.dedent(x).strip())
    code = lambda x: nbf.v4.new_code_cell(textwrap.dedent(x).strip())
    cells = [md('''
    # Week 11 - Machine vision for hydrofoil vapor clouds

    **Run the existing detector, inspect its mistakes, and reproduce its evidence.**
    This extension uses the author's existing 2026-09-12 cavitation code and results.
    It accompanies the final cavitation section of [Lecture 11](../../lectures/week11_shock_vortex_identification.pdf).
    Allow 30-45 minutes of teaching; default execution is CPU inference, not retraining.

    Input: real CFD vapor-volume-fraction rasters and hydrofoil geometry. Output:
    background, attached cavity and disconnected vapor cloud in 2-D. These are
    **CFD fields, not experimental camera photographs**. The network already saw
    the four TRAIN cases; the other three cases were also inspected during development.
    This replay is not a new blind accuracy experiment.
    ''')]
    cells[0].source += badge(NOTEBOOK)
    setup = bootstrap('notebooks/week11').replace('[test]', '[test,reconstruction]')
    cells += [code(setup+'''
from pathlib import Path
import sys, time
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p/'flowmllab/cavitation_detection.py').is_file())
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from flowmllab.cavitation_detection import (
    load_bundle, load_model, predict, scores, inputs, raster_baseline,
    comparison_figure, adapt_from_parent, digest)
torch.set_num_threads(2)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 110})
DATA = ROOT/'data/week11_cavitation'
manifest, cases = load_bundle(DATA)
before = {name:digest(DATA/name) for name in manifest['files']}
model = load_model(DATA/'detector_weights.npz')
print('Model parameters:', sum(p.numel() for p in model.parameters()))
display(pd.DataFrame([{k:c[k] for k in ['name','role','frames']} for c in manifest['cases']]))
'''), md(r'''
    ## 1. What does the model see, and what is a reference?

    The three channels are vapor fraction $\alpha_v$, a solid mask, and the
    clipped distance to solid in raster pixels, $\min(d/48,1)$. The original
    five encoder widths are 8, 16, 24, 32, 48; global average context is added
    at the bottleneck. Bilinear upsampling and skip connections feed a three-logit
    head. A softmax/argmax assigns mutually exclusive classes. Unlike overlapping
    shock/vortex labels, attached and detached are alternative topology states.

    The weak teacher was constructed on the **original Fluent cell-face graph**:
    support is $\alpha_v\geq0.20$; a connected component touching an actual wall
    face is attached. External/crop-uncertain support and solid are ignored (255).
    These algorithmic labels are not independent human truth. The exact native
    references are supplied; this lab does not approximate them with pixel dilation.

    **Predict before running:** Can a thin, poorly resolved ligament change the
    class of an entire vapor region? Does a bright vapor region alone establish
    that the cloud is detached, or that a shedding event occurred?
    '''), code('''
    case = cases['Case1LES']
    example = inputs(case['alpha'][22], case['wall'])
    assert example.shape[0] == 3
    assert np.all(example[0][case['wall']] == 0)
    print('Input tensor:', example.shape, '| reference classes:', np.unique(case['reference']))
    print('Weights source SHA-256:', manifest['source_checkpoint_sha256'])
    '''), md('''
    ## 2. Rerun all seven cases with the frozen detector

    Only the three input channels enter `predict`. Native labels are read afterward
    for evaluation. There is no connectivity repair, gap filling, reference masking
    or geometry veto applied to neural output. We compare every predicted pixel to
    the stored original prediction, and recompute pooled Dice from TP, FP and FN.
    Empty-versus-empty has undefined Dice (`None`), not a fabricated perfect score.
    Solid/uncertain reference pixels are excluded; wall false positives are counted separately.
    '''), code('''
    predictions, rows = {}, []
    started = time.perf_counter()
    for entry in manifest['cases']:
        name = entry['name']; c = cases[name]
        p = predict(model, c['alpha'], c['wall'])
        predictions[name] = p
        np.testing.assert_array_equal(p, c['archived_prediction'])
        measured = scores(p, c['reference'])
        assert measured == entry['archived_metrics']
        rows.append({'case':name, 'role':c['role'], 'frames':len(p),
                     'attached Dice':measured['attached_2d']['dice'],
                     'cloud Dice':measured['disconnected_2d']['dice'],
                     'wall positive pixels':int(((p>0)&c['wall']).sum())})
    display(pd.DataFrame(rows))
    print('Exact archived replay:', sum(len(p) for p in predictions.values()), 'frames')
    print('Local inference seconds:', round(time.perf_counter()-started, 2))
    '''), md('''
    ## 3. Read the cavity and its detection in the same figure

    The left column is the CFD vapor field; the middle is the native weak reference;
    the right is fresh neural inference. Orange is attached, magenta is disconnected
    in 2-D. All panels use the same [0,1] vapor colors and physical coordinates.
    Grey reference support is uncertain; the solid is shown separately.
    The two lead frames (0.36 and 2.12 s) belong to the TRAIN LES trajectory and
    were already in the original review. This figure illustrates the task, not generalization.
    '''), code('''
    comparison_figure(cases['Case1LES'], predictions['Case1LES'], [0,22],
                      'Hydrofoil vapor-cloud detection | Case1LES (TRAIN)')
    plt.show()
    '''), md('''
    ## 4. Retain the difficult nontraining case

    Case 24 is excluded from gradient training but was previously inspected.
    Its pooled cloud Dice is much lower than its attached-cavity Dice. The original
    worst-error frame at 2.00 s is retained below beside an earlier frame.
    A high attached score does not establish reliable detection of detached clouds.
    Compare the physical width of a missed ligament to raster spacing before
    claiming improved topology. Pixel-distance preprocessing is not resolution invariant.
    '''), code('''
    comparison_figure(cases['Case_24'], predictions['Case_24'], [17,24],
                      'Case 24 | inspected nontraining case; retained failure at 2.00 s')
    plt.show()
    '''), md('''
    ## 5. A transparent comparator and a negative control

    A four-connected raster threshold at 0.20 plus one-cell wall contact is a
    **separate classical baseline**, not the native-mesh teacher and not a neural
    postprocessor. Report its disagreements instead of assuming the network wins.
    Then remove all vapor while keeping the same geometry: any positive prediction
    is a false detection in this negative control.
    '''), code('''
    baseline_rows = []
    for name, c in cases.items():
        baseline = np.stack([raster_baseline(a, c['wall']) for a in c['alpha']])
        m = scores(baseline, c['reference'])
        zero = predict(model, np.zeros_like(c['alpha'][0]), c['wall'])
        assert np.count_nonzero(zero) == 0
        baseline_rows.append({'case':name, 'raster attached Dice':m['attached_2d']['dice'],
                              'raster cloud Dice':m['disconnected_2d']['dice'],
                              'zero-vapor positives':int((zero>0).sum())})
    display(pd.DataFrame(baseline_rows))
    '''), md('''
    ## 6. A seeded noise challenge, not a new accuracy claim

    Add Gaussian noise with fixed standard deviation 0.02 to the supplied vapor
    channel, clip to [0,1], and retain geometry. Compare against the same clean
    reference. This is an input-perturbation sensitivity exercise; it is not an
    experimental camera-noise model or a new independent physical validation set.
    Do not select a model or alter thresholds from this already-inspected result.
    '''), code('''
    rng = np.random.default_rng(20260916)
    c = cases['Case1LES']; i = 22
    noisy = np.clip(c['alpha'][i] + rng.normal(0,.02,c['wall'].shape),0,1).astype('float32')
    p_noisy = predict(model, noisy, c['wall'])
    display(pd.DataFrame({'clean': {k:v['dice'] for k,v in scores(predictions[c['name']][i:i+1],c['reference'][i:i+1]).items()},
                         'noisy': {k:v['dice'] for k,v in scores(p_noisy,c['reference'][i:i+1]).items()}}))
    print('Changed labels:', int((p_noisy[0]!=predictions[c['name']][i]).sum()))
    '''), md(r'''
    ## 7. Optional: rerun the original final adaptation stage

    Default Run All keeps this off. The supplied parent is the original native_v4
    model, trained with $\alpha_v\geq0.50$ references. The displayed v6 result used
    **1000 additional** Adam updates at 0.0002 with the revised 0.20 references,
    after 2000 parent updates. This is not a 1000-step model trained from scratch.
    The four complete TRAIN cases, balanced cloud-frame sampling, vertical reflection,
    weighted cross entropy plus foreground Dice, and gradient clipping are preserved.
    Validation was monitored; the fixed final step was selected. There was no fresh
    test after this development correction. No PDE loss or temporal predictor is used.

    Set `RUN_ADAPTATION=True` for the full final-stage rerun (several CPU minutes).
    It returns a separate in-memory model. It cannot overwrite the supplied weights.
    Different PyTorch builds may produce different optimization trajectories.
    A short changed budget is an exercise, not reproduction of the published v6 model.
    '''), code('''
    RUN_ADAPTATION = False
    if RUN_ADAPTATION:
        parent = load_model(DATA/'parent_weights.npz')
        adapted_model, training_history = adapt_from_parent(parent, cases, manifest['protocol'])
        display(pd.DataFrame(training_history))
        validation = cases['Case_19']
        adapted = predict(adapted_model, validation['alpha'], validation['wall'])
        print('Validation weak-reference agreement:', scores(adapted, validation['reference']))
    else:
        print('Optional training skipped; the displayed results use the original frozen v6 weights.')
    assert before == {name:digest(DATA/name) for name in manifest['files']}
    print('Retained inputs and weights unchanged.')
    '''), md('''
    ## Exit assessment and sources

    1. Explain why a vapor-fraction threshold does not by itself identify attachment.
    2. Mark one wrong attached/cloud decision in the displayed nontraining case.
    3. Compare the raster baseline and neural model using both classes, including wall errors.
    4. Specify the expert labels, new trajectories, geometric tests and temporal cadence
       you would need for a defensible detached-cloud tracking or shedding-event claim.

    **Our source:** Ehsan Roohi, author hydrofoil cavitation computer-vision experiment,
    2026-09-12, `native_alpha20_v6`; exact checkpoint, data and source-code hashes in
    [the manifest](../../data/week11_cavitation/manifest.json).
    [Code](../../flowmllab/cavitation_detection.py) ·
    [Full results and limitations](../../results/week11_cavitation/README.md).

    **Related work:** Hatzissawidis et al., *Deep learning semantic segmentation for
    cloud cavitation image analysis*, Physics of Fluids 38, 093331 (2026),
    [doi:10.1063/5.0345365](https://doi.org/10.1063/5.0345365).
    That study segments camera images and then uses a heuristic sheet/cloud split;
    our retained CFD experiment directly predicts three classes. This course lab
    does not reproduce their data, code, results or claim superiority to their method.

    Original course adaptation and verification are AI-assisted. Full CFD solver
    regeneration, mesh convergence, expert ground truth and forecasting are outside
    this replay. Continue to [Week 12](../week12/README.md).
    ''')]
    from build_week11_cavitation_methods import notebook_cells
    cells[-1:-1] = notebook_cells()
    cells[0].source += '\n\n**Multi-method extension:** compare our alpha-input detector, pressure threshold, pressure 3x3 model and two pressure U-Nets on the same 16 moving-case frames.'
    for i, cell in enumerate(cells):
        cell.id = f'w11-cav-{i:02d}'
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
        'language_info': {'name':'python', 'version':'3.12'},
        'flowmllab': {'scope':'Exact replay of existing author vapor-cloud detector; optional final-stage adaptation'}})
    nbf.validate(notebook)
    nbf.write(notebook, ROOT/NOTEBOOK)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path)
    args = parser.parse_args()
    if args.research_root:
        export(args.research_root)
    replay()
    build_notebook()
