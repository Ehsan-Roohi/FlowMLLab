"""Retain and replay the existing alpha/pressure comparison on identical frames."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab.cavitation_detection import digest
from flowmllab.cavitation_methods import (load_comparison, infer_comparison,
    method_mask, binary_score, methods_figure)

DATA = ROOT/'data/week11_cavitation_methods'
OUT = ROOT/'results/week11_cavitation'
METHODS = dict(pressure_patch='combined_static_dynamic_v6',
               pressure_unet='pressure_unet_v8', pressure_topology='pressure_topology_unet_v9')


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def export(research):
    base = Path(research)/'results/cavitation_20260913'
    if (DATA/'manifest.json').exists():
        raise FileExistsError('Already exported; omit --research-root')
    DATA.mkdir(parents=True,exist_ok=True)
    settings = json.loads((base/'pressure_only_v1/PROTOCOL.json').read_text())['settings']
    records = [r for r in json.loads((base/'CONVERTED.json').read_text())
               if r['case'] in ['Plunging3','Oscillation3']]
    manifest = dict(version=1, source='Author cavitation experiments, 2026-09-13',
        cases=['Plunging3','Oscillation3'], seeds=[11,22,33], displayed_seed=11,
        frame_selection='Maximum valid CFD cavity area in each case, not model score; all frames are retained for pooled evaluation',
        scope='Previously inspected optimization-held-out cases; alpha and pressure methods have different inputs and training histories',
        pressure_definition='float32 gauge pressure + operating pressure - configured vapor pressure, then zero solid',
        comparison='Common total-cavity support; alpha/topology outputs union classes 1 and 2; scalar outputs threshold at alpha_hat>=0.20',
        threshold_baseline='p_abs < p_v (pressure deficit < 0); no fitted parameter',
        output_limits='Only alpha-input and pressure-topology networks distinguish attached and disconnected. Others output total cavity only.',
        training={}, files={}, frames=[], source_checkpoints={}, source_code_sha256={})
    counters = {}
    for rec in records:
        name,index = rec['case'],rec['index']
        key = f'{name}_{index:03d}'
        pos = counters.get(name,0); counters[name] = pos+1
        source = base/'processed'/f'{key}.npz'
        pressure_source = base/'pressure_only_v1/fields'/f'{key}.npz'
        with np.load(source,allow_pickle=False) as z:
            fields = dict(alpha=z['alpha_v'],wall=z['wall'],reference=z['target'],
                          x=z['x'],y=z['y'],time=z['time'])
        cfg = settings[key]
        with np.load(pressure_source,allow_pickle=False) as z:
            fields['pressure_gauge'] = z['pressure'].astype('float32')
        p = fields['pressure_gauge']+cfg['operating_pressure']-cfg['p_vap_config']
        p[fields['wall']] = 0
        fields['pressure_deficit'] = p
        with np.load(base/'multicase_v1'/f'{name}_retrained.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['target'][pos],fields['reference'])
            fields['expected_alpha_topology'] = z['prediction'][pos]
        prediction_hashes = {}
        for method,folder in METHODS.items():
            for seed in manifest['seeds']:
                archive = base/folder/f'seed{seed}_{key}_noise0.npz'
                with np.load(archive,allow_pickle=False) as z:
                    expected = z['topology'] if method == 'pressure_topology' else z['prediction']>=.20
                fields[f'expected_{method}_{seed}'] = expected
                prediction_hashes[f'{method}_{seed}'] = digest(archive)
        np.savez_compressed(DATA/f'{key}.npz',**fields)
        manifest['frames'].append(dict(key=key,case=name,index=index,file=key+'.npz',
            time=float(fields['time']), source_sha256=digest(source),
            pressure_source_sha256=digest(pressure_source),pressure_settings=cfg,
            archived_prediction_sha256=prediction_hashes))
    checkpoint = base/'multicase_v1/selected.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    np.savez_compressed(DATA/'alpha_topology.npz',**{k:v.numpy() for k,v in state.items()})
    manifest['source_checkpoints']['alpha_topology'] = digest(checkpoint)
    cfg = json.loads((base/'multicase_v1/PROTOCOL.json').read_text())
    manifest['training']['alpha_topology'] = {k:cfg[k] for k in ['split','steps','seed','learning_rate','parent_sha256','selection','noise','threshold']}
    for method,folder in METHODS.items():
        manifest['training'][method] = json.loads((base/folder/'PROTOCOL.json').read_text())
        # The original v6 prose says BCE, but its executable selector used MAE.
        if method == 'pressure_patch':
            manifest['training'][method]['selection_prose_correction'] = 'Actual executable code selects validation macro MAE at 300-step intervals, not BCE.'
            metrics = json.loads((base/folder/'METRICS.json').read_text())
            manifest['retained_pressure_patch_noise_metrics'] = {
                seed:{k:v for k,v in rows.items() if k.startswith(('Plunging3_', 'Oscillation3_'))}
                for seed,rows in metrics.items()}
        for seed in manifest['seeds']:
            checkpoint = base/folder/f'seed{seed}.pt'
            state = torch.load(checkpoint,map_location='cpu',weights_only=True)
            np.savez_compressed(DATA/f'{method}_{seed}.npz',**{k:v.numpy() for k,v in state.items()})
            manifest['source_checkpoints'][f'{method}_{seed}'] = digest(checkpoint)
    for filename in ['cavitation_multicase_20260913.py','cavitation_combined_static_dynamic_v6.py',
                     'cavitation_pressure_unet_v8.py','cavitation_pressure_topology_unet_v9.py']:
        manifest['source_code_sha256'][filename] = digest(Path(research)/'tools'/filename)
    manifest['files'] = {p.name:digest(p) for p in sorted(DATA.glob('*.npz'))}
    write_json(DATA/'manifest.json',manifest)


def replay():
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    manifest,frames = load_comparison(DATA)
    results, checked, seed11 = [],0,None
    methods = ['alpha_topology','pressure_threshold',*METHODS]
    for seed in manifest['seeds']:
        outputs = infer_comparison(DATA,frames,seed)
        if seed == 11:
            seed11 = outputs
        for frame,out in zip(frames,outputs):
            np.testing.assert_array_equal(out['alpha_topology'],frame['expected_alpha_topology'])
            checked += 1
            for method in METHODS:
                actual = out[method] if method == 'pressure_topology' else out[method]>=.20
                np.testing.assert_array_equal(actual,frame[f'expected_{method}_{seed}'])
                checked += 1
        for case in manifest['cases']:
            members = [(f,o) for f,o in zip(frames,outputs) if f['case']==case]
            for method in methods:
                if seed != 11 and method in ('alpha_topology','pressure_threshold'):
                    continue
                counts = np.zeros(3,dtype='int64')
                for f,o in members:
                    valid = (f['reference']!=255)&~f['wall']
                    m = binary_score(method_mask(method,o[method]),f['alpha']>=.2,valid)
                    counts += [m['tp'],m['fp'],m['fn']]
                tp,fp,fn = map(int,counts)
                results.append(dict(case=case,method=method,seed=seed,frames=len(members),
                    tp=tp,fp=fp,fn=fn,dice=2*tp/(2*tp+fp+fn)))
    write_json(OUT/'methods_metrics.json',dict(scope=manifest['scope'],
        compared_output=manifest['comparison'], rows=results,
        archived_array_checks=checked, archived_binary_predictions_identical=True))
    figures = {}
    for case in manifest['cases']:
        ids = [i for i,f in enumerate(frames) if f['case']==case]
        i = max(ids,key=lambda j:int(((frames[j]['alpha']>=.20)&(frames[j]['reference']!=255)&~frames[j]['wall']).sum()))
        fig = methods_figure(frames[i],seed11[i])
        filename = f'methods_{case}.png'
        fig.savefig(OUT/filename,dpi=170)
        plt.close(fig)
        figures[filename] = dict(sha256=digest(OUT/filename),key=frames[i]['key'],
            time=float(frames[i]['time']),seed=11,
            solid_false_positive_pixels={name:int((method_mask(name,seed11[i][name])&frames[i]['wall']).sum())
                                         for name in methods})
    write_json(OUT/'methods_figure_provenance.json',dict(figures=figures,
        manifest_sha256=digest(DATA/'manifest.json'),code_sha256=digest(ROOT/'flowmllab/cavitation_methods.py'),
        selection=manifest['frame_selection'],
        display='Thicker colored contours with white contrast halos; arrows count unmodified false vapor predictions inside solid. Dice excludes solid/uncertain support.'))
    print(f'Alpha/pressure comparison: {len(frames)} shared frames, 3 pressure seeds, {checked} archived array checks passed.',flush=True)


def notebook_cells():
    import nbformat as nbf
    import textwrap
    md = lambda s: nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
    code = lambda s: nbf.v4.new_code_cell(textwrap.dedent(s).strip())
    return [md(r'''
    ## 8. Compare our alpha, pressure and U-Net methods on identical frames

    This second retained campaign is from 2026-09-13. It includes the original
    alpha-input context U-Net continued on moving cases, a fixed pressure threshold,
    the pressure-only 3x3 model, pressure-only U-Net, and pressure-only topology U-Net.
    The two whole nontraining trajectories are Plunging3 (7 frames) and Oscillation3
    (9 frames); both were inspected in earlier development. They contain no valid
    attached-reference pixels, so they cannot validate attached-class generalization.

    Pressure input is $p_{gauge}+p_{operating}-p_{v,configured}$ in Pa, represented
    at asinh scales 10/100/1000/10000 Pa, plus geometry. **Alpha is not supplied to
    any pressure model.** The classical threshold is $p_{abs}<p_v$. Scalar pressure
    models predict total vapor support at $\hat\alpha\geq0.20$; only the two topology
    networks directly distinguish attached from detached. We compare common total
    support here, not cloud identity for models that do not predict it.

    All three original pressure seeds (11,22,33) are replayed. Plots use the first
    declared seed and the largest valid CFD cavity area, not the best model score. The alpha model
    has a different input advantage and training history: this is an explanation
    of different methods, not a matched-input architecture ranking.
    '''), code('''
    from flowmllab.cavitation_methods import (
        load_comparison, infer_comparison, method_mask, binary_score, methods_figure)
    METHODS_DATA = ROOT/'data/week11_cavitation_methods'
    methods_manifest, frames = load_comparison(METHODS_DATA)
    methods_before = {n:digest(METHODS_DATA/n) for n in methods_manifest['files']}
    method_names = ['alpha_topology','pressure_threshold','pressure_patch','pressure_unet','pressure_topology']
    comparison_rows = []
    for seed in methods_manifest['seeds']:
        outputs = infer_comparison(METHODS_DATA,frames,seed)
        if seed == 11:
            plot_outputs = outputs
        for f,o in zip(frames,outputs):
            np.testing.assert_array_equal(o['alpha_topology'],f['expected_alpha_topology'])
            for method in method_names[2:]:
                actual = o[method] if method == 'pressure_topology' else o[method]>=.20
                np.testing.assert_array_equal(actual,f[f'expected_{method}_{seed}'])
        for case_name in methods_manifest['cases']:
            members = [(f,o) for f,o in zip(frames,outputs) if f['case']==case_name]
            for method in method_names:
                if seed != 11 and method in method_names[:2]:
                    continue
                counts = np.zeros(3,dtype='int64')
                for f,o in members:
                    valid = (f['reference']!=255)&~f['wall']
                    v = binary_score(method_mask(method,o[method]),f['alpha']>=.20,valid)
                    counts += [v['tp'],v['fp'],v['fn']]
                tp,fp,fn = counts
                comparison_rows.append({'case':case_name,'method':method,'seed':seed,
                                        'total-cavity Dice':2*tp/(2*tp+fp+fn)})
    method_metrics = pd.DataFrame(comparison_rows)
    display(method_metrics)
    display(method_metrics.groupby(['case','method'])['total-cavity Dice'].agg(['mean','std','count']))
    print('All 16 shared frames and three pressure seeds reproduce the archived binary predictions.')
    '''), md('''
    ## 9. The homepage comparison: alpha versus pressure methods

    Every panel uses the same CFD field, geometry and time. CFD alpha is the
    background for visual comparison only; it never enters pressure inference.
    Orange/magenta are attached/disconnected classes. Green is total cavity only.
    These are different output semantics, not interchangeable cloud labels.
    The panel Dice values describe this one frame; the table above pools the
    complete two trajectories and retains all pressure seeds.
    White halos improve contour visibility without changing masks. Arrows mark
    false vapor predictions inside the solid hydrofoil: these are model errors,
    not attached cavities. The displayed Dice excludes solid/uncertain support;
    it does not penalize these solid errors, which are reported separately.
    '''), code('''
    for case_name in methods_manifest['cases']:
        ids = [i for i,f in enumerate(frames) if f['case']==case_name]
        i = max(ids,key=lambda j:int(((frames[j]['alpha']>=.20)&(frames[j]['reference']!=255)&~frames[j]['wall']).sum()))
        methods_figure(frames[i],plot_outputs[i])
        plt.show()
    assert methods_before == {n:digest(METHODS_DATA/n) for n in methods_manifest['files']}
    '''), md('''
    **Interpretation:** More spatial context did not uniformly improve the original
    pressure experiments. Keep the U-Net and topology variants as comparisons,
    including their errors. A pressure deficit is a useful feature but is not a
    cavitation transport law; neither model forecasts shedding or tracks cloud identity.
    Pressure sensitivity remains: in the original 3x3 campaign, the three-seed
    Plunging3 total-cavity Dice fell from 0.8711 on clean pressure to 0.7357 at
    1000-Pa input-noise SD. Oscillation3 fell from 0.9109 to 0.8181. Those retained
    noise experiments are documented in the comparison manifest; this gallery
    replays the clean same-frame comparison.

    [Multi-method data and checkpoint provenance](../../data/week11_cavitation_methods/README.md) ·
    [All method code](../../flowmllab/cavitation_methods.py) ·
    [Original campaign metadata](../../data/week11_cavitation_methods/manifest.json).
    ''')]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--research-root',type=Path)
    args = ap.parse_args()
    if args.research_root:
        export(args.research_root)
    replay()
