"""Export hash-checked native fields and frozen predictions for the browser.

No fitting, solver execution, changes to the split, or writes to results/.
All field pixels are native nodes; only streamline paths are rounded for SVG.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm, LinearSegmentedColormap
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmllab import ghia
from flowmllab import cavity_comparison as cc
from flowmllab import cavity_field_operator as op


def write_json(path, data):
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")


def load_cases():
    cases, selection, _ = cc.load()  # verifies the frozen selection, weights and training fields
    folder = ROOT / "results/cavity_field_deeponet"
    trunk = cc.read_npz(folder / "trunk.npz")
    frozen = cc.read_npz(folder / "frozen_predictions.npz")
    receipt = json.loads((folder / "scoring_receipt.json").read_text())
    models = []
    for i in selection['selected_indices']:
        z = cc.read_npz(folder/'models'/(selection['candidate_models'][i]['name']+'.npz'))
        models.append(dict(theta=z['theta'], shapes=z['shapes'].tolist(), scale=z['scale']))
    for re in ghia.REYNOLDS:
        p = ROOT / f'results/cavity_highre_ghia/n512/re{re}.npz'
        meta = json.loads(p.with_suffix('.json').read_text())
        cc.checked(p, meta['field_sha256'])
        if not meta['converged'] or meta['cells'] != 512 or meta['reynolds'] != re:
            raise ValueError('Mismatched CFD configuration')
        truth = cc.read_npz(p)
        if re in op.TEST:
            cc.checked(p, receipt['heldout_field_sha256'][str(re)])
        coefficients = np.array([op.predict_coefficients(m, [re])[0] for m in models])
        if re in op.TEST:
            i = list(frozen['test_re']).index(re)
            fixed = frozen['candidate_coefficients'][selection['selected_indices'], i]
            np.testing.assert_allclose(coefficients, fixed, rtol=1e-12, atol=1e-12)
            coefficients = fixed
        flat = op.reconstruct(trunk, coefficients.mean(axis=0))
        n = truth['u'].size
        nn = dict(x=truth['x'], y=truth['y'], u=flat[:n].reshape(truth['u'].shape), v=flat[n:].reshape(truth['v'].shape))
        if re in cases:
            for k in ('u', 'v'):
                np.testing.assert_allclose(nn[k], cases[re]['POD-DeepONet'][k], rtol=1e-12, atol=1e-12)
        cases[re] = {'CFD': truth, 'POD-DeepONet': nn}
    return cases, selection


def scalars(f, re):
    p = cc.pressure(f, re)
    h = f['x'][1] - f['x'][0]
    omega = (f['v'][1:-1,2:]-f['v'][1:-1,:-2]-f['u'][2:,1:-1]+f['u'][:-2,1:-1])/(2*h)
    return dict(speed=np.hypot(f['u'],f['v']), pressure=p['p'], vorticity=omega)


def paths(f):
    fig, ax = plt.subplots()
    cc.stream(ax, f, 'black', '-')
    result=[]
    for collection in ax.collections:
        for segment in collection.get_segments():
            if len(segment)>1:
                result.append(np.round(segment,6).tolist())
    plt.close(fig)
    return result


def export(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cases, selection = load_cases()
    values = {re:{name:scalars(f,re) for name,f in methods.items()} for re,methods in cases.items()}
    cmap = LinearSegmentedColormap.from_list('flow_spectrum',['#1246b8','#0879ef','#04cddd','#24d482','#d4e746','#ffd139','#f87521','#ce122b'])
    error_cmap=matplotlib.colormaps['RdBu_r']
    labels={'speed':'Speed |u| / U_lid', 'pressure':'Pressure (p − mean(p)) / (ρ U_lid²)', 'vorticity':'Vorticity ω L / U_lid'}
    norms={}; scales={}; error_norms={}
    for field in labels:
        arrays=[v[field] for methods in values.values() for v in methods.values()]
        lo=min(float(a.min()) for a in arrays); hi=max(float(a.max()) for a in arrays)
        bound=max(abs(lo),abs(hi))
        norms[field]=Normalize(0,hi) if field=='speed' else SymLogNorm(.02 if field=='pressure' else 1, vmin=-bound,vmax=bound)
        eb=max(float(np.abs(v['POD-DeepONet'][field]-v['CFD'][field]).max()) for v in values.values())
        error_norms[field]=Normalize(-eb,eb)
        ticks=np.linspace(0,hi,5) if field=='speed' else np.array([-bound,-(.02 if field=='pressure' else 1),0,(.02 if field=='pressure' else 1),bound])
        scales[field]=dict(label=labels[field],min=norms[field].vmin,max=norms[field].vmax,
            normalization='linear' if field=='speed' else 'symmetric-log', linear_threshold=None if field=='speed' else (.02 if field=='pressure' else 1),
            ticks=[dict(value=float(t),position=float(norms[field](t))) for t in ticks],
            colors=['#%02x%02x%02x'%tuple(c[:3]) for c in cmap(np.linspace(0,1,65),bytes=True)],
            error=dict(min=-eb,max=eb,colors=['#%02x%02x%02x'%tuple(c[:3]) for c in error_cmap(np.linspace(0,1,65),bytes=True)]))
    manifest=dict(schema_version=1,reynolds=list(ghia.REYNOLDS),scales=scales,cases={},
        solver='Steady, two-dimensional incompressible Navier–Stokes; 512×512 cells / 513×513 nodes',
        model='Frozen hybrid_2 POD-DeepONet, mean of seeds 7, 23, 61; no Ghia samples used in fitting',
        split={k:selection[k] for k in ('train_re','validation_re','test_re')},
        pressure='Recovered from steady momentum gradients, zero interior mean per field; not an independent pressure validation.',
        sampling='Native 513×513 speed pixels and 511×511 interior pressure/vorticity pixels; no wall extrapolation. Full 513-point centreline curves.',
        interpretation='Agreement with this finite-grid steady CFD branch is not mesh independence or proof of physical stability at high Reynolds number.')
    reference_metrics=list(csv.DictReader((ROOT/'results/cavity_field_deeponet/flow_comparison/comparison_metrics.csv').open()))
    for re in ghia.REYNOLDS:
        methods=cases[re]
        truth=methods['CFD']; nn=methods['POD-DeepONet']
        role='held-out test' if re in op.TEST else 'validation' if re in op.VALIDATION else 'training'
        ref=ghia.reference(re)
        coordinates=truth['x'].tolist()
        profiles={name:{'u':f['u'][:,256].tolist(),'v':f['v'][256,:].tolist()} for name,f in methods.items()}
        difference=np.r_[nn['u'].ravel()-truth['u'].ravel(),nn['v'].ravel()-truth['v'].ravel()]
        reference=np.r_[truth['u'].ravel(),truth['v'].ravel()]
        error=float(np.linalg.norm(difference)/np.linalg.norm(reference))
        pressure_error=float(np.linalg.norm(values[re]['POD-DeepONet']['pressure']-values[re]['CFD']['pressure'])/np.linalg.norm(values[re]['CFD']['pressure']))
        for row in reference_metrics:
            if int(row['Re'])==re and row['method']=='POD-DeepONet':
                np.testing.assert_allclose([error,pressure_error],[float(row['velocity_relative_l2']),float(row['pressure_relative_l2'])],rtol=1e-9,atol=1e-12)
        case=dict(re=re,role=role,velocity_relative_l2=error,pressure_relative_l2=pressure_error,
            coordinate=coordinates,profiles=profiles,ghia={k:v.tolist() for k,v in ref.items()},
            flags={k:dict(position=ghia.FLAGGED[(re,k)][0],reason=ghia.FLAGGED[(re,k)][1]) for k in ('u','v') if (re,k) in ghia.FLAGGED},
            streamlines={name:paths(f) for name,f in methods.items()}, fields={},ghia_scores={})
        for name,f in methods.items():
            case['ghia_scores'][name]={c:ghia.profile_errors(re,c,truth['x'],np.array(profiles[name][c])) for c in ('u','v')}
        for field in labels:
            a=values[re]['CFD'][field]; b=values[re]['POD-DeepONet'][field]
            case['fields'][field]=dict(extent=[0,1,0,1] if field=='speed' else [1/512,511/512,1/512,511/512],
                native_shape=list(a.shape),ranges={name:[float(values[re][name][field].min()),float(values[re][name][field].max())] for name in methods})
            for key,array,norm,cm in [('cfd',a,norms[field],cmap),('nn',b,norms[field],cmap),('error',b-a,error_norms[field],error_cmap)]:
                filename=f're{re}-{field}-{key}.png'
                Image.fromarray(np.flipud(cm(norm(array),bytes=True)[:,:,:3])).save(destination/filename,optimize=True)
                case['fields'][field][key]=filename
        write_json(destination/f're{re}.json',case)
        manifest['cases'][str(re)]=dict(file=f're{re}.json',role=role,velocity_relative_l2=error)
        with (destination/f're{re}-profiles.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['x_or_y_over_L','u_CFD','u_POD_DeepONet','v_CFD','v_POD_DeepONet'])
            writer.writerows(zip(coordinates,profiles['CFD']['u'],profiles['POD-DeepONet']['u'],profiles['CFD']['v'],profiles['POD-DeepONet']['v']))
        print(f'Exported Re={re}: {role}; vector L2={100*error:.5f}%',flush=True)
    manifest['source_sha256']={Path(p).relative_to(ROOT).as_posix():h for p,h in sorted(cc.TRACKED.items())}
    for file in ('flowmllab/ghia.py','flowmllab/cavity_comparison.py','flowmllab/cavity_field_operator.py','qa/build_cavity_explorer.py'):
        manifest['source_sha256'][file]=hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    write_json(destination/'manifest.json',manifest)
    return manifest

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path)
    export(parser.parse_args().destination)
