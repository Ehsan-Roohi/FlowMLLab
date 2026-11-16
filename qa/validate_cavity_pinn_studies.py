"""Independent release-evidence checks; integrity is not scientific acceptance.

This checker never trains, changes thresholds, submits jobs, or rewrites evidence.
Only the compact validation receipt is written. Torch is not required.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = (100, 400, 1000, 3200, 5000, 7500, 10000)
SEEDS = (7, 23, 61)
D5_SEEDS = (12345, 23456, 34567)
D5_METHODS = ('adam_lbfgs', 'soap_ssb')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=512)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked(entry, base=ROOT):
    path = (base / entry['path']).resolve()
    require(path.is_relative_to(base.resolve()), 'Evidence path leaves its declared base')
    require(path.is_file() and sha(path) == entry['sha256'], f'Evidence hash failed: {path}')
    return path


def source_matches(path, expected):
    """Admit the original byte identity or its explicit LF/CRLF serialization."""
    raw = Path(path).read_bytes()
    normal = raw.decode('utf-8-sig').replace('\r\n', '\n').encode('utf-8')
    candidates = {'raw': raw, 'canonical_LF': normal, 'CRLF_serialization': normal.replace(b'\n', b'\r\n')}
    matches = [name for name, data in candidates.items() if hashlib.sha256(data).hexdigest() == expected]
    require(bool(matches), f'Source identity changed: {path}')
    return matches[0]


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Timestamp lacks timezone')
    return result


@lru_cache(maxsize=3)
def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def same(a, b, description, atol=2e-12, rtol=2e-10):
    require(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False), description)


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def fields_ok(data, vertical, variables):
    x, y = data['x'], data[vertical]
    require(x.ndim == y.ndim == 1 and len(x) >= 3 and len(y) >= 3, 'Invalid coordinate dimensions')
    require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite coordinates')
    require(np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0), 'Coordinates must increase')
    for key in variables:
        require(data[key].shape == (len(y), len(x)) and np.isfinite(data[key]).all(), f'Invalid field {key}')


def uniform_trapezoid(shape):
    a = np.ones(shape[0]); b = np.ones(shape[1])
    a[[0, -1]] = .5; b[[0, -1]] = .5
    return a[:, None] * b[None, :]


def ghia_module():
    spec = importlib.util.spec_from_file_location('release_ghia_values', ROOT / 'flowmllab/ghia.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ghia_metrics(pred, ref, mask, w):
    weight = w * mask; total = weight.sum()
    mean = lambda value: float(np.sum(weight * value) / total)
    speed = np.sqrt(mean(ref['u']**2 + ref['v']**2))
    error = np.sqrt(mean((pred['u']-ref['u'])**2 + (pred['v']-ref['v'])**2))
    domega = pred['omega']-ref['omega']; omega = np.sqrt(mean(ref['omega']**2))
    return dict(sample_count=int(mask.sum()), reference_velocity_rms_U=speed,
                velocity_error_rms_U=error, velocity_relative_l2=error/speed,
                u_error_signed_mean_U=mean(pred['u']-ref['u']), v_error_signed_mean_U=mean(pred['v']-ref['v']),
                omega_error_rms_U_over_W=np.sqrt(mean(domega**2)), omega_error_signed_mean_U_over_W=mean(domega),
                omega_relative_l2=np.sqrt(mean(domega**2))/omega)


def validate_ghia(path, variant_records=None):
    m = read_json(path); records = m['records']
    require(m['schema_version'] == 1 and m['campaign_kind'] == 'ghia_constant_lid', 'Wrong Ghia schema')
    variant = variant_records is not None
    reynolds = (100, 7500) if variant else RES
    count = 12 if variant else 42
    expected = {(re, seed, collar) for re in reynolds for seed in SEEDS for collar in (.01, .02)}
    engine_path = ROOT / ('flowmllab/cavity_pinn_stratified.py' if variant else 'flowmllab/cavity_pinn_campaign.py')
    keys = [(r['Re'], r['seed'], r['collar']) for r in records]
    require(len(keys) == count and len(set(keys)) == count and set(keys) == expected, f'Exactly {count} unique expected Ghia cases are required')
    require(len({r['run_id'] for r in records}) == count, 'Duplicate run identifiers')
    require(all(r['outcome'] == 'scored' for r in records), 'All supplied runs must retain a scored endpoint')
    coverage_path = None
    if variant:
        coverage_runs = {}
        for key, record in variant_records.items():
            frozen_path = checked(record['training_provenance']['frozen_manifest.json'])
            frozen = read_json(frozen_path)
            coverage_runs[key] = dict(checkpoint_sha256=frozen['checkpoint_sha256'],
                configuration_sha256=frozen['frozen_model']['config_sha256'],
                frozen_manifest_sha256=sha(frozen_path), model_sha256=frozen['frozen_model']['model_sha256'])
        require(set(coverage_runs) == expected, 'Variant provenance does not cover its exact matrix')
    else:
        coverage_path = path.parent / 'sampling_coverage.json'; coverage = read_json(coverage_path)
        require(len(coverage['pairs']) == 21, 'Missing sampling coverage pairs')
        require(coverage['summary']['identical_objective_pairs'] == 14 and coverage['summary']['collar_independence_established'] is False,
                'Coverage caveat has been lost')
        source_matches(ROOT / 'flowmllab/cavity_pinn_campaign.py', coverage['sampler_source']['canonical_sha256'])
        coverage_runs = {}
        for pair in coverage['pairs']:
            for run in pair['runs']:
                coverage_runs[(pair['Re'], pair['seed'], run['collar'])] = run
        require(set(coverage_runs) == expected, 'Sampling receipt does not cover this exact matrix')
    source_modes = set(); g = ghia_module(); validated = []; by_id = {}
    freezes = []; accesses = []; checks_failed = {}; summaries = {}
    grid = np.arange(1, 512, dtype=float) / 512
    w = uniform_trapezoid((511, 511)); yy = np.broadcast_to(grid[:, None], w.shape)
    masks = {'interior': np.ones(w.shape, bool), 'lower_quarter': yy <= .25, 'upper_quarter': yy >= .75}
    for record in records:
        identifier = record['run_id']; re, seed, collar = record['Re'], record['seed'], record['collar']
        paths = {key: checked(record[key]) for key in ('prediction', 'reference', 'history', 'ghia_profiles')}
        config = record['config']; source = record['source']; freeze = record['freeze']; score = record['score']
        require((config['reynolds'], config['seed'], config['collar']) == (re, seed, collar), 'Record/config mismatch')
        require(source['config'] == config, 'Projection configuration mismatch')
        require(hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == source['config_sha256'], 'Configuration hash mismatch')
        source_modes.add(source_matches(engine_path, source['source_canonical_sha256']))
        source_modes.add(source_matches(ROOT / 'qa/evaluate_cavity_pinn_campaign.py', source['evaluator_source_sha256']))
        source_modes.add(source_matches(ROOT / 'flowmllab/ghia.py', source['ghia_source_sha256']))
        bound = coverage_runs[(re, seed, collar)]
        for coverage_key, source_key in [('checkpoint_sha256', 'checkpoint_sha256'), ('configuration_sha256', 'config_sha256'),
                                          ('frozen_manifest_sha256', 'original_freeze_manifest_sha256'), ('model_sha256', 'model_sha256')]:
            require(bound[coverage_key] == source[source_key], 'Sampling receipt identity mismatch')
        require(record['prediction']['sha256'] == freeze['prediction_sha256'] == score['prediction_sha256'], 'Prediction/freeze/score identity mismatch')
        require(freeze['model_sha256'] == source['model_sha256'], 'Frozen model identity mismatch')
        projection = read_json(paths['prediction'].parent / 'projection_freeze.json')
        require(projection['identity'] == source and projection['prediction_sha256'] == freeze['prediction_sha256'], 'Projection ledger mismatch')
        require(projection['reference_values_read'] is False, 'Projection must precede reference values')
        require(projection['at_utc'] == freeze['at_utc'], 'Projection timestamp mismatch')
        require(read_json(paths['prediction'].parent / 'evaluation_record.json') == record, 'Per-run record differs from campaign manifest')
        require(read_json(paths['prediction'].parent / 'scores.json') == record['scores'], 'Per-run scores differ from manifest')
        ftime = stamp(freeze['at_utc']); atime = stamp(score['reference_accessed_at_utc'])
        require(stamp(source['original_frozen_at_utc']) <= ftime < atime <= stamp(score['cfd_values_accessed_at_utc']), 'Invalid original/projection/reference ordering')
        require(stamp(projection['coordinates_read_at_utc']) <= ftime, 'Coordinates must precede projection freeze')
        freezes.append(ftime); accesses.append(atime)
        pred, ref = arrays(paths['prediction']), arrays(paths['reference'])
        fields_ok(pred, 'y', ('u', 'v', 'p', 'psi', 'omega')); fields_ok(ref, 'y', ('u', 'v', 'psi', 'omega'))
        for name in ('x', 'y'):
            require(np.array_equal(pred[name], grid) and np.array_equal(ref[name], grid), 'Exact 511-node native interior required')
        native = arrays(checked(record['reference_source']))
        for name in ('x', 'y'):
            require(np.array_equal(native[name][1:-1], ref[name]), 'Native coordinate subset mismatch')
        for name in ('u', 'v', 'psi', 'omega'):
            require(np.array_equal(native[name][1:-1, 1:-1], ref[name]), 'Reference values were interpolated or changed')
        require('p' not in native and 'p' not in ref and record['pressure_reference_available'] is False, 'No Ghia CFD pressure truth is available')
        for name, mask in masks.items():
            for key, value in ghia_metrics(pred, ref, mask, w).items():
                same(value, record['scores']['regions'][name][key], f'Native metric mismatch: {identifier}/{name}/{key}')
        truth = g.reference(re); profile_rows = rows(paths['ghia_profiles'])
        require(len(profile_rows) == 34, 'All 34 signed published samples must remain visible')
        for component, coordinate in (('u', 'y'), ('v', 'x')):
            require(np.array_equal(pred['ghia_'+coordinate], truth[coordinate]), 'Direct Ghia query locations changed')
            require(np.array_equal(ref['ghia_'+coordinate], truth[coordinate]) and np.array_equal(ref['ghia_'+component], truth[component]), 'Ghia reference values changed')
            values = pred['ghia_'+component+'_prediction']; keep = g.valid_mask(re, component, truth[coordinate])
            selected = [r for r in profile_rows if r['component'] == component]
            require(len(selected) == 17 and np.isfinite(values).all(), 'Invalid signed Ghia profile')
            for i, row in enumerate(selected):
                for key, value in [('position', truth[coordinate][i]), ('published', truth[component][i]),
                                   ('prediction', values[i]), ('signed_error_U', values[i]-truth[component][i])]:
                    same(float(row[key]), value, 'Signed profile value mismatch')
                require((row['screened_included'].lower() == 'true') == bool(keep[i]), 'Published screening policy changed')
            errors = values-truth[component]; reported = record['scores']['ghia'][component]
            same(np.linalg.norm(errors[keep])/np.linalg.norm(truth[component][keep]), reported['screened_relative_l2'], 'Screened Ghia score mismatch')
            same(np.linalg.norm(errors)/np.linalg.norm(truth[component]), reported['published_relative_l2'], 'Raw Ghia score mismatch')
        history = rows(paths['history'])
        if variant:
            receipt = read_json(checked(variant_records[(re, seed, collar)]['training_provenance']['current_receipt.json']))
            expected_lbfgs_rows = receipt['lbfgs_completed'] + receipt['lbfgs_aborted_steps']
            total = config['adam_steps'] + expected_lbfgs_rows
            require(receipt['closure_calls'] == config['closure_budget'], 'Variant stopped at the wrong closure cap')
        else:
            expected_lbfgs_rows = config['lbfgs_steps']
            total = config['adam_steps']+config['lbfgs_steps']
        require(len(history) == total and record['history']['independent_audit'] is False, 'Training history was shortened or labelled independent audit')
        require(record['history']['metric_name'] == 'training_collar_momentum_rms', 'Wrong training metric label')
        counts = np.array([int(r['objective_evaluations']) for r in history]); times = np.array([float(r['elapsed_seconds']) for r in history])
        require(np.all(np.diff(counts) > 0) and np.all(np.diff(times) >= 0), 'Nonmonotone training counters')
        require(sum(r['stage'] == 'adam' for r in history) == config['adam_steps'], 'Adam history count differs')
        require(sum(r['stage'] == 'lbfgs' for r in history) == expected_lbfgs_rows, 'L-BFGS history count differs')
        if variant: require(counts[-1] == config['closure_budget'], 'Variant history did not spend the exact paired cap')
        for i, row in enumerate(history):
            require(int(row['residual_point_evaluations']) == counts[i]*config['collocation'], 'Residual-point work count mismatch')
            require(np.isfinite(float(row['metric_value'])) and float(row['metric_value']) >= 0, 'Nonfinite training curve')
        outer = record['physics_audit']['outer_uniform']; gh = record['scores']['ghia']
        require(outer['finite'] and len(record['physics_audit']['corner_shells']) == 6, 'Missing finite raw physics audits')
        require(all(s['finite'] for s in record['physics_audit']['corner_shells']), 'Nonfinite corner audit')
        numeric = dict(outer_momentum_rms_le_1e_2=outer['momentum_rms'] <= .01,
                       divergence_rms_le_1e_10=outer['divergence_rms'] <= 1e-10,
                       both_screened_ghia_relative_l2_le_0_02=max(gh['u']['screened_relative_l2'], gh['v']['screened_relative_l2']) <= .02,
                       interior_velocity_relative_l2_le_0_05=record['scores']['regions']['interior']['velocity_relative_l2'] <= .05)
        require(numeric == record['scores']['numeric_checks'], 'Scientific target flags do not match measurements')
        for key, passed in numeric.items():
            checks_failed[key] = checks_failed.get(key, 0) + int(not passed)
        require(record['physical_review_status'] != 'accepted' or all(numeric.values()), 'Failed case was labelled scientifically accepted')
        validated.append({'run_id': identifier, 'numeric_gates_pass': all(numeric.values())})
        by_id[identifier] = record
        summaries.setdefault(re, []).append(record['scores']['regions']['interior']['velocity_relative_l2'])
    require(max(freezes) < min(accesses), 'Every supplied projection must freeze before the first reference read')
    pairs = m['collar_comparisons']; pair_keys = [(p['Re'], p['seed']) for p in pairs]
    expected_pairs = {(r, s) for r in reynolds for s in SEEDS}
    if variant:
        require(len(set(pair_keys)) == len(pair_keys) and set(pair_keys).issubset(expected_pairs),
                'Generic scorer has repeated or out-of-scope collar pairs')
    else:
        require(len(pairs) == 21 and len(set(pair_keys)) == 21 and set(pair_keys) == expected_pairs,
                'All 21 unique baseline collar pairs are required')
    for pair in pairs:
        fine, coarse = by_id[pair['fine_run_id']], by_id[pair['coarse_run_id']]
        require((fine['Re'], fine['seed'], fine['collar']) == (pair['Re'], pair['seed'], .01), 'Fine collar pair mismatch')
        require((coarse['Re'], coarse['seed'], coarse['collar']) == (pair['Re'], pair['seed'], .02), 'Coarse collar pair mismatch')
        a, b = arrays(checked(fine['prediction'])), arrays(checked(coarse['prediction']))
        error = float(np.sqrt(np.sum(w*((a['u']-b['u'])**2+(a['v']-b['v'])**2))/w.sum()))
        scale = float(np.sqrt(np.sum(w*(a['u']**2+a['v']**2))/w.sum()))
        same(error, pair['absolute_velocity_difference_rms_U'], 'Collar absolute score mismatch')
        same(error/scale, pair['relative_velocity_difference'], 'Collar relative score mismatch')
        require(pair['fine_prediction_sha256'] == fine['prediction']['sha256'] and pair['coarse_prediction_sha256'] == coarse['prediction']['sha256'], 'Collar prediction hashes differ')
    return {'integrity': 'passed', 'manifest_sha256': sha(path), 'verified_cases': count, 'verified_collar_pairs': len(pairs),
            'sampling_coverage_sha256': sha(coverage_path) if coverage_path else None, 'uninformative_identical_objective_pairs': 0 if variant else 14,
            'source_serializations_verified': sorted(source_modes), 'all_projections_frozen_before_first_reference_read': not bool(m.get('prior_evaluation_receipt')),
            'all_projections_frozen_before_completed_pass_reference_read': True,
            'native_grid': '511 x 511 exact original interior nodes; no interpolation',
            'vorticity_comparison': 'native AD PINN omega versus retained CFD transported omega',
            'scientific_acceptance': {'accepted_cases': 0, 'numeric_gate_pass_cases': sum(r['numeric_gates_pass'] for r in validated),
                'failed_gate_counts': checks_failed, 'status': 'No case is promoted by integrity verification; physical review and meaningful collar refinement remain separate'},
            'velocity_relative_l2_range_by_Re': {str(re): [min(v), max(v)] for re, v in summaries.items()},
            'pressure_truth': 'unavailable', 'checkpoint_scope': 'Checkpoint identities bound to frozen provenance and independently byte-verified external recoveries; binaries remain external'}


def curl(data):
    return np.gradient(data['v'], data['x'], axis=1, edge_order=2)-np.gradient(data['u'], data['h'], axis=0, edge_order=2)


def d5_regions(x, h):
    X, H = np.meshgrid(x, h); inside = (X > .02)&(X < .98)&(H > .02)&(H < 4.98)
    return {'Interior': inside, 'Upper-left': inside&(X < .25)&(H > 4.75), 'Floor_0.4': inside&(H < .4), 'Floor_0.65': inside&(H < .65)}


def d5_metrics(pred, ref, mask):
    vel = np.sqrt(np.mean(((pred['u']-ref['u'])**2+(pred['v']-ref['v'])**2)[mask]))
    scale = np.sqrt(np.mean((ref['u']**2+ref['v']**2)[mask])); dp = (pred['p']-ref['p'])[mask]
    dw = (pred['omega']-ref['omega'])[mask]; wscale = np.sqrt(np.mean(ref['omega'][mask]**2))
    return dict(count=int(mask.sum()), velocity_RMS_U=vel, CFD_velocity_RMS_U=scale,
                velocity_NRMSE_percent=100*vel/scale, pressure_RMS_rhoU2=np.sqrt(np.mean(dp**2)),
                pressure_signed_mean_offset_rhoU2=dp.mean(), pressure_local_shape_RMS_rhoU2=np.sqrt(np.mean((dp-dp.mean())**2)),
                vorticity_RMS_UoverW=np.sqrt(np.mean(dw**2)), vorticity_NRMSE_percent=100*np.sqrt(np.mean(dw**2))/wscale)


def validate_d5(path, required=False):
    if not path.exists():
        return {'integrity': 'pending', 'reason': 'Fresh six-arm D5 evidence has not arrived', 'required_for_this_invocation': required}
    m = read_json(path); base = path.parent; records = m['records']
    require(m['schema_version'] == 1 and m['campaign_kind'] == 'd5_gaussian_lid' and m['path_base'] == 'manifest_directory', 'Wrong D5 manifest schema')
    require(m['purpose'] == 'controlled_pilot' and m['status'] == 'controlled_pilot_scored', 'Cost smoke cannot stand in for a controlled scientific study')
    expected = {(s, method) for s in D5_SEEDS for method in D5_METHODS}
    keys = [(r['seed'], r['method']) for r in records]
    require(len(keys) == 6 and set(keys) == expected and len(set(keys)) == 6, 'Six complete expected D5 arms are required')
    require(m['CFD_training_labels'] is False and m['all_runs_frozen_before_reference_read'] is True, 'Missing D5 information-boundary declaration')
    require(m['winner_claim'] is False and m['historical_workflow_reproduction'] is False, 'Finite controlled pipeline study must not claim an automatic winner or historical reproduction')
    freezes = []; accesses = []; paired = {}; result_ranges = {}
    for r in records:
        required_entries = ('prediction', 'comparison_prediction', 'reference', 'history', 'training_manifest', 'training_contract')
        p = {key: checked(r[key], base) for key in required_entries}
        training = read_json(p['training_manifest']); contract = read_json(p['training_contract'])
        require(sha(p['training_manifest']) == r['run_manifest_sha256'], 'Original D5 manifest hash changed')
        require(training['status'] == 'frozen' and contract == training['configuration'] == r['configuration'], 'Frozen D5 contract changed')
        require(contract['purpose'] == 'controlled_pilot' and (contract['Re'], contract['W'], contract['D']) == (1000., 1., 5.), 'Wrong D5 physics')
        require(training['convergence_claim'] is False and training['CFD_used_for_training_or_selection'] is False, 'Invalid D5 convergence/data claim')
        source_matches(ROOT/'flowmllab/deep_cavity_training.py', contract['adapter_sha256'])
        source_matches(ROOT/'qa/run_deep_cavity_ablation.py', contract['runner_sha256'])
        freeze = r['freeze']; score = r['score']; config = r['configuration']
        require(training['prediction_sha256'] == r['prediction']['sha256'] == freeze['prediction_sha256'] == score['prediction_sha256'], 'D5 prediction identity chain changed')
        require(training['frozen_model_sha256'] == freeze['model_sha256'] and training['checkpoint_sha256'] == freeze['checkpoint_sha256'], 'D5 model/recovery identity changed')
        require(training['frozen_at_utc'] == freeze['at_utc'], 'D5 freeze timestamp differs')
        require(training['initial_model_sha256'] == r['initial_model_sha256'] and training['sample_sha256'] == r['sample_sha256'], 'D5 initialization/sample identity changed')
        require(training['final_audit'] == r['physical_residual_audit'], 'D5 raw physics audit changed')
        require(hashlib.sha256((json.dumps(training['final_audit'], indent=2, allow_nan=False)+'\n').encode()).hexdigest() == training['final_audit_sha256'], 'D5 embedded final audit does not match original file hash')
        freezes.append(stamp(freeze['at_utc'])); accesses.append(stamp(score['reference_accessed_at_utc']))
        counts = training['counters']; budget = config['warm_evaluations']+config['qn_evaluations']
        require(counts['warm_evaluations'] == config['warm_evaluations'] and counts['qn_evaluations'] == config['qn_evaluations'], 'D5 phase budget not completed')
        require(counts['function_evaluations'] == counts['gradient_evaluations'] == r['function_evaluations'] == r['gradient_evaluations'] == budget, 'D5 work counters differ')
        source_history = checked({'path': r['history']['source_path'], 'sha256': r['history']['source_sha256']}, base)
        require(r['history']['source_sha256'] == training['history_sha256'] and r['history']['independent_audit'] is False, 'D5 history provenance/label changed')
        raw_rows, norm_rows = rows(source_history), rows(p['history'])
        require(len(raw_rows) == len(norm_rows) == budget, 'D5 curve omits completed evaluations')
        for i, (raw, normal) in enumerate(zip(raw_rows, norm_rows), 1):
            require(int(raw['function_evaluations']) == int(raw['gradient_evaluations']) == int(normal['objective_evaluations']) == i, 'D5 history counter changed')
            same(float(raw['loss']), float(normal['metric_value']), 'D5 normalized training loss changed')
            same(float(raw['elapsed_seconds']), float(normal['elapsed_seconds']), 'D5 training time changed')
        pred, comparison, ref = (arrays(p[key]) for key in ('prediction', 'comparison_prediction', 'reference'))
        fields_ok(pred, 'h', ('u', 'v', 'p', 'psi', 'omega')); fields_ok(comparison, 'h', ('u', 'v', 'p', 'psi', 'omega')); fields_ok(ref, 'h', ('u', 'v', 'p', 'omega'))
        for axis in ('x', 'h'):
            same(pred[axis], ref[axis], 'D5 native coordinates differ', atol=2e-14, rtol=0)
            require(np.array_equal(comparison[axis], ref[axis]), 'D5 comparison grid changed')
        require(r['comparison_prediction']['source_prediction_sha256'] == r['prediction']['sha256'], 'Derived D5 comparison lost raw prediction binding')
        for q in ('u', 'v', 'psi'):
            require(np.array_equal(pred[q], comparison[q]), 'Raw D5 velocity/streamfunction was changed')
        masks = d5_regions(ref['x'], ref['h']); shift = float((pred['p']-ref['p'])[masks['Interior']].mean())
        same(shift, r['comparison_prediction']['pressure_shift_subtracted'], 'D5 pressure gauge changed')
        same(comparison['p'], pred['p']-shift, 'D5 pressure requires one common Interior gauge')
        same(comparison['omega'], curl({**pred, 'x':ref['x'], 'h':ref['h']}), 'D5 derived prediction curl mismatch')
        same(ref['omega'], curl(ref), 'D5 common reference curl mismatch')
        for name, mask in masks.items():
            for key, value in d5_metrics(comparison, ref, mask).items():
                same(value, r['scores']['regions'][name][key], f'D5 score mismatch: {r["run_id"]}/{name}/{key}')
        paired[(r['seed'], r['method'])] = r
        result_ranges[r['run_id']] = {name: r['scores']['regions'][name]['velocity_NRMSE_percent'] for name in masks}
    require(max(freezes) < min(accesses), 'All six D5 arms must freeze before the first reference read')
    require(len(m['pair_checks']) == 3 and set(m['paired_seeds']) == set(D5_SEEDS), 'Missing D5 pair checks')
    for seed in D5_SEEDS:
        a, b = (paired[(seed, method)] for method in D5_METHODS)
        require(a['initial_model_sha256'] == b['initial_model_sha256'] and a['sample_sha256'] == b['sample_sha256'], 'D5 arm initialization/sampling not matched')
        ca, cb = (dict(r['configuration']) for r in (a, b)); ca.pop('method'); cb.pop('method')
        require(ca == cb, 'D5 contracts differ beyond optimizer pipeline')
    for entry in m.get('figures', []):
        checked(entry, base)
    if 'metrics_file' in m:
        checked(m['metrics_file'], base)
    return {'integrity': 'passed', 'manifest_sha256': sha(path), 'verified_arms': 6, 'paired_seeds': list(D5_SEEDS),
            'all_arms_frozen_before_reference_read': True, 'raw_and_derived_predictions_verified_separately': True,
            'pressure_gauge': 'one common Interior shift; local-shape diagnostics remain separate',
            'vorticity': 'identical second-order sampled curl for reference and derived prediction; native AD omega retained',
            'scientific_acceptance': 'Finite controlled pilot only; this validator establishes neither convergence nor optimizer superiority',
            'velocity_NRMSE_percent_by_region': result_ranges,
            'checkpoint_scope': 'Binary recoveries remain external; hashes verified through retained original run manifests'}


def load_qa_module(name):
    path = ROOT / 'qa' / (name + '.py')
    spec = importlib.util.spec_from_file_location('release_' + name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_stratified(path, baseline_path, required=False):
    if not path.exists():
        require(not required, 'Required stratified experiment is still pending')
        return {'integrity': 'pending', 'reason': 'The separate 12-run study has not been fully bridged and compared'}
    experiment = read_json(path)
    require(experiment['schema_version'] == 1 and experiment['campaign_kind'] == 'ghia_stratified_matched_work',
            'Wrong stratified evidence schema')
    expected = {(re, seed, collar) for re in (100, 7500) for seed in SEEDS for collar in (.01, .02)}
    key = lambda record: (record['Re'], record['seed'], record['collar'])
    require(len(experiment['records']) == 12 and {key(r) for r in experiment['records']} == expected,
            'Exactly 12 unique stratified provenance records are required')
    bridged = {key(r): r for r in experiment['records']}
    raw_path = checked(experiment['raw_evaluation_manifest'])
    raw = read_json(raw_path)
    variants = {key(r): r for r in raw['records']}
    require(len(raw['records']) == 12 and set(variants) == expected, 'Raw stratified matrix differs')
    baselines = {key(r): r for r in read_json(baseline_path)['records']}
    checked(experiment['bridge_source'])
    bridge = load_qa_module('bridge_cavity_pinn_stratified')
    controls = bridge.controls_at(checked(experiment['baseline_controls']))
    context = bridge.evaluation_context(raw, raw['records'])
    for name, value in context.items():
        require(experiment.get(name) == value, 'Experiment changed the evaluation-attempt chronology')
    compare = load_qa_module('compare_cavity_pinn_stratified')
    comparison_path = path.parent / 'comparison/comparison_manifest.json'
    require(comparison_path.is_file(), 'All twelve paired baseline comparisons are still pending')
    comparison = read_json(comparison_path)
    require(comparison['schema_version'] == 1 and comparison['campaign_kind'] == 'ghia_uniform_vs_stratified_matched_work',
            'Wrong paired-comparison schema')
    source_matches(ROOT / 'qa/compare_cavity_pinn_stratified.py', comparison['comparison_source_sha256'])
    for name, target in [('baseline_manifest', baseline_path), ('variant_raw_manifest', raw_path), ('experiment_manifest', path)]:
        require(checked(comparison['inputs'][name]) == target.resolve(), 'Comparison input points to another study')
    require(comparison.get('evaluation_provenance', {}) == experiment.get('evaluation_provenance', {}),
            'Comparison dropped failed-evaluation or restore evidence')
    for evidence in experiment.get('evaluation_provenance', {}).values(): checked(evidence)
    for name in ('all_projections_frozen_before_first_reference_read',
                 'all_projections_frozen_before_completed_pass_reference_read',
                 'all_training_endpoints_frozen_before_prior_target_access', 'prior_scored_case_count'):
        if name in experiment:
            require(comparison['evaluation_chronology'].get(name) == experiment[name],
                    'Comparison misstates repaired-pass chronology')
    if 'runtime_events' in experiment:
        checked(experiment['runtime_events'])
        require(comparison.get('runtime_events') == experiment['runtime_events'], 'Failed-allocation provenance was dropped')
    require('failed allocations' in experiment['timing_policy'] and 'not campaign wall time' in experiment['timing_policy'],
            'Timing scope must disclose failed allocations')
    canonical = hashlib.sha256((ROOT/'flowmllab/cavity_pinn_stratified.py').read_text(encoding='utf-8-sig').encode()).hexdigest()
    prepared = {}
    for identity in sorted(expected):
        wrapped = bridged[identity]; original = variants[identity]
        require(wrapped['raw_score_record'] == original and wrapped['config'] == original['config'],
                'Bridge changed a raw frozen score record')
        require(wrapped['reference_labels_used_for_training'] is False, 'Variant fit is not reference-free')
        require(wrapped['baseline_control_provenance'] == controls[identity]['evidence'], 'Retained baseline metadata differs')
        metadata = {name: read_json(checked(evidence)) for name, evidence in wrapped['training_provenance'].items()}
        require(set(metadata) == set(bridge.METADATA), 'Incomplete retained training/closure provenance')
        require(wrapped['baseline_contract'] == metadata['budget_contract.json']
                and wrapped['sampling_coverage'] == metadata['sampling_coverage.json'], 'Embedded provenance changed')
        item = compare.validate_pair(ROOT, baselines[identity], original, wrapped, canonical)
        frozen = item['frozen']; receipt = item['receipt']; cfg = item['config']
        require(wrapped['binary_checkpoint']['sha256'] == receipt['checkpoint_sha256'], 'External restart identity differs')
        require(frozen['source_raw_sha256'] == sha(ROOT/'flowmllab/cavity_pinn_stratified.py')
                and frozen['runner_raw_sha256'] == sha(ROOT/'qa/run_cavity_pinn_stratified.py'), 'Exact staged source bytes changed')
        for name in ('history.json', 'closure_history.json', 'sampling_coverage.json'):
            require(wrapped['training_provenance'][name]['sha256'] == frozen['artifact_sha256'][name],
                    'Detailed training state is not bound at freeze')
        closures = metadata['closure_history.json']
        require(len(closures) == cfg['closure_budget']
                and [r['closure_calls'] for r in closures] == list(range(1, cfg['closure_budget']+1)),
                'Every consumed optimizer closure must be retained')
        require(sum(r['stage'] == 'adam' for r in closures) == cfg['adam_steps'], 'Closure history lost warm-stage work')
        require(all(np.isfinite(r['training_loss']) and r['training_loss'] >= 0 for r in closures),
                'Nonfinite consumed training objective')
        normalized = rows(checked(original['history']))
        raw_steps = metadata['history.json']
        require(len(normalized) == len(raw_steps), 'Normalized history lost recorded optimizer steps')
        for displayed, raw_step in zip(normalized, raw_steps):
            require(int(displayed['objective_evaluations']) == raw_step['closure_calls']
                    and int(displayed['residual_point_evaluations']) == raw_step['residual_point_evaluations']
                    and displayed['stage'] == raw_step['stage'] and displayed['loss_location'] == raw_step['loss_location'],
                    'Normalized training curve changed its work/stage/endpoint meaning')
            same(float(displayed['metric_value'])**2, raw_step['training_loss'], 'Weighted training RMS differs from original MSE')
            same(float(displayed['elapsed_seconds']), raw_step['elapsed_seconds'], 'Normalized optimizer timing differs')
            same(raw_step['training_loss'], closures[raw_step['closure_calls']-1]['training_loss'], 'Step history differs from its last consumed closure')
        endpoint = frozen['accepted_endpoint_audit']
        require(endpoint == original['physics_audit']['weighted_accepted_endpoint']
                and endpoint['included_in_training_budget'] is False
                and endpoint['model_sha256'] == frozen['frozen_model']['model_sha256'],
                'Accepted endpoint audit is mismatched or used for fitting')
        require(receipt['audit_closure_calls'] == endpoint['objective_evaluations'] == 1
                and receipt['audit_residual_point_evaluations'] == endpoint['residual_point_evaluations'] == 2048,
                'Accepted endpoint diagnostic cost is missing')
        same(endpoint['weighted_momentum_rms']**2, endpoint['weighted_momentum_loss'], 'Weighted endpoint RMS mismatch')
        require(receipt['initial_model_sha256'] == frozen['frozen_model']['initial_model_sha256'],
                'Paired initial state identity differs')
        prepared[identity] = item
    # Reuse the independent native-grid/Ghia metric checks, with an explicit
    # variant branch for its variable number of accepted L-BFGS steps.
    numerical = validate_ghia(raw_path, bridged)
    pairs = comparison['records']
    require(len(pairs) == 12 and {key(r) for r in pairs} == expected, 'Paired comparison lost a case')
    for row in pairs:
        identity = key(row); old = baselines[identity]; new = variants[identity]; item = prepared[identity]
        require(row['baseline_prediction'] == old['prediction'] and row['stratified_prediction'] == new['prediction'],
                'Comparison substituted prediction files')
        require(row['closure_cap'] == item['closure_cap']
                and row['matched_residual_point_evaluations'] == item['residual_point_work'], 'Comparison work cap differs')
        require(row['baseline_checkpoint_sha256'] == old['source']['checkpoint_sha256']
                and row['stratified_checkpoint_sha256'] == new['source']['checkpoint_sha256'], 'Comparison recovery differs')
        require(row['training_provenance'] == bridged[identity]['training_provenance']
                and row['sampling_coverage'] == item['coverage'], 'Comparison dropped provenance')
    groups = comparison['collar_comparisons']
    require(len(groups) == 6 and {(r['Re'], r['seed']) for r in groups} == {(re, seed) for re in (100, 7500) for seed in SEEDS},
            'All six descriptive collar groups are required')
    unequal = 0
    for row in groups:
        fine = prepared[(row['Re'], row['seed'], .01)]['closure_cap']
        coarse = prepared[(row['Re'], row['seed'], .02)]['closure_cap']
        require(row['epsilon_0_01_closures'] == fine and row['epsilon_0_02_closures'] == coarse
                and row['equal_work'] is (fine == coarse) and row['convergence_claim'] is False,
                'Unequal-work collar comparison was misrepresented')
        unequal += int(fine != coarse)
    metrics = rows(checked(comparison['outputs']['paired_metrics']))
    expected_metrics = {(key(row), row['metric']): row for identity in sorted(expected)
                        for row in compare.metric_rows(baselines[identity], variants[identity])}
    require(len(metrics) == len(expected_metrics), 'Paired metric table has missing or duplicate rows')
    observed = set()
    for row in metrics:
        identity = (int(row['Re']), int(row['seed']), float(row['collar']))
        token = (identity, row['metric'])
        require(token in expected_metrics and token not in observed, 'Unexpected/repeated paired metric')
        observed.add(token); expected_row = expected_metrics[token]
        for name in ('baseline', 'stratified', 'signed_delta', 'error_magnitude_delta'):
            same(float(row[name]), expected_row[name], 'Paired error/bias arithmetic differs')
        reduction = expected_row['relative_error_reduction']
        if reduction is None: require(row['relative_error_reduction'] == '', 'Undefined reduction must remain blank')
        else: same(float(row['relative_error_reduction']), reduction, 'Relative reduction differs')
        require(row['outcome'] == expected_row['outcome'] and row['comparison_basis'] == expected_row['comparison_basis'],
                'Wins/losses or signed-bias policy differ')
    profiles = rows(checked(comparison['outputs']['paired_signed_profiles']))
    expected_profiles = {}
    for identity in sorted(expected):
        item = prepared[identity]
        for row in compare.profile_rows(item['baseline_paths']['ghia_profiles'], item['variant_paths']['ghia_profiles'],
                                        dict(Re=identity[0], seed=identity[1], collar=identity[2])):
            expected_profiles[(identity, row['component'], row['position'])] = row
    require(len(profiles) == len(expected_profiles) == 12*34, 'Every signed published profile sample must be retained')
    observed = set()
    for row in profiles:
        identity = (int(row['Re']), int(row['seed']), float(row['collar']))
        token = (identity, row['component'], float(row['position']))
        require(token in expected_profiles and token not in observed, 'Repeated/changed signed sample location')
        observed.add(token); expected_row = expected_profiles[token]
        for name in ('published', 'baseline_prediction', 'stratified_prediction', 'baseline_signed_error_U',
                     'stratified_signed_error_U', 'absolute_error_reduction_U'):
            same(float(row[name]), expected_row[name], 'Signed paired profile differs')
        require(row['outcome'] == expected_row['outcome']
                and (row['screened_included'].lower() == 'true') == expected_row['screened_included'],
                'Signed profile outcome/screening differs')
    return dict(integrity='passed', experiment_manifest_sha256=sha(path), comparison_manifest_sha256=sha(comparison_path),
                verified_cases=12, verified_matched_baseline_pairs=12, verified_descriptive_collar_groups=6,
                unequal_work_collar_groups=unequal, verified_signed_published_samples=len(profiles),
                native_integrity=numerical, scientific_acceptance=numerical['scientific_acceptance'],
                timing_scope=experiment['timing_policy'],
                runtime_events=experiment.get('runtime_events'), evaluation_chronology=comparison.get('evaluation_chronology'),
                scope='One finite sampling intervention; paired improvements and losses remain descriptive, not general convergence or optimizer superiority')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ghia-manifest', type=Path, default=ROOT/'results/cavity_pinn_campaign/evaluation_manifest.json')
    parser.add_argument('--d5-manifest', type=Path, default=ROOT/'results/deep_cavity_ablation/evaluation_manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'qa/cavity_pinn_studies_validation.json')
    parser.add_argument('--require-d5', action='store_true')
    parser.add_argument('--stratified-manifest', type=Path, default=ROOT/'results/cavity_pinn_stratified/experiment_manifest.json')
    parser.add_argument('--require-stratified', action='store_true')
    args = parser.parse_args(argv)
    report = {'schema_version': 1, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'validator_sha256': sha(Path(__file__).resolve()),
              'purpose': 'Release evidence integrity, explicitly separate from scientific acceptance',
              'visual_QA': 'Not performed by this numerical/provenance validator'}
    for name, action in [('ghia', lambda: validate_ghia(args.ghia_manifest.resolve())),
                         ('d5', lambda: validate_d5(args.d5_manifest.resolve(), args.require_d5)),
                         ('stratified', lambda: validate_stratified(args.stratified_manifest.resolve(), args.ghia_manifest.resolve(), args.require_stratified))]:
        try:
            report[name] = action()
        except Exception as exc:
            report[name] = {'integrity': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
    failed = any(report[key]['integrity'] == 'failed' for key in ('ghia', 'd5', 'stratified'))
    pending_required = ((args.require_d5 and report['d5']['integrity'] != 'passed')
                        or (args.require_stratified and report['stratified']['integrity'] != 'passed'))
    report['release_evidence_check'] = 'failed' if failed or pending_required else 'passed_for_available_evidence'
    report['all_studies_present'] = all(report[key]['integrity'] == 'passed' for key in ('ghia', 'd5', 'stratified'))
    out = args.output.resolve(); require(out.is_relative_to(ROOT.resolve()), 'Validation receipt must stay inside checkout')
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name+'.tmp'); temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n'); temp.replace(out)
    print(json.dumps({'receipt': str(out), 'release_evidence_check': report['release_evidence_check'],
                      'ghia': report['ghia']['integrity'], 'd5': report['d5']['integrity'], 'stratified': report['stratified']['integrity'],
                      'scientific_acceptance': report['ghia'].get('scientific_acceptance'),
                      'failures': {k: report[k].get('reason') for k in ('ghia','d5','stratified') if report[k]['integrity']=='failed'}}, indent=2))
    return int(failed or pending_required)


if __name__ == '__main__':
    sys.exit(main())
