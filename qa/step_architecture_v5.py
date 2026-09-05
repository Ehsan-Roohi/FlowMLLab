#!/usr/bin/env python3
"""Validation-only MLP / DeepONet / Geom-DeepONet comparison for Unity.

This runner snapshots pinned source files and the learning archive. It never
checks out a repository, installs packages, opens the test archive, or refits on
validation cases. See step_architecture_v5.md for the frozen development protocol.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile

BASE = Path('/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-geomdeeponet')
POINTER = 'LATEST_STEP_ARCHITECTURE_V5'
REFERENCE = '2e9199bfa6efafd60c506374029c30aa3b4e009e'
SOURCES = {
    'step_geom_deeponet.py': '4d6efb0d57007c7e3d35a092bd84259ea7725a91097a54c2db5d263696577bd1',
    'mahdavi_deeponet.py': 'c3eea5986096cdfbedad27671c00b6573e6712ce0b00f35525bf3ade172cebc2',
}
LEARNING = 'step_height_learning_7cases.npz'
LEARNING_SHA = '410907d46a040d53cbbd19fd8d44eeb7b41c05150f953fd4d9c6bb479da3479d'
FORBIDDEN_TEST = 'step_height_test_2cases.npz'
DEV, VAL = (16, 21, 25, 50, 75), (33, 58)
MODELS = ('mlp', 'deeponet', 'geom')
SAMPLERS = {'uniform': None, 'zonal': 0.6}
HISTORICAL_VALIDATION = {
    'uniform': (5.445122169, 87.963854380),
    'zonal': (7.320898032, 33.811286542),
}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path, canonical=False):
    raw = Path(path).read_bytes()
    if canonical:
        raw = raw.rstrip(b'\n') + b'\n'
    return hashlib.sha256(raw).hexdigest()


def array_hash(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        value.update(str(array.shape).encode())
        value.update(str(array.dtype).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def forbid_test(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).name == FORBIDDEN_TEST:
            raise RuntimeError('V5_TEST_ARCHIVE_ACCESS_FORBIDDEN')


def metrics(reference, prediction):
    import numpy as np
    y, p = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2 or not len(y):
        raise ValueError('Expected matching nonempty [N,2] velocities')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('NONFINITE_FIELD')
    result = {}
    for region, mask in [('global', np.ones(len(y), dtype=bool)),
                         ('vortex', y[:, 0] < 0), ('main', y[:, 0] >= 0)]:
        if not mask.any() or np.linalg.norm(y[mask]) == 0:
            raise ValueError('Undefined relative error: ' + region)
        delta = p[mask] - y[mask]
        result[region + '_percent'] = float(100 * np.linalg.norm(delta) / np.linalg.norm(y[mask]))
        result[region + '_u_rmse'] = float(np.sqrt(np.mean(delta[:, 0]**2)))
        result[region + '_v_rmse'] = float(np.sqrt(np.mean(delta[:, 1]**2)))
    truth_mask, pred_mask = y[:, 0] < 0, p[:, 0] < 0
    result['negative_u_iou'] = float((truth_mask & pred_mask).sum() / (truth_mask | pred_mask).sum())
    result['negative_u_count_ratio'] = float(pred_mask.sum() / truth_mask.sum())
    result['point_count'] = len(y)
    return result


def average(rows, key):
    return statistics.fmean(row[key] for row in rows)


def sampled_rows(u, size, alpha, seed):
    """Exactly the historical global-pool sampling rule; repeats are retained."""
    import numpy as np
    rng = np.random.default_rng(seed)
    if alpha is None:
        return rng.choice(len(u), min(size, len(u)), replace=False).astype('int32')
    nv = int(round(size * alpha))
    v, m = np.flatnonzero(u < 0), np.flatnonzero(u >= 0)
    if not len(v) or not len(m):
        raise ValueError('Both training regions are required')
    indices = np.concatenate((rng.choice(v, nv, replace=nv > len(v)),
                              rng.choice(m, size-nv, replace=size-nv > len(m))))
    rng.shuffle(indices)
    return indices.astype('int32')


def epoch_batches(indices, case_ids, batch_size, seed, epoch):
    """Identical geometry-pure batches for all three models; no dropped rows."""
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 516]))
    batches = []
    for case in range(len(DEV)):
        rows = indices[case_ids[indices] == case].copy()
        rng.shuffle(rows)
        batches.extend(rows[start:start+batch_size] for start in range(0, len(rows), batch_size))
    rng.shuffle(batches)
    return batches


def select_checkpoint(history, ceiling):
    eligible = [r for r in history if r['global_percent'] <= ceiling]
    return min(eligible, key=lambda r: (r['vortex_percent'], r['global_percent'], r['epoch'])) if eligible else None


def load_data(out):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    source = out / 'source'
    for name, expected in SOURCES.items():
        if digest(source / name, True) != expected:
            raise RuntimeError('SOURCE_HASH_MISMATCH: ' + name)
    geom = module(source / 'step_geom_deeponet.py', 'v5_geometry')
    old = module(source / 'mahdavi_deeponet.py', 'v5_historical')
    archive = out / 'data' / 'results' / 'mahdavi_deeponet' / LEARNING
    if digest(archive) != LEARNING_SHA:
        raise RuntimeError('LEARNING_HASH_MISMATCH')
    cases = old.load_step_height_archive(out / 'data', split='learning')
    if set(cases) != set(DEV + VAL):
        raise RuntimeError('Unexpected learning cases')
    first = cases[DEV[0]]
    bounds = tuple(float(x) for x in (first['x'].min(), first['x'].max(), first['y'].min(), first['y'].max()))
    domain = geom.infer_step_domain(first['x'], first['y'])
    x_scaler = StandardScaler().fit(np.vstack([
        old.step_coordinate_features(h / 100, cases[h]['x'], cases[h]['y'], bounds_m=bounds) for h in DEV]))
    y_scaler = StandardScaler().fit(np.vstack([
        np.column_stack((cases[h]['u'], cases[h]['v'])) for h in DEV]))
    items = {}
    for h in DEV + VAL:
        c = cases[h]
        _, trunk = geom.step_geom_deeponet_inputs(h / 100, c['x'], c['y'], domain=domain)
        features = old.step_coordinate_features(h / 100, c['x'], c['y'], bounds_m=bounds)
        truth = np.column_stack((c['u'], c['v'])).astype(float)
        items[h] = {'branch': np.full((len(truth), 1), h/100, dtype='float32'),
                    'trunk': trunk, 'mlp': x_scaler.transform(features).astype('float32'),
                    'target': y_scaler.transform(truth).astype('float32'), 'truth': truth}
    pooled = {key: np.concatenate([items[h][key] for h in DEV])
              for key in ('branch', 'trunk', 'mlp', 'target', 'truth')}
    pooled['case_id'] = np.concatenate([np.full(len(items[h]['truth']), i, dtype='int32') for i, h in enumerate(DEV)])
    pooled['local_row'] = np.concatenate([np.arange(len(items[h]['truth']), dtype='int32') for h in DEV])
    save_json(out / 'scalers.json', {'fit_heights': DEV, 'target_mean': y_scaler.mean_.tolist(),
              'target_std': y_scaler.scale_.tolist(), 'mlp_mean': x_scaler.mean_.tolist(),
              'mlp_std': x_scaler.scale_.tolist(), 'bounds_m': bounds,
              'domain': vars(domain), 'scaling': 'component StandardScaler, fit before sampling on development only'})
    return old, geom, cases, items, pooled, y_scaler, bounds


def build_model(tf, geom, kind, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    if kind == 'geom':
        return geom.build_step_geom_deeponet(width=48, omega_0=10, seed=seed)
    keras = tf.keras
    if kind == 'mlp':
        x = keras.Input(shape=(None, 8), name='known_geometry_features')
        z = keras.layers.Dense(48, activation='tanh')(x)
        z = keras.layers.Dense(48, activation='tanh')(z)
        return keras.Model(x, keras.layers.Dense(2)(z), name='coordinate_mlp')
    branch = keras.Input(shape=(1,), name='height_ratio')
    query = keras.Input(shape=(None, 2), name='normalized_xy')
    b, t = branch, query
    for width in (128, 128):
        b = keras.layers.Dense(width, activation='tanh')(b)
        t = keras.layers.Dense(width, activation='tanh')(t)
    b = keras.layers.Dense(48)(b)
    t = keras.layers.Dense(96)(t)
    _, _, contraction = geom._keras_components()
    return keras.Model((branch, query), contraction(rank=48, output_dim=2)((b, t)), name='vanilla_deeponet')


def model_inputs(kind, branch, trunk, mlp):
    if kind == 'mlp':
        return mlp[None]
    return (branch[:1], trunk[None, :, :2] if kind == 'deeponet' else trunk[None])


def evaluate(tf, model, kind, items, y_scaler, heights=VAL):
    predictions, rows = {}, []
    for h in heights:
        item = items[h]
        inputs = model_inputs(kind, item['branch'], item['trunk'], item['mlp'])
        prediction = y_scaler.inverse_transform(model(inputs, training=False).numpy()[0].astype(float))
        predictions[h] = prediction
        rows.append({'height_percent': h, **metrics(item['truth'], prediction)})
    return rows, predictions


def context_probe(model, kind, items, y_scaler, reference_predictions):
    import numpy as np
    result = []
    for h in VAL:
        item = items[h]
        indices = np.sort(np.random.default_rng(10_000+h).choice(len(item['truth']), 1024, replace=False))
        inp = model_inputs(kind, item['branch'][indices], item['trunk'][indices], item['mlp'][indices])
        pred = y_scaler.inverse_transform(model(inp, training=False).numpy()[0].astype(float))
        full = reference_predictions[h][indices]
        truth = item['truth'][indices]
        masks = [('global', np.ones(len(indices), dtype=bool)), ('vortex', truth[:, 0] < 0)]
        row = {'height_percent': h, 'query_points': len(indices)}
        for region, mask in masks:
            den = np.linalg.norm(truth[mask])
            row[region + '_context_shift_percent'] = float(100*np.linalg.norm(pred[mask]-full[mask])/den) if den else None
        result.append(row)
    return result


def train_one(tf, geom, kind, sampler, seed, config, out, items, pooled, y_scaler, indices):
    import numpy as np
    directory = out / f'{kind}_{sampler}-seed{seed}'
    directory.mkdir()
    model = build_model(tf, geom, kind, seed)
    initial_weights_hash = array_hash(*model.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=1e-7)
    optimizer.build(model.trainable_variables)
    tensors = {key: tf.constant(pooled[key]) for key in ('branch', 'trunk', 'mlp', 'target')}
    schedule_hash = hashlib.sha256()
    start = time.perf_counter()
    history = []

    @tf.function(input_signature=[tf.RaggedTensorSpec([None, None], tf.int32, ragged_rank=1,
                                                    row_splits_dtype=tf.int32)], jit_compile=False)
    def train_epoch(batches):
        loss_sum, count = tf.constant(0.), tf.constant(0.)
        for i in tf.range(batches.nrows()):
            ids = batches[i]
            branch, trunk, mlp, target = (tf.gather(tensors[key], ids) for key in ('branch', 'trunk', 'mlp', 'target'))
            with tf.GradientTape() as tape:
                prediction = model(model_inputs(kind, branch, trunk, mlp), training=True)[0]
                loss = tf.reduce_mean(tf.square(prediction-target))
                tf.debugging.assert_all_finite(loss, 'NONFINITE_TRAINING_LOSS')
            gradients = tape.gradient(loss, model.trainable_variables)
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, 'NONFINITE_GRADIENT')
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            n = tf.cast(tf.shape(ids)[0], tf.float32)
            loss_sum += n*loss
            count += n
        return loss_sum/count

    for epoch in range(1, config['epochs']+1):
        batches = epoch_batches(indices, pooled['case_id'], config['batch_size'], seed, epoch)
        lengths = np.array([len(b) for b in batches], dtype='int32')
        values = np.concatenate(batches)
        schedule_hash.update(lengths.tobytes())
        schedule_hash.update(values.tobytes())
        loss = float(train_epoch(tf.RaggedTensor.from_row_lengths(values, lengths)).numpy())
        if epoch % config['eval_every'] == 0 or epoch == config['epochs']:
            rows, _ = evaluate(tf, model, kind, items, y_scaler)
            filename = f'epoch_{epoch:04d}.weights.h5'
            model.save_weights(directory / filename)
            row = {'epoch': epoch, 'updates': int(optimizer.iterations.numpy()),
                   'training_loss': loss, 'global_percent': average(rows, 'global_percent'),
                   'vortex_percent': average(rows, 'vortex_percent'), 'validation': rows, 'weights': filename}
            history.append(row)
            save_json(directory / 'history.json', history)
            print(f'V5_CHECKPOINT model={kind} sampler={sampler} seed={seed} epoch={epoch} '
                  f'global={row["global_percent"]:.5f}% vortex={row["vortex_percent"]:.5f}%', flush=True)
    result = {'model': kind, 'sampler': sampler, 'seed': seed, 'directory': directory.name,
              'status': 'trained', 'parameter_count': model.count_params(),
              'updates': int(optimizer.iterations.numpy()), 'sampled_rows_sha256': array_hash(indices),
              'initial_weights_sha256': initial_weights_hash,
              'batch_schedule_sha256': schedule_hash.hexdigest(), 'target_exposures': len(indices)*config['epochs'],
              'elapsed_training_and_validation_seconds': time.perf_counter()-start,
              'terminal': history[-1], 'history_file': directory.name+'/history.json'}
    terminal_rows, terminal_predictions = evaluate(tf, model, kind, items, y_scaler)
    np.savez_compressed(directory / 'terminal_validation.npz', **{f'H{h}': p for h, p in terminal_predictions.items()})
    save_json(directory / 'result.json', result)
    del train_epoch, model, optimizer, tensors
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def historical_anchor(old, cases, bounds, config, out):
    """Original sklearn code, original mixed-geometry batching, validation only."""
    import numpy as np
    rows = []
    for sampler, alpha in SAMPLERS.items():
        start = time.perf_counter()
        fitted = old.fit_step_coordinate_surrogate(cases, list(DEV), alpha, bounds_m=bounds,
                   seed=690, sample_size=60_000, max_iter=90, hidden_layer_sizes=(48, 48))
        case_rows, predictions = [], {}
        for h in VAL:
            p = old.predict_step_coordinate_surrogate(fitted, h, cases[h])
            y = np.column_stack((cases[h]['u'], cases[h]['v']))
            predictions[f'H{h}'] = p
            case_rows.append({'height_percent': h, **metrics(y, p)})
        g, v = average(case_rows, 'global_percent'), average(case_rows, 'vortex_percent')
        expected_g, expected_v = HISTORICAL_VALIDATION[sampler]
        row = {'sampler': sampler, 'seed': 690, 'validation': case_rows, 'global_percent': g,
               'vortex_percent': v, 'recorded_global_percent': expected_g, 'recorded_vortex_percent': expected_v,
               'global_difference_pp': g-expected_g, 'vortex_difference_pp': v-expected_v,
               'within_0_01_pp': max(abs(g-expected_g), abs(v-expected_v)) <= .01,
               'epochs_completed': int(fitted['model'].n_iter_),
               'loss_curve': [float(x) for x in fitted['model'].loss_curve_],
               'elapsed_seconds': time.perf_counter()-start,
               'scope': 'historical reproduction anchor; different framework and batching from controlled arms'}
        weights = {f'coef_{i}': w for i, w in enumerate(fitted['model'].coefs_)}
        weights.update({f'bias_{i}': w for i, w in enumerate(fitted['model'].intercepts_)})
        weights.update(x_mean=fitted['x_scaler'].mean_, x_std=fitted['x_scaler'].scale_,
                       y_mean=fitted['y_scaler'].mean_, y_std=fitted['y_scaler'].scale_)
        np.savez_compressed(out / f'sklearn_{sampler}_anchor.npz', **weights, **predictions)
        rows.append(row)
        save_json(out / 'historical_anchor.json', rows)
        print(f'V5_SKLEARN_ANCHOR sampler={sampler} global={g:.5f}% vortex={v:.5f}% '
              f'reproduced={row["within_0_01_pp"]}', flush=True)
    return rows


def finalize(tf, geom, report, config, out, items, y_scaler):
    import numpy as np
    for seed in config['seeds']:
        same_seed = [r for r in report['results'] if r['seed'] == seed]
        anchor = next(r for r in same_seed if r['model'] == 'mlp' and r['sampler'] == 'uniform')
        anchor_history = json.loads((out / anchor['history_file']).read_text())
        ceiling = min(row['global_percent'] for row in anchor_history) + config['global_guard_pp']
        for sampler in SAMPLERS:
            matched = [r for r in same_seed if r['sampler'] == sampler]
            for key in ('sampled_rows_sha256', 'batch_schedule_sha256', 'updates', 'target_exposures'):
                if len({r[key] for r in matched}) != 1:
                    raise RuntimeError('UNPAIRED_EXPERIMENT: '+key)
        for kind in MODELS:
            paired = [r for r in same_seed if r['model'] == kind]
            if len({r['initial_weights_sha256'] for r in paired}) != 1:
                raise RuntimeError('UNPAIRED_INITIALIZATION: '+kind)
        for result in same_seed:
            directory = out / result['directory']
            history = json.loads((out / result['history_file']).read_text())
            selected = select_checkpoint(history, ceiling)
            result['global_ceiling_percent'] = ceiling
            result['selected'] = selected
            result['status'] = 'selected' if selected else 'no_eligible_checkpoint'
            model = build_model(tf, geom, result['model'], seed)
            evidence = selected or result['terminal']
            model.load_weights(directory / evidence['weights'])
            rows, predictions = evaluate(tf, model, result['model'], items, y_scaler)
            for actual, recorded in zip(rows, evidence['validation'], strict=True):
                for key in ('global_percent', 'vortex_percent'):
                    if abs(actual[key]-recorded[key]) > .002:
                        raise RuntimeError('CHECKPOINT_RESTORE_METRIC_MISMATCH')
            if selected:
                shutil.copy2(directory / selected['weights'], directory / 'selected.weights.h5')
                result['selected_weights_file'] = directory.name + '/selected.weights.h5'
                np.savez_compressed(directory / 'selected_validation.npz', **{f'H{h}': p for h, p in predictions.items()})
            result['context_probe_checkpoint'] = 'selected' if selected else 'terminal'
            result['context_probe'] = context_probe(model, result['model'], items, y_scaler, predictions)
            save_json(directory / 'result.json', result)
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    summaries, pairs = [], []
    for kind in MODELS:
        for sampler in SAMPLERS:
            rows = [r for r in report['results'] if r['model'] == kind and r['sampler'] == sampler]
            complete = all(r['selected'] for r in rows)
            row = {'model': kind, 'sampler': sampler, 'n_seeds': len(rows),
                   'n_eligible': sum(bool(r['selected']) for r in rows), 'parameters': rows[0]['parameter_count']}
            for checkpoint in ('selected', 'terminal'):
                for key in ('global_percent', 'vortex_percent'):
                    values = [r[checkpoint][key] for r in rows] if checkpoint == 'terminal' or complete else []
                    row[checkpoint+'_'+key] = statistics.fmean(values) if values else None
                    row[checkpoint+'_'+key+'_std'] = statistics.stdev(values) if len(values) > 1 else None
            summaries.append(row)
    for seed in config['seeds']:
        for sampler in SAMPLERS:
            reference = next(r for r in report['results'] if r['seed'] == seed and r['sampler'] == sampler and r['model'] == 'mlp')
            for kind in ('deeponet', 'geom'):
                row = next(r for r in report['results'] if r['seed'] == seed and r['sampler'] == sampler and r['model'] == kind)
                for checkpoint in ('selected', 'terminal'):
                    a, b = reference[checkpoint], row[checkpoint]
                    pairs.append({'seed': seed, 'sampler': sampler, 'model': kind, 'reference': 'mlp',
                                  'checkpoint': checkpoint, 'global_delta_pp': b['global_percent']-a['global_percent'] if a and b else None,
                                  'vortex_delta_pp': b['vortex_percent']-a['vortex_percent'] if a and b else None})
            normal = next(r for r in report['results'] if r['seed'] == seed and r['sampler'] == sampler and r['model'] == 'deeponet')
            geo = next(r for r in report['results'] if r['seed'] == seed and r['sampler'] == sampler and r['model'] == 'geom')
            for checkpoint in ('selected', 'terminal'):
                a, b = normal[checkpoint], geo[checkpoint]
                pairs.append({'seed': seed, 'sampler': sampler, 'model': 'geom', 'reference': 'deeponet',
                              'checkpoint': checkpoint, 'global_delta_pp': b['global_percent']-a['global_percent'] if a and b else None,
                              'vortex_delta_pp': b['vortex_percent']-a['vortex_percent'] if a and b else None})
    report['summary'], report['paired_deltas'] = summaries, pairs
    for name, rows in [('summary.csv', summaries), ('paired_deltas.csv', pairs)]:
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(config_path):
    import numpy as np
    import tensorflow as tf
    import sklearn
    sys.addaudithook(forbid_test)
    config_path = Path(config_path).resolve()
    out = config_path.parent
    config = json.loads(config_path.read_text())
    if digest(out / 'step_architecture_v5.py') != config['runner_sha256']:
        raise RuntimeError('RUNNER_HASH_MISMATCH')
    tf.keras.utils.set_random_seed(690)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy('float32')
    devices = tf.config.list_physical_devices('GPU')
    if config['require_gpu'] and not devices:
        raise RuntimeError('NO_TENSORFLOW_GPU_DETECTED')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    with tf.device('/GPU:0' if devices else '/CPU:0'):
        check = tf.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
    if float(check[0, 0].numpy()) != 16 or (config['require_gpu'] and 'GPU:0' not in check.device.upper()):
        raise RuntimeError('GPU_MATMUL_FAILED')
    print('V5_BACKEND_PASS', tf.__version__, check.device, flush=True)
    start = time.perf_counter()
    old, geom, cases, items, pooled, scaler, bounds = load_data(out)
    report = {'status': 'running', 'test_archive_opened': False, 'test_used_for_selection': False,
              'prior_test_already_seen': True, 'split': {'development': DEV, 'validation': VAL},
              'software': {'tensorflow': tf.__version__, 'keras': tf.keras.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
              'hardware': check.device, 'config': config, 'results': [],
              'interpretation': 'Development benchmark on repeatedly used validation cases; no independent test or publication superiority claim.'}
    save_json(out / 'report.json', report)
    if config['historical_anchor']:
        report['historical_anchor'] = historical_anchor(old, cases, bounds, config, out)
    sampling = []
    for seed in config['seeds']:
        for sampler, alpha in SAMPLERS.items():
            indices = sampled_rows(pooled['truth'][:, 0], config['sample_size'], alpha, seed)
            np.savez_compressed(out / f'samples_{sampler}_seed{seed}.npz', pooled_indices=indices,
                               height_percent=np.array(DEV)[pooled['case_id'][indices]], local_rows=pooled['local_row'][indices])
            counts = []
            for i, h in enumerate(DEV):
                ids = indices[pooled['case_id'][indices] == i]
                unique = np.unique(ids)
                counts.append({'height_percent': h, 'draws': len(ids), 'unique_rows': len(unique),
                               'vortex_draws': int((pooled['truth'][ids, 0] < 0).sum()),
                               'unique_vortex_rows': int((pooled['truth'][unique, 0] < 0).sum())})
            sampling.append({'seed': seed, 'sampler': sampler, 'counts': counts, 'sampled_rows_sha256': array_hash(indices)})
            save_json(out / 'sampling.json', sampling)
            for kind in MODELS:
                result = train_one(tf, geom, kind, sampler, seed, config, out, items, pooled, scaler, indices)
                report['results'].append(result)
                save_json(out / 'report.json', report)
    finalize(tf, geom, report, config, out, items, scaler)
    report['status'] = 'development_architecture_comparison_complete' if config['require_gpu'] else 'cpu_execution_smoke_complete'
    report['elapsed_seconds'] = time.perf_counter()-start
    save_json(out / 'report.json', report)
    print_summary(report)
    print('STEP_ARCHITECTURE_V5_COMPLETE TEST_ARCHIVE_OPENED=False', flush=True)


def snapshot(out, root):
    """Use exact reviewed local files when possible; otherwise fetch pinned files."""
    (out / 'source').mkdir()
    data = out / 'data' / 'results' / 'mahdavi_deeponet'
    data.mkdir(parents=True)
    files = [(f'flowmllab/{name}', out/'source'/name, expected, True) for name, expected in SOURCES.items()]
    files.append((f'results/mahdavi_deeponet/{LEARNING}', data/LEARNING, LEARNING_SHA, False))
    for relative, destination, expected, canonical in files:
        candidate = root / relative
        if not candidate.is_file():
            candidate = root / ('source/'+Path(relative).name if canonical else 'data/'+relative)
        if candidate.is_file() and digest(candidate, canonical) == expected:
            shutil.copy2(candidate, destination)
        else:
            url = f'https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/{REFERENCE}/{relative}'
            with urllib.request.urlopen(url, timeout=90) as response:
                destination.write_bytes(response.read())
        if digest(destination, canonical) != expected:
            raise RuntimeError('SNAPSHOT_HASH_MISMATCH: '+relative)
    runner = out / 'step_architecture_v5.py'
    shutil.copy2(Path(__file__).resolve(), runner)
    return runner


def configuration(runner, *, smoke=False):
    return {'epochs': 2 if smoke else 90, 'eval_every': 1 if smoke else 5,
            'sample_size': 1024 if smoke else 60_000, 'batch_size': 1024,
            'seeds': [690] if smoke else [690, 691, 692], 'learning_rate': 8e-4,
            'global_guard_pp': 2.0, 'historical_anchor': not smoke, 'require_gpu': not smoke,
            'runner_sha256': digest(runner), 'source_reference': REFERENCE,
            'source_sha256': SOURCES, 'learning_sha256': LEARNING_SHA,
            'models': MODELS, 'samplers': SAMPLERS,
            'checkpoint_rule': 'For each seed: minimum mean validation vortex error under shared ceiling = minimum MLP-uniform validation global over logged epochs +2 percentage points. No fallback masquerading as selected.',
            'common_training': 'StandardScaler targets fitted to all development rows, sampled global pool, geometry-grouped batches, Adam 8e-4, plain component MSE, float32, no early stop or regularization.',
            'limitations': ['Repeatedly used development validation, not blind evidence.',
                           'MLP has engineered geometry inputs; this is a whole-method comparison, not an isolated SDF ablation.',
                           'Geom pools query points: geometry-grouped sample context differs from full-field inference. Both context shifts and physical errors are reported.',
                           'Same target exposures and updates within a sampler; parameter counts and wall times differ.',
                           'No claim of fewer required DSMC geometries: all arms use five.']}


def command(args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


def submit(args):
    import fcntl
    base = args.base.resolve()
    env = base / 'conda-tf220-clean'
    python = env / 'bin' / 'python'
    if not python.is_file():
        raise RuntimeError('CLEAN_ENV_NOT_FOUND: '+str(python))
    with (base / 'V5_SUBMIT.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = command(['squeue', '--me', '--noheader', '--name=step-arch-v5', '--format=%i %T'])
        if active:
            raise RuntimeError('V5_ALREADY_ACTIVE: '+active)
        # Isolated Python must locate dependencies inside the existing environment.
        libraries = command([str(python), '-I', '-c',
            'import glob,site,importlib.metadata as m; '
            '[m.version(p) for p in ("tensorflow","keras","numpy","scipy","scikit-learn","h5py")]; '
            'print(":".join(sorted({p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/*/lib")})))'])
        if not libraries:
            raise RuntimeError('NVIDIA_LIBRARY_PATH_EMPTY')
        runs = base / 'runs'
        runs.mkdir(exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = Path(tempfile.mkdtemp(prefix='step-architecture-v5-'+stamp+'-', dir=runs))
        runner = snapshot(out, base / 'FlowMLLab')
        config = configuration(runner)
        save_json(out / 'config.json', config)
        q = shlex.quote
        script = '\n'.join([
            '#!/bin/bash', 'set -euo pipefail', 'unset PYTHONPATH PYTHONHOME LD_PRELOAD',
            'export PYTHONNOUSERSITE=1 PYTHONHASHSEED=690 TF_DETERMINISTIC_OPS=1 TF_ENABLE_ONEDNN_OPTS=0',
            'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4',
            'export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 MPLBACKEND=Agg',
            'export LD_LIBRARY_PATH='+q(libraries+':'+str(env/'lib'))+':"${LD_LIBRARY_PATH:-}"',
            'export PATH='+q(str(env/'bin'))+':"$PATH"',
            'nvidia-smi', q(str(python))+' -I -m pip freeze > '+q(str(out/'pip-freeze.txt')),
            'exec '+shlex.join([str(python), '-I', '-u', str(runner), 'run', str(out/'config.json')]), ''])
        batch = out / 'job.sbatch'
        batch.write_text(script)
        raw = command(['sbatch', '--parsable', '--account=pi_roohie_umass_edu', '--partition=gpu',
                       '--gpus=1', '--constraint=a40', '--cpus-per-task=4', '--mem=24G', '--time=02:00:00',
                       '--job-name=step-arch-v5', '--chdir='+str(out), '--output='+str(out/'slurm-%j.out'),
                       '--error='+str(out/'slurm-%j.err'), str(batch)])
        job = raw.split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('UNEXPECTED_SBATCH_RESPONSE: '+raw)
        (out/'JOB_ID').write_text(job+'\n')
        pointer = base / POINTER
        temporary = pointer.with_name(pointer.name+'.tmp')
        temporary.write_text(str(out)+'\n')
        temporary.replace(pointer)
        print('SUBMITTED_JOB='+job+'\nOUT='+str(out))
        print('18 controlled models + 2 historical sklearn anchors; 5 development / 2 validation geometries.')


def print_summary(report):
    print('STATUS=', report['status'], 'TEST_ARCHIVE_OPENED=', report.get('test_archive_opened'))
    print('MODEL       SAMPLER   SELECTED_GLOBAL_%  SELECTED_VORTEX_%  ELIGIBLE  TERMINAL_GLOBAL_% TERMINAL_VORTEX_%')
    for row in report.get('summary', []):
        def fmt(value):
            return 'NA' if value is None else f'{value:.5f}'
        print(f'{row["model"]:<11} {row["sampler"]:<8} {fmt(row["selected_global_percent"]):>17} '
              f'{fmt(row["selected_vortex_percent"]):>18} {row["n_eligible"]}/{row["n_seeds"]} '
              f'{fmt(row["terminal_global_percent"]):>20} {fmt(row["terminal_vortex_percent"]):>18}')
    for row in report.get('historical_anchor', []):
        print('SKLEARN_REPRODUCTION', row['sampler'], 'within_0.01pp=', row['within_0_01_pp'])


def locate(args):
    if getattr(args, 'out', None):
        return args.out.resolve()
    return Path((args.base / POINTER).read_text().strip())


def status(args):
    out = locate(args)
    job_file = out/'JOB_ID'
    job = job_file.read_text().strip() if job_file.exists() else None
    print('JOB='+str(job)+'\nOUT='+str(out))
    if job:
        # squeue --me works for completed jobs too; absence means consult sacct.
        for cmd in (['squeue', '--me', '--noheader', '--name=step-arch-v5', '--format=%i %T %M %R'],
                    ['sacct', '-j', job, '--format=JobID%18,State%20,Elapsed,ExitCode,MaxRSS,NodeList%18']):
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(r.stdout or r.stderr)
    for name in ('failure.json', 'report.json'):
        if (out/name).exists():
            value = json.loads((out/name).read_text())
            print_summary(value) if name == 'report.json' else print(json.dumps(value, indent=2))
    for path in sorted(out.glob('slurm-*.*')):
        print(str(path))
        print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))


def bundle(args):
    out = locate(args)
    target = out.parent / (out.name+'_review.zip')
    checksums = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            # Intermediate checkpoints remain in the original run; include terminal
            # and selected weights, full histories, inputs, sampled rows and fields.
            if path.name.startswith('epoch_') and path.name.endswith('.weights.h5'):
                result_path = path.parent/'result.json'
                if not result_path.exists():
                    continue
                terminal = json.loads(result_path.read_text())['terminal']['weights']
                if path.name != terminal:
                    continue
            relative = path.relative_to(out).as_posix()
            archive.write(path, arcname=relative)
            checksums.append(digest(path)+'  '+relative)
        archive.writestr('SHA256SUMS', '\n'.join(checksums)+'\n')
    print('REVIEW_ZIP='+str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('submit')
    for name in ('status', 'bundle'):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path)
    p = sub.add_parser('run')
    p.add_argument('config', type=Path)
    p = sub.add_parser('smoke')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'run':
        try:
            run(args.config)
        except Exception as exc:
            save_json(args.config.resolve().parent/'failure.json', {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
            raise
    elif args.action == 'submit':
        submit(args)
    elif args.action == 'status':
        status(args)
    elif args.action == 'bundle':
        bundle(args)
    elif args.action == 'smoke':
        args.out.mkdir(parents=True, exist_ok=False)
        runner = snapshot(args.out.resolve(), args.source_root.resolve())
        save_json(args.out/'config.json', configuration(runner, smoke=True))
        run(args.out/'config.json')


if __name__ == '__main__':
    main()
