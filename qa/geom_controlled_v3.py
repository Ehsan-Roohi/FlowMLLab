#!/usr/bin/env python3
"""FlowMLLab / Unity: controlled, validation-only step Geom-DeepONet audit.

Commands (use conda-tf220-clean/bin/python -s):
    geom_controlled_v3.py self-test
    geom_controlled_v3.py submit
    geom_controlled_v3.py status

No installs, git writes, test evaluation, or automatic refit. Each submit creates
a new directory and snapshots only the two reviewed modules + learning archive.
Four arms x three paired seeds, equal optimizer-update budgets. This is a
development experiment, NOT a claim of improvement or a new architecture.
"""
from __future__ import annotations

import argparse
import dataclasses
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

BASE = Path('/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-geomdeeponet')
REFERENCE = '2e9199bfa6efafd60c506374029c30aa3b4e009e'
SOURCES = {
    'step_geom_deeponet.py': '4d6efb0d57007c7e3d35a092bd84259ea7725a91097a54c2db5d263696577bd1',
    'mahdavi_deeponet.py': 'c3eea5986096cdfbedad27671c00b6573e6712ce0b00f35525bf3ade172cebc2',
}
LEARNING = 'step_height_learning_7cases.npz'
LEARNING_SHA = '410907d46a040d53cbbd19fd8d44eeb7b41c05150f953fd4d9c6bb479da3479d'
FORBIDDEN_TEST = 'step_height_test_2cases.npz'
DEV, VAL = (16, 21, 25, 50, 75), (33, 58)
POINTER = 'LATEST_GEOM_CONTROLLED_V3'
ARMS = {
    'uniform_constant': {'alpha': None, 'lr': 8e-4, 'cosine': False},
    'uniform_cosine': {'alpha': None, 'lr': 2e-4, 'cosine': True},
    'zonal_constant': {'alpha': 0.5, 'lr': 8e-4, 'cosine': False},
    'zonal_cosine': {'alpha': 0.5, 'lr': 2e-4, 'cosine': True},
}


def save_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path, canonical=False):
    data = Path(path).read_bytes()
    if canonical:
        data = data.rstrip(b'\n') + b'\n'
    return hashlib.sha256(data).hexdigest()


def checked_source(directory):
    results = {}
    for name, expected in SOURCES.items():
        path = Path(directory) / name
        if sha(path, canonical=True) != expected:
            raise RuntimeError(f'SOURCE_MISMATCH: {path}; do not edit or update automatically. '
                               f'Expected reviewed source from {REFERENCE}.')
        results[name] = sha(path)
    return results


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reject_test_open(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(args[0])).name == FORBIDDEN_TEST:
            raise RuntimeError('TEST_ARCHIVE_ACCESS_FORBIDDEN_IN_V3')


def metric_block(reference, prediction):
    """Physical joint-vector L2 and RMSE. Never discard nonfinite predictions."""
    import numpy as np
    y, p = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2 or not len(y):
        raise ValueError('Expected matching, nonempty [N,2] physical velocities')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('NONFINITE_REFERENCE_OR_PREDICTION')

    def region(mask):
        a, b = y[mask], p[mask]
        if not len(a):
            return {'n': 0, 'relative_l2_percent': None, 'vector_rmse': None,
                    'reference_vector_rms': None, 'component_rmse': None}
        denominator = float(np.linalg.norm(a))
        difference = b - a
        return {
            'n': len(a),
            'relative_l2_percent': (100.0 * float(np.linalg.norm(difference)) / denominator
                                    if denominator > 0 else None),
            'vector_rmse': float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
            'reference_vector_rms': float(np.sqrt(np.mean(np.sum(a**2, axis=1)))),
            'component_rmse': np.sqrt(np.mean(difference**2, axis=0)).tolist(),
        }
    return {'global': region(np.ones(len(y), dtype=bool)),
            'vortex': region(y[:, 0] < 0), 'main': region(y[:, 0] >= 0)}


def mean_metric(rows, region):
    values = [r[region]['relative_l2_percent'] for r in rows]
    if not values or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError(f'Undefined {region} relative error; selection would be invalid')
    return statistics.fmean(values)


def make_loss(tf, alpha):
    """Identical U/V mean reduction for uniform and regional losses.

    Old zonal loss sums U/V whereas old unweighted MSE averages them. Here
    zonal is HALF that old zonal objective. With batch_size=1 this removes
    the extra factor 2 without changing regional priorities. Missing regions
    are renormalized. The implementation also averages cases explicitly.
    """
    def loss(y, p):
        error = tf.reduce_mean(tf.square(p - y), axis=-1)
        if alpha is None:
            return tf.reduce_mean(error)
        vortex = tf.cast(y[..., 0] < 0, error.dtype)
        main = 1.0 - vortex
        nv, nm = tf.reduce_sum(vortex, axis=1), tf.reduce_sum(main, axis=1)
        ev = tf.math.divide_no_nan(tf.reduce_sum(vortex * error, axis=1), nv)
        em = tf.math.divide_no_nan(tf.reduce_sum(main * error, axis=1), nm)
        wv, wm = alpha * tf.cast(nv > 0, error.dtype), (1-alpha) * tf.cast(nm > 0, error.dtype)
        return tf.reduce_mean(tf.math.divide_no_nan(wv * ev + wm * em, wv + wm))
    return loss


def assert_gpu(tf):
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        raise RuntimeError('NO_TENSORFLOW_GPU_DETECTED')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(690)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.optimizer.set_jit(False)
    tf.keras.mixed_precision.set_global_policy('float32')
    with tf.device('/GPU:0'):
        result = tf.matmul(tf.ones((16, 16)), tf.ones((16, 16)))
    if 'GPU:0' not in result.device.upper() or float(result[0, 0].numpy()) != 16:
        raise RuntimeError('GPU_MATMUL_FAILED')
    print('TF_GPU_PREFLIGHT_PASS', tf.__version__, result.device, flush=True)


def gpu_smoke(tf, geom, training, out):
    """Checks real model gradients, loss scale, determinism, and H5 restore."""
    import numpy as np
    target = tf.constant([[[-1., 2.], [1., 0.], [2., 0.]]])
    pred = target + tf.constant([[[1., 1.], [2., 2.], [2., 2.]]])
    for alpha, expected in ((None, 3.0), (0.5, 2.5)):
        np.testing.assert_allclose(make_loss(tf, alpha)(target, pred).numpy(), expected)
    positive = tf.constant([[[1., 1.], [2., 2.]]])
    negative = -positive
    for y in (positive, negative):
        np.testing.assert_allclose(make_loss(tf, 0.5)(y, y+1).numpy(), 1.0)
    points = min(64, training.trunk.shape[1])
    inputs = (tf.constant(training.parameters[:1]), tf.constant(training.trunk[:1, :points]))
    target = tf.constant(training.targets[:1, :points])
    fingerprints = []
    for repeat in range(2):
        tf.keras.backend.clear_session()
        model = geom.build_step_geom_deeponet(width=8, seed=690, omega_0=10)
        optimizer = tf.keras.optimizers.Adam(2e-4)
        for _ in range(3):
            with tf.GradientTape() as tape:
                loss = make_loss(tf, 0.5)(target, model(inputs, training=True))
            gradients = tape.gradient(loss, model.trainable_variables)
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, 'SMOKE_NONFINITE_GRADIENT')
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        predictions = model(inputs, training=False).numpy()
        if not np.isfinite(predictions).all():
            raise RuntimeError('SMOKE_NONFINITE_PREDICTIONS')
        weights = model.get_weights()
        fingerprints.append(hashlib.sha256(b''.join(a.tobytes() for a in weights)).hexdigest())
        if repeat == 0:
            path = out / 'smoke.weights.h5'
            model.save_weights(path)
            model.set_weights([np.zeros_like(a) for a in weights])
            model.load_weights(path)
            np.testing.assert_array_equal(predictions, model(inputs, training=False).numpy())
        del model, optimizer
        gc.collect()
    if fingerprints[0] != fingerprints[1]:
        raise RuntimeError('SHORT_REPEAT_DETERMINISM_FAILED')
    save_json(out / 'gpu_smoke.json', {'status': 'passed', 'weight_hashes': fingerprints,
              'scope': 'two identical 3-update small-model repeats on this node; not proof across all runs'})
    print('GPU_MODEL_LOSS_RESTORE_REPEAT_PASS', flush=True)


def run_arm(tf, geom, cases, training, domain, scale, arm, seed, config, out):
    import numpy as np
    directory = out / f'{arm}-seed{seed}'
    directory.mkdir()
    tf.keras.backend.clear_session()
    recipe = ARMS[arm]
    model = geom.build_step_geom_deeponet(width=48, omega_0=10, seed=seed)
    initial_hash = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    steps = config['epochs'] * len(DEV)
    learning_rate = (tf.keras.optimizers.schedules.CosineDecay(
        recipe['lr'], decay_steps=max(steps-1, 1), alpha=0.01)
        if recipe['cosine'] else recipe['lr'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer.build(model.trainable_variables)
    loss_fn = make_loss(tf, recipe['alpha'])
    parameters, trunks, targets = map(tf.constant, (training.parameters, training.trunk, training.targets))

    @tf.function(jit_compile=False)
    def train_epoch(order):
        total_loss = tf.constant(0., dtype=tf.float32)
        total_gradient = tf.constant(0., dtype=tf.float32)
        for k in tf.range(len(DEV)):
            i = order[k]
            branch = tf.expand_dims(tf.gather(parameters, i), 0)
            trunk = tf.expand_dims(tf.gather(trunks, i), 0)
            target = tf.expand_dims(tf.gather(targets, i), 0)
            with tf.GradientTape() as tape:
                loss = loss_fn(target, model((branch, trunk), training=True))
                tf.debugging.assert_all_finite(loss, 'NONFINITE_TRAINING_LOSS')
            gradients = tape.gradient(loss, model.trainable_variables)
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, 'NONFINITE_TRAINING_GRADIENT')
            total_gradient += tf.linalg.global_norm(gradients)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
        return total_loss / len(DEV), total_gradient / len(DEV)

    full_inputs = {}
    for height in DEV + VAL:
        case = cases[height]
        branch, trunk = geom.step_geom_deeponet_inputs(height / 100., case['x'], case['y'], domain=domain)
        full_inputs[height] = (tf.constant(branch), tf.constant(trunk[None]))

    def predict(height, indices=None):
        branch, trunk = full_inputs[height]
        if indices is not None:
            trunk = tf.gather(trunk, indices, axis=1)
        # No chunking: the current architecture pools the entire supplied query set.
        return scale.inverse_transform(model((branch, trunk), training=False).numpy()[0])

    def evaluate(heights):
        rows, predictions = [], {}
        for height in heights:
            predictions[height] = predict(height)
            reference = np.column_stack((cases[height]['u'], cases[height]['v']))
            rows.append({'height_percent': height, **metric_block(reference, predictions[height])})
        return rows, predictions

    history, best, best_epoch = [], math.inf, None
    rng = np.random.default_rng(seed + 12345)
    start = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        loss, gradient = train_epoch(tf.constant(rng.permutation(len(DEV)), dtype=tf.int32))
        row = {'epoch': epoch, 'updates': int(optimizer.iterations.numpy()),
               'training_loss': float(loss.numpy()), 'mean_gradient_norm': float(gradient.numpy())}
        if not all(math.isfinite(row[k]) for k in ('training_loss', 'mean_gradient_norm')):
            raise RuntimeError('NONFINITE_TRAINING_STATISTICS')
        history.append(row)
        if epoch == 1 or epoch % config['eval_every'] == 0 or epoch == config['epochs']:
            val_rows, _ = evaluate(VAL)
            score = mean_metric(val_rows, 'global')
            row['validation'] = val_rows
            if score < best:
                best, best_epoch = score, epoch
                model.save_weights(directory / 'best_validation.weights.h5')
                save_json(directory / 'best_validation.json', {'epoch': epoch, 'validation': val_rows})
            save_json(directory / 'history.json', history)
            print(f'CHECKPOINT arm={arm} seed={seed} epoch={epoch} updates={row["updates"]} '
                  f'global={score:.5f}% vortex={mean_metric(val_rows, "vortex"):.5f}% '
                  f'best_epoch={best_epoch}', flush=True)
    if int(optimizer.iterations.numpy()) != steps:
        raise RuntimeError('OPTIMIZER_UPDATE_BUDGET_MISMATCH')
    model.save_weights(directory / 'terminal.weights.h5')
    model.load_weights(directory / 'best_validation.weights.h5')
    val_rows, val_predictions = evaluate(VAL)
    dev_rows, dev_predictions = evaluate(DEV)
    if not math.isclose(mean_metric(val_rows, 'global'), best, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError('BEST_CHECKPOINT_RESTORE_METRIC_MISMATCH')
    context_rows = []
    predictions = {**dev_predictions, **val_predictions}
    for height in DEV + VAL:
        reference = np.column_stack((cases[height]['u'], cases[height]['v']))
        if height in DEV:
            indices = training.point_indices[DEV.index(height)]
        else:
            indices = np.sort(np.random.default_rng(10000+height).choice(
                len(reference), config['points'], replace=False))
        subset_prediction = predict(height, indices)
        full_subset_prediction = predictions[height][indices]
        # Strict metrics validate all predictions, including this subset inference.
        subset_metrics = metric_block(reference[indices], subset_prediction)
        full_subset_metrics = metric_block(reference[indices], full_subset_prediction)
        shift = float(np.linalg.norm(subset_prediction-full_subset_prediction))
        denominator = float(np.linalg.norm(reference[indices]))
        context_rows.append({'height_percent': height, 'n_subset': len(indices),
            'prediction_context_delta_over_reference_percent': 100*shift/denominator if denominator else None,
            'prediction_context_delta_vector_rms': shift/math.sqrt(len(indices)),
            'subset_context_global_error_percent': subset_metrics['global']['relative_l2_percent'],
            'full_context_global_error_on_same_points_percent': full_subset_metrics['global']['relative_l2_percent']})
    np.savez_compressed(directory / 'best_predictions_learning_only.npz',
                        **{f'H{h}_uv': p for h, p in predictions.items()})
    result = {'arm': arm, 'seed': seed, 'recipe': recipe, 'optimizer_updates': steps,
              'initial_weights_sha256': initial_hash, 'best_epoch': best_epoch,
              'best_validation_global_percent': mean_metric(val_rows, 'global'),
              'best_validation_vortex_percent': mean_metric(val_rows, 'vortex'),
              'development': dev_rows, 'validation': val_rows, 'query_context': context_rows,
              'elapsed_seconds': time.perf_counter()-start,
              'weights': str(directory / 'best_validation.weights.h5')}
    save_json(directory / 'result.json', result)
    del model, optimizer, train_epoch
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def aggregate(results):
    results = [r for r in results if r.get('status') != 'numerical_divergence']
    summary, paired = {}, []
    for arm in ARMS:
        rows = [r for r in results if r['arm'] == arm]
        if not rows:
            continue
        summary[arm] = {'n_seeds': len(rows)}
        for region in ('global', 'vortex'):
            values = [r[f'best_validation_{region}_percent'] for r in rows]
            summary[arm][region + '_mean_percent'] = statistics.fmean(values)
            summary[arm][region + '_sample_std_percent'] = statistics.stdev(values) if len(values) > 1 else None
        context_values = [c['prediction_context_delta_over_reference_percent'] for r in rows
                          for c in r.get('query_context', []) if c['height_percent'] in VAL
                          and c['prediction_context_delta_over_reference_percent'] is not None]
        summary[arm]['max_validation_context_shift_percent'] = max(context_values) if context_values else None
    for suffix in ('constant', 'cosine'):
        uniform = {r['seed']: r for r in results if r['arm'] == 'uniform_'+suffix}
        zonal = {r['seed']: r for r in results if r['arm'] == 'zonal_'+suffix}
        for seed in sorted(uniform.keys() & zonal.keys()):
            a, b = uniform[seed], zonal[seed]
            if a['initial_weights_sha256'] != b['initial_weights_sha256']:
                raise RuntimeError('PAIRED_INITIALIZATION_MISMATCH')
            dg = b['best_validation_global_percent']-a['best_validation_global_percent']
            dv = b['best_validation_vortex_percent']-a['best_validation_vortex_percent']
            paired.append({'recipe': suffix, 'seed': seed, 'global_delta_pp': dg,
                           'vortex_delta_pp': dv, 'zonal_gate_pass': dg <= 2.0 and dv < 0.0})
    return summary, paired


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    out = Path(config_path).resolve().parent
    sys.addaudithook(reject_test_open)
    checked_source(out / 'source')
    archive = out / 'data' / 'results' / 'mahdavi_deeponet' / LEARNING
    if sha(archive) != LEARNING_SHA:
        raise RuntimeError('LEARNING_ARCHIVE_HASH_MISMATCH')
    import numpy as np
    import tensorflow as tf
    assert_gpu(tf)
    geom = load_module(out / 'source' / 'step_geom_deeponet.py', '_geom_v3_model')
    data = load_module(out / 'source' / 'mahdavi_deeponet.py', '_geom_v3_data')
    cases = data.load_step_height_archive(out / 'data', split='learning')
    if set(cases) != set(DEV+VAL):
        raise RuntimeError('LEARNING_SPLIT_MISMATCH')
    domain = geom.infer_step_domain(cases[DEV[0]]['x'], cases[DEV[0]]['y'])
    scale = geom.fit_step_velocity_scale(cases, DEV)
    training = geom.sample_step_geom_training_batch(cases, DEV, domain=domain,
                 velocity_scale=scale, points_per_case=config['points'], seed=690)
    np.savez_compressed(out / 'training_point_indices.npz', heights=training.height_percent,
                        indices=training.point_indices)
    np.savez_compressed(out / 'training_batch.npz', parameters=training.parameters,
                        trunk=training.trunk, targets=training.targets)
    counts = [{'height_percent': h, 'sampled_vortex_points': int(np.sum(training.targets[i, :, 0] < 0)),
               'sampled_points': config['points'], 'full_vortex_points': int(np.sum(cases[h]['u'] < 0)),
               'full_points': len(cases[h]['u'])} for i, h in enumerate(DEV)]
    save_json(out / 'preprocessing.json', {'domain': dataclasses.asdict(domain),
              'velocity_scale': dataclasses.asdict(scale), 'scale_fit_heights': list(DEV),
              'sample_seed': 690, 'model': {'width': 48, 'omega_0': 10, 'activation': 'swish'},
              'sampling_counts': counts})
    save_json(out / 'software.json', {'python': sys.version, 'executable': sys.executable,
              'numpy': np.__version__, 'tensorflow': tf.__version__, 'keras': tf.keras.__version__,
              'tensorflow_build': tf.sysconfig.get_build_info(),
              'deterministic_ops': True, 'tf32': False, 'jit_compile': False,
              'node': os.environ.get('SLURMD_NODENAME'), 'job_id': os.environ.get('SLURM_JOB_ID')})
    gpu_smoke(tf, geom, training, out)
    results = []
    report = {'status': 'running', 'test_archive_opened': False,
              'test_used_for_selection': False, 'prior_test_already_seen': [44, 67],
              'split': {'development': list(DEV), 'validation': list(VAL)},
              'config': config, 'results': results,
              'checkpoint_rule': 'minimum mean full-field physical global relative L2 on H33/H58; earlier wins ties',
              'zonal_gate': 'paired mean-geometry global delta <= 2 percentage points AND vortex delta < 0',
              'limitations': 'validation used repeatedly for model selection; no new test, no significance claim, '
                             'no comparison with standard DeepONet, no arbitrary-shape claim'}
    save_json(out / 'report.json', report)
    start = time.perf_counter()
    for seed in config['seeds']:
        for arm in ARMS:
            print(f'ARM_START {arm} seed={seed}', flush=True)
            try:
                result = run_arm(tf, geom, cases, training, domain, scale, arm, seed, config, out)
            except (ValueError, RuntimeError, tf.errors.InvalidArgumentError) as exc:
                if 'NONFINITE_' not in str(exc):
                    raise
                result = {'arm': arm, 'seed': seed, 'status': 'numerical_divergence',
                          'error': str(exc), 'traceback': traceback.format_exc()}
                save_json(out / f'{arm}-seed{seed}' / 'failure.json', result)
                print(f'ARM_NUMERICAL_DIVERGENCE {arm} seed={seed}', flush=True)
                tf.keras.backend.clear_session()
                gc.collect()
            results.append(result)
            report['summary'], report['paired_deltas'] = aggregate(results)
            save_json(out / 'report.json', report)
    report['numerical_divergences'] = sum(r.get('status') == 'numerical_divergence' for r in results)
    report['status'] = ('development_validation_audit_complete' if not report['numerical_divergences']
                        else 'audit_complete_with_numerical_divergences')
    report['elapsed_seconds'] = time.perf_counter()-start
    save_json(out / 'report.json', report)
    print_summary(report)
    print('CONTROLLED_V3_COMPLETE TEST_ARCHIVE_OPENED=False', flush=True)


def print_summary(report):
    print('STATUS =', report['status'])
    print('TEST_ARCHIVE_OPENED =', report.get('test_archive_opened'))
    print('ARM                  GLOBAL_MEAN_%  VORTEX_MEAN_%  SEEDS')
    for arm, values in report.get('summary', {}).items():
        print(f'{arm:<21} {values["global_mean_percent"]:13.5f} '
              f'{values["vortex_mean_percent"]:14.5f} {values["n_seeds"]:6}')
    print('PAIRED_ZONAL_MINUS_UNIFORM (percentage points; negative is better)')
    for row in report.get('paired_deltas', []):
        print(f'{row["recipe"]:<9} seed={row["seed"]} global={row["global_delta_pp"]:+.5f} '
              f'vortex={row["vortex_delta_pp"]:+.5f} gate={row["zonal_gate_pass"]}')
    for row in report.get('results', []):
        if row.get('status') == 'numerical_divergence':
            print('NUMERICAL_DIVERGENCE:', row['arm'], 'seed='+str(row['seed']))
    print('VALIDATION_QUERY_CONTEXT_SHIFT_MAX_% (prediction change / reference norm; not model error)')
    for arm, values in report.get('summary', {}).items():
        print(arm, values.get('max_validation_context_shift_percent'))


def command_text(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, check=True, **kwargs).stdout.strip()


def submit(args):
    base = args.base.resolve()
    root, env = base / 'FlowMLLab', base / 'conda-tf220-clean'
    python = env / 'bin' / 'python'
    if not python.is_file():
        raise RuntimeError(f'CLEAN_ENV_NOT_FOUND: {python}')
    sources = checked_source(root / 'flowmllab')
    archive = root / 'results' / 'mahdavi_deeponet' / LEARNING
    if sha(archive) != LEARNING_SHA:
        raise RuntimeError('LEARNING_ARCHIVE_HASH_MISMATCH')
    active = command_text(['squeue', '--me', '--noheader', '--name=geom-v3', '--format=%i %T'])
    if active:
        raise RuntimeError('V3_ALREADY_ACTIVE; check it before submitting a duplicate:\n'+active)
    library_dirs = command_text([str(python), '-I', '-c',
        'import glob,site; print(":".join(sorted({p for s in site.getsitepackages() '
        'for p in glob.glob(s+"/nvidia/*/lib")})))'])
    if not library_dirs:
        raise RuntimeError('NVIDIA_LIBRARY_PATH_EMPTY; no environment changes made')
    runs = base / 'runs'
    runs.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(tempfile.mkdtemp(prefix='geom-controlled-v3-'+stamp+'-', dir=runs))
    (out / 'source').mkdir()
    for name in SOURCES:
        shutil.copy2(root / 'flowmllab' / name, out / 'source' / name)
    target = out / 'data' / 'results' / 'mahdavi_deeponet'
    target.mkdir(parents=True)
    shutil.copy2(archive, target / LEARNING)
    runner = out / 'geom_controlled_v3.py'
    shutil.copy2(Path(__file__).resolve(), runner)
    config = {'epochs': args.epochs, 'eval_every': args.eval_every, 'points': 4096,
              'seeds': args.seeds, 'batch_size': 1, 'updates_per_arm': len(DEV)*args.epochs,
              'arms': ARMS, 'source_reference': REFERENCE, 'source_file_sha256': sources,
              'learning_sha256': LEARNING_SHA, 'runner_sha256': sha(runner),
              'original_repo_head': command_text(['git', '-C', str(root), 'rev-parse', 'HEAD']),
              'repo_worktree_status': command_text(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no']),
              'loss_note': 'component MEAN in both losses; new zonal equals half old zonal at batch=1',
              'no_automatic_refit_or_test': True}
    save_json(out / 'config.json', config)
    q = shlex.quote
    script = '\n'.join([
        '#!/bin/bash', 'set -euo pipefail',
        'unset PYTHONPATH PYTHONHOME LD_PRELOAD',
        'export PYTHONNOUSERSITE=1 PYTHONHASHSEED=690',
        'export TF_DETERMINISTIC_OPS=1 TF_ENABLE_ONEDNN_OPTS=0',
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4',
        'export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1',
        'export MPLBACKEND=Agg',
        'export LD_LIBRARY_PATH='+q(library_dirs+':'+str(env / 'lib'))+':"${LD_LIBRARY_PATH:-}"',
        'export PATH='+q(str(env / 'bin'))+':"$PATH"',
        'nvidia-smi',
        q(str(python))+' -I -m pip freeze > '+q(str(out / 'pip-freeze.txt')),
        'exec '+shlex.join([str(python), '-I', '-u', str(runner), 'run', str(out / 'config.json')]),
        '',
    ])
    (out / 'job.sbatch').write_text(script)
    job = command_text(['sbatch', '--parsable', '--account=pi_roohie_umass_edu',
        '--partition=gpu', '--gpus=1', '--constraint=a40', '--cpus-per-task=4',
        '--mem=24G', '--time=02:00:00', '--job-name=geom-v3', '--chdir='+str(out),
        '--output='+str(out / 'slurm-%j.out'), '--error='+str(out / 'slurm-%j.err'),
        '--mail-type=END,FAIL', '--mail-user=roohie@umass.edu', str(out / 'job.sbatch')]).split(';')[0]
    if not job.isdecimal():
        raise RuntimeError('UNEXPECTED_SBATCH_RESPONSE; inspect '+str(out))
    (out / 'JOB_ID').write_text(job+'\n')
    (base / POINTER).write_text(str(out)+'\n')
    print('SUBMITTED_JOB='+job+'\nOUT='+str(out))
    print(str(len(ARMS)*len(args.seeds))+' models, batch=1, '+str(config['updates_per_arm'])+
          ' updates/model. Test is not copied/opened.')
    print('No packages installed; no repository or existing run modified.')


def status(base):
    pointer = base / POINTER
    if not pointer.is_file():
        print('NO_V3_SUBMISSION_RECORDED')
        return
    out = Path(pointer.read_text().strip())
    job = (out / 'JOB_ID').read_text().strip()
    print('JOB='+job+'\nOUT='+str(out))
    for command in (['squeue', '-j', job, '-o', '%.18i %.18j %.16T %.12M %.30R'],
                    ['sacct', '-j', job, '--format=JobID%18,State%20,Elapsed,ExitCode,MaxRSS,NodeList%18']):
        completed = subprocess.run(command, text=True, capture_output=True)
        print(completed.stdout or completed.stderr)
    for name in ('failure.json', 'report.json'):
        path = out / name
        if path.exists():
            obj = json.loads(path.read_text())
            print_summary(obj) if name == 'report.json' else print(json.dumps(obj, indent=2))
    for suffix in ('out', 'err'):
        path = out / f'slurm-{job}.{suffix}'
        if path.exists():
            print('\nTAIL:', path)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-25:]))


def self_test():
    import unittest
    from types import SimpleNamespace
    import numpy as np

    class AuditTests(unittest.TestCase):
        def test_joint_metrics(self):
            y = np.array([[-3., 4.], [3., 4.]])
            result = metric_block(y, y*1.1)
            self.assertAlmostEqual(result['global']['relative_l2_percent'], 10.)
            self.assertAlmostEqual(result['vortex']['relative_l2_percent'], 10.)
            self.assertEqual(result['vortex']['n'], 1)

        def test_nonfinite_is_failure(self):
            for bad in (np.nan, np.inf):
                with self.assertRaises(ValueError):
                    metric_block([[1., 1.]], [[bad, 1.]])

        def test_component_reduction_and_empty_region(self):
            def divide(a, b):
                return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            backend = SimpleNamespace(reduce_mean=np.mean, square=np.square,
                cast=lambda a, dtype: np.asarray(a, dtype=dtype), reduce_sum=np.sum,
                math=SimpleNamespace(divide_no_nan=divide))
            y = np.array([[[-1., 2.], [1., 0.], [2., 0.]]])
            p = y + np.array([[[1., 1.], [2., 2.], [2., 2.]]])
            self.assertEqual(make_loss(backend, None)(y, p), 3.)
            self.assertEqual(make_loss(backend, 0.5)(y, p), 2.5)
            for sign in (-1, 1):
                a = sign * np.ones((1, 5, 2))
                self.assertEqual(make_loss(backend, 0.5)(a, a+1), 1.)
                self.assertEqual(make_loss(backend, None)(a, a+1), 1.)

        def test_undefined_region_not_zero_error(self):
            result = metric_block([[1., 1.]], [[1., 1.]])
            self.assertIsNone(result['vortex']['relative_l2_percent'])
            with self.assertRaises(ValueError):
                mean_metric([result], 'vortex')

        def test_zero_denominator(self):
            result = metric_block([[0., 0.]], [[1., 1.]])
            self.assertIsNone(result['global']['relative_l2_percent'])
            self.assertAlmostEqual(result['global']['vector_rmse'], math.sqrt(2))

        def test_test_guard(self):
            with self.assertRaises(RuntimeError):
                reject_test_open('open', ('/tmp/'+FORBIDDEN_TEST, 'rb', 0))
            reject_test_open('open', ('/tmp/'+LEARNING, 'rb', 0))
            reject_test_open('open', (1, 'w', 0))

        def test_gate_rejects_previous_winner(self):
            rows = [dict(arm='uniform_constant', seed=690, initial_weights_sha256='same',
                         best_validation_global_percent=14.69919119, best_validation_vortex_percent=83.43398544),
                    dict(arm='zonal_constant', seed=690, initial_weights_sha256='same',
                         best_validation_global_percent=12.90778402, best_validation_vortex_percent=86.72961368)]
            _, pairs = aggregate(rows)
            self.assertFalse(pairs[0]['zonal_gate_pass'])
            rows[1]['best_validation_vortex_percent'] = 80.
            self.assertTrue(aggregate(rows)[1][0]['zonal_gate_pass'])

        def test_initialization_pair_guard(self):
            rows = [dict(arm=a, seed=1, initial_weights_sha256=a,
                         best_validation_global_percent=1., best_validation_vortex_percent=1.)
                    for a in ('uniform_constant', 'zonal_constant')]
            with self.assertRaises(RuntimeError):
                aggregate(rows)

        def test_split_and_budget(self):
            self.assertFalse(set(DEV) & set(VAL))
            self.assertFalse(set(DEV+VAL) & {44, 67})
            self.assertEqual(500*math.ceil(len(DEV)/1), 2500)
            self.assertEqual(2000*math.ceil(len(DEV)/5), 2000)

        def test_strict_json(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'report.json'
                save_json(path, {'a': 1})
                self.assertEqual(json.loads(path.read_text()), {'a': 1})
                with self.assertRaises(ValueError):
                    save_json(path, {'a': float('nan')})
                self.assertEqual(json.loads(path.read_text()), {'a': 1})
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('SELF_TEST_PASS (CPU protocol tests; actual TensorFlow/GPU tests run inside submitted job)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    submit_parser = commands.add_parser('submit')
    submit_parser.add_argument('--base', type=Path, default=BASE)
    submit_parser.add_argument('--epochs', type=int, default=1000)
    submit_parser.add_argument('--eval-every', type=int, default=50)
    submit_parser.add_argument('--seeds', type=int, nargs='+', default=[690, 691, 692])
    commands.add_parser('self-test')
    status_parser = commands.add_parser('status')
    status_parser.add_argument('--base', type=Path, default=BASE)
    run_parser = commands.add_parser('run', help=argparse.SUPPRESS)
    run_parser.add_argument('config', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'submit':
            if args.epochs < 1 or args.eval_every < 1 or len(set(args.seeds)) != len(args.seeds):
                parser.error('positive epochs/eval-every and distinct seeds required')
            if any(seed < 0 or seed > 2**31-1 for seed in args.seeds):
                parser.error('seeds must lie between 0 and 2**31-1')
            submit(args)
        elif args.command == 'status':
            status(args.base)
        elif args.command == 'self-test':
            self_test()
        else:
            run(args.config)
    except Exception as exc:
        if args.command == 'run':
            save_json(args.config.resolve().parent / 'failure.json',
                      {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stdout or '', exc.stderr or '', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
