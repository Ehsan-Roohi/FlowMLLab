"""Reproduce the fixed-domain baseline on the recovered family split (no CFD)."""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
from train_week15_vanilla_deeponet import build_model, case_metrics, geometry_ids, sha256

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / 'results/step_geometry_generalization/source/family_frozen_split_recovered.json'
DATA = ROOT / 'results/step_operator_audit/source/dataset.npz'


def verified_split(masks, split):
    gids = geometry_ids(masks)
    groups = {}
    for name, count in [('train', 103), ('validation', 8), ('heldout', 19)]:
        indices = np.asarray(split[name], dtype=int)
        assert len(indices) == count and len(set(indices)) == count
        assert np.all((indices >= 0) & (indices < len(masks)))
        assert sorted(set(gids[indices])) == split['geometry_ids'][name]
        groups[name] = indices
    assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(len(masks)))
    for a, b in [('train', 'validation'), ('train', 'heldout'), ('validation', 'heldout')]:
        assert not set(gids[groups[a]]) & set(gids[groups[b]])
    assert set(gids[groups['heldout']]) == {12,45,46,47,48,49,50,51}
    return gids, groups


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--steps-per-epoch', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    assert sha256(DATA) == '28d4d4c440cdc4c1ac1d13749ce00b0690d99f29cf20fd56c65fc00b6a8058fd'
    with np.load(DATA, allow_pickle=False) as z:
        raw, xy = z['raw'].astype('float32'), z['queries'][:,:,:2].astype('float32')
        masks, re, shape = z['masks'].astype(bool), z['Re'].astype('float32'), z['shape']
    split = json.loads(SPLIT.read_text())
    gids, groups = verified_split(masks, split)
    if args.verify_only:
        print('PASS: dataset hash, exact recovered split, complete partition, geometry isolation')
        return
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(min(4, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for key in ['train','validation']:
        included = np.zeros(len(raw), dtype=bool)
        included[groups[key]] = True
        rows[key] = np.flatnonzero((included[:,None] & masks).ravel())
    arrays = [np.repeat(re, raw.shape[1])[:,None], xy.reshape(-1,2), raw.reshape(-1,3)]
    scales, normalized = {}, []
    for name, array in zip(['Re','xy','target'], arrays):
        mean, std = array[rows['train']].mean(0), array[rows['train']].std(0)
        std[std == 0] = 1
        scales[name+'_mean'], scales[name+'_std'] = mean, std
        normalized.append(((array-mean)/std).astype('float32'))
    branch, trunk, target = normalized
    np.savez_compressed(args.output/'scaling.npz', **scales)
    model = build_model(tf, args.seed)
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function(reduce_retracing=True)
    def step(b, t, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model((b,t), training=True)-y))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    rng = np.random.default_rng(args.seed)
    vrng = np.random.default_rng(100000+args.seed)
    vi = vrng.choice(rows['validation'], min(65536,len(rows['validation'])), replace=False)
    best, best_epoch, history = float('inf'), None, []
    weights = args.output/f'ordinary_deeponet_seed{args.seed}.weights.h5'
    start = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        losses=[]
        for _ in range(args.steps_per_epoch):
            idx=rng.choice(rows['train'],args.batch_size,replace=False)
            losses.append(float(step(branch[idx],trunk[idx],target[idx])))
        if epoch==1 or epoch%10==0 or epoch==args.epochs:
            pred=model((branch[vi],trunk[vi]),training=False).numpy()
            val=float(np.mean((pred-target[vi])**2))
            assert np.isfinite(val) and np.all(np.isfinite(losses))
            history.append(dict(epoch=epoch,train_mse_scaled=float(np.mean(losses)),validation_mse_scaled=val))
            if val<best:
                best,best_epoch=val,epoch
                model.save_weights(weights)
            print(json.dumps(history[-1]),flush=True)
    model.load_weights(weights)
    metrics=[]
    dest=args.output/'predictions'/'family_holdout'/'ordinary_deeponet'/f'seed_{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    for ci in groups['heldout']:
        idx=np.arange(ci*raw.shape[1],(ci+1)*raw.shape[1])
        pred=model((branch[idx],trunk[idx]),training=False).numpy()*scales['target_std']+scales['target_mean']
        assert np.all(np.isfinite(pred))
        name=f'g{gids[ci]:03d}_Re{int(re[ci])}_medium'
        metrics.append(dict(protocol='diverse_family_v1',case=name,seed=args.seed,checkpoint_epoch=best_epoch,**case_metrics(raw[ci],pred,masks[ci])))
        np.savez_compressed(dest/f'{name}_prediction.npz',prediction=pred,truth=raw[ci],coordinates=xy[ci],mask=masks[ci],shape=shape,row_index=ci,Re=re[ci],evaluation_group='test',pressure_convention='stored p*=Re*p; compare after mean centering')
    with (args.output/'case_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    manifest=dict(protocol='diverse_family_v1',seed=args.seed,epochs=args.epochs,steps_per_epoch=args.steps_per_epoch,batch_size=args.batch_size,
        dataset_sha256=sha256(DATA),split_sha256=sha256(SPLIT),split=split,training_code_sha256=sha256(Path(__file__)),model_code_sha256=sha256(ROOT/'qa/train_week15_vanilla_deeponet.py'),
        checkpoint_sha256=sha256(weights),checkpoint_epoch=best_epoch,history=history,training_seconds=time.perf_counter()-start,
        tensorflow=tf.__version__,numpy=np.__version__,branch_inputs=['Re'],trunk_inputs=['x','y'],
        limitation='Geometry-blind fixed-domain baseline, NOT all DeepONet variants. Same Re and xy imply identical unmasked predictions across geometries.',
        selection='Validation only; test family never used for scaling, gradient updates or checkpoint selection. Retrospective benchmark, not prospectively untouched research test.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('WEEK15_FAMILY_DEEPONET_PASS',flush=True)

if __name__=='__main__':
    main()
