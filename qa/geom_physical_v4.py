#!/usr/bin/env python3
"""Warm-start loss ablation for Unity Geom-DeepONet job 64002555.

Three arms start from EACH seed's same saved uniform_constant checkpoint:
continued_scaled (control), physical_global, physical_regional (lambda=.05).
Only the loss differs; all use fresh Adam, lr=5e-5, 500 epochs, batch=1,
identical sampled points and case orders. No test archive or refit is allowed.
Checkpoint selection: lowest validation vortex error subject to global error
<= that seed's frozen V3 baseline + .5 percentage points. Epoch zero remains
eligible, so a failed refinement retains the starting checkpoint.

This is a development/validation experiment, not confirmatory test evidence.
Commands: python -I geom_physical_v4.py submit | status
"""
from __future__ import annotations
import argparse,datetime as dt,gc,hashlib,importlib.util,json,math,os,shlex,shutil
import statistics,subprocess,sys,tempfile,time,traceback
from pathlib import Path

BASE=Path('/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-geomdeeponet')
SOURCE=BASE/'runs/geom-controlled-v3-20260904T222528Z-qae2wap3'
V3_SHA='89ef8cdce612bc5e606aea8433aa802be412101a255f2fdfb6f2b8736a16ba84'
POINTER='LATEST_GEOM_PHYSICAL_V4'
ARMS=('continued_scaled','physical_global','physical_regional')
SEEDS=(690,691,692)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_origin(source):
    if digest(source/'geom_controlled_v3.py') != V3_SHA:
        raise RuntimeError('V3_RUNNER_HASH_MISMATCH')
    g=load(source/'geom_controlled_v3.py','_physical_v4_base')
    g.checked_source(source/'source')
    if digest(source/'data/results/mahdavi_deeponet'/g.LEARNING) != g.LEARNING_SHA:
        raise RuntimeError('LEARNING_ARCHIVE_HASH_MISMATCH')
    report=json.loads((source/'report.json').read_text())
    if report['status']!='development_validation_audit_complete' or report['test_archive_opened']:
        raise RuntimeError('ORIGIN_MUST_BE_COMPLETED_VALIDATION_ONLY_V3')
    for seed in SEEDS:
        if not (source/f'uniform_constant-seed{seed}/best_validation.weights.h5').is_file():
            raise RuntimeError(f'MISSING_STARTING_CHECKPOINT seed={seed}')
    return g,report


def physical_loss(tf,scale,regional_weight):
    """Training-only, per-case physical relative squared-vector errors.

    L = sum(||e_phys||^2)/sum(||u_ref||^2)
        + lambda * sum_reverse(||e_phys||^2)/sum_reverse(||u_ref||^2).
    The reference U<0 mask is used in the loss only; never in model inputs.
    Empty reverse-flow subsets contribute zero; each case has equal weight.
    Normalization of model outputs does not change this physical objective.
    """
    scales=tf.constant(scale,dtype=tf.float32)
    def loss(y,p):
        physical_y=y*scales;error=(p-y)*scales
        energy=tf.reduce_sum(tf.square(physical_y),axis=-1)
        squared=tf.reduce_sum(tf.square(error),axis=-1)
        norm=tf.reduce_sum(energy,axis=1)
        tf.debugging.assert_positive(norm,'ZERO_TRAINING_REFERENCE_ENERGY')
        global_error=tf.reduce_sum(squared,axis=1)/norm
        if regional_weight == 0:
            return tf.reduce_mean(global_error)
        mask=tf.cast(y[...,0]<0,squared.dtype)
        regional=tf.math.divide_no_nan(tf.reduce_sum(mask*squared,axis=1),
                                       tf.reduce_sum(mask*energy,axis=1))
        return tf.reduce_mean(global_error+regional_weight*regional)
    return loss


def loss_tests(tf):
    import numpy as np
    y=np.array([[[-1.,2.],[1.,0.],[2.,0.]]],dtype=np.float32)
    p=y+np.array([[[1.,1.],[2.,2.],[2.,2.]]],dtype=np.float32)
    for scale in ([1.,1.],[128.65195,7.879271]):
        scales=np.array(scale,dtype=np.float32)
        for lam,expected in ((0.,1.8),(.05,1.82)):
            fn=physical_loss(tf,scales,lam)
            prediction=tf.Variable(p/scales)
            with tf.GradientTape() as tape:
                value=fn(tf.constant(y/scales),prediction)
            np.testing.assert_allclose(value.numpy(),expected,rtol=2e-6)
            tf.debugging.assert_all_finite(tape.gradient(value,prediction),'NONFINITE_LOSS_GRADIENT')
        positive=tf.ones((1,4,2));fn=physical_loss(tf,[1.,1.],.05)
        np.testing.assert_allclose(fn(positive,positive*2).numpy(),1.)
    print('PHYSICAL_LOSS_SCALE_INVARIANCE_AND_GRADIENT_PASS',flush=True)


def fit(g,geom,tf,cases,training,domain,scale,source,output,seed,arm,config):
    import numpy as np
    tf.keras.backend.clear_session()
    directory=output/f'{arm}-seed{seed}';directory.mkdir()
    model=geom.build_step_geom_deeponet(width=48,omega_0=10,seed=seed)
    initial_path=source/f'uniform_constant-seed{seed}/best_validation.weights.h5'
    model.load_weights(initial_path)
    initial_hash=hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    loss_fn=(g.make_loss(tf,None) if arm=='continued_scaled' else
             physical_loss(tf,scale.array,.05 if arm=='physical_regional' else 0.))
    optimizer=tf.keras.optimizers.Adam(config['learning_rate']);optimizer.build(model.trainable_variables)
    params,trunks,targets=map(tf.constant,(training.parameters,training.trunk,training.targets))
    @tf.function(jit_compile=False)
    def train_epoch(order):
        total=tf.constant(0.,tf.float32)
        for k in tf.range(len(g.DEV)):
            i=order[k]
            inputs=(tf.expand_dims(tf.gather(params,i),0),tf.expand_dims(tf.gather(trunks,i),0))
            y=tf.expand_dims(tf.gather(targets,i),0)
            with tf.GradientTape() as tape:
                value=loss_fn(y,model(inputs,training=True))
                tf.debugging.assert_all_finite(value,'NONFINITE_TRAINING_LOSS')
            gradients=tape.gradient(value,model.trainable_variables)
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient,'NONFINITE_TRAINING_GRADIENT')
            optimizer.apply_gradients(zip(gradients,model.trainable_variables));total+=value
        return total/len(g.DEV)
    full={};unique_indices={}
    for h in g.DEV+g.VAL:
        branch,trunk=geom.step_geom_deeponet_inputs(h/100,cases[h]['x'],cases[h]['y'],domain=domain)
        full[h]=(tf.constant(branch),tf.constant(trunk[None]))
        unique_indices[h]=np.unique(np.column_stack((cases[h]['x'],cases[h]['y'])),axis=0,return_index=True)[1]
    def evaluate(heights):
        rows,predictions=[],{}
        for h in heights:
            predictions[h]=scale.inverse_transform(model(full[h],training=False).numpy()[0])
            y=np.column_stack((cases[h]['u'],cases[h]['v']))
            indices=unique_indices[h];truth=y[indices,0]<0;guess=predictions[h][indices,0]<0
            intersection=int(np.sum(truth&guess));union=int(np.sum(truth|guess))
            rows.append({'height_percent':h,**g.metric_block(y,predictions[h]),
                         'reverse_flow_iou_unique_cells':intersection/union if union else 1.,
                         'reference_reverse_cells':int(np.sum(truth)),
                         'predicted_reverse_cells':int(np.sum(guess))})
        return rows,predictions
    baseline,_=evaluate(g.VAL)
    base_global=g.mean_metric(baseline,'global');base_vortex=g.mean_metric(baseline,'vortex')
    base_saved=next(r for r in json.loads((source/'report.json').read_text())['results']
                    if r['arm']=='uniform_constant' and r['seed']==seed)
    if abs(base_global-base_saved['best_validation_global_percent'])>0.002 or \
       abs(base_vortex-base_saved['best_validation_vortex_percent'])>0.02:
        raise RuntimeError('STARTING_CHECKPOINT_RESTORE_MISMATCH')
    best_epoch=0;best_vortex=base_vortex;best_global=base_global
    weights=directory/'best_validation.weights.h5';model.save_weights(weights)
    history=[{'epoch':0,'updates':0,'validation':baseline,'eligible_global_guard':True}]
    rng=np.random.default_rng(seed+12345);start=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        loss=float(train_epoch(tf.constant(rng.permutation(len(g.DEV)),dtype=tf.int32)).numpy())
        if not math.isfinite(loss): raise RuntimeError('NONFINITE_TRAINING_LOSS')
        row={'epoch':epoch,'updates':int(optimizer.iterations.numpy()),'training_loss':loss}
        history.append(row)
        if epoch==1 or epoch%config['eval_every']==0 or epoch==config['epochs']:
            validation,_=evaluate(g.VAL);gv=g.mean_metric(validation,'global');vv=g.mean_metric(validation,'vortex')
            eligible=gv<=base_global+config['global_guard_pp']
            row.update(validation=validation,eligible_global_guard=eligible)
            if eligible and vv<best_vortex:
                best_epoch,best_global,best_vortex=epoch,gv,vv;model.save_weights(weights)
            g.save_json(directory/'history.json',history)
            print(f'V4_CHECKPOINT arm={arm} seed={seed} epoch={epoch} global={gv:.5f} '
                  f'vortex={vv:.5f} eligible={eligible} best_epoch={best_epoch}',flush=True)
    if int(optimizer.iterations.numpy())!=len(g.DEV)*config['epochs']:
        raise RuntimeError('UPDATE_BUDGET_MISMATCH')
    terminal_validation=history[-1]['validation']
    model.save_weights(directory/'terminal.weights.h5');model.load_weights(weights)
    val,vp=evaluate(g.VAL);dev,dp=evaluate(g.DEV)
    if not math.isclose(g.mean_metric(val,'vortex'),best_vortex,rel_tol=1e-5,abs_tol=1e-5):
        raise RuntimeError('SELECTED_CHECKPOINT_RESTORE_MISMATCH')
    np.savez_compressed(directory/'best_predictions_learning_only.npz',**{f'H{h}_uv':p for h,p in {**vp,**dp}.items()})
    result={'arm':arm,'seed':seed,'starting_weights_sha256':digest(initial_path),
       'initial_parameter_hash':initial_hash,'baseline_global_percent':base_global,'baseline_vortex_percent':base_vortex,
       'best_epoch':best_epoch,'global_percent':g.mean_metric(val,'global'),'vortex_percent':g.mean_metric(val,'vortex'),
       'global_delta_pp':g.mean_metric(val,'global')-base_global,'vortex_delta_pp':g.mean_metric(val,'vortex')-base_vortex,
       'selected_new_checkpoint':best_epoch>0,'development':dev,'validation':val,
       'terminal_global_percent':g.mean_metric(terminal_validation,'global'),
       'terminal_vortex_percent':g.mean_metric(terminal_validation,'vortex'),
       'terminal_eligible_global_guard':history[-1]['eligible_global_guard'],
       'updates':int(optimizer.iterations.numpy()),'elapsed_seconds':time.perf_counter()-start}
    g.save_json(directory/'result.json',result)
    del model,optimizer,train_epoch;tf.keras.backend.clear_session();gc.collect()
    return result


def summarize(report):
    summary={}
    for arm in ARMS:
        rows=[r for r in report['results'] if r['arm']==arm and r.get('status')!='numerical_divergence']
        if not rows: continue
        summary[arm]={'n_seeds':len(rows),'new_checkpoints':sum(r['selected_new_checkpoint'] for r in rows)}
        for key in ('global_percent','vortex_percent','global_delta_pp','vortex_delta_pp'):
            vals=[r[key] for r in rows];summary[arm][key]=statistics.fmean(vals)
            summary[arm][key+'_sample_std']=statistics.stdev(vals) if len(vals)>1 else None
    for seed in SEEDS:
        hashes={r['initial_parameter_hash'] for r in report['results'] if r['seed']==seed and 'initial_parameter_hash' in r}
        if len(hashes)>1: raise RuntimeError('PAIRED_INITIALIZATION_MISMATCH')
    return summary


def show(report):
    print('STATUS=',report['status'],' TEST_ARCHIVE_OPENED=',report['test_archive_opened'])
    print('ARM                   GLOBAL_%    VORTEX_%   GLOBAL_DELTA_PP  VORTEX_DELTA_PP   N_NEW/N_SEEDS')
    for arm,r in report.get('summary',{}).items():
        print(f'{arm:<21} {r["global_percent"]:10.5f} {r["vortex_percent"]:11.5f} '
              f'{r["global_delta_pp"]:+16.5f} {r["vortex_delta_pp"]:+16.5f} '
              f'{r["new_checkpoints"]}/{r["n_seeds"]}')
    controls={r['seed']:r for r in report['results'] if r['arm']=='continued_scaled' and 'global_percent' in r}
    for r in report['results']:
        if r['arm']=='continued_scaled' or 'global_percent' not in r or r['seed'] not in controls: continue
        c=controls[r['seed']]
        print(f'PAIRED_MINUS_CONTINUED arm={r["arm"]} seed={r["seed"]} '
              f'global_pp={r["global_percent"]-c["global_percent"]:+.5f} '
              f'vortex_pp={r["vortex_percent"]-c["vortex_percent"]:+.5f}')


def run(config_path,require_gpu=True):
    import numpy as np
    import tensorflow as tf
    output=config_path.resolve().parent;config=json.loads(config_path.read_text());source=output/'origin'
    for relative,expected in config['origin_file_sha256'].items():
        if digest(source/relative)!=expected:
            raise RuntimeError('ORIGIN_SNAPSHOT_HASH_MISMATCH: '+relative)
    g,original=load_origin(source);sys.addaudithook(g.reject_test_open)
    if require_gpu: g.assert_gpu(tf)
    else:
        tf.keras.utils.set_random_seed(690);tf.config.experimental.enable_op_determinism()
    loss_tests(tf)
    geom=g.load_module(source/'source/step_geom_deeponet.py','_physical_v4_geom')
    data=g.load_module(source/'source/mahdavi_deeponet.py','_physical_v4_data')
    cases=data.load_step_height_archive(source/'data',split='learning')
    prep=json.loads((source/'preprocessing.json').read_text());domain=geom.StepDomain(**prep['domain'])
    scale=geom.StepVelocityScale(**prep['velocity_scale'])
    training=geom.sample_step_geom_training_batch(cases,g.DEV,domain=domain,velocity_scale=scale,points_per_case=4096,seed=690)
    with np.load(source/'training_point_indices.npz') as expected:
        if not np.array_equal(training.point_indices,expected['indices']):
            raise RuntimeError('TRAINING_SAMPLE_MISMATCH')
    report={'status':'running','test_archive_opened':False,'test_used_for_selection':False,'prior_test_seen':[44,67],
            'config':config,'results':[], 'tensorflow':tf.__version__,'keras':tf.keras.__version__,
            'hardware_mode':'GPU' if require_gpu else 'CPU exploratory pilot',
            'checkpoint_rule':'min validation vortex under frozen V3 global +0.5 pp; epoch zero allowed',
            'limitations':'development selection; not a new test, not an architecture comparison; no SDF change'}
    g.save_json(output/'report.json',report)
    for seed in config['seeds']:
        for arm in ARMS:
            try:
                result=fit(g,geom,tf,cases,training,domain,scale,source,output,seed,arm,config)
            except (ValueError,RuntimeError,tf.errors.InvalidArgumentError) as exc:
                if 'NONFINITE_' not in str(exc): raise
                result={'arm':arm,'seed':seed,'status':'numerical_divergence','error':str(exc)}
                g.save_json(output/f'{arm}-seed{seed}'/'failure.json',result)
            report['results'].append(result);report['summary']=summarize(report)
            g.save_json(output/'report.json',report)
    report['status']=('development_validation_refinement_complete' if len(report['results'])==len(config['seeds'])*len(ARMS)
                      and not any(r.get('status')=='numerical_divergence' for r in report['results']) else 'complete_with_divergences')
    g.save_json(output/'report.json',report);show(report)
    print('PHYSICAL_V4_COMPLETE TEST_ARCHIVE_OPENED=False',flush=True)


def prepare(source,output,epochs=500,seeds=SEEDS,eval_every=25):
    g,_=load_origin(source)
    relative=['geom_controlled_v3.py','report.json','preprocessing.json','training_point_indices.npz',
              'source/step_geom_deeponet.py','source/mahdavi_deeponet.py',
              'data/results/mahdavi_deeponet/'+g.LEARNING]
    relative += [f'uniform_constant-seed{s}/best_validation.weights.h5' for s in SEEDS]
    hashes={}
    for name in relative:
        destination=output/'origin'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,destination);hashes[name]=digest(destination)
    shutil.copy2(Path(__file__).resolve(),output/'geom_physical_v4.py')
    config={'epochs':epochs,'seeds':list(seeds),'eval_every':eval_every,'learning_rate':5e-5,'batch_size':1,
            'global_guard_pp':.5,'regional_weight':.05,'points_per_case':4096,'sample_seed':690,
            'source_run':str(source),'origin_file_sha256':hashes,'runner_sha256':digest(output/'geom_physical_v4.py'),
            'source_job':64002555,'optimizer':'fresh Adam for each arm','updates_per_arm':5*epochs,
            'exploratory_cpu_seed_690_already_examined':True}
    g.save_json(output/'config.json',config)
    return config


def cmd(command):
    return subprocess.run(command,text=True,capture_output=True,check=True).stdout.strip()


def submit(args):
    base=args.base.resolve();source=args.source_run.resolve();env=base/'conda-tf220-clean';python=env/'bin/python'
    load_origin(source)
    active=cmd(['squeue','--me','--noheader','--name=geom-v4','--format=%i %T'])
    if active: raise RuntimeError('V4_ALREADY_ACTIVE: '+active)
    libraries=cmd([str(python),'-I','-c','import glob,site; print(":".join(sorted({p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/*/lib")})))'])
    if not libraries: raise RuntimeError('NVIDIA_LIBRARY_PATH_EMPTY')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(tempfile.mkdtemp(prefix='geom-physical-v4-'+stamp+'-',dir=base/'runs'))
    prepare(source,out,args.epochs)
    q=shlex.quote
    batch='\n'.join(['#!/bin/bash','set -euo pipefail','unset PYTHONPATH PYTHONHOME LD_PRELOAD',
       'export PYTHONNOUSERSITE=1 PYTHONHASHSEED=690 TF_ENABLE_ONEDNN_OPTS=0 TF_DETERMINISTIC_OPS=1',
       'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1',
       'export LD_LIBRARY_PATH='+q(libraries+':'+str(env/'lib'))+':"${LD_LIBRARY_PATH:-}"',
       'export PATH='+q(str(env/'bin'))+':"$PATH"','nvidia-smi',
       'exec '+shlex.join([str(python),'-I','-u',str(out/'geom_physical_v4.py'),'run',str(out/'config.json')]),''])
    (out/'job.sbatch').write_text(batch)
    job=cmd(['sbatch','--parsable','--account=pi_roohie_umass_edu','--partition=gpu','--gpus=1','--constraint=a40',
             '--cpus-per-task=4','--mem=24G','--time=01:00:00','--job-name=geom-v4','--chdir='+str(out),
             '--output='+str(out/'slurm-%j.out'),'--error='+str(out/'slurm-%j.err'),
             '--mail-type=END,FAIL','--mail-user=roohie@umass.edu',str(out/'job.sbatch')]).split(';')[0]
    if not job.isdecimal(): raise RuntimeError('UNEXPECTED_SUBMISSION_RESPONSE: '+job)
    (out/'JOB_ID').write_text(job+'\n');(base/POINTER).write_text(str(out)+'\n')
    print('SUBMITTED_JOB='+job+'\nOUT='+str(out)+'\n9 warm-start models; no test or refit.')


def status(base):
    out=Path((base/POINTER).read_text().strip());job=(out/'JOB_ID').read_text().strip()
    print('JOB='+job+'\nOUT='+str(out))
    print(cmd(['sacct','-j',job,'--format=JobID%18,State%20,Elapsed,ExitCode,MaxRSS,NodeList%18']))
    if (out/'report.json').is_file(): show(json.loads((out/'report.json').read_text()))
    if (out/'failure.json').is_file(): print((out/'failure.json').read_text())
    for ext in ('out','err'):
        p=out/f'slurm-{job}.{ext}'
        if p.exists(): print(str(p)+'\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-20:]))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('submit');s.add_argument('--base',type=Path,default=BASE)
    s.add_argument('--source-run',type=Path,default=SOURCE);s.add_argument('--epochs',type=int,default=500)
    s=sub.add_parser('status');s.add_argument('--base',type=Path,default=BASE)
    s=sub.add_parser('run');s.add_argument('config',type=Path)
    args=p.parse_args()
    try:
        if args.command=='submit':
            if args.epochs<1: p.error('epochs must be positive')
            submit(args)
        elif args.command=='status': status(args.base)
        else: run(args.config)
    except Exception as exc:
        if args.command=='run':
            (args.config.resolve().parent/'failure.json').write_text(json.dumps({'status':'failed','traceback':traceback.format_exc()},indent=2))
        traceback.print_exc()
        if isinstance(exc,subprocess.CalledProcessError):
            print(exc.stdout or '',exc.stderr or '',file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__': main()
