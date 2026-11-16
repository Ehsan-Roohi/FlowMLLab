#!/usr/bin/env python3
"""Train or audit the bounded constant-lid Ghia PINN experiment.

No reference file is read here. Independent post-freeze scoring is a separate
consumer of the immutable predictions and their SHA256 manifest.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'flowmllab'))
import numpy as np
import torch
from cavity_pinn_campaign import Config,Trainer,numerical_admission,final_audit,predict,source_hash,tensor_hash

STOP_REQUESTED=False

def stop_requested(_signum,_frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def write_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def acquire_lock(path):
    # OS lock is released automatically on process death; a marker alone is not a lock.
    stream=path.open('a+b')
    if os.name=='nt':
        import msvcrt
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            stream.close(); raise RuntimeError('another process owns this output directory')
    else:
        import fcntl
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            stream.close(); raise RuntimeError('another process owns this output directory')
    return stream


def export_frozen(trainer,output,n):
    if not trainer.finished or trainer.frozen is None: raise RuntimeError('predictions require a frozen endpoint')
    if tensor_hash(trainer.model.state_dict())!=trainer.frozen['model_sha256']: raise RuntimeError('model changed after freeze')
    if n!=trainer.config.export_grid: raise ValueError('export grid differs from frozen configuration')
    manifest_path=output/'frozen_manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing['frozen_model']!=trainer.frozen or existing['config']!=asdict(trainer.config):
            raise ValueError('existing frozen bundle has a different model/configuration')
        if existing['prediction_grid_n']!=n: raise ValueError('existing frozen grid mismatch')
        for key,digest_key in (('prediction_file','prediction_sha256'),('physics_audit_file','physics_audit_sha256')):
            retained=output/existing[key]
            if not retained.is_file() or hashlib.sha256(retained.read_bytes()).hexdigest()!=existing[digest_key]:
                raise ValueError('existing frozen artifact hash failed; preserve it for diagnosis')
        return existing
    # A missing manifest denotes an interrupted export. Rebuild only from the
    # already frozen model, original freeze time and configuration-fixed grid.
    axis=np.linspace(0,1,n); xx,yy=np.meshgrid(axis,axis)
    xy=np.column_stack((xx.ravel(),yy.ravel()))
    singular=(xy[:,1]==1)&((xy[:,0]==0)|(xy[:,0]==1))
    values=predict(trainer.model,xy[~singular],str(trainer.device),trainer.config.chunk_size)
    arrays={'x':axis,'y':axis,'reynolds':np.array(trainer.config.reynolds),
            'singular_corner_mask':singular.reshape(n,n)}
    for key,value in values.items():
        full=np.full(n*n,np.nan); full[~singular]=value; arrays[key]=full.reshape(n,n)
    profile=np.linspace(0,1,513)
    vertical=predict(trainer.model,np.column_stack((np.full_like(profile,.5),profile)),str(trainer.device))
    horizontal=predict(trainer.model,np.column_stack((profile,np.full_like(profile,.5))),str(trainer.device))
    arrays.update(centerline_y=profile,centerline_u=vertical['u'],centerline_x=profile,centerline_v=horizontal['v'])
    destination=output/'frozen_predictions.npz'; temporary=output/'frozen_predictions.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,destination)
    audit=final_audit(trainer.model,trainer.audit_points,str(trainer.device),trainer.config.chunk_size)
    write_json(output/'final_physics_audit.json',audit)
    manifest={'schema':1,'experiment':'constant-lid punctured-collocation PINN',
              'config':asdict(trainer.config),'frozen_model':trainer.frozen,
              'source_sha256':source_hash(),'prediction_file':destination.name,
              'prediction_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
              'frozen_utc':trainer.frozen['frozen_utc'],'prediction_grid_n':n,
              'physics_audit_file':'final_physics_audit.json',
              'physics_audit_sha256':hashlib.sha256((output/'final_physics_audit.json').read_bytes()).hexdigest(),
              'reference_labels_used':False,'reference_evaluation_performed':False,
              'array_layout':'(y,x); exact top corner values are NaN and explicitly masked',
              'pressure_gauge':'p(0.5,0.5)=0',
              'limits':'finite budget; explicit collar; no accepted standard-Ghia accuracy claim'}
    write_json(output/'frozen_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admit-only',action='store_true')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reynolds',type=int,default=100)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--depth',type=int,default=4)
    parser.add_argument('--collar',type=float,default=.02)
    parser.add_argument('--collocation',type=int,default=2048)
    parser.add_argument('--adam-steps',type=int,default=2000)
    parser.add_argument('--lbfgs-steps',type=int,default=500)
    parser.add_argument('--adam-lr',type=float,default=.001)
    parser.add_argument('--chunk-size',type=int,default=256)
    parser.add_argument('--lbfgs-history',type=int,default=50)
    parser.add_argument('--audit-points',type=int,default=2048)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-steps',type=int,default=0,help='maximum complete updates in this process; zero means all remaining')
    parser.add_argument('--max-wall-seconds',type=float,default=0,help='stop after a complete step, preserving the declared schedule')
    parser.add_argument('--checkpoint-seconds',type=float,default=120)
    parser.add_argument('--export-grid',type=int,default=129)
    args=parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    admission=numerical_admission(args.device)
    print(json.dumps({'admission':admission}),flush=True)
    if not admission['passed']: raise RuntimeError('numerical admission failed')
    if args.admit_only: return
    if args.output is None: parser.error('--output is required for training')
    if args.max_steps<0 or args.max_wall_seconds<0 or args.export_grid<3: parser.error('invalid process bound or export grid')
    config=Config(**{key:getattr(args,key) for key in ('reynolds','seed','width','depth','collar','collocation',
                    'adam_steps','lbfgs_steps','adam_lr','chunk_size','lbfgs_history','audit_points','export_grid')})
    args.output.mkdir(parents=True,exist_ok=True)
    lock=acquire_lock(args.output/'run.lock')
    try:
        checkpoint=args.output/'checkpoint.pt'
        if args.resume:
            if not checkpoint.is_file(): raise FileNotFoundError('resume requested without checkpoint')
            trainer=Trainer.load(checkpoint,args.device,config=config)
        else:
            if checkpoint.exists() or (args.output/'frozen_manifest.json').exists():
                raise RuntimeError('existing run: use --resume with the identical configuration')
            trainer=Trainer(config,args.device)
        write_json(args.output/'config.json',asdict(config))
        write_json(args.output/'admission.json',admission)
        for signame in ('SIGTERM','SIGINT','SIGUSR1'):
            if hasattr(signal,signame): signal.signal(getattr(signal,signame),stop_requested)
        started=time.perf_counter(); saved=started; executed=0
        # A valid initial recovery exists before the first expensive step.
        trainer.save(checkpoint)
        while not trainer.finished and not STOP_REQUESTED:
            if args.max_steps and executed>=args.max_steps: break
            if args.max_wall_seconds and time.perf_counter()-started>=args.max_wall_seconds: break
            row=trainer.step(); executed+=1
            if executed==1 or executed%10==0: print(json.dumps(row),flush=True)
            if time.perf_counter()-saved>=args.checkpoint_seconds:
                trainer.save(checkpoint); saved=time.perf_counter()
        digest=trainer.save(checkpoint)
        write_json(args.output/'history.json',trainer.history)
        receipt={'finished':trainer.finished,'adam_completed':trainer.adam_completed,
                 'lbfgs_completed':trainer.lbfgs_completed,'closure_calls':trainer.closure_calls,
                 'residual_point_evaluations':trainer.residual_point_evaluations,
                 'elapsed_training_seconds':trainer.elapsed_seconds,'process_seconds':time.perf_counter()-started,
                 'checkpoint_sha256':digest,'runtime_history':trainer.runtime_history,
                 'stop_requested':STOP_REQUESTED,'source_sha256':source_hash()}
        if trainer.finished:
            receipt['frozen_manifest']=export_frozen(trainer,args.output,args.export_grid)
        write_json(args.output/'current_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    finally:
        lock.close()

if __name__=='__main__': main()

