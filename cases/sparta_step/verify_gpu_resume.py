#!/usr/bin/env python3
"""Kill a real solver mid-sampling, resume particles and verify independent averages."""
import argparse
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from unittest import mock

spec=importlib.util.spec_from_file_location('preempt',Path(__file__).with_name('gpu_preempt.py'))
preempt=importlib.util.module_from_spec(spec);spec.loader.exec_module(preempt)
gpu=preempt.gpu


def solver_recovery(out, binary, backend, launcher, serial):
    root=out/backend; path=root/'attempts/0001'
    row=gpu.smoke_row()
    gpu.campaign.generate_case(path,row,smoke=True)
    gpu.checkpoint.checkpoint_deck(path)
    if backend=='kokkos':gpu.kk_deck(path/'in.step')
    cmd=gpu.command(binary,backend,launcher,ranks=2,host_kokkos=True,serial=serial)
    # Insert a test-only pause after a genuine partial averaging window. Kill
    # while the solver still owns its in-memory running average.
    deck=(path/'in.step').read_text().replace('dump final grid', 'shell sleep 10\ndump final grid')
    (path/'in.step').write_text(deck)
    with open(path/'in.step') as inp, open(path/'solver.stdout','w') as log:
        proc=subprocess.Popen(cmd,cwd=path,stdin=inp,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                if (path/'grid.block.450.gz').exists() and (path/'WARM_CHECKPOINT.json').exists():break
                if proc.poll() is not None:raise AssertionError((path/'solver.stdout').read_text()[-5000:])
                time.sleep(.02)
            else:raise AssertionError('No partial sampling/checkpoint')
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        finally:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    assert proc.returncode != 0 and not (root/'COMPLETE.json').exists()
    assert gpu.checkpoint.latest_warm(root)==path/'restart.warm'
    before=gpu.pilot.sha(path/'solver.stdout')
    # A newer interrupted write must not shadow the sealed checkpoint.
    bad=root/'attempts/0002';bad.mkdir()
    for name in ['WARM_CHECKPOINT.json','case.json']:shutil.copy2(path/name,bad/name)
    (bad/'restart.warm').write_bytes(b'truncated')
    (bad/'restart.warm.pending').write_bytes(b'uncommitted')
    assert gpu.checkpoint.latest_warm(root)==path/'restart.warm'
    final,timing,report=gpu.resumable_case(root,row,None,cmd,backend,timeout=120)
    assert timing['checkpoint_resumed'] and final.name=='0003'
    assert [v['steps'] for v in timing['loops']]==[50,350,50]
    meta=gpu.load(final/'case.json')
    assert meta['sampling_steps']==400 and meta['final_step']==450
    assert [b['step'] for b in report['blocks']]==list(range(100,451,50))
    assert gpu.pilot.sha(path/'solver.stdout')==before
    with mock.patch.object(gpu,'execute',side_effect=AssertionError('Completed arm reran')):
        again,_,_=gpu.resumable_case(root,row,None,cmd,backend)
        assert again==final
    print(f'KILL_RESUME_FULL_WINDOW_AND_COMPLETED_REUSE_PASS backend={backend}',flush=True)


def scheduler_guards(out):
    base=out/'scheduler';previous=base/'runs/old';previous.mkdir(parents=True)
    (base/gpu.POINTER).write_text(str(previous))
    (previous/'JOB_ID').write_text('64022083')
    gpu.pilot.write_json(previous/'manifest.json',{'jobs':{'benchmark':'64022083'}})
    pilot=base/'runs'/gpu.campaign.PILOT_RUN/'pilot';pilot.mkdir(parents=True)
    (pilot/'restart.final').write_bytes(b'scheduler fixture only')
    gpu.pilot.write_json(pilot/'case.json',dict(sparta_commit=gpu.pilot.SPARTA_COMMIT,
        final_step=60000,nx=1000,ny=200,h_over_H=.5,ppc_outlet_reference=20))
    calls=[]
    def fake(cmd,**kw):
        calls.append(cmd)
        if cmd[:3]==['scontrol','show','job']:
            job=cmd[-1]
            return f'JobName=step-gpu-bench JobState=PENDING Reason=JobHeldUser UserId=test({os.getuid()}) WorkDir={previous} JobId={job}'
        if cmd[0]=='sbatch':return '700002'
        return ''
    with mock.patch.object(preempt,'call',side_effect=fake), \
         mock.patch.object(gpu.subprocess,'check_output',return_value='700001;unity\n') as submit, \
         mock.patch.object(gpu.shutil,'disk_usage',return_value=type('Usage',(),{'free':40*2**30})()), \
         mock.patch.object(gpu,'status'):
        preempt.migrate(Path(__file__).parent,base,'a'*40,'64022083')
        assert submit.call_count==1
        cmd=submit.call_args.args[0]
        assert all(v in cmd for v in ['--hold','--partition=gpu-preempt','--requeue','--mem=48G','--constraint=a40'])
        assert calls.index(['scancel','--state=PENDING','64022083']) < calls.index(['scontrol','release','700001'])
        assert any('--dependency=afterany:700001' in c for c in calls)
        preempt.migrate(Path(__file__).parent,base,'a'*40,'64022083')
        assert submit.call_count==1
        run=Path((base/gpu.POINTER).read_text().strip())
    with mock.patch.object(preempt,'launch_again') as retry:
        for state in ['FAILED','CANCELLED','OUT_OF_MEMORY','COMPLETED','PREEMPTED','NODE_FAIL','TIMEOUT']:
            with mock.patch.object(preempt,'call',return_value=f'700001|{state}|'):
                prior=retry.call_count
                preempt.guard(run,'700001')
                assert retry.call_count-prior==int(state in ['PREEMPTED','NODE_FAIL','TIMEOUT'])
    print('PENDING_MIGRATION_IDEMPOTENCY_AND_RETRY_POLICY_PASS',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-binary',required=True);p.add_argument('--kokkos-binary')
    p.add_argument('--launcher',default='mpirun');p.add_argument('--serial',action='store_true')
    a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='step-resume-test-') as tmp:
        out=Path(tmp)
        solver_recovery(out,str(Path(a.cpu_binary).resolve()),'cpu',a.launcher,a.serial)
        if a.kokkos_binary:
            solver_recovery(out,str(Path(a.kokkos_binary).resolve()),'kokkos',a.launcher,a.serial)
        scheduler_guards(out)
    print('GPU_CHECKPOINT_RECOVERY_VERIFIED ACTUAL_CUDA_TEST=False',flush=True)


if __name__=='__main__':main()
