#!/usr/bin/env python3
"""Exercise actual Kokkos styles, cross-backend restarts, particle budget and Slurm guards.

The CI/local host backend test is not evidence of a working CUDA build.
"""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from unittest import mock

spec=importlib.util.spec_from_file_location('gpu_benchmark',Path(__file__).with_name('gpu_benchmark.py'))
gpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(gpu)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-binary',required=True);p.add_argument('--kokkos-binary',required=True)
    p.add_argument('--launcher',default='mpirun');p.add_argument('--serial',action='store_true')
    a=p.parse_args()
    cpu=str(Path(a.cpu_binary).resolve());kk=str(Path(a.kokkos_binary).resolve())
    for seed in gpu.campaign.SEEDS:
        assert gpu.check_particle_budget(gpu.benchmark_row(seed))==5100000
    for change in [dict(nx=3500,ny=700),dict(ppc=40)]:
        try:gpu.check_particle_budget(dict(gpu.benchmark_row(20260905),**change))
        except ValueError as exc:assert 'PARTICLE_BUDGET_EXCEEDED' in str(exc)
        else:raise AssertionError('Unrequested particle enlargement was accepted')
    assert not hasattr(gpu,'capacity_deck')
    print('PILOT_PARTICLE_BUDGET_GUARD_PASS',flush=True)
    with tempfile.TemporaryDirectory(prefix='step-kk-test-') as temp:
        out=Path(temp)
        gpu.preflight(out,cpu,kk,a.launcher,host_kokkos=True,serial=a.serial)
        print('HOST_KOKKOS_FRESH_AND_BIDIRECTIONAL_RESTART_PASS',flush=True)
        # Exercise exact production benchmark time segmentation on a cheap mesh.
        row=gpu.benchmark_row(gpu.campaign.SEEDS[0]);row.update(nx=20,ny=100,ppc=8)
        path=out/'timed-small'
        gpu.campaign.generate_case(path,row,out/'preflight/cpu-fresh-h50/restart.final','smoke',smoke=True)
        gpu.kk_deck(path/'in.step')
        result=gpu.execute(path,gpu.command(kk,'kokkos',a.launcher,host_kokkos=True,serial=a.serial),timeout=300)
        gpu.validate_case(path)
        assert [r['steps'] for r in result['loops']]==[700,1050,350]
        assert all(r['seconds']>0 for r in result['loops'])
        probes=gpu.campaign.diagnostic_probes(path/'grid.final.gz',gpu.load(path/'case.json'))
        values={(r[0],r[1]):dict(zip(['x_m','y_m','area_m2','u_m_s','v_m_s','p_Pa'],r[2:])) for r in probes}
        assert all(x==0 for x in gpu.campaign.field_difference(values,values).values())
        print('BENCHMARK_TIME_SEGMENTATION_AND_DIAGNOSTICS_PASS',flush=True)
        assert '-k' not in gpu.command(cpu,'cpu',ranks=16)
        cmd=gpu.command(kk,'kokkos')
        assert cmd[:3]==['mpirun','-np','1']
        assert cmd[cmd.index('-np')+1]=='1'
        assert cmd[cmd.index('-k'):cmd.index('-k')+4]==['-k','on','g','1']
        assert cmd[-4:]==['-pk','kokkos','gpu/aware','off']
        base=out/'submit-base'
        pilotpath=base/'runs'/gpu.campaign.PILOT_RUN/'pilot';pilotpath.mkdir(parents=True)
        (pilotpath/'restart.final').write_bytes(b'test fixture only')
        meta=dict(sparta_commit=gpu.pilot.SPARTA_COMMIT,final_step=60000,nx=1000,ny=200,h_over_H=.5,ppc_outlet_reference=20)
        gpu.pilot.write_json(pilotpath/'case.json',meta)
        with mock.patch.object(gpu.shutil,'disk_usage',return_value=type('Usage',(),{'free':40*2**30})()), \
             mock.patch.object(gpu.subprocess,'check_output',return_value='123456;unity\n') as sbatch:
            with contextlib.redirect_stdout(io.StringIO()):gpu.submit(Path(__file__).parent,base,'a'*40)
            assert sbatch.call_count==1
            cmd=sbatch.call_args.args[0]
            assert all(x in cmd for x in ['--gpus=1','--ntasks=16','--constraint=a40','--export=ALL','--mem=48G'])
            try:gpu.submit(Path(__file__).parent,base,'a'*40)
            except ValueError as exc:assert 'already recorded' in str(exc)
            else:raise AssertionError('Duplicate benchmark was submitted')
            assert sbatch.call_count==1
        assert not (base/gpu.campaign.POINTER).exists()
        print('GPU_SUBMISSION_AND_DUPLICATE_GUARDS_PASS',flush=True)
    print('HOST_KOKKOS_VERIFICATION_COMPLETE ACTUAL_CUDA_TEST=False',flush=True)


if __name__=='__main__':main()
