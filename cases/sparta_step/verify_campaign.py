#!/usr/bin/env python3
"""Real MPI fresh/restart checks and campaign dependency/failure regressions."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest import mock

spec=importlib.util.spec_from_file_location('step_campaign',Path(__file__).with_name('campaign.py'))
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def smoke_row(height=50):
    r=campaign.matrix()[6].copy()
    r.update(id=f'smoke_h{height}',height_percent=height,nx=20,ny=100,ppc=8,
             warmup_steps=100,sampling_steps=400,block_steps=50,sample_every=10)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',required=True);p.add_argument('--launcher',default='mpirun')
    p.add_argument('--ranks',type=int,default=2)
    p.add_argument('--serial',action='store_true',help='Actual serial solver checks; MPI checks must still run in CI/Slurm')
    a=p.parse_args();binary=str(Path(a.binary).resolve())
    if a.ranks<1:raise ValueError('Positive rank count required')
    rows=campaign.matrix()
    assert len(rows)==39 and len([r for r in rows if r['phase']=='validation'])==11
    assert len({r['height_percent'] for r in rows})==15
    assert len({r['dt_s'] for r in rows[:9]})==1
    for r in rows:
        assert abs(r['sampling_steps']*r['dt_s']-120000*campaign.REFERENCE_DT)<1e-18
        assert r['sampling_steps']//r['block_steps']==12
        assert abs(r['sample_every']*r['dt_s']-10*campaign.REFERENCE_DT)<1e-20
    print('MATCHED_PHYSICAL_TIME_AND_GRID_STUDY_PASS',flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OMPI_MCA_rmaps_base_oversubscribe='1')
    if os.geteuid()==0:env.update(OMPI_ALLOW_RUN_AS_ROOT='1',OMPI_ALLOW_RUN_AS_ROOT_CONFIRM='1')
    with tempfile.TemporaryDirectory(prefix='step-campaign-check-') as tmp:
        root=Path(tmp)
        def execute(path,ranks):
            with open(path/'in.step') as inp,open(path/'solver.stdout','w') as out:
                launch=[binary] if a.serial else [a.launcher,'-np',str(ranks),binary]
                r=subprocess.run(launch,cwd=path,stdin=inp,
                                 stdout=out,stderr=subprocess.STDOUT,env=env,timeout=180)
            if r.returncode:
                print((path/'solver.stdout').read_text()[-14000:]);raise AssertionError('MPI smoke failed')
            with contextlib.redirect_stdout(io.StringIO()):report=campaign.summarize_case(path)
            assert report['status']=='campaign_case_execution_complete'
            assert report['stuck_particles']==0 and len(report['blocks'])==8
            assert (path/'restart.warm').is_file() and (path/'restart.final').is_file()
            assert (path/'comparison_probes.csv').is_file()
        for height in [16,50,75]:
            path=root/f'fresh{height}'
            campaign.generate_case(path,smoke_row(height),smoke=True)
            execute(path,a.ranks)
            print(f'CAMPAIGN_{"SERIAL" if a.serial else "MPI"}_FRESH_PASS h={height} ranks={1 if a.serial else a.ranks}',flush=True)
        restart=root/'continued'
        campaign.generate_case(restart,smoke_row(),root/'fresh50'/'restart.warm','smoke',smoke=True)
        # Different rank count explicitly exercises read-time ghost-cell handling.
        execute(restart,2*a.ranks)
        print(f'CAMPAIGN_{"SERIAL" if a.serial else "MPI"}_RESTART_PASS ranks={1 if a.serial else 2*a.ranks}',flush=True)
        bad=smoke_row();bad['height_percent']=75
        try:campaign.generate_case(root/'wrong_restart',bad,root/'fresh50'/'restart.warm','smoke',smoke=True)
        except ValueError as exc:assert 'Restart case mismatch' in str(exc)
        else:raise AssertionError('Mismatched geometry restart accepted')
        try:campaign.generate_case(root/'fresh50',smoke_row(),smoke=True)
        except ValueError:pass
        else:raise AssertionError('Existing run overwritten')
        print('RESTART_PROVENANCE_AND_OVERWRITE_GUARDS_PASS',flush=True)
        ids=[str(n)+'\n' for n in range(901,908)]
        with mock.patch.object(campaign.subprocess,'check_output',side_effect=ids) as sbatch:
            with contextlib.redirect_stdout(io.StringIO()):
                campaign.submit(Path(__file__).parent,root/'submit','a'*40,min_free_gib=0)
            calls=[x.args[0] for x in sbatch.call_args_list]
            assert len(calls)==7
            assert '--dependency=afterok:902:903:904' in calls[4]
            assert '--dependency=afterok:905' in calls[5]
            assert '--dependency=afterok:906' in calls[6]
            assert all('--nodes=1' in c and '--export=ALL' in c for c in calls)
            assert all('--kill-on-invalid-dep=yes' in c for c in calls[1:])
            assert '--array=0,1,2%1' in calls[1]
            assert '--array=6,7,8,9,10%3' in calls[3]
            assert '--array='+','.join(map(str,range(11,39)))+'%4' in calls[5]
        print('CAMPAIGN_DEPENDENCIES_AND_RESOURCE_LIMITS_PASS',flush=True)
        with mock.patch.object(campaign.subprocess,'check_output',side_effect=['1001\n',subprocess.CalledProcessError(1,'sbatch')]):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    campaign.submit(Path(__file__).parent,root/'partial','b'*40,min_free_gib=0)
            except subprocess.CalledProcessError:pass
            else:raise AssertionError('Partial submission was hidden')
            out=Path((root/'partial'/campaign.POINTER).read_text().strip())
            meta=campaign.read_json(out/'manifest.json')
            assert meta['status']=='partial_submission' and meta['jobs']=={'build':'1001'}
        print('CAMPAIGN_PARTIAL_SUBMISSION_RECORDED',flush=True)
        with mock.patch.object(campaign.subprocess,'check_output',return_value='1001\n') as queued:
            try:campaign.resume(out)
            except ValueError as exc:assert 'active/pending' in str(exc)
            else:raise AssertionError('An active campaign was resubmitted')
            assert queued.call_count==1
        with mock.patch.object(campaign.subprocess,'check_output',side_effect=['']+[f'{n}\n' for n in range(1101,1108)]) as resumed:
            with contextlib.redirect_stdout(io.StringIO()):campaign.resume(out)
            calls=[c.args[0] for c in resumed.call_args_list]
            assert calls[0][:2]==['squeue','--me'] and len(calls)==8
            assert '--dependency=afterok:1102:1103:1104' in calls[5]
            assert '--dependency=afterok:1105' in calls[6]
            newmeta=campaign.read_json(out/'manifest.json')
            assert newmeta['job_history'][0]['jobs']=={'build':'1001'}
        print('RESUME_REFUSES_ACTIVE_JOBS_AND_REBUILDS_DEPENDENCIES_PASS',flush=True)
        # Scientific gate rejection must retain its report and not release the sweep.
        def fake_attempt(out,row):
            return root, {'checks':{'positive_net_flow':True},'mass_imbalance_fraction':.3,
                          'bulk_velocity_half_drift_fraction':.01,'regional_reverse_half_drift_fraction':.01,
                          'boundary_pressure_error_fraction':[.01,.01],'maximum_dt_over_sampled_tau':.1,
                          'maximum_cell_over_minimum_sampled_lambda':.28}
        probes={(1,1):{'x_m':30e-6,'y_m':2e-6,'area_m2':1.,'u_m_s':100.,'v_m_s':0.,'p_Pa':30000.}}
        with mock.patch.object(campaign,'completed_attempt',side_effect=fake_attempt),mock.patch.object(campaign,'ensemble_probes',return_value=probes):
            with contextlib.redirect_stdout(io.StringIO()):rc=campaign.validation_gate(out)
            assert rc==3 and not (out/'VALIDATION_PASS').exists()
            assert not campaign.read_json(out/'validation_report.json')['geometry_sweep_released']
        print('FAILED_SCIENTIFIC_GATE_BLOCKS_GEOMETRY_PASS',flush=True)
        def passing_attempt(out,row):
            path,r=fake_attempt(out,row);r['mass_imbalance_fraction']=.01;return path,r
        with mock.patch.object(campaign,'completed_attempt',side_effect=passing_attempt),mock.patch.object(campaign,'ensemble_probes',return_value=probes):
            with contextlib.redirect_stdout(io.StringIO()):rc=campaign.validation_gate(out)
            assert rc==0 and (out/'VALIDATION_PASS').is_file()
            assert not campaign.read_json(out/'validation_report.json')['training_data_approved']
        print('PASSING_SCREEN_DOES_NOT_CLAIM_SCIENTIFIC_RELEASE_PASS',flush=True)
    print('SPARTA_CAMPAIGN_VERIFICATION_COMPLETE',flush=True)


if __name__=='__main__':main()
