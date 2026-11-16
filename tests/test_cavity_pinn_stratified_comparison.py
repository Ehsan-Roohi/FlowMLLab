"""Synthetic metadata-only tests; these fixtures are not scientific results."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compare_cavity_pinn_stratified',ROOT/'qa/compare_cavity_pinn_stratified.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def fixture(root):
    def digest(value): return hashlib.sha256(str(value).encode()).hexdigest()
    def write(path,value):
        path=root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        return c.file_entry(root,path)
    def blob(path,value):
        path=root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value,encoding='utf-8')
        return c.file_entry(root,path)
    blob(Path('flowmllab/cavity_pinn_stratified.py'),'# synthetic fixture engine, never executed\n')
    variant_source=c.canonical_sha(root/'flowmllab/cavity_pinn_stratified.py')
    region=dict(sample_count=4,reference_velocity_rms_U=.5,velocity_error_rms_U=.1,velocity_relative_l2=.2,
        u_error_signed_mean_U=-.02,v_error_signed_mean_U=.01,omega_error_rms_U_over_W=.4,
        omega_error_signed_mean_U_over_W=-.03,omega_relative_l2=.3)
    ghia=dict(screened_relative_l2=.2,published_relative_l2=.2,screened_rmse_U=.1,screened_linf_U=.2,n_screened=2,n_published=2)
    scores=dict(regions={k:deepcopy(region) for k in ('interior','lower_quarter','upper_quarter')},
                ghia={k:deepcopy(ghia) for k in ('u','v')})
    physics=dict(outer_uniform=dict(momentum_rms=.5,divergence_rms=1e-15),
                 corner_shells=[dict(r_min=.001,r_max=.01,count=256,momentum_rms=4.)])
    baseline_records=[];variant_records=[];bridges=[]
    for re,seed,eps in sorted(c.BASE_EXPECTED):
        key=f're{re}_s{seed}_eps{eps}'
        if (re,seed,eps) not in c.EXPECTED:
            baseline_records.append(dict(Re=re,seed=seed,collar=eps,outcome='scored'));continue
        cap=3010+(1 if eps==.01 and seed==23 else 0)
        cfg=dict(reynolds=re,seed=seed,width=64,depth=4,collar=eps,collocation=2048,adam_steps=2000,
                 lbfgs_steps=500,adam_lr=.001,chunk_size=2048,lbfgs_history=50,audit_points=2048,audit_seed=983451,export_grid=129)
        base_freeze_sha=digest('original-baseline-freeze'+key)
        reference=blob(Path('reference')/(key+'.npz'),'synthetic bytes, not CFD')
        def record(arm,config,sourcehash):
            folder=Path('scores')/arm/key
            prediction=blob(folder/'prediction.npz','synthetic predicted bytes '+arm+key)
            history=blob(folder/'history.csv',f'objective_evaluations,residual_point_evaluations,elapsed_seconds\n{cap},{cap*2048},10\n')
            shift=.02 if arm=='uniform' else .01
            profiles=blob(folder/'profiles.csv',
                'component,position,published,prediction,signed_error_U,screened_included\n'
                f'u,0.4,-0.2,{-0.2+shift},{shift},True\nu,0.7,0.3,{0.3+shift},{shift},True\n'
                f'v,0.4,-0.1,{-0.1-shift},{-shift},True\nv,0.7,0.2,{0.2-shift},{-shift},True\n')
            model=digest(arm+'model'+key)
            source=dict(config=config,config_sha256=c.config_sha(config),source_canonical_sha256=sourcehash,
                model_sha256=model,checkpoint_sha256=digest(arm+'checkpoint'+key),
                original_freeze_manifest_sha256=base_freeze_sha,original_prediction_sha256=digest(arm+'original-pred'+key),
                original_frozen_at_utc='2026-11-16T02:00:00Z',evaluator_source_sha256=digest('evaluator'),
                ghia_source_sha256=digest('ghia'),coordinate_sha256=digest('coordinates'),
                ghia_coordinate_sha256=digest('ghia-coordinates'),reference_file_sha256=digest('CFD-source'+str(re)))
            return dict(run_id=arm+'_'+key,Re=re,seed=seed,collar=eps,outcome='scored',config=config,source=source,
                prediction=prediction,reference=reference,history=history,ghia_profiles=profiles,
                freeze=dict(prediction_sha256=prediction['sha256'],model_sha256=model,at_utc='2026-11-16T02:01:00Z'),
                score=dict(prediction_sha256=prediction['sha256'],reference_accessed_at_utc='2026-11-16T02:02:00Z',cfd_values_accessed_at_utc='2026-11-16T02:02:01Z'),
                reference_source=dict(path='synthetic-source',sha256=digest('CFD-source'+str(re))),native_subset='same synthetic native subset',
                scores=deepcopy(scores),physics_audit=deepcopy(physics))
        b=record('uniform',cfg,c.BASE_SOURCE_SHA256);baseline_records.append(b)
        control_folder=Path('controls')/key
        control_entries={}
        control_values={'config.json':cfg,
            'current_receipt.json':dict(finished=True,closure_calls=cap,residual_point_evaluations=cap*2048,checkpoint_sha256=b['source']['checkpoint_sha256']),
            'frozen_manifest.json':dict(config=cfg,frozen_model=dict(model_sha256=b['source']['model_sha256']))}
        for name,value in control_values.items(): control_entries[name]=write(control_folder/name,value)
        base_freeze_sha=control_entries['frozen_manifest.json']['sha256']
        b['source']['original_freeze_manifest_sha256']=base_freeze_sha
        control_provenance=dict(archive_sha256=c.ARCHIVE_SHA256,checkpoint_sha256=b['source']['checkpoint_sha256'],
            members={name:dict(archive_member='synthetic-archive/'+key+'/'+name,sha256=entry['sha256']) for name,entry in control_entries.items()})
        control_entries['baseline_provenance.json']=write(control_folder/'baseline_provenance.json',control_provenance)
        vcfg={**cfg,'closure_budget':cap,'study_kind':'controlled_variant','sampling':'fixed_conditional_uniform_area_strata_v1'}
        v=record('stratified',vcfg,variant_source)
        for r in v['scores']['regions'].values():
            r['velocity_error_rms_U']*=.8;r['velocity_relative_l2']*=.8
            r['omega_error_rms_U_over_W']*=1.1;r['omega_relative_l2']*=1.1
        v['physics_audit']['weighted_accepted_endpoint']=dict(included_in_training_budget=False,weighted_momentum_loss=.25)
        contract=dict(baseline_config=cfg,baseline_closure_calls=cap,baseline_residual_point_evaluations=cap*2048,
            baseline_config_sha256=c.config_sha(cfg),baseline_checkpoint_sha256=b['source']['checkpoint_sha256'],
            baseline_frozen_model_sha256=b['source']['model_sha256'],baseline_source_sha256=c.BASE_SOURCE_SHA256,
            archive_sha256=c.ARCHIVE_SHA256,metadata=control_provenance['members'],
            baseline_provenance_sha256=control_entries['baseline_provenance.json']['sha256'])
        import math
        areas=dict(top_corner_annuli=math.pi*(.05**2-eps**2)/2,remaining_wall_band=1-.9**2-math.pi*.05**2/2,interior=.81)
        total=1-math.pi*eps**2/2;objective=digest('weighted-samples'+key)
        coverage=dict(objective_sha256=objective,counts=dict(top_corner_annuli=512,remaining_wall_band=512,interior=1024),
            collar=eps,top_corner_radius=.05,adaptive_sampling=False,total_area=total,areas=areas,weight_sums={k:x/total for k,x in areas.items()})
        folder=Path('provenance')/key
        entries={}
        for name,value in {'config.json':vcfg,'budget_contract.json':contract,'sampling_coverage.json':coverage,
                           'final_physics_audit.json':v['physics_audit.json'] if 'physics_audit.json' in v else v['physics_audit'],
                           'history.json':[dict(closure_calls=cap,residual_point_evaluations=cap*2048)]}.items():
            entries[name]=write(folder/name,value)
        frozen=dict(config=vcfg,baseline_contract_sha256=c.plain_state_hash(contract),source_sha256=variant_source,
            baseline_source_sha256=c.BASE_SOURCE_SHA256,prediction_sha256=v['source']['original_prediction_sha256'],
            frozen_utc=v['source']['original_frozen_at_utc'],checkpoint_sha256=v['source']['checkpoint_sha256'],
            closure_calls=cap,residual_point_evaluations=cap*2048,
            frozen_model=dict(model_sha256=v['source']['model_sha256'],closure_calls=cap,objective_sha256=objective),
            artifact_sha256={k:e['sha256'] for k,e in entries.items()},accepted_endpoint_audit=v['physics_audit']['weighted_accepted_endpoint'])
        entries['frozen_manifest.json']=write(folder/'frozen_manifest.json',frozen)
        receipt=dict(finished=True,adam_completed=2000,lbfgs_completed=500,lbfgs_aborted_steps=1,closure_calls=cap,
            residual_point_evaluations=cap*2048,checkpoint_sha256=v['source']['checkpoint_sha256'],frozen_manifest=frozen,
            budget_contract_sha256=c.plain_state_hash(contract),objective_sha256=objective,initial_model_sha256=digest('initial'+str(seed)))
        entries['current_receipt.json']=write(folder/'current_receipt.json',receipt)
        v['source']['original_freeze_manifest_sha256']=entries['frozen_manifest.json']['sha256']
        variant_records.append(v)
        bridges.append(dict(run_id=v['run_id'],Re=re,seed=seed,collar=eps,config=vcfg,training_provenance=entries,
                            baseline_contract=contract,sampling_coverage=coverage,
                            baseline_control_provenance=control_entries,raw_score_record=deepcopy(v)))
    base_path=root/'baseline.json';var_path=root/'variant.json';bridge_path=root/'experiment.json'
    write(Path('baseline.json'),dict(records=baseline_records))
    raw_entry=write(Path('variant.json'),dict(records=variant_records))
    events = write(Path('runtime_events.json'), dict(synthetic_fixture=True, events=[]))
    write(Path('experiment.json'),dict(campaign_kind='ghia_stratified_matched_work',raw_evaluation_manifest=raw_entry,records=bridges,runtime_events=events))
    return base_path,var_path,bridge_path,root/'comparison'


def test_all12_pair_work_signed_profiles_and_immutable_rerun(tmp_path):
    args=fixture(tmp_path);report=c.execute(*args,root=tmp_path)
    assert len(report['records'])==12 and len(report['collar_comparisons'])==6
    assert report['runtime_events']['sha256'] == c.sha(tmp_path/'runtime_events.json')
    assert all('excludes failed allocations' in row['time_comparison'] for row in report['records'])
    unequal=[r for r in report['collar_comparisons'] if not r['equal_work']]
    assert {(r['Re'],r['seed']) for r in unequal}=={(100,23),(7500,23)}
    assert all(not r['convergence_claim'] for r in report['collar_comparisons'])
    import csv
    with (tmp_path/report['outputs']['paired_metrics']['path']).open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert any(r['outcome']=='improved' and 'velocity_relative_l2' in r['metric'] for r in rows)
    assert any(r['outcome']=='worsened' and 'omega_relative_l2' in r['metric'] for r in rows)
    before=c.sha(args[3]/'comparison_manifest.json')
    assert c.execute(*args,root=tmp_path)==report
    assert c.sha(args[3]/'comparison_manifest.json')==before


def test_last_projection_must_precede_first_target_access():
    row=dict(source=dict(original_frozen_at_utc='2026-11-16T01:00:00Z'),
             freeze=dict(at_utc='2026-11-16T02:00:00Z'),
             score=dict(reference_accessed_at_utc='2026-11-16T01:30:00Z',cfd_values_accessed_at_utc='2026-11-16T01:30:01Z'))
    with pytest.raises(ValueError,match='all 12 queried'): c.freeze_barrier([row])


def test_counterpart_work_mismatch_rejected_even_with_valid_file_hash(tmp_path):
    args=fixture(tmp_path);baseline=c.read_json(args[0])
    row=next(r for r in baseline['records'] if (r['Re'],r['seed'],r['collar'])==(100,7,.01))
    history=tmp_path/row['history']['path']
    history.write_text('objective_evaluations,residual_point_evaluations,elapsed_seconds\n4000,8192000,10\n')
    row['history']['sha256']=c.sha(history);args[0].write_text(json.dumps(baseline))
    with pytest.raises(ValueError,match='closure work'): c.execute(*args,root=tmp_path)
    assert not args[3].exists()


def test_mutated_bridged_budget_metadata_rejected_before_profiles(tmp_path):
    args=fixture(tmp_path);bridge=c.read_json(args[2]);entry=bridge['records'][0]['training_provenance']['budget_contract.json']
    (tmp_path/entry['path']).write_text('{}')
    with pytest.raises(ValueError,match='SHA256'): c.execute(*args,root=tmp_path)


def test_signed_bias_uses_magnitude_not_negative_is_better():
    result=c.error_change(-.1,-.2,signed=True)
    assert result['outcome']=='worsened' and result['signed_delta']==pytest.approx(-.1)
    assert c.error_change(-.1,.05,signed=True)['outcome']=='improved'
    assert c.error_change(0,0)['relative_error_reduction'] is None


def test_changed_output_is_not_silently_rewritten(tmp_path):
    args=fixture(tmp_path);report=c.execute(*args,root=tmp_path)
    path=tmp_path/report['outputs']['paired_signed_profiles']['path'];path.write_text('corrupt')
    with pytest.raises(ValueError,match='SHA256'): c.execute(*args,root=tmp_path)


def test_existing_comparison_revalidates_nested_provenance(tmp_path):
    args=fixture(tmp_path);c.execute(*args,root=tmp_path)
    bridge=c.read_json(args[2])
    path=tmp_path/bridge['records'][0]['baseline_control_provenance']['current_receipt.json']['path']
    path.write_text('{}')
    with pytest.raises(ValueError,match='SHA256'): c.execute(*args,root=tmp_path)


def repaired_attempt_fixture(root,args):
    bridge=c.read_json(args[2])
    evidence={}
    for name in ('completed_pass_receipt','prior_failed_attempt','checkpoint_forensics','evaluation_wrapper_source'):
        path=root/(name+'.json');path.write_text(json.dumps(dict(synthetic_fixture=True,kind=name)))
        evidence[name]=c.file_entry(root,path)
    bridge.update(evaluation_provenance=evidence,
        all_projections_frozen_before_completed_pass_reference_read=True,
        all_projections_frozen_before_first_reference_read=False,
        all_training_endpoints_frozen_before_prior_target_access=True,prior_scored_case_count=4)
    args[2].write_text(json.dumps(bridge))
    return evidence


def test_repaired_pass_discloses_prior_four_case_access_without_metric_changes(tmp_path):
    args=fixture(tmp_path);evidence=repaired_attempt_fixture(tmp_path,args)
    report=c.execute(*args,root=tmp_path)
    assert report['evaluation_provenance']==evidence
    chronology=report['evaluation_chronology']
    assert chronology['all_projections_frozen_before_first_reference_read'] is False
    assert chronology['all_training_endpoints_frozen_before_prior_target_access'] is True
    assert 'scored 4 cases' in chronology['disclosure'] and 'no fitting or model selection' in chronology['disclosure']
    assert 'completed scoring pass' in report['freeze_barrier']['scope']
    assert 'first_target_access_in_completed_pass' in report['freeze_barrier']
    assert 'first_target_access' not in report['freeze_barrier']


def test_prior_forensic_receipt_tamper_is_rejected_on_comparison_reuse(tmp_path):
    args=fixture(tmp_path);evidence=repaired_attempt_fixture(tmp_path,args)
    c.execute(*args,root=tmp_path)
    (tmp_path/evidence['checkpoint_forensics']['path']).write_text('{}')
    with pytest.raises(ValueError,match='SHA256'):c.execute(*args,root=tmp_path)
