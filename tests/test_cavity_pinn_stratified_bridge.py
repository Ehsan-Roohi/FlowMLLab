"""Small synthetic provenance tests; these are not scientific result fixtures."""
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stratified_evidence_bridge', ROOT/'qa/bridge_cavity_pinn_stratified.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


@pytest.fixture
def study(tmp_path, monkeypatch):
    checkout = tmp_path/'checkout'; output = checkout/'results/cavity_pinn_stratified'
    for name in ('flowmllab/cavity_pinn_stratified.py', 'flowmllab/cavity_pinn_campaign.py',
                 'qa/run_cavity_pinn_stratified.py', 'qa/bridge_cavity_pinn_stratified.py',
                 'qa/evaluate_cavity_pinn_stratified.py'):
        target=checkout/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    shutil.copytree(ROOT/'results/cavity_pinn_stratified/baseline_controls', output/'baseline_controls')
    monkeypatch.setattr(b, 'ROOT', checkout)
    monkeypatch.setattr(b, '__file__', str(checkout/'qa/bridge_cavity_pinn_stratified.py'))
    controls = b.controls_at(output/'baseline_controls/controls_manifest.json')
    engine=checkout/'flowmllab/cavity_pinn_stratified.py'; runner=checkout/'qa/run_cavity_pinn_stratified.py'
    original=checkout/'flowmllab/cavity_pinn_campaign.py'
    runs=[]; records=[]
    for key in sorted(b.EXPECTED):
        re, seed, collar=key; control=controls[key]
        identifier=f'synthetic_re{re}_seed{seed}_eps{collar}'
        run=tmp_path/'external_runs'/identifier; run.mkdir(parents=True)
        config=dict(control['config'], closure_budget=control['receipt']['closure_calls'],
                    study_kind='controlled_variant', sampling='fixed_conditional_uniform_area_strata_v1')
        contract=dict(baseline_config=control['config'],
            baseline_checkpoint_sha256=control['receipt']['checkpoint_sha256'],
            baseline_frozen_model_sha256=control['frozen']['frozen_model']['model_sha256'],
            baseline_provenance_sha256=control['evidence']['baseline_provenance.json']['sha256'],
            metadata=control['provenance']['members'], archive_sha256=control['provenance']['archive_sha256'],
            baseline_closure_calls=config['closure_budget'],
            baseline_residual_point_evaluations=config['closure_budget']*config['collocation'],
            baseline_source_sha256=b.canonical_sha(original),
            runner_source_sha256=b.canonical_sha(runner), runner_raw_sha256=b.sha(runner))
        coverage=dict(objective_sha256='synthetic-objective',
            counts=dict(top_corner_annuli=512, remaining_wall_band=512, interior=1024))
        for name,value in [('config.json',config), ('budget_contract.json',contract),
                           ('sampling_coverage.json',coverage), ('history.json',[{'synthetic':True}]),
                           ('closure_history.json',[{'synthetic':True}]), ('final_physics_audit.json',{'synthetic':True})]:
            write(run/name,value)
        (run/'checkpoint.pt').write_bytes(('synthetic checkpoint identity only: '+identifier).encode())
        (run/'frozen_predictions.npz').write_bytes(b'synthetic original prediction identity only')
        work=config['closure_budget']*config['collocation']
        model=dict(model_sha256='synthetic-model',objective_sha256='synthetic-objective',
                   baseline_contract_sha256='synthetic-contract',frozen_utc='2026-11-15T00:00:00Z')
        artifact_names=('checkpoint.pt','config.json','budget_contract.json','sampling_coverage.json',
                        'history.json','closure_history.json','final_physics_audit.json','frozen_predictions.npz')
        frozen=dict(config=config,reference_labels_used=False,closure_calls=config['closure_budget'],
            residual_point_evaluations=work,frozen_model=model,baseline_contract_sha256='synthetic-contract',
            source_sha256=b.canonical_sha(engine),source_raw_sha256=b.sha(engine),
            runner_sha256=b.canonical_sha(runner),runner_raw_sha256=b.sha(runner),
            baseline_source_sha256=b.canonical_sha(original),artifact_sha256={name:b.sha(run/name) for name in artifact_names},
            checkpoint_sha256=b.sha(run/'checkpoint.pt'),prediction_sha256=b.sha(run/'frozen_predictions.npz'),
            frozen_utc=model['frozen_utc'])
        write(run/'frozen_manifest.json',frozen)
        receipt=dict(finished=True,closure_calls=config['closure_budget'],residual_point_evaluations=work,
            adam_completed=config['adam_steps'],objective_sha256='synthetic-objective',
            budget_contract_sha256='synthetic-contract',checkpoint_sha256=b.sha(run/'checkpoint.pt'))
        write(run/'current_receipt.json',receipt)
        public=output/'raw_scoring'/identifier; public.mkdir(parents=True)
        artifacts={}
        for name in ('prediction','reference','history','ghia_profiles','reference_source'):
            artifact=public/(name+'.fixture'); artifact.write_bytes(('synthetic '+name).encode())
            artifacts[name]=b.entry(artifact)
        record=dict(run_id=identifier,Re=re,seed=seed,collar=collar,config=config,outcome='scored',**artifacts,
            source=dict(checkpoint_sha256=receipt['checkpoint_sha256'],
                original_freeze_manifest_sha256=b.sha(run/'frozen_manifest.json'),
                model_sha256=model['model_sha256'],original_prediction_sha256=frozen['prediction_sha256'],
                source_canonical_sha256=frozen['source_sha256'],original_frozen_at_utc=frozen['frozen_utc']),
            freeze=dict(at_utc='2026-11-15T01:00:00Z'),
            score=dict(reference_accessed_at_utc='2026-11-15T02:00:00Z'))
        records.append(record); runs.append(run)
    raw=output/'evaluation_manifest.json'
    write(raw,dict(schema_version=1,campaign_kind='ghia_constant_lid',records=records))
    return raw,runs,output


def test_twelve_bindings_are_immutable_and_checkpoints_stay_external(study):
    raw,runs,output=study
    result=b.bridge(raw,runs,output); manifest=b.read_json(result)
    assert len(manifest['records']) == 12
    assert manifest['raw_evaluation_manifest']['sha256'] == b.sha(raw)
    assert not list(output.rglob('checkpoint.pt'))
    for record in manifest['records']:
        assert set(record['training_provenance']) == set(b.METADATA)
        for evidence in record['training_provenance'].values(): b.checked(evidence)
        assert record['raw_score_record'] in b.read_json(raw)['records']
    before={p.relative_to(output).as_posix():b.sha(p) for p in output.rglob('*') if p.is_file()}
    assert b.bridge(raw,runs,output) == result
    assert before == {p.relative_to(output).as_posix():b.sha(p) for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('filename', ['checkpoint.pt','budget_contract.json'])
def test_changed_recovery_or_contract_is_rejected(study,filename):
    raw,runs,output=study
    (runs[0]/filename).write_bytes(b'changed')
    with pytest.raises((ValueError,json.JSONDecodeError)): b.bridge(raw,runs,output)
    assert not (output/'experiment_manifest.json').exists()


def test_global_prediction_freeze_precedes_first_reference(study):
    raw,runs,output=study
    value=b.read_json(raw)
    value['records'][-1]['freeze']['at_utc']='2026-11-15T02:30:00Z'
    value['records'][-1]['score']['reference_accessed_at_utc']='2026-11-15T03:00:00Z'
    write(raw,value)
    with pytest.raises(ValueError,match='All 12 native projections'): b.bridge(raw,runs,output)


def test_changed_archived_control_is_rejected(study):
    raw,runs,output=study
    control=next((output/'baseline_controls').glob('*/current_receipt.json'))
    control.write_text('{}')
    with pytest.raises(ValueError,match='byte identity'): b.bridge(raw,runs,output)


def test_runtime_failure_receipt_is_bound_and_cannot_change(study):
    raw,runs,output=study
    runtime=output/'runtime_events.json'
    write(runtime,dict(synthetic=True, failed_allocation_seconds=18, original_closures=0))
    result=b.bridge(raw,runs,output)
    assert b.read_json(result)['runtime_events'] == b.entry(runtime)
    write(runtime,dict(synthetic=True, failed_allocation_seconds=0))
    with pytest.raises(ValueError,match='differing completed experiment'):
        b.bridge(raw,runs,output)

def test_repaired_pass_discloses_prior_targets_and_preserves_original_endpoints(study):
    import copy
    raw,runs,output=study; data=b.read_json(raw); records=data['records']
    forensic=output/'sampler_restore_forensics.json';write(forensic,dict(synthetic=True))
    old=copy.deepcopy(records[:4])
    for row in old:
        row['freeze']['at_utc']='2026-11-15T00:20:00Z'
        row['score']['reference_accessed_at_utc']='2026-11-15T00:30:00Z'
    endpoints=[]
    for run in runs:
        frozen=b.read_json(run/'frozen_manifest.json')
        endpoints.append(dict(config=frozen['config'],manifest_sha256=b.sha(run/'frozen_manifest.json'),
            model_sha256=frozen['frozen_model']['model_sha256'],frozen_utc=frozen['frozen_utc']))
    prior=output/'prior_evaluation_attempt.json'
    write(prior,dict(original_manifest=dict(records=old),prior_scored_case_count=4,
        prior_reference_values_read=True,model_or_fitting_changes_after_prior_access=False,
        all_training_endpoints=endpoints,first_prior_reference_accessed_at_utc='2026-11-15T00:30:00Z',
        training_endpoints_frozen_before_prior_target_access=True,sampler_forensics=b.entry(forensic)))
    barrier=output/'completed_pass/projection_barrier.json'
    write(barrier,dict(predictions=[dict(run_id=row['run_id'],prediction_sha256=row['prediction']['sha256'],
        model_sha256=row['source']['model_sha256'],at_utc=row['freeze']['at_utc']) for row in records],
        prior_evaluation_receipt=b.entry(prior),sampler_forensics=b.entry(forensic)))
    data.update(evaluation_wrapper_sha256=b.sha(b.ROOT/'qa/evaluate_cavity_pinn_stratified.py'),
        projection_barrier=b.entry(barrier),prior_evaluation_receipt=b.entry(prior),sampler_forensics=b.entry(forensic),
        model_or_fitting_changes_after_prior_access=False)
    write(raw,data)
    result=b.read_json(b.bridge(raw,runs,output))
    assert result['prior_scored_case_count']==4
    assert result['all_projections_frozen_before_first_reference_read'] is False
    assert result['all_projections_frozen_before_completed_pass_reference_read'] is True
    assert result['all_training_endpoints_frozen_before_prior_target_access'] is True
    assert set(result['evaluation_provenance'])=={'completed_pass_receipt','prior_failed_attempt','checkpoint_forensics','evaluation_wrapper_source'}
    changed=b.read_json(prior);changed['all_training_endpoints'][-1]['model_sha256']='changed'
    write(prior,changed);data['prior_evaluation_receipt']=b.entry(prior);write(raw,data)
    with pytest.raises(ValueError,match='trained endpoint changed'):
        b.bridge(raw,runs,output)
