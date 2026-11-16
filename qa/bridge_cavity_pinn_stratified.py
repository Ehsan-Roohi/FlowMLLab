"""Bind stratified PINN runs to immutable training and scoring evidence.

No training, reference-value evaluation, model selection or checkpoint mirroring.
The original generic score records remain unchanged provenance.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {(re, seed, collar) for re in (100, 7500) for seed in (7, 23, 61) for collar in (.01, .02)}
METADATA = ('config.json', 'budget_contract.json', 'current_receipt.json',
            'frozen_manifest.json', 'sampling_coverage.json', 'final_physics_audit.json',
            'history.json', 'closure_history.json')


def require(condition, message):
    if not condition: raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def canonical_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode()).hexdigest()


def entry(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Published evidence must remain inside the checkout')
    return dict(path=path.relative_to(ROOT.resolve()).as_posix(), sha256=sha(path))


def checked(evidence):
    path = (ROOT / evidence['path']).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Public evidence path leaves the checkout')
    require(path.is_file() and sha(path) == evidence['sha256'], f'Public evidence hash failed: {path}')
    return path


def copy_once(source, destination):
    expected = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha(destination) == expected, f'Refusing to overwrite differing provenance: {destination}')
    else:
        temporary = destination.with_name(destination.name + '.tmp')
        shutil.copyfile(source, temporary)
        require(sha(temporary) == expected, f'Provenance copy failed: {source}')
        temporary.replace(destination)
    require(sha(destination) == expected, f'Provenance changed during copy: {destination}')
    return entry(destination)


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Evidence timestamp lacks timezone')
    return result


def controls_at(path):
    path = Path(path); manifest = read_json(path)
    archive = 'd041091ee84b5e1150fc186b2f15087852f0e1d2929be78ce96d81558c953235'
    require(manifest['schema'] == 1 and manifest['archive_sha256'] == archive, 'Wrong baseline archive identity')
    controls = {}
    for control in manifest['controls']:
        key = (control['Re'], control['seed'], float(control['collar']))
        require(key in EXPECTED and key not in controls, 'Unexpected or repeated baseline control')
        directory = (path.parent / control['directory']).resolve()
        require(directory.is_relative_to(path.parent.resolve()), 'Baseline directory leaves controls')
        evidence = {}
        for name, digest in control['sha256'].items():
            require(Path(name).name == name, 'Baseline metadata path leaves its directory')
            evidence[name] = entry(directory / name)
            require(evidence[name]['sha256'] == digest, 'Baseline metadata byte identity failed')
        require(set(evidence) == {'config.json', 'current_receipt.json', 'frozen_manifest.json', 'baseline_provenance.json'}, 'Incomplete original baseline metadata')
        provenance = read_json(directory / 'baseline_provenance.json')
        receipt = read_json(directory / 'current_receipt.json')
        require(provenance['archive_sha256'] == archive and provenance['checkpoint_sha256'] == receipt['checkpoint_sha256'], 'Baseline archive/recovery mismatch')
        require(receipt['finished'] and receipt['closure_calls'] == control['closure_cap']
                and receipt['residual_point_evaluations'] == control['residual_points'], 'Baseline indexed budget differs')
        for name, member in provenance['members'].items():
            require(member['sha256'] == evidence[name]['sha256'], 'Baseline archive member mismatch')
        controls[key] = dict(evidence=evidence, config=read_json(directory / 'config.json'),
            receipt=receipt, frozen=read_json(directory / 'frozen_manifest.json'), provenance=provenance)
    require(set(controls) == EXPECTED, 'Exactly 12 archived baseline controls are required')
    return controls


def bind_run(run, raw_record, output, control):
    run = Path(run)
    configuration = read_json(run / 'config.json')
    frozen = read_json(run / 'frozen_manifest.json')
    receipt = read_json(run / 'current_receipt.json')
    contract = read_json(run / 'budget_contract.json')
    coverage = read_json(run / 'sampling_coverage.json')
    key = (configuration['reynolds'], configuration['seed'], configuration['collar'])
    require(key in EXPECTED, 'Run is outside the declared 12-case variant')
    require(contract['baseline_config'] == control['config']
            and contract['baseline_checkpoint_sha256'] == control['receipt']['checkpoint_sha256']
            and contract['baseline_frozen_model_sha256'] == control['frozen']['frozen_model']['model_sha256'],
            'Budget contract does not match the retained original counterpart')
    require(contract['baseline_provenance_sha256'] == control['evidence']['baseline_provenance.json']['sha256']
            and contract['metadata'] == control['provenance']['members']
            and contract['archive_sha256'] == control['provenance']['archive_sha256'],
            'Budget contract provenance differs from the verified archive')
    require((raw_record['Re'], raw_record['seed'], raw_record['collar']) == key, 'Scored/run case mismatch')
    require(raw_record['config'] == frozen['config'] == configuration, 'Configuration mismatch')
    require(configuration['sampling'] == 'fixed_conditional_uniform_area_strata_v1'
            and configuration['study_kind'] == 'controlled_variant', 'Wrong sampling study')
    require(receipt['finished'] and frozen['reference_labels_used'] is False, 'Endpoint not frozen without labels')
    require(receipt['closure_calls'] == frozen['closure_calls'] == configuration['closure_budget']
            == contract['baseline_closure_calls'], 'Paired complete-closure budget mismatch')
    require(receipt['residual_point_evaluations'] == frozen['residual_point_evaluations']
            == contract['baseline_residual_point_evaluations']
            == configuration['collocation'] * configuration['closure_budget'], 'Matched point-work mismatch')
    require(receipt['adam_completed'] == configuration['adam_steps'], 'Warm stage is incomplete')
    require({k: configuration[k] for k in contract['baseline_config']} == contract['baseline_config'],
            'Baseline physical/network/optimizer settings changed')
    require(coverage['objective_sha256'] == receipt['objective_sha256']
            == frozen['frozen_model']['objective_sha256'], 'Weighted objective identity mismatch')
    require(coverage['counts'] == dict(top_corner_annuli=512, remaining_wall_band=512, interior=1024),
            'Declared stratified sample counts changed')
    require(frozen['baseline_contract_sha256'] == frozen['frozen_model']['baseline_contract_sha256']
            == receipt['budget_contract_sha256'], 'Budget evidence identity mismatch')
    engine = ROOT / 'flowmllab/cavity_pinn_stratified.py'
    runner = ROOT / 'qa/run_cavity_pinn_stratified.py'
    original = ROOT / 'flowmllab/cavity_pinn_campaign.py'
    require(frozen['source_sha256'] == canonical_sha(engine), 'Stratified engine identity changed')
    require(frozen['runner_sha256'] == canonical_sha(runner), 'Stratified runner identity changed')
    require(frozen['source_raw_sha256'] == sha(engine) and frozen['runner_raw_sha256'] == sha(runner),
            'Original staged source bytes changed')
    require(frozen['baseline_source_sha256'] == canonical_sha(original)
            == contract['baseline_source_sha256'], 'Original physics identity changed')
    require(contract['runner_source_sha256'] == frozen['runner_sha256']
            and contract['runner_raw_sha256'] == frozen['runner_raw_sha256'], 'Declared runner identity mismatch')
    for name, digest in frozen['artifact_sha256'].items():
        require(Path(name).name == name, 'Frozen artifact path leaves run')
        require(sha(run / name) == digest, f'Frozen artifact changed: {run / name}')
    require(sha(run / 'checkpoint.pt') == receipt['checkpoint_sha256'] == frozen['checkpoint_sha256'],
            'Complete external checkpoint hash failed')
    source = raw_record['source']
    require(source['checkpoint_sha256'] == receipt['checkpoint_sha256']
            and source['original_freeze_manifest_sha256'] == sha(run / 'frozen_manifest.json'),
            'Native score is not bound to this original recovery')
    require(source['model_sha256'] == frozen['frozen_model']['model_sha256']
            and source['original_prediction_sha256'] == frozen['prediction_sha256']
            and source['source_canonical_sha256'] == frozen['source_sha256'],
            'Frozen model/source/prediction identity mismatch')
    require(source['original_frozen_at_utc'] == frozen['frozen_utc']
            == frozen['frozen_model']['frozen_utc'], 'Original freeze timestamp mismatch')
    require(timestamp(frozen['frozen_utc']) <= timestamp(raw_record['freeze']['at_utc'])
            < timestamp(raw_record['score']['reference_accessed_at_utc']), 'Invalid freeze/reference ordering')
    for name in ('prediction', 'reference', 'history', 'ghia_profiles', 'reference_source'):
        checked(raw_record[name])
    destination = output / 'run_provenance' / raw_record['run_id']
    require(destination.resolve().is_relative_to(output.resolve()), 'Run identifier leaves public output')
    metadata = {name: copy_once(run / name, destination / name) for name in METADATA}
    return dict(run_id=raw_record['run_id'], Re=key[0], seed=key[1], collar=key[2],
                method='Adam-to-L-BFGS, fixed area-weighted stratified collocation',
                config=configuration, training_provenance=metadata, baseline_control_provenance=control['evidence'], baseline_contract=contract,
                sampling_coverage=coverage, raw_score_record=raw_record,
                binary_checkpoint=dict(sha256=receipt['checkpoint_sha256'],
                    location='external recovery; byte-verified at bridge creation; not mirrored into repository'),
                reference_labels_used_for_training=False)


def evaluation_context(raw, records):
    if 'evaluation_wrapper_sha256' not in raw:
        return dict(all_projections_frozen_before_first_reference_read=True,
                    all_projections_frozen_before_completed_pass_reference_read=True)
    mapping = dict(completed_pass_receipt=raw['projection_barrier'],
                   prior_failed_attempt=raw['prior_evaluation_receipt'],
                   checkpoint_forensics=raw['sampler_forensics'])
    wrapper = ROOT/'qa/evaluate_cavity_pinn_stratified.py'
    require(sha(wrapper) == raw['evaluation_wrapper_sha256'], 'Completed evaluation wrapper changed')
    mapping['evaluation_wrapper_source'] = entry(wrapper)
    documents = {name: read_json(checked(evidence)) for name, evidence in mapping.items()
                 if name != 'evaluation_wrapper_source'}
    prior = documents['prior_failed_attempt']; barrier = documents['completed_pass_receipt']
    scored = [row for row in prior['original_manifest']['records'] if row['outcome'] == 'scored']
    require(len(scored) == prior['prior_scored_case_count'] and 0 < len(scored) < 12,
            'Prior partial reference access must remain explicit')
    require(prior['prior_reference_values_read'] is True
            and prior['model_or_fitting_changes_after_prior_access'] is False
            and raw['model_or_fitting_changes_after_prior_access'] is False,
            'Repair must not hide prior access or change trained endpoints')
    endpoints = {(row['config']['reynolds'], row['config']['seed'], row['config']['collar']): row
                 for row in prior['all_training_endpoints']}
    require(len(prior['all_training_endpoints']) == 12 and set(endpoints) == EXPECTED,
            'Prior attempt lacks all twelve original trained endpoints')
    first_prior = min(timestamp(row['score']['reference_accessed_at_utc']) for row in scored)
    require(timestamp(prior['first_prior_reference_accessed_at_utc']) == first_prior,
            'Prior access timestamp changed')
    projected = {row['run_id']: row for row in barrier['predictions']}
    require(len(barrier['predictions']) == 12 and len(projected) == 12, 'Completed native barrier is incomplete')
    for record in records:
        identity = (record['Re'], record['seed'], record['collar']); endpoint = endpoints[identity]
        require(endpoint['config'] == record['config']
                and endpoint['manifest_sha256'] == record['source']['original_freeze_manifest_sha256']
                and endpoint['model_sha256'] == record['source']['model_sha256']
                and timestamp(endpoint['frozen_utc']) < first_prior, 'A trained endpoint changed after earlier target access')
        projection = projected[record['run_id']]
        require(projection['prediction_sha256'] == record['prediction']['sha256']
                and projection['model_sha256'] == endpoint['model_sha256']
                and projection['at_utc'] == record['freeze']['at_utc'], 'Completed native barrier identity differs')
    require(barrier['prior_evaluation_receipt'] == raw['prior_evaluation_receipt']
            and barrier['sampler_forensics'] == raw['sampler_forensics']
            and prior['sampler_forensics'] == raw['sampler_forensics'], 'Repair provenance links differ')
    require(prior['training_endpoints_frozen_before_prior_target_access'] is True,
            'Original training freeze chronology was not established')
    return dict(evaluation_provenance=mapping, prior_scored_case_count=len(scored),
                all_projections_frozen_before_first_reference_read=False,
                all_projections_frozen_before_completed_pass_reference_read=True,
                all_training_endpoints_frozen_before_prior_target_access=True,
                evaluation_chronology=f'Earlier partial scoring accessed targets for {len(scored)} cases; all twelve training endpoints were already frozen. The repaired pass completed all twelve native projections before its own target reads, without fitting or model choice')


def bridge(raw_manifest, runs, output, baseline_controls=None, runtime_events=None):
    raw_manifest, output = Path(raw_manifest).resolve(), Path(output).resolve()
    require(output.is_relative_to(ROOT.resolve()), 'Output must remain inside the checkout')
    baseline_controls = Path(baseline_controls) if baseline_controls else output / 'baseline_controls/controls_manifest.json'
    controls = controls_at(baseline_controls)
    raw = read_json(raw_manifest); records = raw['records']
    require(raw['schema_version'] == 1 and raw['campaign_kind'] == 'ghia_constant_lid', 'Wrong raw score schema')
    keys = [(r['Re'], r['seed'], r['collar']) for r in records]
    require(len(keys) == len(set(keys)) == 12 and set(keys) == EXPECTED, 'Exactly the declared 12 scored cases are required')
    require(all(r['outcome'] == 'scored' for r in records), 'A full experiment requires every frozen endpoint to be scored')
    by_key = {key: record for key, record in zip(keys, records)}
    directories = {}
    for run in runs:
        configuration = read_json(Path(run) / 'config.json')
        key = (configuration['reynolds'], configuration['seed'], configuration['collar'])
        require(key not in directories, 'Duplicate supplied training trajectory')
        directories[key] = Path(run)
    require(set(directories) == EXPECTED, 'Exactly all 12 original completed runs are required')
    frozen = [timestamp(r['freeze']['at_utc']) for r in records]
    accesses = [timestamp(r['score']['reference_accessed_at_utc']) for r in records]
    require(max(frozen) < min(accesses), 'All 12 native projections must freeze before the first target-value access')
    output.mkdir(parents=True, exist_ok=True)
    bound = [bind_run(directories[key], by_key[key], output, controls[key]) for key in sorted(EXPECTED)]
    manifest = dict(schema_version=1, campaign_kind='ghia_stratified_matched_work',
        status='completed frozen experiments; integrity is separate from scientific acceptance',
        expected_reynolds=[100, 7500], expected_seeds=[7, 23, 61], expected_collars=[.01, .02],
        raw_evaluation_manifest=entry(raw_manifest), bridge_source=entry(__file__), baseline_controls=entry(baseline_controls),
        records=bound, **evaluation_context(raw, records),
        scope='One prespecified sampling-only intervention; no architecture or target-based model selection',
        comparison_policy='Each run matches its own original baseline point-work budget; collar budgets can differ',
        history_policy='Raw generic-scorer history is retained unchanged; training loss here is area-weighted momentum MSE and its plotted square root is not an independent audit',
        checkpoint_policy='Original complete restart files remain external; small metadata only are copied',
        scientific_acceptance='No success, convergence, or optimizer superiority is implied by this evidence bridge')
    runtime_path = Path(runtime_events) if runtime_events else output / 'runtime_events.json'
    if runtime_events is not None: require(runtime_path.is_file(), 'Requested runtime-event receipt is unavailable')
    if runtime_path.is_file(): manifest['runtime_events'] = entry(runtime_path)
    manifest['timing_policy'] = 'Optimizer-step seconds exclude failed allocations, requeue/scheduler time, checkpoint I/O and independent audits; they are not campaign wall time'
    destination = output / 'experiment_manifest.json'
    if destination.exists():
        require(read_json(destination) == manifest, 'Refusing to replace differing completed experiment evidence')
    else:
        temporary = destination.with_name(destination.name + '.tmp')
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-manifest', type=Path, default=ROOT / 'results/cavity_pinn_stratified/evaluation_manifest.json')
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/cavity_pinn_stratified')
    parser.add_argument('--baseline-controls', type=Path, help='Defaults to output/baseline_controls/controls_manifest.json')
    parser.add_argument('--runtime-events', type=Path, help='Optional compact failed-allocation/recovery evidence; discovered under output by default')
    args = parser.parse_args()
    result = bridge(args.raw_manifest, args.run, args.output, args.baseline_controls, args.runtime_events)
    print(json.dumps(dict(experiment_manifest=str(result), sha256=sha(result), verified_cases=12)))


if __name__ == '__main__': main()

