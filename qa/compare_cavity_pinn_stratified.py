#!/usr/bin/env python3
"""Compare the single stratified intervention with its 12 completed controls.

This post-score consumer does not train, select endpoints, or load CFD arrays.
It verifies the bridged training metadata and immutable scorer evidence first,
then compares existing physical-error metrics and signed Ghia profile samples.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE_SHA256 = '507a6e0a6d632a173c8bdc8748564beffee77ce3664aeffcee25edc5ca305b9e'
ARCHIVE_SHA256 = 'd041091ee84b5e1150fc186b2f15087852f0e1d2929be78ce96d81558c953235'
EXPECTED = {(re, seed, collar) for re in (100, 7500) for seed in (7, 23, 61) for collar in (.01, .02)}
BASE_EXPECTED = {(re, seed, collar) for re in (100, 400, 1000, 3200, 5000, 7500, 10000)
                 for seed in (7, 23, 61) for collar in (.01, .02)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def canonical_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition: raise ValueError(message)


def checked_entry(root, entry):
    path = (root / entry['path']).resolve()
    require(path.is_relative_to(root.resolve()), 'Evidence escapes repository root')
    require(path.is_file() and sha(path) == entry['sha256'], 'Evidence SHA256 mismatch: ' + entry['path'])
    return path


def file_entry(root, path):
    return dict(path=Path(path).resolve().relative_to(root.resolve()).as_posix(), sha256=sha(path))


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def plain_state_hash(value):
    """The training engine's structural hash restricted to JSON contract types."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode()); visit(len(item))
            for child in item: visit(child)
        elif isinstance(item, (str, bytes, bool, int, float)) or item is None:
            data = repr(item).encode()
            digest.update(type(item).__name__.encode() + str(len(data)).encode() + b':' + data)
        else: raise TypeError('Non-JSON budget contract')
    visit(value)
    return digest.hexdigest()


def key_of(row):
    return (int(row['Re']), int(row['seed']), float(row['collar']))


def indexed(records, expected, require_scored=True):
    result = {}
    for row in records:
        key = key_of(row)
        require(key not in result, 'Duplicate scientific counterpart: ' + repr(key))
        if require_scored: require(row['outcome'] == 'scored', 'Incomplete/rejected score: ' + repr(key))
        result[key] = row
    require(set(result) == expected, 'Missing or unexpected controlled cases')
    return result


def timestamp(value):
    moment = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(moment.tzinfo is not None, 'Unqualified freeze/access timestamp')
    return moment


def freeze_barrier(records):
    originals = [timestamp(row['source']['original_frozen_at_utc']) for row in records]
    projections = [timestamp(row['freeze']['at_utc']) for row in records]
    accesses = [timestamp(row['score']['reference_accessed_at_utc']) for row in records]
    require(max(originals) <= min(accesses), 'Targets accessed before all 12 model endpoints froze')
    require(max(projections) <= min(accesses), 'Targets accessed before all 12 queried predictions froze')
    for row in records:
        require(timestamp(row['source']['original_frozen_at_utc']) <= timestamp(row['freeze']['at_utc']),
                'Projection precedes the original model freeze')
        require(timestamp(row['score']['reference_accessed_at_utc']) <= timestamp(row['score']['cfd_values_accessed_at_utc']),
                'Inconsistent target-access ordering')
    return dict(all_12_models_frozen=True, all_12_projections_frozen=True,
                scope='native-projection barrier for the completed scoring pass; not a claim that an earlier failed pass never accessed targets',
                last_model_freeze=max(originals).isoformat(), last_projection_freeze=max(projections).isoformat(),
                first_target_access_in_completed_pass=min(accesses).isoformat())


def checked_history(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), 'Empty optimizer history')
    calls = [int(row['objective_evaluations']) for row in rows]
    require(all(b > a for a, b in zip(calls, calls[1:])), 'History work is not strictly increasing')
    return dict(closure_calls=calls[-1], residual_points=int(rows[-1]['residual_point_evaluations']),
                elapsed_seconds=float(rows[-1]['elapsed_seconds']))


def validate_record(root, row, expected_source):
    require(row['source']['source_canonical_sha256'] == expected_source, 'Wrong training source identity')
    require(row['source']['config'] == row['config'], 'Projection configuration mismatch')
    require(row['source']['config_sha256'] == config_sha(row['config']), 'Scored configuration hash mismatch')
    digest = row['prediction']['sha256']
    require(digest == row['freeze']['prediction_sha256'] == row['score']['prediction_sha256'],
            'Scored prediction differs from frozen prediction')
    require(row['freeze']['model_sha256'] == row['source']['model_sha256'], 'Scored model identity mismatch')
    paths = {name: checked_entry(root, row[name]) for name in ('prediction', 'reference', 'history', 'ghia_profiles')}
    return paths, checked_history(paths['history'])


def validate_pair(root, baseline, variant, bridge, variant_source):
    bpaths, bhistory = validate_record(root, baseline, BASE_SOURCE_SHA256)
    vpaths, vhistory = validate_record(root, variant, variant_source)
    require(bridge['run_id'] == variant['run_id'] and bridge['raw_score_record'] == variant,
            'Bridge embeds a different raw score record')
    required = ('config.json', 'budget_contract.json', 'current_receipt.json', 'frozen_manifest.json',
                'sampling_coverage.json', 'final_physics_audit.json', 'history.json')
    entries = bridge['training_provenance']
    require(all(name in entries for name in required), 'Bridge is missing required training metadata')
    paths = {name: checked_entry(root, entries[name]) for name in required}
    metadata = {name: read_json(path) for name, path in paths.items()}
    cfg = metadata['config.json']; contract = metadata['budget_contract.json']
    receipt = metadata['current_receipt.json']; frozen = metadata['frozen_manifest.json']
    coverage = metadata['sampling_coverage.json']; history = metadata['history.json']
    control_entries = bridge['baseline_control_provenance']
    control_names = ('config.json', 'current_receipt.json', 'frozen_manifest.json', 'baseline_provenance.json')
    require(set(control_entries) == set(control_names), 'Incomplete archived control provenance')
    controls = {name: read_json(checked_entry(root, control_entries[name])) for name in control_names}
    control_receipt = controls['current_receipt.json']; control_provenance = controls['baseline_provenance.json']
    require(controls['config.json'] == baseline['config'] == controls['frozen_manifest.json']['config'],
            'Archived control configuration changed')
    require(control_provenance['archive_sha256'] == ARCHIVE_SHA256
            and control_provenance['checkpoint_sha256'] == control_receipt['checkpoint_sha256']
            == baseline['source']['checkpoint_sha256'], 'Archived control recovery differs')
    require(control_receipt['finished'] and control_receipt['closure_calls'] == contract['baseline_closure_calls']
            and control_receipt['residual_point_evaluations'] == contract['baseline_residual_point_evaluations'],
            'Archived control work differs')
    require(controls['frozen_manifest.json']['frozen_model']['model_sha256'] == baseline['source']['model_sha256'],
            'Archived control model differs')
    require(contract['metadata'] == control_provenance['members']
            and contract['baseline_provenance_sha256'] == control_entries['baseline_provenance.json']['sha256'],
            'Contract no longer matches archived metadata')
    for name, member in control_provenance['members'].items():
        require(control_entries[name]['sha256'] == member['sha256'], 'Archived member hash failed')
    require(cfg == variant['config'] == frozen['config'], 'Bridged configuration differs from scored model')
    require(contract['baseline_config'] == baseline['config'], 'Baseline counterpart configuration differs')
    require({k: cfg[k] for k in baseline['config']} == baseline['config'], 'Intervention changed paired physics or network')
    require(cfg['study_kind'] == 'controlled_variant' and cfg['sampling'] == 'fixed_conditional_uniform_area_strata_v1',
            'This is not the declared single sampling intervention')
    require(cfg['width'] == 64 and cfg['depth'] == 4 and cfg['collocation'] == 2048 and cfg['adam_steps'] == 2000,
            'Undeclared architecture, points or warm stage')
    require(contract['archive_sha256'] == ARCHIVE_SHA256, 'Wrong baseline archival evidence')
    require(contract['baseline_source_sha256'] == BASE_SOURCE_SHA256, 'Different baseline physical source')
    require(contract['baseline_config_sha256'] == baseline['source']['config_sha256'], 'Baseline config binding differs')
    require(contract['baseline_checkpoint_sha256'] == baseline['source']['checkpoint_sha256'], 'Different baseline recovery')
    require(contract['baseline_frozen_model_sha256'] == baseline['source']['model_sha256'], 'Different baseline endpoint')
    require(contract['metadata']['frozen_manifest.json']['sha256'] == baseline['source']['original_freeze_manifest_sha256'],
            'Different original baseline freeze manifest')
    require(plain_state_hash(contract) == frozen['baseline_contract_sha256'] == receipt['budget_contract_sha256'],
            'Budget contract structural hash differs')
    require(frozen['source_sha256'] == variant_source and frozen['baseline_source_sha256'] == BASE_SOURCE_SHA256,
            'Bridged implementation differs')
    require(sha(paths['frozen_manifest.json']) == variant['source']['original_freeze_manifest_sha256'],
            'Original frozen manifest is not the GPU-scored one')
    # A crash after final-manifest creation can leave the earlier completed receipt
    # without its redundant embedded manifest; the standalone manifest is authoritative.
    if 'frozen_manifest' in receipt:
        require(frozen['frozen_model'] == receipt['frozen_manifest']['frozen_model'], 'Receipt embeds a different freeze')
    require(frozen['frozen_model']['model_sha256'] == variant['source']['model_sha256'], 'Bridged model differs')
    require(frozen['prediction_sha256'] == variant['source']['original_prediction_sha256'], 'Original prediction differs')
    require(frozen['frozen_utc'] == variant['source']['original_frozen_at_utc'], 'Original freeze time changed')
    require(frozen['checkpoint_sha256'] == receipt['checkpoint_sha256'] == variant['source']['checkpoint_sha256'],
            'Checkpoint identity mismatch')
    for name in required:
        if name in frozen['artifact_sha256']:
            require(entries[name]['sha256'] == frozen['artifact_sha256'][name], 'Bridged metadata was not hash-bound at freeze')
    cap = cfg['closure_budget']; points = cap * cfg['collocation']
    require(receipt['finished'] and receipt['adam_completed'] == 2000, 'Variant did not finish the declared stages')
    require(cap == contract['baseline_closure_calls'] == receipt['closure_calls'] == frozen['closure_calls']
            == frozen['frozen_model']['closure_calls'] == bhistory['closure_calls'] == vhistory['closure_calls'],
            'Total training closure work is not matched')
    require(points == contract['baseline_residual_point_evaluations'] == receipt['residual_point_evaluations']
            == frozen['residual_point_evaluations'] == bhistory['residual_points'] == vhistory['residual_points'],
            'Residual-point work is not matched')
    require(history and history[-1]['closure_calls'] == cap and history[-1]['residual_point_evaluations'] == points,
            'Frozen step history ended before the work cap')
    require(coverage['objective_sha256'] == frozen['frozen_model']['objective_sha256'] == receipt['objective_sha256'],
            'Coverage/objective identity differs')
    require(coverage['counts'] == dict(top_corner_annuli=512, remaining_wall_band=512, interior=1024),
            'Stratum allocation changed')
    require(coverage['collar'] == cfg['collar'] and coverage['top_corner_radius'] == .05 and not coverage['adaptive_sampling'],
            'Undeclared sampler adaptation or support')
    eps = cfg['collar']; areas = dict(top_corner_annuli=math.pi*(.05**2-eps**2)/2,
        remaining_wall_band=1-.9**2-math.pi*.05**2/2, interior=.81)
    total = 1-math.pi*eps**2/2
    require(abs(coverage['total_area']-total) < 1e-14, 'Retained area differs')
    for name, area in areas.items():
        require(abs(coverage['areas'][name]-area) < 1e-14 and abs(coverage['weight_sums'][name]-area/total) < 1e-14,
                'Weights do not represent retained-domain area')
    require(metadata['final_physics_audit.json'] == variant['physics_audit'], 'Scored physical audit differs')
    require(variant['physics_audit']['weighted_accepted_endpoint']['included_in_training_budget'] is False,
            'Accepted-endpoint diagnostic was counted as fitting')
    for key in ('evaluator_source_sha256', 'ghia_source_sha256', 'coordinate_sha256', 'ghia_coordinate_sha256', 'reference_file_sha256'):
        require(baseline['source'][key] == variant['source'][key], 'Unmatched scoring/reference identity: '+key)
    require(baseline['reference_source'] == variant['reference_source'], 'Different CFD source')
    require(baseline['native_subset'] == variant['native_subset'], 'Different sampled spatial support')
    return dict(baseline_paths=bpaths, variant_paths=vpaths, baseline_history=bhistory, variant_history=vhistory,
                config=cfg, contract=contract, receipt=receipt, frozen=frozen, coverage=coverage,
                provenance=entries, baseline_control_provenance=control_entries, closure_cap=cap, residual_point_work=points)


def error_change(baseline, variant, signed=False, diagnostic=False):
    require(math.isfinite(baseline) and math.isfinite(variant), 'Nonfinite reported metric')
    b = abs(baseline) if signed else baseline; v = abs(variant) if signed else variant
    delta = v-b; tolerance = 1e-12*max(1., b, v)
    outcome = 'tie' if abs(delta) <= tolerance else ('improved' if delta < 0 else 'worsened')
    if diagnostic: outcome = 'diagnostic_only'
    return dict(baseline=baseline, stratified=variant, signed_delta=variant-baseline,
                error_magnitude_delta=delta, relative_error_reduction=(b-v)/b if b > 0 else None,
                comparison_basis='absolute signed bias' if signed else 'reported error magnitude', outcome=outcome)


def metric_rows(baseline, variant):
    rows = []; identity = dict(Re=variant['Re'], seed=variant['seed'], collar=variant['collar'])
    def add(name, b, v, signed=False, diagnostic=False):
        rows.append(dict(**identity, metric=name, **error_change(float(b), float(v), signed, diagnostic)))
    names = ('velocity_error_rms_U', 'velocity_relative_l2', 'u_error_signed_mean_U', 'v_error_signed_mean_U',
             'omega_error_rms_U_over_W', 'omega_error_signed_mean_U_over_W', 'omega_relative_l2')
    for region in ('interior', 'lower_quarter', 'upper_quarter'):
        b=baseline['scores']['regions'][region]; v=variant['scores']['regions'][region]
        require(b['sample_count'] == v['sample_count'], 'Spatial metric sample coverage differs')
        require(math.isclose(b['reference_velocity_rms_U'], v['reference_velocity_rms_U'], rel_tol=1e-13, abs_tol=1e-15),
                'Reference normalization differs')
        for name in names: add('field.'+region+'.'+name, b[name], v[name], signed='signed_mean' in name)
    for component in ('u', 'v'):
        b=baseline['scores']['ghia'][component]; v=variant['scores']['ghia'][component]
        require((b['n_screened'], b['n_published']) == (v['n_screened'], v['n_published']), 'Ghia screening differs')
        for name in ('screened_relative_l2', 'published_relative_l2', 'screened_rmse_U', 'screened_linf_U'):
            add('ghia.'+component+'.'+name, b[name], v[name])
    for name in ('momentum_rms', 'divergence_rms'):
        add('physics.outer_uniform.'+name, baseline['physics_audit']['outer_uniform'][name],
            variant['physics_audit']['outer_uniform'][name], diagnostic=True)
    bs, vs = baseline['physics_audit']['corner_shells'], variant['physics_audit']['corner_shells']
    require(len(bs) == len(vs), 'Different independent corner shell audit')
    for b,v in zip(bs,vs):
        require((b['r_min'],b['r_max'],b['count']) == (v['r_min'],v['r_max'],v['count']), 'Corner shell support differs')
        add(f"physics.corner_{b['r_min']:g}_{b['r_max']:g}.momentum_rms", b['momentum_rms'],v['momentum_rms'],diagnostic=True)
    return rows


def profile_rows(baseline_path, variant_path, identity):
    def load(path):
        result = {}
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            for raw in csv.DictReader(stream):
                key = (raw['component'], float(raw['position']))
                require(key not in result and key[0] in ('u','v'), 'Duplicate/invalid profile coordinate')
                require(raw['screened_included'] in ('True','False','true','false'), 'Invalid profile screening flag')
                row = {name:float(raw[name]) for name in ('published','prediction','signed_error_U')}
                require(all(math.isfinite(value) for value in row.values()), 'Nonfinite profile value')
                require(abs(row['prediction']-row['published']-row['signed_error_U'])<1e-12, 'Inconsistent signed profile error')
                row['screened_included']=raw['screened_included'].lower()=='true'; result[key]=row
        return result
    baseline,variant=load(baseline_path),load(variant_path)
    require(set(baseline)==set(variant), 'Different profile sample coverage; interpolation is forbidden')
    result=[]
    for key in sorted(baseline):
        b,v=baseline[key],variant[key]
        require(b['published']==v['published'] and b['screened_included']==v['screened_included'], 'Profile reference/screening differs')
        outcome=error_change(b['signed_error_U'],v['signed_error_U'],signed=True)['outcome']
        result.append(dict(**identity,component=key[0],position=key[1],published=b['published'],
            baseline_prediction=b['prediction'],stratified_prediction=v['prediction'],
            baseline_signed_error_U=b['signed_error_U'],stratified_signed_error_U=v['signed_error_U'],
            absolute_error_reduction_U=abs(b['signed_error_U'])-abs(v['signed_error_U']),
            screened_included=b['screened_included'],outcome=outcome))
    return result


def descriptive_summary(metrics):
    groups=defaultdict(list)
    for row in metrics: groups[(row['Re'],row['collar'],row['metric'])].append(row)
    result=[]
    for (re,collar,name),rows in sorted(groups.items()):
        require({row['seed'] for row in rows}=={7,23,61}, 'Summary lost a predeclared seed')
        result.append(dict(Re=re,collar=collar,metric=name,n_paired_seeds=3,
            improved=sum(row['outcome']=='improved' for row in rows),worsened=sum(row['outcome']=='worsened' for row in rows),
            tied=sum(row['outcome']=='tie' for row in rows),diagnostic_only=all(row['outcome']=='diagnostic_only' for row in rows),
            median_error_magnitude_delta=statistics.median(row['error_magnitude_delta'] for row in rows),
            minimum_error_magnitude_delta=min(row['error_magnitude_delta'] for row in rows),
            maximum_error_magnitude_delta=max(row['error_magnitude_delta'] for row in rows)))
    return result


def collar_groups(pairs):
    by_key={key_of(row):row for row in pairs}; result=[]
    for re in (100,7500):
        for seed in (7,23,61):
            a,b=by_key[(re,seed,.01)],by_key[(re,seed,.02)]
            equal=a['closure_cap']==b['closure_cap']
            result.append(dict(Re=re,seed=seed,epsilon_0_01_closures=a['closure_cap'],epsilon_0_02_closures=b['closure_cap'],
                equal_work=equal,convergence_claim=False,
                interpretation='equal total work; descriptive collar comparison only' if equal else
                'unequal work across collars; cannot establish equal-work collar sensitivity or convergence'))
    return result


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def atomic_csv(path,rows):
    require(bool(rows),'Empty comparison output')
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def execute(baseline_manifest,variant_manifest,experiment_manifest,output,root=ROOT):
    root=Path(root).resolve();output=Path(output).resolve()
    require(output.is_relative_to(root),'Published comparison must stay in the checkout')
    inputs={name:file_entry(root,path) for name,path in [('baseline_manifest',baseline_manifest),
            ('variant_raw_manifest',variant_manifest),('experiment_manifest',experiment_manifest)]}
    destination=output/'comparison_manifest.json'
    old = None
    if destination.exists():
        old=read_json(destination)
        require(old['inputs']==inputs and old['comparison_source_sha256']==sha(__file__),
                'Refusing to replace a comparison from different evidence/source')
        for entry in old['outputs'].values():checked_entry(root,entry)
    baseline=read_json(baseline_manifest);variant=read_json(variant_manifest);bridge=read_json(experiment_manifest)
    require(bridge['campaign_kind']=='ghia_stratified_matched_work','Wrong experiment bridge')
    runtime_events = bridge.get('runtime_events')
    if runtime_events is not None: checked_entry(root, runtime_events)
    evaluation_provenance = bridge.get('evaluation_provenance', {})
    require(isinstance(evaluation_provenance, dict), 'Invalid evaluation provenance mapping')
    for entry in evaluation_provenance.values(): checked_entry(root, entry)
    chronology_keys = ('all_projections_frozen_before_completed_pass_reference_read',
                       'all_projections_frozen_before_first_reference_read',
                       'all_training_endpoints_frozen_before_prior_target_access', 'prior_scored_case_count')
    evaluation_chronology = {key: bridge[key] for key in chronology_keys if key in bridge}
    if 'prior_failed_attempt' in evaluation_provenance:
        require(bridge.get('all_projections_frozen_before_completed_pass_reference_read') is True
                and bridge.get('all_projections_frozen_before_first_reference_read') is False
                and bridge.get('all_training_endpoints_frozen_before_prior_target_access') is True,
                'Repaired-pass chronology is missing or misrepresented')
        count = bridge.get('prior_scored_case_count')
        require(isinstance(count, int) and not isinstance(count, bool) and 0 < count < 12,
                'Prior partial-scoring count is missing')
        evaluation_chronology['disclosure'] = (f'The earlier failed evaluation scored {count} cases and accessed targets. '
            'All 12 original trained endpoints had already frozen then. The repaired completed pass restores exact '
            'saved states, freezes all native projections before its own reference read, and performs no fitting or model selection.')
    else:
        evaluation_chronology['disclosure'] = ('The native-projection barrier below concerns the supplied completed scoring pass. '
            'It is not proof that no earlier evaluation attempt accessed references.')
    checked_entry(root,bridge['raw_evaluation_manifest'])
    require(bridge['raw_evaluation_manifest']['sha256']==inputs['variant_raw_manifest']['sha256'], 'Bridge refers to another score manifest')
    b=indexed(baseline['records'],BASE_EXPECTED);v=indexed(variant['records'],EXPECTED)
    bridged=indexed(bridge['records'],EXPECTED,require_scored=False)
    barrier=freeze_barrier(list(v.values()))
    variant_source=canonical_sha(root/'flowmllab/cavity_pinn_stratified.py')
    # Validate all original work/source/provenance before reading any profile targets.
    prepared={key:validate_pair(root,b[key],v[key],bridged[key],variant_source) for key in sorted(EXPECTED)}
    if old is not None:
        require(old.get('runtime_events') == runtime_events, 'Runtime-event evidence differs')
        require(old.get('evaluation_provenance', {}) == evaluation_provenance
                and old.get('evaluation_chronology') == evaluation_chronology, 'Evaluation-attempt evidence differs')
        return old
    pairs=[];metrics=[];profiles=[]
    for key,item in prepared.items():
        old,new=b[key],v[key];identity=dict(Re=key[0],seed=key[1],collar=key[2])
        metrics.extend(metric_rows(old,new))
        profiles.extend(profile_rows(item['baseline_paths']['ghia_profiles'],item['variant_paths']['ghia_profiles'],identity))
        pairs.append(dict(**identity,baseline_run_id=old['run_id'],stratified_run_id=new['run_id'],
            closure_cap=item['closure_cap'],matched_residual_point_evaluations=item['residual_point_work'],
            baseline_checkpoint_sha256=old['source']['checkpoint_sha256'],stratified_checkpoint_sha256=new['source']['checkpoint_sha256'],
            initial_model_sha256=item['receipt']['initial_model_sha256'],weighted_objective_sha256=item['receipt']['objective_sha256'],
            lbfgs_accepted_updates=item['receipt']['lbfgs_completed'],lbfgs_budget_aborts=item['receipt']['lbfgs_aborted_steps'],
            baseline_optimizer_seconds=item['baseline_history']['elapsed_seconds'],stratified_optimizer_seconds=item['variant_history']['elapsed_seconds'],
            time_comparison='optimizer-step seconds only; excludes failed allocations, scheduler/requeue time, checkpoint I/O and audits; not campaign wall time; equal GPU hardware is not established',
            baseline_prediction=old['prediction'],stratified_prediction=new['prediction'],
            baseline_profiles=old['ghia_profiles'],stratified_profiles=new['ghia_profiles'],
            training_provenance=item['provenance'],baseline_control_provenance=item['baseline_control_provenance'],
            sampling_coverage=item['coverage'],
            accepted_endpoint_audit=item['frozen']['accepted_endpoint_audit'],
            pressure_validation='unavailable; retained CFD reference has no pressure field'))
    for entry in inputs.values(): checked_entry(root,entry)
    output.mkdir(parents=True,exist_ok=True)
    metric_path=output/'paired_metrics.csv';profile_path=output/'paired_signed_profiles.csv'
    atomic_csv(metric_path,metrics);atomic_csv(profile_path,profiles)
    report=dict(schema_version=1,campaign_kind='ghia_uniform_vs_stratified_matched_work',
        created_at_utc=datetime.now(timezone.utc).isoformat(),comparison_source_sha256=sha(__file__),
        status='all12 frozen matched-work pairs compared; no automatic scientific acceptance',inputs=inputs,
        outputs=dict(paired_metrics=file_entry(root,metric_path),paired_signed_profiles=file_entry(root,profile_path)),
        freeze_barrier=barrier,runtime_events=runtime_events,evaluation_provenance=evaluation_provenance,
        evaluation_chronology=evaluation_chronology,records=pairs,collar_comparisons=collar_groups(pairs),
        descriptive_three_seed_summaries=descriptive_summary(metrics),
        metric_policy='Signed biases retained; their error comparison uses absolute bias alongside RMS. Lower error is better. No combined winner score.',
        sampling_comparison='original fixed uniform samples versus conditional-uniform strata with unbiased global area weights',
        initial_state_evidence='same original seeded model source; exact initialization equivalence admitted before fitting',
        limits=['one bounded intervention; no additional architecture/sampling search',
                'three paired seeds are descriptive, not a universal optimizer/accuracy claim',
                'finite punctured-domain fit does not establish full-domain corner integrability',
                'CFD is a numerical reference; native sampled support and Ghia screening are unchanged',
                'optimizer-step seconds exclude failed allocations and requeue/scheduler time; they are not campaign wall time or a controlled hardware speed comparison',
                'checkpoint bytes were verified by the upstream GPU scorer/bridge; this compact consumer verifies their bound identities',
                'completed-pass target-access chronology does not erase an earlier failed partial evaluation; supplied attempt/forensic receipts remain explicit'])
    atomic_json(destination,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-manifest',type=Path,default=ROOT/'results/cavity_pinn_campaign/evaluation_manifest.json')
    parser.add_argument('--variant-manifest',type=Path,default=ROOT/'results/cavity_pinn_stratified/evaluation_manifest.json')
    parser.add_argument('--experiment-manifest',type=Path,default=ROOT/'results/cavity_pinn_stratified/experiment_manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/cavity_pinn_stratified/comparison')
    args=parser.parse_args();report=execute(args.baseline_manifest,args.variant_manifest,args.experiment_manifest,args.output)
    print(json.dumps(dict(status=report['status'],pairs=len(report['records']),outputs=report['outputs']),indent=2))


if __name__=='__main__':main()
