"""Strict completed-pass evaluation of frozen stratified PINNs.

The v1 training/checkpoint code is preserved. This adapter restores serialized
samples and final parameters exactly; it never regenerates the sampled objective
and exposes no optimizer continuation. Same-runtime restart tests do not certify
bitwise cross-platform sampler regeneration or cross-platform training continuation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {(re, seed, collar) for re in (100, 7500) for seed in (7, 23, 61) for collar in (.01, .02)}
CORE_SHA256 = '1d7118052fcf0eea1550433484cb799a598e8a4c23130b1c3a85cd2b3610817c'
ENGINE_RAW_SHA256 = 'e1eeab53d6e284b8e90ba50ca7200299ad7b62db42450a2c6110a65edb2678b3'


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def canonical_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode()).hexdigest()


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dependencies():
    core_path = ROOT/'qa/evaluate_cavity_pinn_campaign.py'
    require(sha(core_path) == CORE_SHA256, 'Original scorer changed')
    engine_path = ROOT/'flowmllab/cavity_pinn_stratified.py'
    require(sha(engine_path) == ENGINE_RAW_SHA256, 'Original stratified training source changed')
    return (module_at(core_path, 'immutable_pinn_score_functions'),
            module_at(engine_path, 'immutable_stratified_training_source'))


class FrozenPredictor:
    """Inference only; deliberately no step(), optimizer, save(), or resume()."""
    def __init__(self, model, config, frozen, device, restoration):
        self.model, self.config, self.frozen = model, config, frozen
        self.device, self.restoration = device, restoration
        self.finished = True

    def runtime(self):
        import torch
        cuda = self.device.type == 'cuda'
        return dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=str(self.device),
                    gpu_name=torch.cuda.get_device_name(self.device) if cuda else None,
                    hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    role='frozen inference only', restoration=self.restoration)


def load_frozen(path, engine, device='cpu', config=None):
    import torch
    path = Path(path); before = sha(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(sha(path) == before, 'Checkpoint changed while reading')
    payload = {key: value for key, value in state.items() if key != 'checkpoint_state_sha256'}
    require(engine.state_hash(payload) == state['checkpoint_state_sha256'], 'Complete serialized checkpoint hash failed')
    require(state['schema'] == engine.SCHEMA and state['source_sha256'] == engine.source_hash()
            and state['baseline_source_sha256'] == engine.BASE_SOURCE_SHA256, 'Frozen source/schema changed')
    require(state['source_raw_sha256'] == sha(engine.__file__)
            and state['baseline_source_raw_sha256'] == sha(engine.BASE_PATH), 'Original staged source bytes changed')
    restored = engine.Config(**state['config'])
    require(engine.config_hash(restored) == state['config_sha256']
            and (config is None or restored == config), 'Frozen configuration differs')
    require(engine.tensor_hash(state['model']) == state['model_sha256'], 'Saved final model hash failed')
    require(engine.state_hash(state['optimizer']) == state['optimizer_sha256'], 'Saved optimizer state hash failed')
    objective = engine.tensor_hash(dict(points=state['points'], weights=state['weights'], stratum_ids=state['stratum_ids']))
    require(objective == state['objective_hash'], 'Saved weighted objective hash failed')
    require(engine.tensor_hash(dict(points=state['audit_points'])) == state['audit_points_sha256'],
            'Saved independent audit locations hash failed')
    require(state['closure_calls'] == restored.closure_budget
            and state['residual_point_evaluations'] == restored.closure_budget*restored.collocation,
            'Only complete budget-matched endpoints can be evaluated')
    require(state['adam_completed'] == restored.adam_steps, 'Warm stage incomplete')
    frozen = state['frozen']
    require(frozen is not None and frozen['model_sha256'] == state['model_sha256']
            and frozen['config_sha256'] == state['config_sha256']
            and frozen['objective_sha256'] == objective
            and frozen['initial_model_sha256'] == state['initial_model_sha256']
            and frozen['baseline_contract_sha256'] == engine.state_hash(state['baseline_contract'])
            and frozen['closure_calls'] == state['closure_calls'], 'Frozen endpoint identity differs')
    require(frozen['reference_labels_used'] is False, 'Frozen fit used reference labels')
    for name in ('points', 'weights', 'audit_points'):
        require(bool(torch.isfinite(state[name]).all()), 'Nonfinite serialized sampling data')
    require(state['points'].shape == (restored.collocation, 2)
            and state['weights'].shape == state['stratum_ids'].shape == (restored.collocation,),
            'Saved objective shape differs')
    require(bool((state['weights'] > 0).all()) and abs(float(state['weights'].sum())-1) < 1e-14,
            'Saved objective area weights invalid')
    require(torch.equal(torch.bincount(state['stratum_ids'], minlength=3),
                        torch.tensor([restored.collocation//4, restored.collocation//4, restored.collocation//2])),
            'Saved stratum quotas changed')
    if restored.study_kind == 'controlled_variant':
        contract = state['baseline_contract']
        require(contract['baseline_closure_calls'] == restored.closure_budget
                and contract['baseline_config'] == {k: asdict(restored)[k] for k in contract['baseline_config']},
                'Saved baseline budget/physics contract differs')
    # Construct only the parameter containers, preserving the process CPU RNG.
    # Load the actual saved parameters; do not invoke Trainer or any sampler.
    with torch.random.fork_rng(devices=[]):
        model = engine.CavityPINN(restored.width, restored.depth, restored.reynolds)
    model.load_state_dict(state['model'], strict=True)
    model = model.to(torch.device(device))
    require(engine.tensor_hash(model.state_dict()) == frozen['model_sha256'], 'Loaded inference parameters differ')
    rng = {name: state[name] for name in ('sampler_rng', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda')}
    restoration = dict(adapter_sha256=sha(__file__), checkpoint_sha256=before,
        complete_checkpoint_state_sha256=state['checkpoint_state_sha256'],
        serialized_objective_sha256=objective, serialized_rng_state_sha256=engine.state_hash(rng),
        fresh_sampler_regeneration_required=False, saved_objective_tensors_preserved=True,
        training_continuation_certified=False,
        policy='Read-only inference from exact saved final tensors; all complete-state hashes verified')
    require(sha(path) == before, 'Checkpoint changed during inference restoration')
    return FrozenPredictor(model, restored, frozen, torch.device(device), restoration)


def prepare_run(run, output, core, engine, device):
    run = Path(run)
    config = core.read_json(run/'config.json')
    require((config['reynolds'], config['seed'], config['collar']) in EXPECTED
            and config['study_kind'] == 'controlled_variant', 'Run outside declared twelve-case study')
    frozen = core.read_json(run/'frozen_manifest.json')
    receipt = core.read_json(run/'current_receipt.json')
    for name, digest in frozen['artifact_sha256'].items():
        require(Path(name).name == name and sha(run/name) == digest, 'Original frozen artifact hash failed: '+name)
    require(receipt['finished'] and sha(run/'checkpoint.pt') == receipt['checkpoint_sha256']
            == frozen['checkpoint_sha256'], 'Completed recovery identity failed')
    predictor = load_frozen(run/'checkpoint.pt', engine, device=device, config=engine.Config(**config))
    require(predictor.frozen == frozen['frozen_model'] and config == frozen['config'],
            'Restored endpoint differs from original frozen export')
    identifier = core.run_id(config); directory = output/'completed_pass/runs'/identifier
    reference = ROOT/'results/cavity_highre_ghia/n512'/f"re{int(config['reynolds'])}.npz"
    coordinates = core.native_coordinates(reference)
    ghia_source = ROOT/'flowmllab/ghia.py'
    identity = dict(run_id=identifier, Re=config['reynolds'], seed=config['seed'], config=config,
        model_sha256=predictor.frozen['model_sha256'], config_sha256=predictor.frozen['config_sha256'],
        source_canonical_sha256=engine.source_hash(), projection_runtime=predictor.runtime(), evaluation_device=device,
        ghia_source_sha256=sha(ghia_source), evaluator_source_sha256=sha(core.__file__),
        evaluation_wrapper_sha256=sha(__file__), restoration=predictor.restoration,
        checkpoint_sha256=receipt['checkpoint_sha256'],
        original_freeze_manifest_sha256=sha(run/'frozen_manifest.json'),
        original_prediction_sha256=frozen['prediction_sha256'], original_frozen_at_utc=frozen['frozen_utc'],
        training_runtime_history=receipt.get('runtime_history', []))
    existing = directory/'projection_freeze.json'
    if existing.exists():
        # Reuse the original immutable numerical projection across evaluation
        # allocations. Preserve its true runtime; do not relabel it as a new run.
        old = core.read_json(existing)['identity']
        identity['projection_runtime'] = old['projection_runtime']
    projection, prediction = core.freeze_projection(directory, identity, coordinates,
        core.coordinate_literals(ghia_source),
        lambda xy: engine.predict(predictor.model, xy, device, predictor.config.chunk_size))
    return dict(pending=False, run=run, run_id=identifier, directory=directory, reference=reference,
        coords=coordinates, projection=projection, prediction=prediction, config=config,
        physics=core.read_json(run/'final_physics_audit.json'))


def preserve_prior_attempt(output, runs, core, forensics):
    receipt_path = output/'prior_evaluation_attempt.json'
    current = output/'evaluation_manifest.json'
    if receipt_path.exists():
        old = core.read_json(receipt_path)
        require(old.get('sampler_forensics') == forensics, 'Forensic provenance changed after repair')
        for entry in old['all_training_endpoints']:
            matching = [Path(run)/'frozen_manifest.json' for run in runs
                        if core.read_json(Path(run)/'config.json') == entry['config']]
            require(len(matching) == 1 and sha(matching[0]) == entry['manifest_sha256'],
                    'Original trained endpoint changed after prior evaluation')
        return core.file_entry(receipt_path)
    if not current.exists(): return None
    original = core.read_json(current)
    require('evaluation_wrapper_sha256' not in original, 'A completed wrapper manifest lacks its prior-attempt receipt')
    scored = [row for row in original['records'] if row['outcome'] == 'scored']
    require(scored and len(scored) < 12, 'Expected a documented partial original evaluation')
    for row in scored:
        for name in ('prediction', 'reference', 'history', 'ghia_profiles'): core.checked_entry(row[name])
    endpoint_records = []
    for run in runs:
        path = Path(run)/'frozen_manifest.json'; frozen = core.read_json(path)
        endpoint_records.append(dict(config=frozen['config'], manifest_sha256=sha(path),
            model_sha256=frozen['frozen_model']['model_sha256'], frozen_utc=frozen['frozen_utc']))
    first_access = min(row['score']['reference_accessed_at_utc'] for row in scored)
    from datetime import datetime
    stamp = lambda text: datetime.fromisoformat(text.replace('Z', '+00:00'))
    require(max(stamp(row['frozen_utc']) for row in endpoint_records) < stamp(first_access),
            'A training endpoint was not frozen before prior target access')
    receipt = dict(schema_version=1, sampler_forensics=forensics, original_manifest_sha256=sha(current),
        original_manifest=original, prior_scored_case_count=len(scored),
        first_prior_reference_accessed_at_utc=first_access, all_training_endpoints=endpoint_records,
        training_endpoints_frozen_before_prior_target_access=True,
        native_projection_barrier_completed=False, prior_reference_values_read=True,
        model_or_fitting_changes_after_prior_access=False,
        interpretation=f'The first evaluation scored {len(scored)} frozen cases and rejected {len(original["records"])-len(scored)} at platform-dependent annular-coordinate regeneration; this record is retained, not treated as a fully blinded native-projection pass')
    core.write_json(receipt_path, receipt)
    return core.file_entry(receipt_path)


def completed_pass(runs, output, core, engine, device, forensics=None):
    keys = []
    for run in runs:
        cfg = core.read_json(Path(run)/'config.json')
        keys.append((cfg['reynolds'], cfg['seed'], cfg['collar']))
    require(len(keys) == len(set(keys)) == 12 and set(keys) == EXPECTED, 'Exactly all twelve original frozen runs are required')
    prior = preserve_prior_attempt(output, runs, core, forensics)
    prepared = []
    # Any failure here aborts before importing target values or scoring any case.
    for run in runs:
        prepared.append(prepare_run(run, output, core, engine, device))
    require(len(prepared) == 12, 'Native projection barrier incomplete')
    prepared.sort(key=lambda item: (item['config']['reynolds'], item['config']['seed'], item['config']['collar']))
    predictions = [dict(run_id=item['run_id'], at_utc=item['projection']['at_utc'],
                        prediction_sha256=item['projection']['prediction_sha256'],
                        model_sha256=item['projection']['identity']['model_sha256']) for item in prepared]
    barrier_path = output/'completed_pass/projection_barrier.json'
    if barrier_path.exists():
        barrier = core.read_json(barrier_path)
        require(barrier['predictions'] == predictions and barrier['evaluation_wrapper_sha256'] == sha(__file__)
                and barrier['prior_evaluation_receipt'] == prior and barrier.get('sampler_forensics') == forensics,
                'Completed-pass projection barrier changed')
        target_accessed = barrier['target_access_started_at_utc']
    else:
        target_accessed = core.utc()
        core.write_json(barrier_path, dict(schema_version=1, evaluation_wrapper_sha256=sha(__file__),
            scope='All twelve native predictions frozen before THIS completed-pass target read',
            target_access_started_at_utc=target_accessed, prior_evaluation_receipt=prior,
            sampler_forensics=forensics, predictions=predictions))
    if prior: core.checked_entry(prior)
    if forensics: core.checked_entry(forensics)
    ghia = core.module_at(ROOT/'flowmllab/ghia.py', 'stratified_completed_pass_targets')
    records = [core.score_prepared(item, output/'completed_pass', ghia, target_accessed) for item in prepared]
    require(len(records) == 12 and all(row['outcome'] == 'scored' for row in records), 'Completed scoring pass incomplete')
    ordered = sorted(records, key=lambda row: (row['Re'], row['seed'], row['collar']))
    manifest = dict(schema_version=1, campaign_kind='ghia_constant_lid',
        status='All twelve immutable endpoints scored in a complete post-freeze evaluation pass',
        records=ordered, collar_comparisons=core.collar_comparisons(ordered), expected_reynolds=[100, 7500],
        evaluation_wrapper_sha256=sha(__file__), original_evaluator_source_sha256=sha(core.__file__),
        projection_barrier=core.file_entry(output/'completed_pass/projection_barrier.json'),
        information_policy='All twelve trained endpoints predate any recorded target access; prior partial evaluation is disclosed and no endpoint was changed',
        projection_policy='All twelve native projections preceded target reads in the completed pass; not a claim of no earlier partial target access',
        restoration_policy='Exact serialized weighted objective and model state; evaluation only, no cross-platform optimizer-continuation certification',
        prior_evaluation_receipt=prior, sampler_forensics=forensics, model_or_fitting_changes_after_prior_access=False,
        limitations='CFD has no pressure truth; incomplete original evaluation retained; no model selection or retraining')
    destination = output/'evaluation_manifest.json'
    if destination.exists() and core.read_json(destination).get('evaluation_wrapper_sha256'):
        old = core.read_json(destination)
        # The barrier ledger may be inspected again without inventing a new
        # scientific pass; retain the first completed manifest unchanged.
        require(old == manifest, 'Refusing to replace different completed frozen evidence')
        core.checked_entry(old['projection_barrier'])
        return destination
    core.write_json(destination, manifest)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/cavity_pinn_stratified')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--forensics', type=Path, default=ROOT/'results/cavity_pinn_stratified/sampler_restore_forensics.json')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT.resolve()), 'Evaluation output must stay in the checkout')
    output.mkdir(parents=True, exist_ok=True)
    core, engine = dependencies()
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    core.validate_evaluation_device(args.device)
    require(args.forensics.is_file(), 'The compact platform-regeneration forensic receipt is required')
    forensic_entry = core.file_entry(args.forensics)
    require(forensic_entry['sha256'] == '834814f4aec9c6542de4e0096a46795bf40de103b5a5ca5d3c75d76a9c29af68',
            'The independently inspected forensic receipt changed')
    with core.writer_lock(output/'evaluation.lock'):
        path = completed_pass(args.run, output, core, engine, args.device, forensic_entry)
    print(json.dumps(dict(manifest=str(path), sha256=sha(path), completed_cases=12)))


if __name__ == '__main__': main()

