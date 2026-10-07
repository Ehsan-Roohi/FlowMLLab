"""Read-only exercise audit; --report PATH exclusively creates a new report.

Finite frozen-output and semantic mutations test canonical checks and the actual
notebook check-cell context. They do not prove resistance to every tailored answer.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import ast
import contextlib
import copy
import inspect
import hashlib
import io
import json
from types import SimpleNamespace
import numpy as np
import torch
from flowmllab import transformer_course as lab
from qa.build_transformer_course import notebook


def fixtures():
    """Synthetic audit data, independent of retained experiment results."""
    lab.seed_all(1709)
    rng = np.random.default_rng(1709)
    rep = lab.Representation.fit(rng.normal(size=(30, 96)), 8)
    return dict(np=np, torch=torch, lab=lab, Q=torch.randn(16,8),
                K=torch.randn(16,8), V=torch.randn(16,8),
                decoder=lab.CausalDecoder(), x=torch.randn(2,4,8),
                field=rng.normal(size=(8,12)), representation=rep,
                qr_ids=lab.qr_sensors(rep,16))


def namespace(spec):
    ns = fixtures()
    for task in spec['tasks']:
        exec(task['solution'], ns)
    return ns


def function_name(task):
    return next(node.name for node in ast.parse(task['solution']).body
                if isinstance(node, ast.FunctionDef))


def run_check(checks, ns):
    try:
        exec(checks, ns)
        return True, None
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def notebook_check(spec, task_number, ns):
    """Execute the exact generated cell, including its exception/status handling."""
    cells = [cell.source for cell in notebook(spec, False).cells
             if cell.cell_type == 'code' and cell.source.startswith('if RUN_EXERCISES:')]
    assert len(cells) == len(spec['tasks'])
    ns.update(RUN_EXERCISES=True,
              task_status={i: None for i in range(1,len(spec['tasks'])+1)})
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(cells[task_number-1], ns)
    return ns['task_status'][task_number], stream.getvalue()


def apply_semantic_mutation(ns, key):
    """Install one plausible wrong implementation while preserving its API."""
    if key == 'unsquared_representation_fraction':
        ns['representation_limited'] = lambda metrics: (
            metrics['representation_floor'] /
            max(metrics['field_relative_l2'],1e-20) > .95)
    elif key == 'zero_floor_global_condition':
        original = ns['sensor_audit']
        def wrong_sensor_audit(rep, truth, designs, observations=None):
            rows = original(rep,truth,designs,observations)
            for row in rows:
                row['floor'] = 0.
                row['condition'] = float(np.linalg.cond(rep.modes))
            return rows
        ns['sensor_audit'] = wrong_sensor_audit
    elif key == 'mean_per_step_as_global_error':
        original = ns['rollout_audit']
        def wrong_rollout(*args, **kwargs):
            result = original(*args, **kwargs)
            result['field_error'] = float(np.mean(result['per_step_error']))
            return result
        ns['rollout_audit'] = wrong_rollout
    elif key == 'control_flags_read_pretrained_arm':
        original = ns['audit_transfer_comparisons']
        def wrong_transfer(records):
            rows = original(records)
            for row in rows:
                row['control_selected_step'] = row['pretrained_selected_step']
                row['control_untrained'] = row['pretrained_selected_step'] == 0
                row['control_boundary'] = row['pretrained_boundary']
            return rows
        ns['audit_transfer_comparisons'] = wrong_transfer
    elif key == 'near_ceiling_is_boundary':
        ns['boundary_flag'] = lambda record: record['selected_step'] >= record['ceiling']-20
    else:
        raise ValueError(f'unknown semantic mutation: {key}')


SEMANTIC_MUTATIONS = (
    (22,3,'unsquared_representation_fraction'),
    (19,4,'zero_floor_global_condition'),
    (21,4,'mean_per_step_as_global_error'),
    (22,4,'control_flags_read_pretrained_arm'),
    (20,3,'near_ceiling_is_boundary'),
)


def build_report():
    sources = {week: (ROOT/f'course/lessons/week{week}.json').read_bytes()
               for week in range(17,23)}
    specs = {week: json.loads(data.decode('utf-8')) for week,data in sources.items()}
    rows = []
    for week,spec in specs.items():
        for i,task in enumerate(spec['tasks'],1):
            name = function_name(task)
            reference_pass,reference_error = run_check(task['checks'],namespace(spec))
            ns = namespace(spec)
            original, cache = ns[name], []
            def frozen(*args,_original=original,_cache=cache,**kwargs):
                if not _cache:
                    _cache.append(copy.deepcopy(_original(*args,**kwargs)))
                return copy.deepcopy(_cache[0])
            ns[name] = frozen
            mutation_pass,mutation_error = run_check(task['checks'],ns)
            rows.append(dict(week=week,task=i,function=name,
                reference_pass=reference_pass,reference_error=reference_error,
                frozen_result_rejected=not mutation_pass,mutation_error=mutation_error))
    semantic_rows = []
    for week,task_number,key in SEMANTIC_MUTATIONS:
        spec = specs[week]
        checks = spec['tasks'][task_number-1]['checks']
        reference_pass,reference_error = run_check(checks,namespace(spec))
        ns = namespace(spec)
        apply_semantic_mutation(ns,key)
        mutation_pass,mutation_error = run_check(checks,ns)
        reference_status,reference_output = notebook_check(spec,task_number,namespace(spec))
        ns = namespace(spec)
        apply_semantic_mutation(ns,key)
        mutation_status,mutation_output = notebook_check(spec,task_number,ns)
        semantic_rows.append(dict(week=week,task=task_number,mutation=key,
            reference_pass=reference_pass,reference_error=reference_error,
            rejected=not mutation_pass,mutation_error=mutation_error,
            notebook_reference_pass=reference_status is True,
            notebook_reference_output=reference_output,
            notebook_mutation_rejected=mutation_status is False,
            notebook_mutation_output=mutation_output))

    spec = specs[17]
    signature = list(inspect.signature(namespace(spec)['permutation_error']).parameters)
    signature_check = dict(actual=signature,
        expected=['q','k','v','order','attention_fn'],
        passed=signature == ['q','k','v','order','attention_fn'])
    nb = notebook(spec,False)
    reports = []
    for mode in ('partial','failed'):
        ns = fixtures()
        ns.update(RUN_EXERCISES=True, task_status={i:None for i in range(1,5)},
                  workspace=SimpleNamespace(cleanup=lambda:None))
        for i,task in enumerate(spec['tasks'],1):
            exec(task['solution'] if i<=2 else task['starter'],ns)
        if mode == 'failed':
            ns['student_attention'] = lambda *args: torch.tensor(0.)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            for cell in nb.cells:
                if cell.cell_type=='code' and (cell.source.startswith('if RUN_EXERCISES:')
                        or cell.source.startswith('print("Coding tasks passed:"')):
                    exec(cell.source,ns)
        reports.append(dict(mode=mode,status=ns['task_status'],output=stream.getvalue()))
    status_pass = (reports[0]['status']=={1:True,2:True,3:None,4:None}
                   and 'Coding tasks passed: 2 / 4' in reports[0]['output']
                   and reports[1]['status'][1] is False
                   and reports[1]['status'][3] is None)
    passed = (all(r['reference_pass'] and r['frozen_result_rejected'] for r in rows)
              and all(r['reference_pass'] and r['rejected']
                      and r['notebook_reference_pass'] and r['notebook_mutation_rejected']
                      for r in semantic_rows)
              and signature_check['passed'] and status_pass)
    return dict(scope='Independent synthetic fixtures; instructor functions and actual generated notebook check cells. Full notebook execution is separate.',
        lesson_sha256={str(week):hashlib.sha256(data).hexdigest() for week,data in sources.items()},
        passed=passed,reference_passes=sum(r['reference_pass'] for r in rows),checks=len(rows),
        frozen_result_rejections=sum(r['frozen_result_rejected'] for r in rows),
        semantic_mutations=len(semantic_rows),
        semantic_rejections=sum(r['rejected'] for r in semantic_rows),
        notebook_semantic_rejections=sum(r['notebook_mutation_rejected'] for r in semantic_rows),
        limitation='Finite mutation testing cannot certify all possible constant or tailored solutions; surviving mutations are explicitly reported.',
        rows=rows,semantic_rows=semantic_rows,permutation_callable_contract=signature_check,
        builder_status_checks=reports,builder_status_pass=status_pass)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Create a NEW report file; existing paths are rejected.')
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        parser.error(f'refusing to overwrite existing report: {args.report}')
    report = build_report()
    text = json.dumps(report,indent=2)+'\n'
    if args.report is not None:
        # Exclusive creation also protects against a path appearing during the audit.
        with args.report.open('x',encoding='utf-8',newline='\n') as handle:
            handle.write(text)
    print(text,end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())




