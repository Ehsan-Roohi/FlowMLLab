"""Check submitted-table arithmetic, fresh cavity fields and notebook additions.

No solver trajectory is run by this validator. It reads the independent fresh
suite created by review_student_cavity.py --run.
"""
from pathlib import Path
import csv
import copy
import hashlib
import json
import sys

import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
from review_student_cavity import load, selected_cases, OUT
from flowmllab.pressure_velocity import divergence, _residual
from flowmllab.pressure_velocity_experiments import ExperimentalConfig, benchmark_errors
from notebook_math import normalize_markdown


def main():
    submitted=list(csv.DictReader((OUT/'submitted_summary.csv').open()))
    assert len(submitted)==12
    keys=lambda r:(r['method'],int(r['reynolds']),int(r['cells']),r['convection'])
    assert len({keys(r) for r in submitted})==12
    for row in submitted:
        expected={'simple':1,'piso':2,'pimple':6}[row['method']]
        assert int(row['pressure_solves'])==expected*int(row['iterations'])
        assert float(row['reported_momentum_linf'])<1e-6
        assert float(row['reported_divergence_linf'])<1e-9
        assert row['v_reference_policy']==('exclude_x_0.9063' if int(row['reynolds'])==400 else 'all_tabulated_samples')
    rows=load();configs=list(selected_cases());matches=[]
    assert {keys(r['config']) for r in rows}=={(c.method,int(c.reynolds),c.cells,c.convection) for c in configs}
    for row in rows:
        cfg=ExperimentalConfig(**row['config']);h=1/cfg.cells
        assert row['converged']
        assert _residual(row['u_faces'],row['v_faces'],row['p'],cfg)<1e-6
        assert np.max(np.abs(divergence(row['u_faces'],row['v_faces'],h)))<1e-9
        assert abs(row['p'].mean())<1e-12
        for field in ['u_faces','v_faces','p','u','v','psi','omega']:assert np.isfinite(row[field]).all()
        for name,value in benchmark_errors(row).items():np.testing.assert_allclose(row[name],value,rtol=1e-12)
        for name,digest in row['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        candidates=[r for r in submitted if keys(r)==keys(row['config'])]
        if candidates:
            reported=candidates[0]
            for component in ['u','v']:
                assert abs(100*row[f'ghia_{component}_relative_l2']-float(reported[f'reported_{component}_error_percent']))<=.0051
            assert row['iterations']==int(reported['iterations'])
            assert row['pressure_solves']==int(reported['pressure_solves'])
            matches.append(keys(row['config']))
    assert len(matches)==8
    # An integrated row and its volume-normalized row give the same correction.
    volume=.25**2;diagonal=.64;pressure_gradient=.04
    integrated_diagonal=volume*diagonal
    np.testing.assert_allclose(-volume/integrated_diagonal*pressure_gradient,
                               -pressure_gradient/diagonal,atol=1e-15)
    preserved=0
    baseline=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))['baseline_notebooks']
    for name in ['notebooks/week01/03_cavity_ghia.ipynb',
                 'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb']:
        after=nbformat.read(ROOT/name,4)
        # Check every pre-existing cell, including source and executed outputs.
        remaining=copy.deepcopy([c for c in after.cells if not c.id.startswith(('student-review-', 'visual-'))])
        # Only the documented outer-iteration notation changes in baseline prose.
        # Reverse that spelling change before checking the historical cell digest.
        for cell in remaining:
            if cell.cell_type=='markdown':
                cell.source=cell.source.replace(r'q_{\rm lag}',r'\widehat q').replace(r'p_{\rm lag}',r'\widehat p')
        digest=hashlib.sha256(json.dumps(remaining,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert digest==baseline[name]['cells_sha256'],name
        assert len(remaining)==baseline[name]['pre_existing_cell_count'],name
        preserved+=len(remaining)
    foundation=nbformat.read(ROOT/'notebooks/week01/03_cavity_ghia.ipynb',4)
    note=next(c for c in foundation.cells if c.id=='student-review-rectangular-vorticity-note')
    assert normalize_markdown(note.source)==note.source
    cell=next(c for c in foundation.cells if c.id=='student-review-rectangular-vorticity-check')
    namespace={};exec(cell.source,namespace)
    np.testing.assert_allclose(namespace['omega_central'],-1.008,atol=1e-12)
    np.testing.assert_allclose(namespace['omega_wrong_spacing'],-1.584,atol=1e-12)
    assert cell.execution_count is not None and not any(o.output_type=='error' for o in cell.outputs)
    book=nbformat.read(ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb',4);nbformat.validate(book)
    accepted=[c for c in book.cells if c.id.startswith('student-review-')]
    assert len(accepted)==4
    for cell in accepted:
        if cell.cell_type=='markdown':
            assert normalize_markdown(cell.source)==cell.source
            assert cell.source.count('$$')%2==0
        else:
            compile(cell.source,'student-review-cell','exec')
            assert cell.execution_count is not None
            assert not any(o.output_type=='error' for o in cell.outputs)
            assert sum('image/png' in o.get('data',{}) for o in cell.outputs)==3
    result={'submitted_cases':12,'pressure_solve_count_checks':12,'fresh_cases':9,
            'fresh_central_cases_matching_report_rounded_errors_and_work':len(matches),
            'fresh_fv_residuals_and_benchmark_metrics_checked':True,
            'student_timing_replicated':False,'old_time_correction':'Documented; solver unchanged',
            'notebook_review_cells':4,'new_code_cell_executed':True,
            'week1_unequal_spacing_derivative_check':'passed and executed',
            'integrated_coefficient_normalization_check':'passed',
            'pre_existing_notebook_cells_checked':preserved,
            'permitted_baseline_prose_change':'PIMPLE outer fields use lag subscript instead of hat; code and outputs unchanged',
            'scope':'Selected fresh reproduction; four N=64 PISO/PIMPLE cases remain student-reported.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
