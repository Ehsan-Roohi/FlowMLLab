#!/usr/bin/env python3
"""Validate retained summaries and reproduce small/large comparisons (stdlib).

Usage: python results/week15_capacity_study/analyze_results.py
This does not retrain a model or replay an unavailable checkpoint.
"""
import csv
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
NUMERIC=('velocity_error_pct','pressure_error_pct','vorticity_error_pct','filtered_reverse_iou','reverse_velocity_error_pct','extent_abs_error_H','parameter_entries','real_scalar_parameters')

def load_csv(name):
    with (HERE/name).open(newline='') as f: rows=list(csv.DictReader(f))
    assert len({(r['model'],r['capacity']) for r in rows})==len(rows), 'duplicate run'
    for r in rows:
        assert r['seed']=='17' and r['test_cases']=='19'
        expected_available = name=='original_transolver_results.csv' and r['model']=='Original Transolver' and r['capacity']=='small'
        assert r['checkpoint_available']==str(expected_available), 'availability metadata changed; review ledger'
        for k in NUMERIC:
            r[k]=float(r[k]) if r[k] else None
            if r[k] is not None:
                assert math.isfinite(r[k]) and r[k]>=0,(r['model'],k)
        assert 0<=r['filtered_reverse_iou']<=1
    return rows

def main():
    rows=load_csv('native_capacity_results.csv')
    original=load_csv('original_transolver_results.csv')
    assert len(rows)==9 and len(original)==3
    assert len([r for r in rows if r['capacity']=='large'])==6
    assert all(r['protocol']=='native_capacity' for r in rows)
    assert all(r['protocol']=='original_transolver' for r in original)
    report=[]
    wins={'DeepONet':16,'Geom-DeepONet':14,'SMART':6}
    for model in ('DeepONet','Geom-DeepONet','SMART'):
        small=next(r for r in rows if r['model']==model and r['capacity']=='small')
        large=next(r for r in rows if r['model']==model and r['capacity']=='large')
        report.append({
          'model':model,
          'velocity_error_reduction_pct':100*(small['velocity_error_pct']-large['velocity_error_pct'])/small['velocity_error_pct'],
          'filtered_iou_change':large['filtered_reverse_iou']-small['filtered_reverse_iou'],
          'extent_error_change_H':large['extent_abs_error_H']-small['extent_abs_error_H'],
          'cases_with_lower_velocity_error':wins[model],
          'case_win_count_provenance':'Retained run summary; per-case arrays unavailable for re-counting',
          'case_count':19,
          'seed_count':1,
        })
    native_json=json.loads((HERE/'reported_results.json').read_text())['native_capacity']
    assert len(native_json)==len(rows)
    for r,stored in zip(rows,native_json):
        for k in NUMERIC:
            assert r[k]==stored[k], (r['model'],k,'CSV/JSON mismatch')
    result={
      'meaning':'Positive velocity reduction means lower error for the larger model; negative means deterioration. Changes describe this one seed only.',
      'paired_native_capacity':report,
      'cancelled_controls':['DoMINO small','Geo-FNO small','GeoTransolver small'],
      'cross_protocol_ranking_permitted':False,
      'checkpoint_replay_performed_by_this_script':False,
    }
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__': main()
