#!/usr/bin/env python3
"""Reconstruct reported tables from retained, previously verified run summaries.

This is a data-curation script, not a training or checkpoint-replay implementation.
No missing metric or unavailable weight is reconstructed or imputed.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ['protocol','model','capacity','seed','test_cases','velocity_error_pct','pressure_error_pct','vorticity_error_pct','filtered_reverse_iou','reverse_velocity_error_pct','extent_abs_error_H','parameter_entries','real_scalar_parameters','evidence_status','checkpoint_available']

def row(model,capacity,v,p,w,iou,rv=None,extent=None,params=None,real=None,protocol='native_capacity'):
    return dict(zip(FIELDS,[protocol,model,capacity,17,19,v,p,w,iou,rv,extent,params,real,'previously_verified_summary',False]))

ROWS = [
 row('DeepONet','small',17.000066205074912,53.331282565468236,36.500333183690124,.33720510587959796,312.33702288175886,.5956140350877194),
 row('DeepONet','large',13.801042004635459,41.22469801651804,26.6138108905993,.43810971886436734,230.60344374807258,.7166666666666667,5925123,5925123),
 row('Geom-DeepONet','small',12.31530533338848,36.68630630091617,22.248413738451507,.3872077212414149,182.52355053550318,.28508771929824556),
 row('Geom-DeepONet','large',10.817912001358835,30.699403210690146,18.512060265792044,.39554264384468274,119.21265752691971,.21754385964912282,5465089,5465089),
 row('SMART','small',9.942652150204307,39.08893374392861,17.764357466446725,.4909585717097718,158.94817673532586,.28859649122807013),
 row('SMART','large',10.664244978051437,49.33064320212916,21.161891987449245,.46289104255575736,142.8496228268272,.33508771929824566,11380676,11380676),
 row('DoMINO','large',17.57054002661454,None,None,.46017398573563123,params=2941051,real=2941051),
 row('Geo-FNO','large',15.819560527801514,None,None,.3134422407525868,params=6730325,real=12038741),
 row('GeoTransolver','large',20.8319569386934,None,None,.32351670724462617,params=9367179,real=9367179),
]
ORIGINAL = [
 row('Original Transolver','small',14.426055230592427,210.42786307083932,None,.41859670376788155,params=100351,real=100351,protocol='original_transolver'),
 row('Original Transolver scratch','large',5.512791181865492,59.323251824629935,None,.5525939706160283,params=3859011,real=3859011,protocol='original_transolver'),
 row('Original Transolver GeoPT transfer','large',6.115033802233245,57.261428331073965,None,.49113876808703527,params=3859011,real=3859011,protocol='original_transolver'),
]
ORIGINAL[0]['checkpoint_available']=True
CASES = [
 row('DeepONet','large',14.436291694641113,44.25212478637695,28.710662841796875,.5865102639296188,112.20812225341797,.85,5925123,5925123),
 row('Geom-DeepONet','large',13.445074081420898,35.551902770996094,21.132265090942383,.5179817498658078,98.85494232177734,.016666666666666607,5465089,5465089),
 row('SMART','large',10.803688049316406,38.97336959838867,22.218544006347656,.7374677002583979,56.798885345458984,.25,11380676,11380676),
 row('DoMINO','large',16.65517234802246,102.91171264648438,18.33420753479004,.7171336206896551,64.10631561279297,.16666666666666652,2941051,2941051),
 row('Geo-FNO','large',13.180060386657715,37.17748260498047,23.75387191772461,.5588688946015424,95.39813232421875,.5,6730325,12038741),
 row('GeoTransolver','large',23.884170532226562,122.34320068359375,34.36735153198242,.6082981715893109,43.44955825805664,1.8166666666666664,9367179,9367179),
 row('Original Transolver','small',21.5030098,None,None,.650102459,params=100351,real=100351,protocol='original_transolver'),
 row('Original Transolver scratch','large',5.3602338,None,None,.74650913,params=3859011,real=3859011,protocol='original_transolver'),
 row('Original Transolver GeoPT transfer','large',4.5732784,None,None,.800768386,params=3859011,real=3859011,protocol='original_transolver'),
]
for r in CASES:
    r['test_cases']=1
    r['case']='g049_Re100'
    if r['model']=='Original Transolver' and r['capacity']=='small':
        r['checkpoint_available']=True


def write_csv(name,rows,fields):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)

def main():
    write_csv('native_capacity_results.csv',ROWS,FIELDS)
    write_csv('original_transolver_results.csv',ORIGINAL,FIELDS)
    write_csv('g049_Re100_results.csv',CASES,['case']+FIELDS)
    (HERE/'reported_results.json').write_text(json.dumps({
      'schema_version':1,
      'provenance':'Values transcribed from retained run summaries after the transient execution environment reset; not remeasured from checkpoints in this publication.',
      'aggregation':'Arithmetic mean of 19 case-level metrics; not pooled field error. Case table is one illustrative geometry/Re only.',
      'missing_values':'null in JSON, empty in CSV; not zero',
      'precision_note':'Original-Transolver g049 case values were retained at the shown precision; no unavailable digits are inferred.',
      'native_capacity':ROWS,'original_transolver':ORIGINAL,'g049_Re100':CASES,
    },indent=2)+'\n')

if __name__=='__main__':main()
