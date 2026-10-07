"""Compare a fresh protocol run to retained evidence with predeclared tolerance."""
import argparse,json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('reference',type=Path);parser.add_argument('candidate',type=Path)
parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
assert json.loads((args.reference/'plan.json').read_text())==json.loads((args.candidate/'plan.json').read_text()),'Protocols differ'
rows=[]
for experiment in ('sensors','forecast','transfer'):
    expected={r['key']:r for r in json.loads((args.reference/f'{experiment}.json').read_text())['rows']}
    actual={r['key']:r for r in json.loads((args.candidate/f'{experiment}.json').read_text())['rows']}
    assert set(expected)==set(actual)
    for key in expected:
        for metric in ('field_relative_l2','representation_floor','in_subspace_relative_l2'):
            a,b=expected[key]['metrics'][metric],actual[key]['metrics'][metric]
            rows.append({'experiment':experiment,'key':key,'metric':metric,'reference':a,'candidate':b,
                         'passed':bool(np.isclose(a,b,rtol=.02,atol=1e-5))})
report={'relative_tolerance':.02,'absolute_tolerance':1e-5,'passed':all(r['passed'] for r in rows),'comparisons':rows}
args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('PASS' if report['passed'] else 'FAIL',len(rows),'numerical comparisons')
raise SystemExit(0 if report['passed'] else 1)
