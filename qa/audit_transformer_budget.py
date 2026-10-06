"""Validation-only checkpoint selection under a doubled forecast training ceiling.

This sensitivity analysis does not replace the frozen 1200-step release protocol.
Test metrics are descriptive; no selection uses them.
"""
from pathlib import Path
from dataclasses import replace,asdict
import json
import numpy as np
import torch
from flowmllab import transformer_course as lab

root=Path(__file__).resolve().parents[1]
out=root/'results/transformer_budget_audit'
if out.exists() and any(out.iterdir()):raise FileExistsError('Use an empty audit directory')
out.mkdir(parents=True,exist_ok=True)
p=replace(lab.Protocol(),max_steps=2400)
cases,_=lab.load_data(root)
record,fields,states=lab.forecast_experiment(cases,p)
lab.json_write(out/'plan.json',asdict(p));lab.json_write(out/'forecast.json',record)
np.savez_compressed(out/'predictions.npz',**{key+'__'+suffix:a for key,v in fields.items() for suffix,a in lab.pack_prediction(v).items()})
torch.save(states,out/'checkpoints.pt')
base=json.loads((root/'results/transformer_course_v3/forecast.json').read_text())
baseline={r['key']:r for r in base['rows']}
summary=[]
for row in record['rows']:
    if 'training' in row:
        summary.append({'key':row['key'],'base_validation_mse':baseline[row['key']]['training']['validation_mse'],
                        'audit_validation_mse':row['training']['validation_mse'],'selected_step':row['training']['selected_step'],
                        'at_budget_boundary':row['training']['at_budget_boundary'],
                        'base_field_relative_l2':baseline[row['key']]['metrics']['field_relative_l2'],
                        'audit_field_relative_l2':row['metrics']['field_relative_l2']})
lab.json_write(out/'summary.json',{'interpretation':'Budget sensitivity, not a proof of convergence; test metrics do not select checkpoints.','rows':summary})
lab.json_write(out/'manifest.json',{'source_hashes':{x:lab.canonical_digest(root/x) for x in ['flowmllab/transformer_course.py','qa/audit_transformer_budget.py']},'files':{x.name:lab.digest(x) for x in out.iterdir() if x.is_file()}})
print(json.dumps(summary,indent=2),flush=True)
