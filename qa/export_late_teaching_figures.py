"""Export selected actual notebook outputs for the corresponding lecture sources.

This reads saved executed outputs only; it performs no training or field editing.
The manifest binds every exported image to its notebook, cell and output hash.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/late_course_teaching'
CASES={
    '15':('week15/W15_Complete_Geometry_Generalization.ipynb',
          {'late-operator-controls':'geometry','late-operator-fit':'history','late-operator-test':'test'}),
    '16':('week16/W16_Supersonic_Shape_Optimization.ipynb',
          {'late-w16-all-models-code':'validation_models'}),
    '17':('week17/W17_CFD_Transformer.ipynb',{'w17-11':'sensors'}),
    '18':('week18/W18_CFD_Transformer.ipynb',{'w18-09':'causal_response'}),
    '19':('week19/W19_CFD_Transformer.ipynb',{'w19-09':'reconstruction'}),
    '20':('week20/W20_CFD_Transformer.ipynb',{'w20-09':'classroom_fit','w20-11':'retained_sensing'}),
    '21':('week21/W21_CFD_Transformer.ipynb',{'w21-05':'full_rollout_error','w21-09':'rollout'}),
    '22':('week22/W22_CFD_Transformer.ipynb',{'w22-07':'transfer_label_curves','w22-09':'transfer'}),
    'bridge':('optional_geometry_time_bridge/Geometry_Time_Operator_Bridge.ipynb',
              {'late-bridge-feedback-plot-code':'feedback_phase'}),
}

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weeks',nargs='+',default=list(CASES),choices=list(CASES))
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'figure_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for week in args.weeks:
        relative,cells=CASES[week];path=ROOT/'notebooks'/relative
        data=path.read_bytes();nb=nbf.reads(data.decode(),as_version=4)
        rows=[]
        for cell_id,stem in cells.items():
            cell=next(c for c in nb.cells if c.id==cell_id)
            if cell.execution_count is None:raise RuntimeError(f'{path.name}:{cell_id} is not executed')
            images=[o.data['image/png'] for o in cell.outputs if 'image/png' in o.get('data',{})]
            if not images:raise RuntimeError(f'{path.name}:{cell_id} has no figure')
            for i,encoded in enumerate(images,1):
                png=base64.b64decode(encoded)
                target=OUT/f'week{week}_{stem}_{i:02d}.png';target.write_bytes(png)
                rows.append(dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(png),
                                 cell_id=cell_id,image_index=i,code_sha256=sha(cell.source.encode())))
        manifest[week]=dict(notebook=path.relative_to(ROOT).as_posix(),notebook_sha256=sha(data),figures=rows)
        print('Exported',week,len(rows),'executed figures',flush=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')

if __name__=='__main__':main()
