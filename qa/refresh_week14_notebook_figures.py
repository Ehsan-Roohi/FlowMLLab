"""Refresh only presentation PNGs in the executed notebook, without retraining.

Cell sources, execution records, numeric/HTML outputs and metadata are preserved.
This is not represented as a new numerical execution.
"""
import base64
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/week14/W14_pyCALC_RANS_PINN_NN.ipynb'
notebook=json.loads(path.read_text(encoding='utf-8'))
count=0
for cell in notebook['cells']:
    source=''.join(cell.get('source',[]))
    for name in ['profiles_legend_below','coefficients','gap','convergence','inverse']:
        old_name='profiles' if name=='profiles_legend_below' else name
        if "'"+old_name+".png'" not in source and "'"+name+".png'" not in source:
            continue
        if old_name!=name:
            cell['source']=[part.replace("'profiles.png'","'profiles_legend_below.png'") for part in cell['source']]
        outputs=[o for o in cell.get('outputs',[]) if 'image/png' in o.get('data',{})]
        assert len(outputs)==1, (cell['id'],name)
        outputs[0]['data']['image/png']=base64.b64encode(
            (ROOT/'results/week14_validation'/f'{name}.png').read_bytes()).decode('ascii')
        count+=1
assert count==5, count
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
print('Refreshed five figure outputs only; no new scientific execution claimed.')
