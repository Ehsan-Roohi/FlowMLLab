"""Rebuild the existing RGB field atlas and add matched pressure comparisons.

Uses only checksum-verified retained fields; no CFD trajectory is launched.
Run before qa.build_week01_2_lecture and qa.validate_week01_2_figure_guide.
"""
from pathlib import Path
import hashlib
import json
import re as regex
import sys
import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qa.refresh_week01_2_figures import (OUT, archived_rows, fields, refinement,
    integrate, CAPTIONS, caption_tex, supplementary_field_caption)
from flowmllab.pressure_velocity_visuals import color_normalization, values_for


def main():
    rows=archived_rows()
    def arrays_digest():
        h=hashlib.sha256()
        for r in rows:
            h.update(json.dumps(r['config'],sort_keys=True).encode())
            for key in sorted(k for k,v in r.items() if isinstance(v,np.ndarray)):
                a=np.ascontiguousarray(r[key])
                h.update(key.encode());h.update(str(a.dtype).encode())
                h.update(str(a.shape).encode());h.update(a.tobytes())
        return h.hexdigest()
    before=arrays_digest();names=[]
    scales={}
    for key in ['speed','p','omega']:
        norm=color_normalization(key,[values_for(r,key) for r in rows])
        scales[key]={'minimum':float(norm.vmin),'maximum':float(norm.vmax),
                     'normalization':'linear' if key=='speed' else 'symmetric log',
                     'linear_threshold':None if key=='speed' else norm.linthresh}
        for re in [100,400,1000]:
            fields(rows,key,re=re,cells=64 if re==100 else 32)
            names.append(f'field_{key}_re{re}')
    for re in [400,1000]:
        refinement(rows,re);names.append(f'refinement_re{re}')
        name=f'field_p_re{re}_n64'
        fields(rows,'p',re=re,cells=64,name=name);names.append(name)
    assert arrays_digest()==before, 'Rendering modified a native data array'
    source=ROOT/'lectures/source/week01_2_pressure_velocity.tex'
    text=source.read_text(encoding='utf-8')
    addition=r'\input{week01_2_pressure_contours.tex}'
    if addition not in text:
        anchor=r'\topic{23}{References}'
        assert text.count(anchor)==1
        text=text.replace(anchor,addition+'\n\\newpage\n'+anchor)
        source.write_text(text,encoding='utf-8',newline='\n')
    integrate()
    extra_path=OUT/'supplementary_captions.json'
    extra=json.loads(extra_path.read_text(encoding='utf-8'))
    for re in [400,1000]:
        for key in ['speed','p','omega']:
            extra[f'field_{key}_re{re}']=supplementary_field_caption(re,key)
    extra_path.write_text(json.dumps(extra,indent=2)+'\n',encoding='utf-8',newline='\n')
    from qa.integrate_week01_2_figure_guide import main as make_index, URL
    make_index()
    index=json.loads((OUT/'figure_index.json').read_text(encoding='utf-8'))
    path=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
    book=nbformat.read(path,4)
    old_ids=[c.id for c in book.cells]
    for cell in book.cells:
        if cell.cell_type!='markdown':continue
        # Only the existing one-paragraph caption is replaced; teaching text,
        # exercises, cell identities and numerical outputs are preserved.
        for name in names:
            e=index[name]
            if e['image'] not in cell.source:continue
            pattern=r'\*\*Figure '+regex.escape(e['number'])+r'\.\*\*[^\n]*'
            cell.source,count=regex.subn(pattern,lambda _:f"**Figure {e['number']}.** {e['caption']}",cell.source)
            assert count==1,(cell.id,name,count)
        cell.source=cell.source.replace('Figures 1-23 share identifiers','Figures 1-25 share identifiers')
    for re in [400,1000]:
        id=f'rgb-pressure-atlas-re{re}'
        if any(c.id==id for c in book.cells):continue
        e=index[f'field_p_re{re}_n64']
        text=(f'### Matched pressure comparison: Re={re}\n\n'
              f"![Pressure and streamlines at Re={re}]({URL}{e['image']})\n\n"
              f"**Figure {e['number']}.** {e['caption']}\n\n"
              'Compare these fields with Figure 15 at Re=100: the unit-square geometry, '
              'unit lid speed, zero-mean pressure gauge, spacing and color scale are common. '
              'The other walls are stationary. Pressure is divided by $\\rho U_{\\mathrm{lid}}^2$. '
              'Filled contours interpolate native samples without smoothing; nearest native '
              'values fill the half-cell edge bands. The colorbar reports physical values, '
              'although its color mapping is nonlinear.\n\n'
              '**Interpretation task.** Locate the low-pressure vortex core and upper-right '
              'pressure rise. Identify a change with Reynolds number and a difference between '
              'formulations. Explain why agreeing streamlines and small residuals do not '
              'establish pressure accuracy. Consult the pressure-gradient discrepancy in '
              'Table 5 and distinguish that consistency diagnostic from reference error.')
        at=next(i for i,c in enumerate(book.cells) if c.id=='visual-caption-resources')
        book.cells.insert(at,nbformat.v4.new_markdown_cell(text,id=id))
    assert all(any(c.id==id for c in book.cells) for id in old_ids)
    nbformat.validate(book)
    path.write_text(nbformat.writes(book),encoding='utf-8',newline='\n')
    from PIL import Image
    outputs={}
    for name in names:
        with Image.open(OUT/(name+'.png')) as im:
            assert im.mode=='RGB'
            assert im.width==1980
        outputs[name]={ext+'_sha256':hashlib.sha256((OUT/(name+'.'+ext)).read_bytes()).hexdigest()
                       for ext in ['pdf','png']}
    receipt={'verified_archive_cases':len(rows),'native_arrays_sha256_before_and_after':before,
             'fresh_cfd_runs':0,'native_arrays_changed':False,'scales':scales,
             'rendering':'129 filled contour levels; no filtering or clipping; native edge bands',
             'png_mode':'RGB','raster_dpi':300,'figures':outputs}
    destination=ROOT/'results/week01_2_visuals';destination.mkdir(exist_ok=True)
    (destination/'render_validation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'rgb_figures':len(names),'native_arrays_unchanged':True,
                      'notebook_cells':len(book.cells),'fresh_cfd_runs':0}))


if __name__=='__main__':main()
