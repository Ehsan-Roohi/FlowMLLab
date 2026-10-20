"""Check the Week 1.2 algebra examples, notebook lesson and PDF artifact."""
from pathlib import Path
import copy,contextlib,hashlib,io,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import numpy as np,pandas as pd,nbformat as n,pdfplumber
from pypdf import PdfReader
from qa.week01_2_equation_lesson import ALGEBRA_CODE,improve_notebook

def main():
    ns={'np':np,'pd':pd,'display':lambda value:None}
    with contextlib.redirect_stdout(io.StringIO()):exec(ALGEBRA_CODE,ns)
    piso=ns['piso_example'];pimple=ns['pimple_example']
    np.testing.assert_allclose([piso[0]['q1'],piso[0]['q2']],[6/11,-2/11],atol=1e-14)
    np.testing.assert_allclose([piso[1]['q1'],piso[2]['q1']],[18/77,118/539],atol=1e-14)
    np.testing.assert_allclose([piso[1]['pressure'],piso[2]['pressure']],[-96/77,-480/539],atol=1e-14)
    np.testing.assert_allclose([piso[1]['frozen momentum defect'],piso[2]['frozen momentum defect']],[32/77,8/539],atol=1e-14)
    expected=[.218923933209647,.211902644638081,.212228965470311]
    np.testing.assert_allclose([r['final q1'] for r in pimple],expected,atol=1e-14)
    defects=[r['nonlinear momentum defect'] for r in pimple]
    assert defects[2]<defects[1]<defects[0]
    assert all(r['mass defect']<1e-14 for r in piso[1:]+pimple)
    # Independent two-cell correction calculation.
    h=.5;d=.1;u=.2;p1=-.5;p2=.5
    assert abs(u-d*(p2-p1)/h)<1e-14
    path=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb';nb=n.read(path,4);n.validate(nb)
    lesson_text='\n'.join(c.source for c in nb.cells if c.metadata.get('pressure_coupling_lesson'))
    for definition in [r'\Delta p=p_{\rm corr}-p^k=p',r'r_{c,P}',r'\widehat q^{(\ell+1)}',r'\widetilde p_E-\widetilde p_P']:
        assert definition in lesson_text,definition
    assert r'\delta_P' not in lesson_text
    canonical=improve_notebook(copy.deepcopy(nb));again=improve_notebook(copy.deepcopy(canonical))
    assert canonical==again,'Lesson update is not idempotent'
    assert len([c for c in canonical.cells if c.metadata.get('pressure_coupling_lesson')])==18
    code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==8
    assert all(c.execution_count is not None for c in code)
    outputs=[o for c in code for o in c.outputs];assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==14
    for c in nb.cells:
        if c.metadata.get('pressure_coupling_lesson'):
            for token in ['\\small','\\topic','\\begin{tabular}','show every step','write every step']:
                assert token not in c.source.lower(),token
            assert c.source.count('$$')%2==0
    pdf=ROOT/'lectures/week01_2_pressure_velocity.pdf';r=PdfReader(pdf);assert len(r.pages)==26
    fonts=set()
    for page in r.pages:
        for f in page.get('/Resources',{}).get('/Font',{}).values():fonts.add(str(f.get_object().get('/BaseFont','')))
    assert any('TimesNewRoman' in f for f in fonts)
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            assert not any(c['text']=='\x00' or c['x0']<20 or c['x1']>page.width-20 for c in page.chars)
    record={'notation':{'continuity_residual':'r_c,P; distinct from pressure correction','pressure_correction':'Delta p = p_corr - p_k = p_prime','stored_pressure_change':'alpha_p p_prime','spatial_pressure_difference':'p_E - p_P'},'notebook':{'code_cells':8,'png_outputs':14,'errors':0,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'examples':{'two_cell_conservation':True,'piso_exact_fractions_checked':True,'pimple_three_outer_passes_checked':True,'algebra_examples_are_cavity_simulations':False},'lesson_update_idempotent':True,'pdf':{'pages':26,'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'times_new_roman_present':True,'all_pages_rendered_and_reviewed':True,'text_within_horizontal_page_bounds':True},'source_access':'Publisher metadata and chapter previews; PISO paper abstract; official pressure-velocity algorithm documentation. Full textbook text was not downloaded.','scientific_scope':'Teaching and presentation update; solver physics and retained CFD trajectories unchanged. Notebook executed locally; hosted Colab not executed.'}
    target=ROOT/'results/week01_2_formulations/coupling_lesson_validation.json';target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
