"""Validate teaching overlays, accepted-cell provenance and bounded solver fixes."""
from pathlib import Path
import hashlib,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import nbformat
import numpy as np
from notebook_math import normalize_markdown
from pypdf import PdfReader

def main():
    patches=json.loads((ROOT/'qa/notebook_review_patches.json').read_text(encoding='utf-8'))
    accepted=json.loads((ROOT/'qa/course_refresh_overlay_replacements.json').read_text(encoding='utf-8'))['cells']
    affected=[key for key,items in patches.items() if any(p.get('id','').startswith('oct20-') for p in items)]
    code_count=0
    for key in affected:
        n=nbformat.read(ROOT/key,as_version=4);nbformat.validate(n)
        ids=[c.id for c in n.cells];assert len(set(ids))==len(ids),key
        for c in n.cells:
            if c.cell_type=='markdown':assert normalize_markdown(c.source)==c.source,key
            else:
                compile(c.source,key,'exec');code_count+=1
                assert not any(o.output_type=='error' for o in c.outputs),key
        for patch in patches[key]:
            if patch.get('kind')=='insert' and patch.get('id','').startswith('oct20-'):
                cells=[c for c in n.cells if c.metadata.get('review_id')=='review-'+patch['id']]
                assert len(cells)==1,(key,patch['id'])
                expected=patch['replacement'].strip()
                current=cells[0].source.strip()
                if current!=expected:
                    # Exact reviewed replacements retain the original overlay provenance.
                    change=accepted.get(key,{}).get(patch['id'])
                    assert change is not None,(key,patch['id'],'unreviewed overlay change')
                    assert change['cell_type']==cells[0].cell_type
                    assert change['previous_source_sha256']==hashlib.sha256(expected.encode()).hexdigest()
                    assert change['accepted_source_sha256']==hashlib.sha256(current.encode()).hexdigest()
                    if cells[0].cell_type=='code':
                        assert cells[0].execution_count is not None,(key,patch['id'],'repaired code not executed')
    # No optional training is run: reject dangerous destinations before fitting.
    from flowmllab.stokes_refined import run
    try:run('same_data',output_dir=ROOT/'results/forbidden-student-output')
    except ValueError:pass
    else:raise AssertionError('Retained evidence directory was accepted')
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as scratch:
        (Path(scratch)/'same_data').mkdir()
        try:run('same_data',output_dir=scratch)
        except FileExistsError:pass
        else:raise AssertionError('Existing student output was accepted')
    from flowmllab.gas_dynamics import rayleigh_inverse_t0,rayleigh_ratios
    for mach in [0.1,0.8,1.2,4.,100.]:
        recovered=rayleigh_inverse_t0(float(rayleigh_ratios(mach)[4]),'subsonic' if mach<1 else 'supersonic')
        np.testing.assert_allclose(recovered,mach,rtol=1e-9)
    manifest=json.loads((ROOT/'qa/feedback_lecture_manifest.json').read_text())
    for name,record in manifest.items():
        path=ROOT/'lectures'/f'{name}.pdf';reader=PdfReader(path)
        assert len(reader.pages)==record['pages']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        # Topic guides can precede retained protocol pages. Audit their explicit
        # one-based positions instead of assuming every extension is an appendix.
        numbers=record.get('teaching_page_numbers',list(range(record['base_pages']+1,len(reader.pages)+1)))
        assert len(numbers)==record['teaching_pages'] and len(set(numbers))==len(numbers)
        assert all(1<=number<=len(reader.pages) for number in numbers)
        assert (ROOT/record['section']).is_file()
        for number in numbers:
            page=reader.pages[number-1]
            assert len(page.extract_text())>120
            fonts=page['/Resources']['/Font']
            assert any('TimesNewRoman' in str(f.get_object().get('/BaseFont','')) for f in fonts.values())
    result={'affected_notebooks':len(affected),'code_cells_syntax_checked':code_count,
            'lecture_pdfs_checked':len(manifest),'portable_math':True,'overlay_cells_match':True,
            'stokes_output_guards':'passed without training','rayleigh_branch_roundtrips':'passed',
            'scope':'Structural checks and bounded regressions; full notebook execution is recorded separately.'}
    (ROOT/'qa/feedback_structural_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
