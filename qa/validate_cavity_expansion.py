"""Check preserved lessons/pages, executed additions and retained science identity."""
from pathlib import Path
import hashlib,io,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import nbformat
from pypdf import PdfReader
from build_cavity_expansion_lessons import TAG,BOOT
from flowmllab.cavity_course import verify_frozen_models
from flowmllab.deep_cavity_audit import load_bundle,evaluate


def git_bytes(path):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT}','-C',str(ROOT),'show','042669e6ed6f7b6fdb67b57cdf41b24e95890ee1:'+path])


def main():
    receipt={'notebooks':{},'lecture_base_pages':{},'scope':'New extension cells executed; original code/output preservation checked. Not a new hosted Run All.'}
    # The current bootstrap is executed once in a clean namespace. It resolves
    # the local updated checkout; the optional network-clone branch is untested.
    import os
    os.chdir(ROOT);exec(compile(BOOT,'current_bootstrap','exec'),{})
    for path in sorted((ROOT/'notebooks').glob('week*/*.ipynb')):
        document=nbformat.read(path,as_version=4)
        additions=[c for c in document.cells if c.metadata.get(TAG)]
        if not additions:continue
        nbformat.validate(document)
        rel=path.relative_to(ROOT).as_posix()
        old_count=0
        if path.name not in {'W13_Deep_Cavity_D5_Regional_Accuracy.ipynb', 'W4_Lab5_All_Re_Ghia_Validation.ipynb', 'W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb'}:
            old=nbformat.reads(git_bytes(rel).decode('utf8'),as_version=4)
            kept=[c for c in document.cells if not c.metadata.get(TAG)]
            assert len(kept)==len(old.cells),(rel,'old cell count')
            for a,b in zip(old.cells,kept):
                assert a.source==b.source,(rel,'old cell source changed')
                if a.cell_type=='code':assert a.outputs==b.outputs,(rel,'old outputs changed')
            old_count=len(old.cells)
        for c in additions:
            if c.cell_type=='code':
                compile(c.source,rel,'exec')
                assert c.execution_count is not None,(rel,'unexecuted addition')
                assert not any(o.output_type=='error' for o in c.outputs)
            else:
                assert '\\\\frac' not in c.source and '\\\\sin' not in c.source
        receipt['notebooks'][rel]={'original_cells_preserved':old_count,'new_cells':len(additions),
                                   'new_code_cells':sum(c.cell_type=='code' for c in additions)}
    manifest=json.loads((ROOT/'qa/cavity_lecture_manifest.json').read_text())
    for name,info in manifest.items():
        rel='lectures/'+name+'.pdf';old=PdfReader(io.BytesIO(git_bytes(rel)));current=PdfReader(ROOT/rel)
        assert len(old.pages)==info['base_pages_preserved']
        for a,b in zip(old.pages,current.pages):
            assert a.get_contents().get_data()==b.get_contents().get_data(),(name,'base page stream changed')
            assert a.mediabox==b.mediabox
        assert len(current.pages)==info['total_pages']
        for page in list(current.pages)[len(old.pages):]:assert len(page.extract_text().strip())>150
        receipt['lecture_base_pages'][name]=len(old.pages)
    frozen=verify_frozen_models(ROOT)
    for name,digest in frozen['source_sha256'].items():
        p=ROOT/name.replace('\\','/')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
    fields,_=load_bundle(ROOT/'data/week13_d5');rows=evaluate(fields)
    assert len(rows)==16
    verification=json.loads((ROOT/'results/week13_d5/verification.json').read_text())
    assert verification['numeric_values_reproduced']==128 and verification['original_unchanged']
    receipt.update({'frozen_models_hash_checked':len(frozen['models_sha256']),
        'frozen_study_sources_unchanged':True,'deep_cavity_regional_numbers_reproduced':128,
        'deep_cavity_regions_recomputed':len(rows),'bootstrap_local_passed':True,
        'colab_clone_branch_tested':False})
    (ROOT/'qa/cavity_expansion_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
