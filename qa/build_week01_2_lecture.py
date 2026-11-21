"""Compile the authoritative Week 1.2 LaTeX lecture with Tectonic.
Requires licensed Times New Roman and Tectonic on PATH or TECTONIC_BIN.
"""
from pathlib import Path
import os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
BASE_LECTURE_PAGES=72
BASE_WITH_GHIA_PAGES=74

def build():
    from pypdf import PdfReader
    sys.path.insert(0,str(ROOT))
    if os.name=='nt':
        os.environ.setdefault('FLOWMLLAB_TNR_DIR',str(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'))
    from qa.build_cavity_expansion_lectures import build as append_extensions
    from lecture_typography import font_paths
    font_paths()  # Validate the separately authored appendix's fonts first.
    executable=os.environ.get('TECTONIC_BIN') or shutil.which('tectonic')
    if not executable:
        local=ROOT/'tmp/tex_runtime/tectonic.exe'
        if local.is_file():executable=str(local)
    if not executable:raise RuntimeError('Install Tectonic or set TECTONIC_BIN; LaTeX source is lectures/source/week01_2_pressure_velocity.tex')
    env=os.environ.copy()
    cache=ROOT/'tmp/tex_runtime/cache';fontconfig=ROOT/'tmp/tex_runtime/fontconfig.conf'
    if cache.is_dir():env.setdefault('TECTONIC_CACHE_DIR',str(cache))
    if os.name=='nt' and fontconfig.is_file():env.setdefault('FONTCONFIG_FILE',str(fontconfig))
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    subprocess.run([executable,'--keep-logs','--outdir',str(out),str(ROOT/'lectures/source/week01_2_pressure_velocity.tex')],cwd=ROOT,env=env,check=True)
    target=out/'week01_2_pressure_velocity.pdf'
    pages=len(PdfReader(target).pages)
    if pages!=BASE_LECTURE_PAGES:raise RuntimeError(f'Expected {BASE_LECTURE_PAGES} base pages, found {pages}; inspect layout before publishing.')
    published=ROOT/'lectures'/target.name
    protected=(published, ROOT/'qa/cavity_lecture_manifest.json', ROOT/'qa/cavity_startup_lecture.json')
    previous={p:p.read_bytes() if p.exists() else None for p in protected}
    # The existing two-page seven-Re extension is authored separately. Rebuild
    # it after the base lecture rather than silently dropping it on recompiles.
    try:
        shutil.copy2(target,published)
        append_extensions(targets={'week01_2_pressure_velocity'})
        final_pages=len(PdfReader(published).pages)
        if final_pages!=BASE_WITH_GHIA_PAGES:raise RuntimeError(f'Expected {BASE_WITH_GHIA_PAGES} preserved pages before startup, found {final_pages}')
        from qa.append_cavity_startup_lecture import build as append_startup
        final_pages=append_startup()['total_pages']
    except Exception:
        for path, payload in previous.items():
            if payload is None:
                if path.exists():path.unlink()
            else:path.write_bytes(payload)
        raise
    return final_pages

if __name__=='__main__':print('Lecture pages:',build())
