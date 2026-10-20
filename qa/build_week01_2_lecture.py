"""Compile the authoritative Week 1.2 LaTeX lecture with Tectonic.
Requires licensed Times New Roman and Tectonic on PATH or TECTONIC_BIN.
"""
from pathlib import Path
import os,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
LECTURE_PAGES=25

def build():
    from pypdf import PdfReader
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
    if pages!=LECTURE_PAGES:raise RuntimeError(f'Expected {LECTURE_PAGES} pages, found {pages}; inspect layout before publishing.')
    shutil.copy2(target,ROOT/'lectures'/target.name)
    return pages

if __name__=='__main__':print('Lecture pages:',build())
