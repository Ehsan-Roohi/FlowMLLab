"""Typeset the reading guides and integrate them into the relevant lecture PDFs.

The shared TeX sections are authoritative. PDF metadata records where each base
lecture ends, making repeated assembly idempotent. No base pages are rasterized.
"""
from pathlib import Path
import hashlib,io,json,os,re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
from build_student_feedback_lessons import LESSONS
from pypdf import PdfReader,PdfWriter
from lecture_typography import register_reportlab

TARGETS={
 'week01_1':['week01_1_ai_assisted_scientific_software'],
 'week02_1':['week02_1_probabilistic_uq'],
 'week04_1':['week04_cavity_surrogates_deeponet'],
 'week04_2':['week04_2_stokes_to_navier_stokes','week04_2_grid51_validation'],
 'week07':['week07_cylinder_lbm_neural_surrogate','week07_modal_forecasting'],
 'week07_1':['week07_1_hypersonic_rarefied_cylinder'],
 'week07_2':['week07_2_cylinder_state_estimation'],
 'week07_3':['week07_3_masked_pretraining'],
 'week07_4':['week07_4_diverse_wake_pretraining'],
 'week08':['week08_gas_dynamics_sciml'],
 'week09':['week09_rarefied_deeponet_case_studies','week09_3_nozzle_data_alignment'],
 'week10':['week10_dsmc_data_driven_surrogates'],
 'week10_1':['week10_1_collision_map_audit'],
 'week11':['week11_shock_vortex_identification'],
}

def prose(s):
    escape={'&':r'\&','%':r'\%','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    s=''.join(escape.get(c,c) for c in s)
    s=re.sub(r'`([^`]+)`',r'\\texttt{\1}',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'\\textbf{\1}',s)
    return s.replace('--',r'\textendash{}')

def tex_body(md):
    title,body=md.split('\n',1)
    # Preserve math tokens; escaping is applied only to prose.
    chunks=re.split(r'(\$\$.*?\$\$|\$[^$]+\$)',body,flags=re.S)
    body=''.join((r'\['+s[2:-2]+r'\]') if s.startswith('$$') else s if s.startswith('$') else prose(s) for s in chunks)
    return r'\section*{'+prose(title.lstrip('# '))+'}\n'+body

def main():
    from reportlab.pdfgen import canvas
    fonts=register_reportlab('FeedbackFooter')
    out=ROOT/'output/pdf/reading_audits';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'lectures/source/reading_audits';source.mkdir(parents=True,exist_ok=True)
    exe=os.environ.get('TECTONIC_BIN') or shutil.which('tectonic')
    if not exe and (ROOT/'tmp/tex_runtime/tectonic.exe').is_file():
        exe=str(ROOT/'tmp/tex_runtime/tectonic.exe')
    if not exe:raise RuntimeError('Install Tectonic or set TECTONIC_BIN.')
    env=os.environ.copy();env.setdefault('TECTONIC_CACHE_DIR',str(ROOT/'tmp/tex_runtime/cache'))
    if os.name=='nt' and (ROOT/'tmp/tex_runtime/fontconfig.conf').is_file():
        env.setdefault('FONTCONFIG_FILE',str(ROOT/'tmp/tex_runtime/fontconfig.conf'))
    manifest={}
    for week,targets in TARGETS.items():
        body=tex_body(LESSONS[week])
        # A section can be reused in another authoring pipeline with \input.
        (source/f'{week}_section.tex').write_text(body+'\n',encoding='utf-8')
        title='Week '+week.removeprefix('week').replace('_','.').lstrip('0')+' | Reading the evidence'
        document=r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=23mm]{geometry}
\usepackage{amsmath,amssymb,xcolor,hyperref,fancyhdr}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{Times New Roman}
\usepackage[italic]{mathastext}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{7pt}\setlength{\parindent}{0pt}
\pagestyle{fancy}\fancyhf{}\setlength{\headheight}{14pt}
\lhead{FlowMLLab}\rhead{Reading and experiment guide}\cfoot{\thepage}
\begin{document}
'''+r'\noindent\textbf{'+prose(title)+r'}\par\medskip\hrule\medskip'+'\n'+body+r'''
\paragraph{Check your understanding.}
Identify the input information, the reference, the metric denominator and the
evaluation split in one worked notebook comparison. Predict which conclusion
would change if any one of those four changed, then check your prediction.
\end{document}
'''
        tex=source/f'{week}.tex';tex.write_text(document,encoding='utf-8')
        proc=subprocess.run([str(exe),'--keep-logs','--outdir',str(out),str(tex)],cwd=ROOT,env=env,capture_output=True,text=True)
        (out/f'{week}.console.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError(week+': '+proc.stderr[-1500:])
        appendix=PdfReader(out/f'{week}.pdf')
        for target in targets:
            path=ROOT/'lectures'/f'{target}.pdf'
            writer=PdfWriter();base_count=0
            if path.exists():
                base=PdfReader(path)
                base_count=int(base.metadata.get('/FlowMLLabBasePages',len(base.pages)))
                if base_count:writer.append(base,pages=(0,base_count),import_outline=True)
            for index,page in enumerate(appendix.pages):
                # Continue the existing lecture's visible page numbers.
                overlay=io.BytesIO();width,height=float(page.mediabox.width),float(page.mediabox.height)
                c=canvas.Canvas(overlay,pagesize=(width,height))
                c.setFillColorRGB(1,1,1);c.rect(width/2-35,15,70,30,fill=1,stroke=0)
                c.setFillColorRGB(0,0,0);c.setFont(fonts['regular'],10)
                c.drawCentredString(width/2,28,str(base_count+index+1));c.save()
                page=page.clone(PdfWriter())
                page.merge_page(PdfReader(overlay).pages[0])
                writer.add_page(page)
            writer.add_outline_item('Reading the evidence and extending the experiment',base_count)
            writer.add_metadata({'/Title':target.replace('_',' '),'/Author':'Ehsan Roohi',
                                 '/FlowMLLabBasePages':str(base_count),
                                 '/FlowMLLabTeachingSection':'reading_audits/'+week+'_section.tex'})
            assembled=out/f'{target}.pdf'
            with assembled.open('wb') as stream:writer.write(stream)
            path.write_bytes(assembled.read_bytes())
            manifest[target]={'base_pages':base_count,'teaching_pages':len(appendix.pages),
                              'pages':base_count+len(appendix.pages),'section':f'lectures/source/reading_audits/{week}_section.tex',
                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(week,'typeset and integrated',flush=True)
    (ROOT/'qa/feedback_lecture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
