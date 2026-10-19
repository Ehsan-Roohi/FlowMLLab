"""Render the optional editable Markdown lecture with genuine TNR and black text.

Requires pandoc, XeLaTeX and the font contract in qa/LECTURE_TYPOGRAPHY.md.
The source Markdown is canonical; intermediate TeX/logs remain in revision_work.
"""
from pathlib import Path
import shutil
import re
import subprocess
from lecture_typography import font_paths

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'lectures/source/optional_geometry_time_bridge.md'
OUT=ROOT/'revision_work/geometry_time_pdf'

PREAMBLE=r'''\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=22mm,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb,longtable,booktabs,array,calc,graphicx}
\usepackage{xcolor,hyperref,xurl,fancyhdr,enumitem,titlesec}
\input{lectures/source/lecture_typography.tex}
% This short lecture has inline identifiers, not aligned source listings.
\setmonofont{Times New Roman}
\urlstyle{same}
\hypersetup{pdftitle={Optional geometry-time operator evaluation},pdfauthor={Ehsan Roohi | FlowMLLab}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\widowpenalty=10000\clubpenalty=10000
\setlist{itemsep=2pt,topsep=4pt,leftmargin=1.5em}
\providecommand{\tightlist}{\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}}
\titleformat{\section}{\large\bfseries}{\thesection}{.5em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{.5em}{}
\titlespacing*{\section}{0pt}{14pt}{5pt}
\titlespacing*{\subsection}{0pt}{9pt}{4pt}
\setcounter{secnumdepth}{0}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small FlowMLLab | Optional self-study}
\fancyhead[R]{\small Geometry and time}
\fancyfoot[L]{\small Manufactured examples | No CFD or paper reproduction}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{.3pt}
\renewcommand{\footrulewidth}{0pt}
\begin{document}
{\LARGE\bfseries Geometry, time and trustworthy\\[3pt]operator evaluation}\par
\vspace{4pt}
'''


def main():
    font_paths()  # fail if fontconfig resolves a substitute
    OUT.mkdir(parents=True,exist_ok=True)
    markdown=SOURCE.read_text(encoding='utf-8').split('\n',1)[1]
    markdown=re.sub(r'(?<![<(])(https://[^\s>]+)',r'<\1>',markdown)
    converted=subprocess.run(['pandoc','--from=markdown','--to=latex','--shift-heading-level-by=-1'],
                             input=markdown,text=True,capture_output=True,check=True).stdout
    converted=converted.replace('\\hypertarget{four-exercises-and-a-self-study-rubric}',
                                '\\clearpage\n\\hypertarget{four-exercises-and-a-self-study-rubric}')
    converted=converted.replace('\\hypertarget{references-and-evidence-limits}',
                                '\\clearpage\n\\hypertarget{references-and-evidence-limits}')
    tex=OUT/'optional_geometry_time_bridge.tex'
    tex.write_text(PREAMBLE+converted+'\n\\end{document}\n',encoding='utf-8')
    for _ in range(2):
        result=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',
                               '-output-directory',str(OUT),str(tex)],cwd=ROOT,text=True,capture_output=True)
        (OUT/'build_stdout.txt').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError('XeLaTeX failed; inspect '+str(OUT/'build_stdout.txt'))
    target=ROOT/'lectures/optional_geometry_time_bridge.pdf'
    shutil.copyfile(OUT/'optional_geometry_time_bridge.pdf',target)
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
