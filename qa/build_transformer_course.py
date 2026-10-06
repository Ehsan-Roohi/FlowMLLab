"""Render canonical editable sources into an explicitly selected output tree.

Default writes only build/. Neither this nor the compatibility entry point
rewrites source notes, lesson specifications, existing results or course guides.
"""
from pathlib import Path
import argparse
import hashlib
import html
import io
import json
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Preformatted, KeepTogether
from reportlab import rl_config

ROOT=Path(__file__).resolve().parents[1]
class CourseKernelManager(KernelManager):
    def format_kernel_cmd(self,extra_arguments=None):
        command=super().format_kernel_cmd(extra_arguments)
        command[0]=sys.executable
        return command
SETUP='''from pathlib import Path
import sys, tempfile, json
ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'flowmllab/transformer_course.py').exists()),None)
if ROOT is None:
    raise RuntimeError('Extract the supplied course bundle and start Jupyter in its root; see docs/TRANSFORMER_COURSE.md for Colab upload setup.')
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from flowmllab import transformer_course as lab
lab.seed_all(17)
cases,input_manifest=lab.load_data(ROOT)
EVIDENCE=ROOT/'results/transformer_course_v3'
# Any optional experiment must write into this disposable workspace.
workspace=tempfile.TemporaryDirectory(prefix='flowmllab-student-')
RUN_OUTPUT=Path(workspace.name)
task_status={}
print('Loaded checksummed simulated wake fields. No retained files will be rewritten.')
'''


def notebook(spec,solutions=False):
    cells=[nbf.v4.new_markdown_cell(f'# Week {spec["week"]}: {spec["title"]}\n\n{spec["question"]}\n\n**Prerequisites:** {spec["prerequisites"]}\n\nAllow 120 minutes plus the written analysis. '+('Instructor solution edition.' if solutions else 'Student edition: worked examples run immediately; set RUN_EXERCISES=True after completing the four functions. Incomplete tasks are reported, never counted as passes.')),
           nbf.v4.new_code_cell(SETUP),nbf.v4.new_code_cell('RUN_EXERCISES='+str(solutions))]
    for demo in spec['demos']:
        cells += [nbf.v4.new_markdown_cell('## '+demo['title']+'\n\n'+demo['prose']),nbf.v4.new_code_cell(demo['code'])]
    for i,task in enumerate(spec['tasks'],1):
        cells.append(nbf.v4.new_markdown_cell(f'## Task {i}: {task["title"]}\n\n{task["prompt"]}'))
        cells.append(nbf.v4.new_code_cell(task['solution'] if solutions else task['starter']))
        checks='\n'.join('    '+line for line in task['checks'].splitlines())
        cells.append(nbf.v4.new_code_cell(f'if RUN_EXERCISES:\n{checks}\n    task_status[{i}]=True\nelse:\n    task_status[{i}]=False\nprint("Task {i}:","PASS" if task_status[{i}] else "NOT SUBMITTED")'))
    cells += [nbf.v4.new_markdown_cell('## Interpretation and submission\n\nSubmit the four completed functions, the requested plots/tables, and a 300-word claim ledger. Distinguish observations, fitted quantities, independent evaluation and unsupported extrapolation. Retain failed seeds. Use the lecture source for the rubric and assumptions.'),
              nbf.v4.new_code_cell('print("Coding tasks passed:",sum(task_status.values()),"/",len(task_status))\n'+('assert all(task_status.values())\n' if solutions else '')+'workspace.cleanup()')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'course':{'week':spec['week'],'edition':'instructor' if solutions else 'student'}})
    for i,c in enumerate(nb.cells):c['id']=f'w{spec["week"]}-{i:02d}'
    return nb


def equation(text):
    # A compact genuine mathematical display, embedded with transparent background.
    if '\\begin{cases}' in text:
        text=r'M_{ij}=0\ (j\leq i),\qquad M_{ij}=-\infty\ (j>i)'
    fig=plt.figure(figsize=(7,.6));fig.text(.5,.5,'$'+text+'$',ha='center',va='center',fontsize=14)
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=180,transparent=True,bbox_inches='tight',pad_inches=.12);plt.close(fig);buffer.seek(0)
    pic=Image(buffer);factor=min(72/180,475/pic.imageWidth)
    pic.drawWidth=pic.imageWidth*factor;pic.drawHeight=pic.imageHeight*factor
    return pic


def render_pdf(source,destination,figure=None):
    rl_config.invariant=1
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCourse',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=9,allowWidows=0,allowOrphans=0,textColor=colors.HexColor('#243247')))
    styles.add(ParagraphStyle(name='TableCourse',fontName='Helvetica',fontSize=8,leading=11))
    styles['Heading1'].textColor=colors.HexColor('#123b59');styles['Heading1'].fontSize=23;styles['Heading1'].leading=28
    styles['Heading2'].textColor=colors.HexColor('#0e6572');styles['Heading2'].spaceBefore=15;styles['Heading2'].spaceAfter=8
    for heading in ('Heading1','Heading2','Heading3'):styles[heading].keepWithNext=True
    story=[];lines=source.read_text(encoding='utf-8').splitlines();i=0
    def para(s,style='BodyCourse'):
        s=html.escape(s).replace('**','').replace('`','')
        return Paragraph(s,styles[style])
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):story.append(para(line[2:],'Heading1'));story.append(Spacer(1,14));i+=1;continue
        if line.startswith('### '):story.append(para(line[4:],'Heading3'));i+=1;continue
        if line.startswith('## '):story.append(para(line[3:],'Heading2'));i+=1;continue
        if line.startswith('$$'):
            story.append(KeepTogether([Spacer(1,6),equation(line.strip('$ ')),Spacer(1,10)]));i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(set(c)<=set('-: ') for c in cells):table.append([para(c,'TableCourse') for c in cells])
                i+=1
            t=Table(table,colWidths=[475/len(table[0])]*len(table[0]),repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#b5c6d0')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.extend([t,Spacer(1,12)]);continue
        text=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','$$')):
            text.append(lines[i].strip());i+=1
        story.append(para(' '.join(text)))
    if figure and figure.exists():
        pic=Image(str(figure));factor=min(475/pic.imageWidth,300/pic.imageHeight);pic.drawWidth=pic.imageWidth*factor;pic.drawHeight=pic.imageHeight*factor
        story.append(KeepTogether([para('Executed evidence and interpretation','Heading2'),pic,
            para('This figure is generated from the retained protocol. Use the notebook to inspect all seeds and recompute the plotted quantities. The earlier lecture sections specify what the diagnostic can and cannot establish.')]))
    def page(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#0e6572'));canvas.line(48,799,547,799)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#53677d'));canvas.drawString(48,809,'FLOWMLLAB  /  TRANSFORMERS AND FLUID LEARNING')
        canvas.drawString(48,27,'Original teaching notes | Simulated wake data | Retained-case evidence');canvas.drawRightString(547,27,str(doc.page));canvas.restoreState()
    destination.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(destination),pagesize=(595,842),leftMargin=54,rightMargin=66,topMargin=56,bottomMargin=48,title=lines[0][2:],author='FlowMLLab',pageCompression=1).build(story,onFirstPage=page,onLaterPages=page)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'build/transformer-course')
    parser.add_argument('--release',action='store_true',help='Explicitly allow writing notebooks/PDFs into this checkout')
    parser.add_argument('--pdf-only',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--check',action='store_true',help='Check canonical notebook input cells without writing')
    parser.add_argument('--weeks',type=int,nargs='+',choices=range(17,23),default=list(range(17,23)),help='Render or retry only the selected weeks')
    args=parser.parse_args(argv)
    if args.output.resolve()==ROOT and not args.release and not args.check:parser.error('Writing into the source checkout requires --release')
    protected=[*list((ROOT/'course/lessons').glob('*.json')),*list((ROOT/'lectures/source').glob('week*_transformer_course.md')),*list((ROOT/'results/transformer_course_v3').glob('*'))]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    execution_log=args.output/'notebook_execution.json'
    records=json.loads(execution_log.read_text()) if execution_log.exists() and args.weeks!=list(range(17,23)) else []
    for week in args.weeks:
        spec=json.loads((ROOT/f'course/lessons/week{week}.json').read_text())
        for solutions in (False,True):
            relative=Path(f'instructor/week{week}/W{week}_Solutions.ipynb' if solutions else f'notebooks/week{week}/W{week}_CFD_Transformer.ipynb')
            nb=notebook(spec,solutions)
            if args.check:
                existing=nbf.read(ROOT/relative,as_version=4)
                assert [c.source for c in existing.cells]==[c.source for c in nb.cells],relative
                continue
            if args.pdf_only:continue
            if args.execute:
                started=time.perf_counter()
                # A NEW client and kernel for every notebook; working directory is explicit.
                client=NotebookClient(nb,timeout=600,startup_timeout=180,kernel_name='python3',kernel_manager_class=CourseKernelManager,resources={'metadata':{'path':str(ROOT)}})
                client.execute()
                seconds=time.perf_counter()-started
                records=[r for r in records if r['notebook']!=relative.as_posix()]
                records.append({'notebook':relative.as_posix(),'seconds':seconds,'code_cells':sum(c.cell_type=='code' for c in nb.cells)})
                print(relative,round(seconds,2),'s',flush=True)
            destination=args.output/relative;destination.parent.mkdir(parents=True,exist_ok=True);nbf.write(nb,destination)
            if args.execute:execution_log.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
        if not args.check:
            render_pdf(ROOT/f'lectures/source/week{week}_transformer_course.md',args.output/f'lectures/week{week}_cfd_transformer.pdf',ROOT/f'results/transformer_course_v3/week{week}.png')
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    assert before==after,'Builder modified canonical source or retained evidence'
    if records:
        (args.output/'notebook_execution.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print('Canonical sources and retained evidence preserved.',flush=True)

if __name__=='__main__':main()
