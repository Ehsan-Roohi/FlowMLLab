"""Build the Week 7.2 executable notebook and lecture PDF."""
from pathlib import Path
import hashlib, html, shutil, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'qa'))
from add_colab_entrypoints import badge, bootstrap


def make_notebook():
    import nbformat as nbf
    def md(text): return nbf.v4.new_markdown_cell(text.strip())
    def code(text): return nbf.v4.new_code_cell(text.strip())
    path = ROOT/'notebooks/week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb'
    cells = [
        md('''# Week 7.2 - Sparse-sensor state estimation of a cylinder wake
<!-- MIE690A article-aligned validation v4 -->

Can present noisy measurements correct a reduced forecast without using future
observations? We compare a causal POD-space Kalman filter with matched open-loop,
persistence and sensor-only baselines on existing FlowMLLab LBM fields.

Prerequisites: Weeks 5 and 7; allow 75-90 minutes; CPU only. This is a previously
inspected Re110 trajectory with synthetic measurement noise, not new CFD, new-Re
generalization or grid-independent truth. Lecture:
`lectures/week07_2_cylinder_state_estimation.pdf`.'''),
        code('''from pathlib import Path
import hashlib, json, sys, tempfile
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'flowmllab/state_estimation.py').is_file())
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'qa'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from flowmllab.modal_experiments import load_cases
from flowmllab.state_estimation import *
from run_week72_state_estimation import PLAN, experiment, make_figures
tracked = [p for folder in ('data/modal_labs','results/week07_2_state_estimation') for p in (ROOT/folder).rglob('*') if p.is_file()]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
cases = load_cases(ROOT/'data/modal_labs')
display(PLAN)'''),
        md(r'''## 1. Information contracts before equations

Write down what each method may observe after training:

- **Open-loop:** the last training state only.
- **Causal filter:** measurements through the current time.
- **Sensor-only:** the same current measurements, but no temporal model.
- **Smoother:** current and future measurements; defined here but not evaluated.

Comparing filtering with sensor-only is essential: otherwise improvement over
open-loop could be caused merely by receiving new data.'''),
        md(r'''## 2. Reduced state and observation models

With row-vector modal coefficients,
$q_k=\bar q+\Phi a_k$, $a_{k+1}=a_kA+w_k$, and
$y_k=\bar q_s+Ha_k+v_k$. Fit POD, $A$, residual covariance and sensor positions
on training only. The implementation uses QR-seeded D-optimal oversampling.

Derive the predict/update equations. Explain why explicitly forming the
innovation-covariance inverse is unnecessary, and why Joseph's covariance form
is numerically safer.'''),
        code('''# Exact low-rank causal control before the CFD experiment.
rng = np.random.default_rng(4)
phi, _ = np.linalg.qr(rng.normal(size=(24,2)))
z = np.array([[np.cos(.12*k), np.sin(.12*k)] for k in range(100)])
control = z @ phi.T
model0 = fit_reduced_kalman(control[:60], 2, 5, .01)
y0 = control[60:,model0.sensor_indices] + rng.normal(0,.01,(40,5))
states0, cov0, _ = kalman_filter(model0,y0)
error0 = np.linalg.norm(reconstruct_states(model0,states0)-control[60:])/np.linalg.norm(control[60:])
assert error0 < .15 and np.all(np.linalg.eigvalsh(cov0) >= -1e-12)
full = kalman_filter(model0,y0)[0]
prefix = kalman_filter(model0,y0[:5])[0]
np.testing.assert_allclose(full[:5],prefix)
print('Known-system relative error:',error0)
print('PASS: covariance PSD and prefix invariance (causality).')'''),
        md('''## 3. Freeze selection before the test

Training frames are `[0,160)`, validation `[160,210)`, test `[210,281)`.
Validation compares ranks 4/6/8, sensor counts 8/16/32, and process-covariance
scales 1/10/100/1000 using five noise seeds. Noise is 10% of training-field RMS.
Five different seeds are reserved for final evaluation. Test truth selects nothing.'''),
        code('''scratch = Path(tempfile.mkdtemp(prefix='flowmllab-week72-'))
result, model, examples = experiment(cases)
make_figures(scratch,cases,result,model,examples)
display(pd.DataFrame(result['candidates']).sort_values('validation_relative_l2_mean').head(12))
print('Selected:',result['selected'])
print('Fresh figures:',scratch)'''),
        md('''## 4. Compare matched baselines

Open-loop DMD receives no future measurements. Persistence repeats frame 159.
Sensor-only independently solves the POD observation least-squares problem at
each time and receives exactly the same noisy measurements as the Kalman filter.

Report mean and sample SD across the five measurement-noise seeds. These seeds
are not independent CFD realizations and do not support population confidence
intervals.'''),
        code('''rows=[]
for name,records in result['methods'].items():
    values=100*np.array([r['test']['relative_l2'] for r in records])
    rows.append({'method':name,'test_relative_L2_mean_percent':values.mean(),
                 'sample_SD_percent':values.std(ddof=1)})
display(pd.DataFrame(rows).sort_values('test_relative_L2_mean_percent'))
display(Image(filename=str(scratch/'state_estimation_fields.png')))
display(Image(filename=str(scratch/'state_estimation_scores.png')))'''),
        md('''## 5. Audit uncertainty, not only the point estimate

Project the filtered modal covariance through the POD basis. The nominal 95%
marginal intervals cover only about 55% of test field values: a severe retained
under-coverage failure. They are not simultaneous spatial bands.

Explain how truncation, structural dynamics error, correlated field errors and
white-Gaussian assumptions can make accurate point estimates overconfident.
Propose validation-only calibration, then a genuinely untouched evaluation.'''),
        code('''coverage=pd.DataFrame(result['coverage'])
display(coverage)
print('Mean nominal-95% pointwise coverage:',coverage.test_pointwise_95_coverage.mean())
assert coverage.test_pointwise_95_coverage.mean() < .75, 'Update the retained failure statement if behavior changes' '''),
        md('''## 6. Claim boundary and next experiment

The bounded conclusion is that current observations plus a reduced dynamical
prior improve this one retained trajectory under the declared artificial noise.
It is not evidence of new-Re generalization, pressure/force sensing, exact
continuum accuracy, or offline smoothing.

Next: predeclare missing-sensor bursts and a complete new-Re trajectory; select
all settings on development sequences. Reference: Särkkä and Svensson,
*Bayesian Filtering and Smoothing*, 2nd ed., 2023,
https://doi.org/10.1017/9781108917407. Code and figures here are independently
authored; no restricted handout material is incorporated.'''),
        code('''after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
assert after == before, 'Notebook modified retained data/evidence'
print('PASS: retained data and evidence unchanged.')'''),
    ]
    cells[0].source += badge(path.relative_to(ROOT).as_posix())
    cells[1].source = bootstrap(path.parent.relative_to(ROOT).as_posix()) + cells[1].source
    for index, cell in enumerate(cells):
        cell.id = hashlib.sha256(f'week72-{index}-{cell.source}'.encode()).hexdigest()[:12]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    nbf.write(notebook,path); print(path)


def make_pdf():
    import json
    from statistics import mean, stdev
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer, Image, Table, TableStyle
    source = ROOT/'lectures/source/week07_2_cylinder_state_estimation.md'
    out = ROOT/'output/pdf/week07_2_cylinder_state_estimation.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
    evidence = json.loads((ROOT/'results/week07_2_state_estimation/metrics.json').read_text())
    styles = getSampleStyleSheet()
    ink = colors.HexColor('#173b56'); muted = colors.HexColor('#506070')
    styles.add(ParagraphStyle(name='Title72', fontName='Helvetica-Bold', fontSize=23, leading=27, textColor=ink, spaceAfter=12))
    styles.add(ParagraphStyle(name='Heading72', fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=ink, spaceBefore=10, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name='Body72', fontName='Helvetica', fontSize=10.2, leading=14.4, spaceAfter=7))
    styles.add(ParagraphStyle(name='Small72', fontName='Helvetica', fontSize=8.8, leading=12, textColor=muted, spaceAfter=7))
    math_font = Path('C:/Windows/Fonts/timesi.ttf')
    if not math_font.is_file():
        from matplotlib.font_manager import FontProperties, findfont
        math_font = Path(findfont(FontProperties(family='DejaVu Serif', style='italic')))
    pdfmetrics.registerFont(TTFont('Equation72Embedded', str(math_font)))
    styles.add(ParagraphStyle(name='Equation72', fontName='Equation72Embedded', fontSize=12, leading=18, leftIndent=12, spaceAfter=4))
    sections = [page.strip().split('\n\n') for page in source.read_text(encoding='utf-8').split('\n---\n')]
    story = []
    def paragraph(text, style='Body72'):
        story.append(Paragraph(text, styles[style]))
    def section(index, skip=0):
        blocks = sections[index]
        paragraph(html.escape(blocks[0].removeprefix('# ')), 'Heading72')
        for block in blocks[1+skip:]:
            paragraph(html.escape(block).replace('\n', ' '))
    def table(rows, widths, header=True):
        obj = Table(rows, colWidths=widths, hAlign='LEFT')
        settings = [('FONTNAME',(0,0),(-1,-1),'Helvetica'), ('FONTSIZE',(0,0),(-1,-1),9.4),
                    ('TOPPADDING',(0,0),(-1,-1),9), ('BOTTOMPADDING',(0,0),(-1,-1),9),
                    ('LINEBELOW',(0,0),(-1,0),.7,ink), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5f7'),colors.white])]
        if header:
            settings += [('BACKGROUND',(0,0),(-1,0),ink), ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                         ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')]
        obj.setStyle(TableStyle(settings)); story.extend([obj, Spacer(1,8)])

    # Page 1: concept, measurement model and the full causal update in one place.
    paragraph('07.2  /  CYLINDER WAKE', 'Small72')
    paragraph('State estimation<br/>from sparse sensors', 'Title72')
    for block in sections[0][1:]: paragraph(html.escape(block), 'Body72')
    paragraph('01  Reduced model and observations', 'Heading72')
    paragraph('Let x<sub>k</sub> be a <b>column</b> of POD coefficients; the code stores its transpose a<sub>k</sub>. The POD basis contains training modes, and H selects its sensor rows. With F = A<super>T</super>, the model is:')
    for equation in ('q<sub>k</sub> = q&#772; + Φx<sub>k</sub>',
                     'x<sub>k+1</sub> = Fx<sub>k</sub> + w<sub>k</sub>, &nbsp; w<sub>k</sub> ~ N(0,Q)',
                     'y<sub>k</sub> = q&#772;<sub>s</sub> + Hx<sub>k</sub> + v<sub>k</sub>, &nbsp; v<sub>k</sub> ~ N(0,R)'):
        paragraph(equation.replace('q&#772;', 'q<sub>mean</sub>'), 'Equation72')
    for block in sections[1][2:]: paragraph(html.escape(block))
    paragraph('02  Causal predict-update recursion', 'Heading72')
    for equation in ('x<super>-</super> = Fx<super>+</super>, &nbsp; P<super>-</super> = FP<super>+</super>F<super>T</super> + Q',
                     'r = y - q<sub>mean,s</sub> - Hx<super>-</super>, &nbsp; S = HP<super>-</super>H<super>T</super> + R',
                     'K = P<super>-</super>H<super>T</super>S<super>-1</super>, &nbsp; x<super>+</super> = x<super>-</super> + Kr',
                     'P<super>+</super> = (I-KH)P<super>-</super>(I-KH)<super>T</super> + KRK<super>T</super>'):
        paragraph(equation, 'Equation72')
    paragraph('Superscripts - and + denote predicted and updated states. The code solves for the gain and uses the Joseph covariance update. A prefix-invariance test checks that future measurements cannot alter earlier estimates.', 'Small72')
    paragraph(html.escape(sections[2][-1]), 'Small72')

    # Page 2: native text table generated from retained metrics, followed by UQ.
    story.append(PageBreak())
    paragraph('Protocol and quantitative evidence', 'Title72')
    section(3)
    paragraph('Test error across five measurement-noise seeds', 'Heading72')
    rows = [['Method', 'Mean relative L2 (%)', 'Sample SD (pp)']]
    for name, label in [('kalman','Kalman filter'),('sensor_only','Sensor-only POD'),('open_loop','Open-loop DMD'),('persistence','Persistence')]:
        values = [100*r['test']['relative_l2'] for r in evidence['methods'][name]]
        rows.append([label, f'{mean(values):.3f}', f'{stdev(values):.3f}'])
    table(rows, [218,160,133])
    paragraph('SD is measured in percentage points (pp). Open-loop and persistence ignore sensor noise, so their across-seed SD is zero. All rows use the same test frames; these are repeated measurements of one CFD trajectory.', 'Small72')
    paragraph(html.escape(sections[4][-1]))
    section(5)

    # Page 3: explicit physical aspect ratio; the PDF never stretches an image.
    story.append(PageBreak())
    paragraph('Wake reconstruction and local errors', 'Title72')
    figure = ROOT/'results/week07_2_state_estimation/state_estimation_fields.png'
    width, height = ImageReader(str(figure)).getSize()
    story.append(Image(str(figure), width=511, height=511*height/width))
    paragraph('Final test frame, first evaluation seed (31). Panels (a-d) share the velocity scale; (e-f) share the absolute-error scale. Circles mark the 32 sensors. Equal x/D and y/D scales preserve physical geometry. The cylinder itself lies outside this wake region.', 'Small72')
    paragraph('Reading the figure', 'Heading72')
    paragraph('The velocity contours show the coherent wake; the absolute-error panels reveal differences that a signed color scale can hide. Contour rendering interpolates level crossings for display; all scores use the original 32 x 78 samples.', 'Body72')
    section(6)
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#d4dfe6')); canvas.line(42,39,553,39)
        canvas.setFont('Helvetica',8); canvas.setFillColor(muted)
        canvas.drawString(42,25,'FlowMLLab  /  Ehsan Roohi  /  Week 7.2')
        canvas.drawRightString(553,25,str(doc.page))
    SimpleDocTemplate(str(out), pagesize=(595,842), rightMargin=42,leftMargin=42,
        topMargin=38,bottomMargin=52, title='Week 7.2 - Cylinder wake state estimation',
        author='Ehsan Roohi / FlowMLLab').build(story,onFirstPage=footer,onLaterPages=footer)
    shutil.copy2(out,ROOT/'lectures'/out.name); print(out)


if __name__=='__main__':
    make_notebook(); make_pdf()
