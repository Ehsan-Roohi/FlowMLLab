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
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer, Image
    source = ROOT/'lectures/source/week07_2_cylinder_state_estimation.md'
    out = ROOT/'output/pdf/week07_2_cylinder_state_estimation.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CourseTitle',fontName='Helvetica-Bold',fontSize=21,leading=25,textColor=colors.HexColor('#173b56'),spaceAfter=18))
    styles.add(ParagraphStyle(name='CourseBody',fontName='Helvetica',fontSize=10.4,leading=15,spaceAfter=10,alignment=TA_LEFT))
    story=[]
    pages=source.read_text(encoding='utf-8').split('\n---\n')
    for page_index,page in enumerate(pages):
        if page_index: story.append(PageBreak())
        for block in page.strip().split('\n\n'):
            heading,sep,body=block.partition('\n')
            if heading.startswith('# '):
                story.append(Paragraph(html.escape(heading[2:]),styles['CourseTitle']))
                if body: story.append(Paragraph(html.escape(body).replace('\n','<br/>'),styles['CourseBody']))
            else: story.append(Paragraph(html.escape(block).replace('\n','<br/>'),styles['CourseBody']))
        if page_index==4:
            story.extend([Spacer(1,6),Image(str(ROOT/'results/week07_2_state_estimation/state_estimation_fields.png'),width=500,height=381)])
        if page_index==5:
            story.extend([Spacer(1,8),Image(str(ROOT/'results/week07_2_state_estimation/state_estimation_scores.png'),width=480,height=270)])
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#c7d5df'));canvas.line(42,38,553,38)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#506070'))
        canvas.drawString(42,25,'FlowMLLab | Week 7.2 | State estimation')
        canvas.drawRightString(553,25,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=52,
        title='Week 7.2 - Cylinder wake state estimation',author='Ehsan Roohi / FlowMLLab').build(story,onFirstPage=footer,onLaterPages=footer)
    shutil.copy2(out,ROOT/'lectures'/out.name); print(out)


if __name__=='__main__':
    make_notebook(); make_pdf()
