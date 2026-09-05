"""Build the two original companion notebooks and four-page lecture notes."""
from pathlib import Path
import argparse
import hashlib
import html
import shutil
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'qa'))
from add_colab_entrypoints import badge,bootstrap


def md(s):return nbf.v4.new_markdown_cell(s.strip())
def code(s):return nbf.v4.new_code_cell(s.strip())


SETUP='''from pathlib import Path
import sys, hashlib, tempfile
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'flowmllab/modal_experiments.py').is_file())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'qa'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from threadpoolctl import threadpool_limits
from flowmllab.modal_tools import *
from flowmllab.field_metrics import field_metrics, temporal_spectrum
from flowmllab.modal_experiments import PLAN, load_cases, forecast_experiment, sensor_experiment
from run_modal_labs import figures
tracked_evidence = [p for folder in ('data/modal_labs','results/modal_labs') for p in (ROOT/folder).rglob('*') if p.is_file()]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_evidence}
cases = load_cases(ROOT/'data/modal_labs')
display(PLAN)
print('Checksummed author LBM; previously inspected cases, not a new blind CFD study.')'''

RUN='''# Fresh CPU fits, not cached retained predictions. Limit BLAS threads for reproducibility.
with threadpool_limits(limits=1):
    forecast, predictions = forecast_experiment(cases)
    sensing, examples = sensor_experiment(cases)
scratch = Path(tempfile.mkdtemp(prefix='flowmllab-modal-'))
figures(scratch, cases, forecast, predictions, sensing, examples)
print('Exploratory figure output:', scratch)'''

COMMON=r'''## A shared metric contract (PDEBench-inspired, not identical scores)
Report $\|q-\hat q\|_{2,w}/\|q\|_{2,w}$, area-weighted RMSE, maximum error,
worst-frame relative error, ROI-edge error, and error of the supplied scalar integral.
An ROI edge is **not a physical wall**. An integral of vorticity or velocity is
**not automatically mass conservation**. Declare weights, units and geometry.
For zero reference norm, relative error is undefined (`None`), not epsilon-regularized.

Test a known offset before trusting the CFD score: adding one to a field of two
must give relative L2 = 0.5 and RMSE = 1. With 20 cells of area 0.25, the
absolute scalar-integral error must be 5. These are numerical identities, not fitted tolerances.'''

CONTROL='''truth = np.full((3,4,5), 2.0)
metrics = field_metrics(truth+1, truth, np.full((4,5), .25))
assert np.isclose(metrics['relative_l2'], .5)
assert np.isclose(metrics['rmse'], 1)
assert np.isclose(metrics['mean_absolute_scalar_integral_error'], 5)
assert field_metrics(np.ones_like(truth), 0*truth)['relative_l2'] is None
display(metrics)'''

SOURCES='''## Sources, provenance and submission
The data are earlier **author-generated FlowMLLab LBM cases**, published in
[cylinder-cfd-v1](https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/cylinder-cfd-v1).
They are not copied package examples or data from the separate hypersonic DSMC article.
See `data/modal_labs/README.md` and its original/derived file hashes.

Original textbook implementations were inspired by
[PyDMD](https://github.com/PyDMD/PyDMD),
[PySensors](https://github.com/dynamicslab/pysensors),
[PySINDy](https://github.com/dynamicslab/pysindy), and
[PDEBench](https://github.com/pdebench/PDEBench).
No code or figures were copied; these are not wrappers, complete replacements,
or a claim of exact equivalence to the packages' advanced algorithms.

Submit the complete split, selected settings, all candidate/seed scores, one
failure explanation, and one proposed **new** validation experiment. Do not
retune on the retained test or overwrite reference evidence. Short CPU runtime
reflects reuse of already-generated CFD, not a fresh high-fidelity simulation.'''


def make_notebook(kind):
    if kind=='week05':
        path=ROOT/'notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb'
        cells=[md('''# Week 5 companion - Sparse sensing and sparse dynamics
<!-- MIE690A article-aligned validation v4 -->
**Development companion:** local Run All is tested; the main-branch Colab link
becomes available only after this work is merged. This adds no new course week.

Learn to reconstruct a real wake using a few transverse-velocity sensors and
to identify an interpretable low-dimensional ODE. Prerequisites: Week 4/5 POD,
least squares and case-wise splitting. Allow 75-90 minutes; CPU only.
Lecture: `lectures/week05_modal_sensing.pdf`.

The ROI contains 32 x 78 fluid points, not the cylinder wall. Its coarse LBM
reference has 12 native cells per diameter and is not grid-independent CFD.
Filled contours interpolate level crossings only for display. No extra spatial
resolution, denoising or super-resolution is implied.'''),code(SETUP),md(r'''## 1. Warm-up: recover a known low-rank state
Fit $q=\bar q+\Phi a$ using training snapshots. Given observations
$y=Cq+\epsilon$, solve $(C\Phi)\hat a\simeq y-C\bar q$.
Pivoted QR selects $r$ independent rows; D-optimal greedy selection adds
oversampling rows. Arbitrary QR tail entries do not provide that criterion.
The pseudoinverse norm controls measurement-noise amplification; a full-rank
but ill-conditioned sensor matrix may still be poor.

First verify exact noiseless recovery on an explicitly rank-three control.
This synthetic control is a unit experiment; the next section uses actual LBM.'''),code('''rng = np.random.default_rng(42)
control = rng.normal(size=(30,3)) @ rng.normal(size=(3,40))
pod = fit_pod(control, rank=3)
sensor_ids = select_sensors(pod.modes, 6)
recovered = reconstruct_sensors(pod, sensor_ids, control[:,sensor_ids])
assert np.allclose(recovered, control, atol=1e-12)
print('Condition number:', np.linalg.cond(pod.modes[sensor_ids]))'''),md('''## 2. Real-data protocol - freeze before execution
Training: complete Re90 and Re110 trajectories of v/U. Validation: Re100.
Retained test: Re105, previously inspected in other work, not a new blind case.
Fit an eight-mode POD and ALL sensor locations using training cases only.
Try 8, 16 and 32 sensors; select the smallest budget with mean optimized
validation relative L2 <= 5%, otherwise the best validation budget.

Compare five random layouts with optimized placement. Measurement noise is
Gaussian with standard deviation 1% of training v RMS, paired at coincident
locations. The five seeds describe artificial sensor noise/layout variation,
not independent CFD realizations. They do not support population confidence intervals.

The shared runner also performs the Week 7 forecast needed by the SINDy bridge;
all fits below are new and outputs go to a fresh temporary directory.'''),code(RUN),code('''rows = pd.DataFrame(sensing['records'])
display(rows[rows['split']=='validation'][['method','budget','seed','relative_l2']])
print('Validation-selected sensor budget:', sensing['selected_budget'])
display(rows[rows['split']=='test'][['method','budget','seed','condition_number','relative_l2','edge_relative_l2']])
display(Image(filename=str(scratch/'sensor_fields.png')))
display(Image(filename=str(scratch/'sensor_audit.png')))'''),md('''## 3. Separate representation, observation and dynamics
The full-field POD oracle uses the complete target snapshot. It diagnoses
representation error and is not an operational sensor method. Compare methods
at equal sensor count. Explain why adding sensors cannot eliminate truncated modes.

For the dynamics bridge, use only Re110 frames 0:160 to fit a rank-two POD and
scale its coefficients. Build all monomials through degree three, integrate
their values by four-interval trapezoids, and regress coefficient increments.
Sequential thresholded least squares removes small terms and refits.
Validation chooses among thresholds 0.01, 0.05 and 0.1; the test continuation
must not alter this decision. There is no clipping or truth reset.

Inspect all candidates, including failed integrations, and the discovered
equations. A two-mode orbit cannot uniquely identify off-orbit cubic dynamics;
this is not discovery of Navier-Stokes. Its field error must be compared with
the **rank-two** oracle, not only with higher-rank models.'''),code('''display(pd.DataFrame(forecast['sindy_candidates']).T)
print('Selected sparse ODE:', forecast['selected_sindy'])
if forecast['selected_sindy']:
    for i, equation in enumerate(forecast['sindy_candidates'][forecast['selected_sindy']]['equations']):
        print(f'dz{i+1}/dt = {equation}')
display(pd.DataFrame({k:v['test'] for k,v in forecast['methods'].items()
    if k.startswith('SINDy') or k in ('DMD-r2','POD-r2-oracle')}).T)
display(Image(filename=str(scratch/'forecast_audit.png')))'''),md(COMMON),code(CONTROL),md(SOURCES)]
    else:
        path=ROOT/'notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb'
        cells=[md('''# Week 7 companion - Modal forecasting of a real cylinder wake
<!-- MIE690A article-aligned validation v4 -->
**Development companion:** local Run All is tested; main-branch Colab becomes
available only after merge. No new release or DOI is implied.

Compare projected DMD with a freshly trained nonlinear modal baseline, persistence,
and POD representation floors. Prerequisites: POD, complex eigenvalues and Week 7
LBM; allow 60-75 minutes. Lecture: `lectures/week07_modal_forecasting.pdf`.

All fields come from earlier author-generated CFD, not synthetic wake formulas.
The 12-node-per-diameter LBM is educational, not grid independent. This wake has
vortical structures but no shock. DMD forecasts a field; it does not identify cores.'''),code(SETUP),md(r'''## 1. Derive and test the row-vector convention
For training POD rows $a_k$, let $X=[a_0;\ldots;a_{m-2}]$ and
$Y=[a_1;\ldots;a_{m-1}]$. Solve $A=X^\dagger Y$, so $a_{k+1}=a_kA$.
Reconstruct with the frozen training mean and basis. For eigenvalue $\lambda$,
$f=\arg(\lambda)/(2\pi\Delta t)$ and $\sigma=\log|\lambda|/\Delta t$.
Frequency is subject to sampling and the principal logarithm branch.

Verify a known oscillator before interpreting wake eigenvalues. Its angular
frequency is 2, so cycles per unit time must be $1/\pi$.'''),code('''dt = .02
t = np.arange(400)*dt
oscillator = np.column_stack((np.cos(2*t), np.sin(2*t)))
operator = fit_dmd(oscillator[:200])
assert np.allclose(rollout_dmd(operator, oscillator[199], 200), oscillator[200:], atol=1e-12)
assert np.isclose(max(m['frequency'] for m in dmd_modes(operator,dt)), 1/np.pi)
display(pd.DataFrame(dmd_modes(operator,dt)))'''),md('''## 2. Forecast, do not assimilate the answer
Re110 vorticity: training frames [0,160), validation [160,210), test [210,281).
Fit centered POD only on training. Choose DMD rank among 2/4/6/8 by validation
field error. Initialize at frame159 and advance through both held windows
without resetting. This tests future time in one case, not a new Reynolds number.

The fresh MLP uses eight POD coordinates, four-frame history, two 32-unit tanh
layers and fixed seed17. Scaling and supervised pairs use training only.
It has more initial history than one-state DMD; report that distinction.
Persistence repeats the last training field; the mean repeats the training mean.
Full-field POD projections see target fields and are oracle floors, not forecasts.
Do not compare to archived phase-decoder results with different information/splits.'''),code(RUN),code('''print('Validation-selected DMD:', forecast['selected_dmd'])
display(pd.DataFrame({k:{'validation_l2':v['validation']['relative_l2'],
    'test_l2':v['test']['relative_l2'], 'worst_test_frame':v['test']['worst_frame_relative_l2']}
    for k,v in forecast['methods'].items()}).T)
display(forecast['methods']['MLP-r8'])
display(Image(filename=str(scratch/'forecast_fields.png')))
display(Image(filename=str(scratch/'forecast_audit.png')))'''),md('''## 3. A frequency estimate is not a resolution guarantee
The source full-history force Strouhal is a diagnostic only; it was never used
to train or select a model. Inspect all selected-DMD eigenvalues and distinguish
wake modes, harmonics and possible box/acoustic modes. Do not automatically
declare the largest frequency to be shedding.

At a geometrically selected probe near (x/D,y/D)=(4,0.5), compare mean-removed,
Hann-windowed one-sided temporal spectra. Report the actual FFT spacing and
phase error at the reference peak. The test window is only about 7.40 D/U,
giving spacing about 0.135 U/D: too short for a precise Strouhal measurement.
A model-based eigenfrequency does not improve the raw sampling resolution.

The Week 5 SINDy result is also retained, including failed candidates. Its
rank-two field limitation is not evidence that all sparse dynamics fail.'''),code('''selected = forecast['selected_dmd']
display(pd.DataFrame(forecast['methods'][selected]['modes']))
print('Full-history force Strouhal (diagnostic only):', forecast['source_full_history_strouhal_diagnostic_only'])
print('Test duration:', forecast['test_duration'])
display(pd.DataFrame(forecast['spectra']).T)
display(pd.DataFrame(forecast['sindy_candidates']).T)'''),md(COMMON),code(CONTROL),md('''## 4. Interpret and extend without changing the retained test
Explain the gap between the eight-mode POD oracle and DMD. Distinguish
discretization error, representation error and autonomous dynamics error.
Why would resetting to a true state at frame210 make the forecast easier?
Propose a new-Re case and a longer-duration test, with thresholds frozen before
seeing either. For a noise-robust DMD study, add a declared observation-noise
protocol and a matched robust method; this exercise does not implement PyDMD's
noise-aware variants. Do not infer vortex-segmentation performance from a good
forecast contour. Common contour levels are for display only; no data smoothing
or resolution enhancement is used.'''),md(SOURCES)]
    cells[0].source+=badge(path.relative_to(ROOT).as_posix())
    cells[1].source=bootstrap(path.parent.relative_to(ROOT).as_posix())+cells[1].source
    cells.append(code('''after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_evidence}
assert after == before, 'Notebook modified retained data/evidence'
print('PASS: retained data and evidence unchanged.')'''))
    for i,c in enumerate(cells):c.id=hashlib.sha256((kind+str(i)+c.source).encode()).hexdigest()[:12]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    nbf.write(nb,path);print(path)


def make_pdf(kind):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak
    from reportlab.lib.enums import TA_LEFT
    name={'week05':'week05_modal_sensing','week07':'week07_modal_forecasting'}[kind]
    source=ROOT/'lectures/source'/f'{name}.md'
    out=ROOT/'output/pdf'/f'{name}.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CourseTitle',fontName='Helvetica-Bold',fontSize=21,leading=25,textColor=colors.HexColor('#173b56'),spaceAfter=16))
    styles.add(ParagraphStyle(name='CourseBody',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=10,alignment=TA_LEFT))
    story=[]
    for pi,page in enumerate(source.read_text().split('\n---\n')):
        if pi:story.append(PageBreak())
        for block in page.strip().split('\n\n'):
            if block.startswith(('# ','## ')):
                heading,_,body=block.partition('\n')
                level=2 if heading.startswith('## ') else 1
                story.append(Paragraph(html.escape(heading[level+1:]),styles['Heading2'] if level==2 else styles['CourseTitle']))
                if body:story.append(Paragraph(html.escape(body).replace('\n','<br/>'),styles['CourseBody']))
            else:
                story.append(Paragraph(html.escape(block).replace('\n','<br/>'),styles['CourseBody']))
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#c7d5df'));canvas.line(42,38,553,38)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#506070'))
        canvas.drawString(42,25,'FlowMLLab | Original teaching companion | Development branch')
        canvas.drawRightString(553,25,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=52,
        title=name.replace('_',' '),author='Ehsan Roohi / FlowMLLab').build(story,onFirstPage=footer,onLaterPages=footer)
    shutil.copy2(out,ROOT/'lectures'/out.name);print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pdf',action='store_true');args=parser.parse_args()
    for kind in ('week05','week07'):
        make_notebook(kind)
        if args.pdf:make_pdf(kind)
