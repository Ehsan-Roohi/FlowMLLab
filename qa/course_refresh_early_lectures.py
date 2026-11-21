"""Focused lecture companions for the existing Weeks2,3,5 lessons.

The validated original Week2 (24p) and Week3 (37p) cores are retained verbatim
as PDF pages. New pages are generated from this editable teaching source. The
Week5 guide retains its original technical/source paragraphs and adds the
worked reconstruction algorithm and figures. No solver or training is run.
"""
from pathlib import Path
import os,sys,json,hashlib,re,html
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
os.environ.setdefault('FLOWMLLAB_TNR_DIR',r'C:\Windows\Fonts' if os.name=='nt' else '')
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from qa.lecture_typography import register_reportlab
from qa.week01_2_math import Equation
import pymupdf as fitz

OUT=ROOT/'tmp/course_refresh_early/lectures';OUT.mkdir(parents=True,exist_ok=True)
fonts=register_reportlab('EarlyTNR')
STEM1='AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student'
STEM2='AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student'
STEM5='W5_Lab2_Sparse_Sensing_Dynamics'

def body(s):return ('text',s)
def eq(s):return ('equation',s)
def fig(stem,index,maxheight=None):
 receipt=json.loads((ROOT/'results/course_refresh_early'/stem/'execution.json').read_text())
 return ('figure',ROOT/receipt['inline_figures'][index]['path'],maxheight)
def named(name):return ('figure',ROOT/'results/course_refresh_early'/STEM5/name,None)
def old_modal_page(index):
 text=(ROOT/'lectures/source/week05_modal_sensing.md').read_text(encoding='utf-8').split('\n---\n')[index]
 text=text.replace("Study this lab after Week 7's modal dynamics; its directory does not impose an\nadditional mandatory pre-project dynamics unit.", "The required sensing lesson uses POD and least squares. Complete the optional autonomous-dynamics extension after Week 7.")
 return [body(block) for block in text.strip().split('\n\n') if block.strip()]

def render(pages,target,size,page_offset=0):
 landscape=size[0]>size[1];margin=35 if landscape else 43
 fs=15.2 if landscape else 11
 styles={'body':ParagraphStyle('Body',fontName=fonts['regular'],fontSize=fs,leading=fs*1.27,textColor=colors.black,spaceAfter=10),
 'title':ParagraphStyle('Title',fontName=fonts['bold'],fontSize=24 if landscape else 21,leading=28 if landscape else 25,textColor=colors.black,spaceAfter=15)}
 width=size[0]-2*margin
 story=[]
 for number,(title,blocks) in enumerate(pages):
  if number:story.append(PageBreak())
  story.append(Paragraph(html.escape(title),styles['title']))
  for block in blocks:
   if block[0]=='text':
    raw=block[1]
    if raw.startswith('#') and '\n' in raw:
     heading,rest=raw.split('\n',1);t='<b>'+html.escape(heading.lstrip('# '))+'</b><br/>'+html.escape(rest).replace('\n',' ')
    else:t=html.escape(raw).replace('\n',' ')
    t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t);t=re.sub(r'`([^`]+)`',r'<i>\1</i>',t)
    t=re.sub(r'^#+\s*','',t)
    story.append(Paragraph(t,styles['body']))
   elif block[0]=='equation':story.append(Equation(block[1],width,size=20 if landscape else 15))
   elif block[0]=='matrix_example':story.append(MatrixExample(width))
   elif block[0]=='figure':
    p=block[1];image=Image(str(p));scale=min(width/image.imageWidth,(block[2] or (345 if landscape else 350))/image.imageHeight)
    image.drawWidth=image.imageWidth*scale;image.drawHeight=image.imageHeight*scale;story.extend([image,Spacer(1,10)])
 def footer(canvas,doc):
  canvas.setFont(fonts['regular'],9);canvas.setFillColor(colors.black)
  canvas.drawString(margin,18,'FlowMLLab | Worked computational laboratory')
  canvas.drawRightString(size[0]-margin,18,str(doc.page+page_offset))
 SimpleDocTemplate(str(target),pagesize=size,leftMargin=margin,rightMargin=margin,topMargin=28 if landscape else 40,bottomMargin=34,title='FlowMLLab: worked teaching guide',author='Ehsan Roohi').build(story,onFirstPage=footer,onLaterPages=footer)
 doc=fitz.open(target)
 if len(doc)!=len(pages):raise RuntimeError(f'Unexpected page overflow: {target.name}: {len(doc)} vs {len(pages)}')
 return len(doc)

def append_core(name,base_count,pages):
 path=ROOT/'lectures'/name
 old=fitz.open(path)
 if len(old)<base_count:raise ValueError('Missing protected core pages')
 hashes=[hashlib.sha256(old[i].get_pixmap(matrix=fitz.Matrix(.75,.75),alpha=False).samples).hexdigest()for i in range(base_count)]
 extra=OUT/(path.stem+'_new_pages.pdf');render(pages,extra,(old[0].rect.width,old[0].rect.height),page_offset=base_count)
 merged=fitz.open();merged.insert_pdf(old,from_page=0,to_page=base_count-1);add=fitz.open(extra);merged.insert_pdf(add)
 result=OUT/path.name;merged.save(result,garbage=3,deflate=True);merged.close();add.close();old.close()
 final=fitz.open(result)
 assert hashes==[hashlib.sha256(final[i].get_pixmap(matrix=fitz.Matrix(.75,.75),alpha=False).samples).hexdigest()for i in range(base_count)]
 final.close();path.write_bytes(result.read_bytes())
 return {'path':path.relative_to(ROOT).as_posix(),'protected_core_pages':base_count,'protected_render_sha256':hashes,'changed_pages':list(range(base_count+1,base_count+len(pages)+1)),'page_count':base_count+len(pages),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def week2_pages():
 return [('From the width tuple to an actual network',[
 body('Aim: make a controlled depth/width experiment whose code implements the stated architecture. The notebook now builds one Dense layer for every entry in HIDDEN_UNITS. Thus (32,), (32,32) and (32,32,32) have one, two and three hidden layers, respectively. ACTIVATION is the nonlinear function used in each hidden layer; the scalar output remains linear.'),
 eq(r'h^{(l)}=\sigma(W^{(l)}h^{(l-1)}+b^{(l)}),\qquad N_\theta=\sum_l(n_{l-1}+1)n_l'),
 body('The input h(0) contains four standardized features: log10(Kn), accommodation coefficient alpha, pressure ratio Pi and temperature ratio Theta. W(l) maps the previous layer to the next; b(l) is its bias. The parameter-count assertion includes every bias and catches an architecture that disagrees with the exposed control.'),
 body('Keep the same physical samples, training-only scalers, seed, optimizer and maximum update budget for the first comparison. Select using validation loss. Record the number of parameters, actual accepted epochs, elapsed time and validation error. More depth increases compositional capacity but may make optimization more difficult; it is not an automatic accuracy improvement.'),
 body('Required experiment: compare the three tuples above, then change only activation for the validation-selected depth. Retain the original default result. Report whether a change reduces signed profile error, merely reduces training loss, or changes optimization cost. A CPU time depends on the recorded machine and thread settings.')]),
 ('Interpret the correct error target',[
 body('Aim: distinguish fitting noisy observations, approximating an analytic function and validating a physical flow solver. The Week2 data are a deliberately constructed positive mass-flow proxy. They do not establish a real transition-regime constitutive law.'),
 body('The random split is appropriate here because each row is an independently generated operating condition. It would be misleading for many nearby grid points from the same CFD realization. Training fits weights and scalers; validation chooses stopping/model settings; the final evaluation must not choose another model.'),
 body('The aggregate test table compares against noisy labels. The fixed-alpha, fixed-Pi, fixed-Theta curve compares against the noise-free formula. A model can be smoother than the labels and still have a systematic signed discrepancy. The uncertainty floor of noisy labels and a deterministic physical bias are different quantities.'),
 body('For the extrapolation section, low-Kn rows define both fit and validation. Higher-Kn rows are outside that training support. A good validation loss within the low-Kn interval is not evidence of extrapolation accuracy. Compare the supplied high-Kn error distribution, negativity and monotonicity, and explain which regime is absent from training.'),
 body('Submit the split sizes and feature ranges, architecture/optimizer controls, validation choice, noisy-target metrics, noise-free signed profile, and the first condition where the declared physical diagnostic fails. Do not tune a network against the public analytic profile and then describe that same profile as an untouched blind test.')])]

def week3_pages():
 return [
 ('Lab1 and Lab2: what each calculation establishes',[
 body('Lab1 derives Maxwellian samples and their macroscopic moments, then separates finite-population bias from random sampling error. Its final neural comparison changes label noise while retaining identical model settings. The analytic slip curve is a teaching proxy, not a kinetic-flow solution.'),
 body('Lab2 traces movement, diffuse walls, local elastic collisions and sampling in a deliberately simplified CPU cavity. Its required diagnostics are configuration, averaging-window times, pair conservation, native profiles and cell-level field statistics. The independently retained HS-NTC wall-pressure comparison uses a different calibrated model and physical case.'),
 body('Controls: N_SAMPLE changes the displayed cloud; N_VALUES and N_REPEATS change the sampling study. In the cavity, nx/ny control cells, ppc controls particles, dt controls physical stepping, and sample_start/sample_stride define the averaging window. Seed changes a realization, not the physical model.'),
 body('Required route: complete the CPU model first. The optional GPU flags remain false by default. Neither a missing optional GPU folder nor an unrun production stage invalidates a completed CPU lesson; neither can be presented as new validated production evidence.')]),
 ('Lab1: derive and check the moment estimators',[
 eq(r'\widehat{\mathbf{U}}=\frac{1}{N}\sum_i\mathbf{c}_i,\quad\mathbf{C}_i=\mathbf{c}_i-\widehat{\mathbf{U}}'),
 eq(r'T_N=\frac{m}{3k_BN}\sum_i|\mathbf{C}_i|^2,\quad p_N=n_0k_BT_N'),
 eq(r'\mathbf{q}_N=\frac{mn_0}{2N}\sum_i|\mathbf{C}_i|^2\mathbf{C}_i'),
 body('Here c is molecular velocity [m/s], U is bulk drift, C is thermal velocity, m is molecular mass [kg], kB is Boltzmann constant [J/K], and n0 is number density [1/m3]. Temperature, pressure and heat flux have units K, Pa and W/m2. Removing the drift prevents organized motion being counted as temperature.'),
 body('For independent fixed-N equilibrium samples, E[T_N]=(N-1)T/N. Multiplication by N/(N-1) removes that particular estimator bias for N>1. It does not prove a nonstationary, correlated, variably populated cavity has been corrected. The heat-flux vector has zero equilibrium expectation, but the average of its noisy norm is positive.')]),
 ('Lab1: normalized random error and signed bias',[
 fig(STEM1,1,340),
 body('Velocity RMSE is divided by the exact thermal component scale; temperature RMSE is divided by generating temperature. Their units are no longer mixed. The middle panel tests the signed finite-N bias separately from RMSE. More independent repeats sharpen the estimate of bias; they do not remove the denominator-N bias.')]),
 ('Lab1: label quality can dominate the learned trend',[
 fig(STEM1,2,330),
 body('The two MLPs have the same 75 training coordinates, initialization, two 24-unit tanh layers and L-BFGS settings. Only prescribed label-noise amplitude changes. The noise-free curve is public analytic teaching evidence, not an independent DSMC reference. The zero clipping of noisy labels can itself introduce bias near zero.')]),
 ('Lab1: inspect the signed discrepancy',[
 fig(STEM1,3,325),
 body('Use this profile with the monotonicity/negativity table and accepted iteration counts. A lower training loss can fit more noise. The fixed default comparison has no architecture search; if students tune depth or regularization, they must introduce a separate validation design before claiming independent evaluation.')]),
 ('Lab2: collision conservation is necessary, not sufficient',[
 eq(r'\mathbf{V}=\frac{\mathbf{v}_1+\mathbf{v}_2}{2},\quad g=|\mathbf{v}_1-\mathbf{v}_2|'),
 eq(r'\mathbf{v}_1^{\prime}=\mathbf{V}+\frac{g}{2}\mathbf{e},\quad\mathbf{v}_2^{\prime}=\mathbf{V}-\frac{g}{2}\mathbf{e}'),
 body('The random unit vector e is isotropic: draw cos(theta) uniformly, then the azimuth uniformly. Pair momentum and kinetic energy remain unchanged, which the notebook checks numerically. The expected local pair count is approximately Ncell*cbar*dt/(2*Kn*L), with stochastic rounding to an integer.'),
 body('This mini model does not implement pair-speed-dependent VHS-NTC acceptance. A conserved collision can still have an inaccurate collision frequency. Keep the simplified physics identified when discussing slip, temperature and pressure proxies; no calibrated Kn transfer has been established.')]),
 ('Lab2: compare the actual sampling windows',[
 body('Sampling occurs after a move. For zero-based step k, the observation time is (k+1)dt. Equal numbers of recorded fields do not imply equal physical windows or equal independent sample counts.'),
 body('Quick: 16 by 16 cells, 8 particles per cell, dt = 0.003, 650 steps. Sampling begins at 0.903 and ends at 1.923, with spacing 0.03 and 35 recorded fields; final time 1.95.'),
 body('Better sampled: 20 by 20 cells, 24 particles per cell, dt = 0.0025, 2200 steps. Sampling begins at 1.7525 and ends at 5.495, with spacing 0.0075 and 500 recorded fields; final time 5.5.'),
 body('These cases also use different seeds. Therefore their difference combines grid, population, duration, averaging and realization effects. It is a diagnostic comparison, not an isolated refinement error. A controlled time-step experiment halves dt, doubles steps and sample_stride, and sets sample_start_new = 2*sample_start + 1. The extra one accounts for sampling after a zero-based move; merely doubling sample_start shifts every observation earlier by half of the old time step.'),
 body('The collision-scale indicators dx/lambda and dt/tau are screening diagnostics, not a proof of convergence. Inspect stationarity and independent realizations before using a smoother averaged field as an ML target.')]),
 ('Lab2: read unsmoothed native fields honestly',[
 fig(STEM2,3,330),
 body('The historical temperature estimator and the cellwise pressure proxy are identified explicitly. Cell-center samples do not supply boundary values; blank strips are not filled by wall extrapolation. The quick field remains available, but its noisy topology is not presented as a reliable vortex measurement.')]),
 ('Lab2: compare profiles before interpreting topology',[
 fig(STEM2,4,330),
 body('The vertical profile uses linear interpolation between the two cell centers straddling x/L=0.5 and retains native y locations. No data are invented at the wall. The quick/better discrepancy has no confidence-interval interpretation because their physical averaging windows and numerical controls differ.')]),
 ('Week3: required engineering decisions',[
 body('Experiment 1: compare denominator-N and denominator-(N-1) temperature estimates at N = 30, 60, 300. Report signed bias and normalized RMSE. Doubling the number of independent repeats should reduce uncertainty in the measured bias without changing its exact expectation.'),
 body('Experiment 2: reduce bulk drift from 100 to 10 m/s at fixed temperature. Compare absolute drift error and drift-relative error; do not divide by an exactly zero component. Explain the cost of resolving a small mean against large molecular fluctuations.'),
 body('Experiment 3: keep the mini-cavity grid and matched physical sampling window fixed while changing particles per cell. Use at least three declared seeds. Show individual native profiles and their mean; distinguish realization spread, estimator bias and unknown discretization error.'),
 body('Submit both completed CPU notebooks and the combined report. Include every changed control, random seed, averaging-window time, measured runtime, an error definition and one limitation. Keep useful failed results. A smooth contour alone is not an acceptance criterion.')])]

def week5_pages():
 pages= [
 ('Sparse sensing: reconstruction before forecasting',[
 body('Aim: reconstruct a real cylinder-wake transverse-velocity field from a small set of point measurements, and explain why placement, rank and noise change the answer. The required notebook sections use CPU linear algebra. The later sparse-dynamics branch is optional until the Week7 time-series prerequisites are complete.'),
 body('A sensor reconstruction receives new observations at every frame. An autonomous forecast receives an initial state and then evolves without target updates. These are different information budgets. The companion teaches both, but scores them separately.'),
 body('Prerequisites are SVD/POD, least squares, standardization and complete-case splitting. Work through the algebraic sensitivity calculation before opening the wake fields. Then select a sensor budget on a different Reynolds case, inspect the frozen physical reconstruction and explain which part of the error additional measurements cannot remove.'),
 body('The fields are the original FlowMLLab LBM data released in cylinder-cfd-v1. They predate this teaching experiment. The low-Mach wake is distinct from the rarefied hypersonic cylinder study: this case has no shock. The plotted region is an extracted downstream rectangle, not the physical computational boundary.'),
 body('By the end of the required lesson, you should be able to derive the observation system, relate noise amplification to a singular value, justify a layout and budget without test leakage, and identify a spatial or temporal failure that one aggregate norm hides. The later extension asks a different question: whether a sparse polynomial ODE predicts modal evolution without future measurements.'),
 body('Keep a record of the basis rank, sensor locations, every noise seed, validation rule, full-field projection floor and measured CPU time. The default sensor study is complete by itself. The optional dynamics path retains its own time split, model selection and continuous-rollout assessment.')]),
 ('Derive the gappy-POD observation equation',[
 body('Flatten a field into q with n spatial entries. Fit the training mean qbar and a matrix Phi of r orthonormal POD columns. Coefficients a describe the retained part, and residual e represents unresolved structure. C is a row-selection matrix for m sensors; epsilon is observation noise.'),
 eq(r'q=\bar q+\Phi a+e,\qquad y=Cq+\epsilon'),
 eq(r'\widehat a=\arg\min_a\|(C\Phi)a-(y-C\bar q)\|_2^2'),
 eq(r'\widehat q=\bar q+\Phi\widehat a'),
 body('At least r independent measurements are necessary for full column rank. The implementation uses a least-squares solve, not an explicitly formed matrix inverse. Center the measurements with the training mean at the selected coordinates. A mean from the final case would leak target information.'),
 body('The full-field POD oracle projects the entire target field into the fixed basis. It establishes a representation floor and is intentionally given more information than any sensor model. It is not a deployable reconstruction baseline.')]),
 ('A numerical conditioning calculation',[
 *old_modal_page(4),
 body('The notebook now solves this system explicitly and plots coefficient error against the measured matrix condition number. Change only perturbation magnitude first. Then change row separation. A particular noise direction need not attain the worst-case singular-value bound.')]),
 ('Choose informative, distinct measurement locations',[
 body('Pivoted QR of the transposed spatial basis selects r independent rows. For budgets above r, greedy D-optimal oversampling adds the row with largest leverage against the current Gram matrix. This seeks independent information rather than simply taking the tail of a QR permutation.'),
 eq(r'G=\Phi_C^T\Phi_C,\qquad\ell_i=\phi_i^TG^{\dagger}\phi_i'),
 body('Here Phi_C contains the selected rows; phi_i is a candidate row expressed as a column. The code excludes already selected coordinates. The pseudoinverse protects the numerical leverage calculation, while reconstruction still checks full column rank.'),
 body('Random layouts use the same point budget. A fixed noise seed produces a full-grid artificial perturbation that both layouts sample, so the method comparison is paired. Five noise/layout seeds are five measurement experiments on one CFD trajectory, not five independent flow realizations.'),
 body('More sensors can stabilize the observation system but cannot recover a missing spatial mode. Rank, sensor count and noise must therefore be interpreted together.')]),
 ('Freeze physical cases and select a budget',[
 *old_modal_page(1),
 body('The exposed controls now pass directly to the sensing function: SENSOR_RANK, SENSOR_BUDGETS, NOISE_FRACTION, NOISE_SEEDS and VALIDATION_TARGET. Rank changes require validation-based selection too. The default experiment preserves rank 8, budgets 8/16/32 and noise 1% of training RMS. The final public case does not choose a budget.')]),
 ('Physical reconstruction on the held-out case',[
 named('sensor_diagnostic_1.png'),
 body('The filled LBM reference and the reconstructed contours share identical physical levels. Markers are actual selected sensor coordinates; the error map is linear and includes the full maximum. Data remain at the native 32 by 78 ROI resolution. Contour interpolation does not create additional CFD resolution.')]),
 ('A field norm does not replace local and temporal checks',[
 named('sensor_diagnostic_2.png'),
 body('The spatial curve uses one declared native row, not a smoothed fit. The temporal diagnostic evaluates every retained Re = 105 frame. A small global mean can hide a badly reconstructed phase or local extremum. These time frames are correlated observations; do not count them as 281 independent physical cases.')]),
 ('Conditioning, measurement count and the representation floor',[
 named('sensor_diagnostic_3.png'),
 body('Every declared noise/layout seed is shown, along with the mean across those replicates. This does not provide a confidence interval over independent CFD solutions. The oracle floor isolates information lost by the fixed POD basis. Error above that floor includes observation and inversion effects, while the original LBM has its own discretization limits.')]),
 ('Optional bridge: identify a sparse differential equation',[
 *old_modal_page(2),
 eq(r'z(t_{k+w})-z(t_k)\simeq\left[\int_{t_k}^{t_{k+w}}\Theta(z(t))\,dt\right]\Xi'),
 body('z contains standardized POD coefficients; Theta contains constant, linear, quadratic and cubic monomials; Xi is the coefficient matrix. The four-interval window uses trapezoidal integration. Sequential thresholded least squares fits, removes small coefficients, then refits remaining terms. Integral regression avoids directly differentiating noisy data but does not eliminate identifiability or stability limitations.')]),
 ('Assess autonomous dynamics without resetting to the answer',[
 body('Fit the two-mode basis and coefficient scaling only on the first 160 Re = 110 frames. Frames 160-209 select the sparsity threshold. Start from the final training state and integrate continuously through validation and the final 71 assessment frames. Resetting to a true intermediate state would make a different, easier problem.'),
 body('Inspect the selected polynomial terms, signs and units. A near-periodic orbit constrains its neighborhood; it cannot uniquely identify arbitrary off-orbit cubic dynamics. A model with a tiny local fit can diverge during autonomous integration. Failed candidates are retained rather than clipped.'),
 body('Compare the sparse ODE with a rank-two full-field POD oracle and DMD. The oracle sees target states; DMD and the ODE do not. Distinguish representation error, coefficient-dynamics error and coarse-grid LBM error. The optional shared helper also fits the Week7 nonlinear history baseline, so its runtime exceeds the two-state regression cost.'),
 body('The default covers sensing; set RUN_DYNAMICS_BRIDGE=True after Week 7 to execute the additional dynamics study.')]),
 ('Five controlled investigations',[
 body('1. Vary algebraic observation perturbation 0.001/0.01/0.1 at fixed sensor matrix. Compare coefficient-error amplification with the smallest singular value. Then vary row separation and explain the change.'),
 body('2. At rank 8, compare 8/16/32 sensors with the same five noise seeds. Select using Re = 100 only. Report the selected count, every seeded Re = 105 score and a native profile; state explicitly if the validation target is missed.'),
 body('3. At a frozen rank/layout, compare noise fractions 0, 0.01, 0.03. Inspect signed spatial discrepancies relative to the oracle. Zero observation noise is not zero reconstruction error.'),
 body('4. Compare rank 4 and 8 with valid sensor budgets. Record SVD/offline cost and reconstruction/online cost separately. More captured energy may increase the measurement requirement or worsen conditioning.'),
 body('5. After Week7, enable sparse dynamics. List retained terms and compare a continuous rollout with the oracle and DMD. Explain the earliest failure rather than presenting only the most favorable trajectory.')]),
 ('Assessment, provenance and scope',[
 *old_modal_page(3),
 body('Record rank, budget, noise fraction, seed, split role, condition number, field error, worst-frame error and runtime. Preserve the default result before post-hoc sensitivity studies. The same public cases cannot become newly blind by rerunning the notebook. A genuinely new physical assessment requires withheld cases and a protocol frozen before their inspection.')])]

 # Merge the derivation with its hand-checkable numerical example, and selection
 # algorithm with the physical-case protocol. No exercise or result is removed.
 worked=[body('Worked calculation: an exactly observed state in the POD span has no representation residual. For an identity observation matrix, the coefficient error equals measurement noise. Now use nearly parallel rows:'),
         ('matrix_example',),
         eq(r'\epsilon=(0,0.01)^T,\qquad \widehat a_1=1,\quad\widehat a_2=\frac{1.03-1}{0.01}=3'),
         body('The exact coefficients are (1,2). An observation perturbation of norm 0.01 causes coefficient error 1: a hundred-fold amplification. This local matrix is an algebraic observation system, not a claimed orthonormal full POD basis. Replace the row separation 0.01 by 0.1; the same perturbation then gives coefficient error 0.1. The notebook solves both cases and plots error against condition number.'),
         body('Exercise: add a component outside the retained span. Even exact, well-conditioned sensors cannot remove that residual. Then vary perturbation direction: a particular noise realization need not attain the worst-case singular-value bound.')]
 merged_derivation=('Gappy POD: derive it, then calculate it',pages[1][1]+worked)
 merged_selection=('Sensor selection and the frozen-case protocol',pages[3][1]+pages[4][1])
 return [pages[0],merged_derivation,merged_selection,*pages[5:]]


class MatrixExample(Flowable):
 """Clean vector brackets for the two-sensor numerical example."""
 def __init__(self,width):
  Flowable.__init__(self);self.width=width;self.height=64
 def draw(self):
  canvas=self.canv;canvas.setFillColor(colors.black);canvas.setStrokeColor(colors.black)
  x=(self.width-328)/2
  for label,rows,boxwidth in [('A =',[[1,0],[1,.01]],100),('a =',[[1],[2]],70),('y =',[[1],[1.02]],90)]:
   canvas.setFont(fonts['italic'],14);canvas.drawString(x,27,label);left=x+28;right=left+boxwidth-40
   for side,direction in [(left,1),(right,-1)]:
    canvas.line(side,13,side,49);canvas.line(side,49,side+direction*5,49);canvas.line(side,13,side+direction*5,13)
   canvas.setFont(fonts['regular'],13)
   for i,row in enumerate(rows):
    for j,value in enumerate(row):canvas.drawCentredString(left+10+(j+.5)*(right-left-20)/len(row),36-18*i,str(value))
   x+=boxwidth+24

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--only',nargs='+',choices=['week2','week3','week5'],default=['week2','week3','week5']);args=ap.parse_args()
 receipt_path=ROOT/'qa/course_refresh_early_lectures.json'
 if receipt_path.exists():receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
 else:
  consolidated=ROOT/'qa/course_refresh_early_final.json'
  prior=json.loads(consolidated.read_text(encoding='utf-8')) if consolidated.exists() else {}
  receipt=prior.get('lecture_build',{'scope':'Teaching only; no CFD or training performed by this builder','artifacts':[]})
 artifacts={a['path']:a for a in receipt['artifacts']}
 if 'week2'in args.only:
  a=append_core('week02_supervised_learning_rarefaction.pdf',24,week2_pages());artifacts[a['path']]=a
 if 'week3'in args.only:
  a=append_core('week03_kinetic_dsmc.pdf',37,week3_pages());artifacts[a['path']]=a
 if 'week5'in args.only:
  path=ROOT/'lectures/week05_modal_sensing.pdf';pages=week5_pages();render(pages,path,(595,842))
  a={'path':path.relative_to(ROOT).as_posix(),'changed_pages':list(range(1,len(pages)+1)),'page_count':len(pages),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};artifacts[a['path']]=a
 receipt['artifacts']=list(artifacts.values());receipt['builder']='qa/course_refresh_early_lectures.py';receipt['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({a['path']:a['page_count']for a in receipt['artifacts']},indent=2))
if __name__=='__main__':main()
