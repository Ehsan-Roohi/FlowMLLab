"""Append PINN theory and verified computation-cost lessons; preserve earlier pages."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys
import numpy as np
import pymupdf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'qa'))
from lecture_typography import configure_matplotlib
NAMES=['week04_cavity_surrogates_deeponet','week13_rectangular_cavity_pinn']
TAG='fresh_pinn_protocol_20261115'

def digest(payload):return hashlib.sha256(payload).hexdigest()
def body(s):return {'type':'body','text':s}
def math(s):return {'type':'math','latex':s}
def small(s):return {'type':'small','text':s}
def table(rows,widths):return {'type':'table','rows':rows,'widths':widths}
def page(title,*blocks):return {'title':title,'blocks':list(blocks),'campaign_extension_id':TAG}

def prior_receipt():
    target=ROOT/'qa/cavity_pinn_lecture_prior_pages.json'
    if target.exists():return
    record={}
    for name in NAMES:
        path=ROOT/'lectures'/f'{name}.pdf'
        with pymupdf.open(path) as doc:
            record[name]={'pages':len(doc),'pdf_sha256':digest(path.read_bytes()),
                          'page_text_sha256':[digest(p.get_text().encode()) for p in doc],
                          'page_rgb72_sha256':[digest(p.get_pixmap(matrix=pymupdf.Matrix(1,1),alpha=False).samples) for p in doc]}
    target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

def cost_figure(source):
    source=Path(source);data=json.loads(source.read_text(encoding='utf-8-sig'))
    rows=data['rows']
    assert [r['chunk_size'] for r in rows]==[256,512,1024,2048]
    for row in rows:
        assert len(row['times_seconds'])==5
        assert np.isclose(np.median(row['times_seconds'][-3:]),row['median_last3_seconds'],rtol=1e-13)
        assert row['gradient_max_abs_difference']<=5e-15
    assert max(r['initial_loss'] for r in rows)==min(r['initial_loss'] for r in rows)
    dest=ROOT/'results/cavity_pinn_campaign/cost_study';dest.mkdir(exist_ok=True,parents=True)
    retained=dest/'chunk_cost_probe.json'
    if retained.exists() and retained.read_bytes()!=source.read_bytes():raise RuntimeError('Different cost evidence already exists.')
    retained.write_bytes(source.read_bytes());assert digest(retained.read_bytes())==digest(source.read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    configure_matplotlib()
    fig,ax=plt.subplots(figsize=(6.1,3.8))
    chunks=np.array([r['chunk_size'] for r in rows]);medians=np.array([r['median_last3_seconds'] for r in rows])
    for i,row in enumerate(rows):
        ax.scatter(np.repeat(row['chunk_size'],5),row['times_seconds'],s=19,c='.65',zorder=2,
                   label='Five consecutive Adam updates' if i==0 else None)
    ax.plot(chunks,medians,'o-',color='black',lw=1.3,markersize=5,label='Median of last three')
    ax.set_xscale('log',base=2);ax.set_xticks(chunks,labels=[str(x) for x in chunks])
    ax.set(xlabel='Points per computational chunk',ylabel='Time per Adam update (s)',ylim=(0,1.8))
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False,ncol=1,fontsize=9)
    fig.subplots_adjust(left=.15,right=.98,top=.96,bottom=.33)
    path=dest/'chunk_cost_profile.png';fig.savefig(path,dpi=220);plt.close(fig)
    (dest/'manifest.json').write_text(json.dumps({'kind':data['kind'],'source_probe':{'path':retained.relative_to(ROOT).as_posix(),'sha256':digest(retained.read_bytes())},'figure':{'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path.read_bytes())},'timing_scope':'five consecutive Adam updates per chunk; median of last three; not independent replicates'},indent=2)+'\n',encoding='utf-8')
    return data,retained,path

def make_pages(data,retained,figure):
    rows=data['rows'];ratio=rows[0]['median_last3_seconds']/rows[-1]['median_last3_seconds']
    ghia=[
    page('A label-free solution is a different experiment',
         body('Week 4 Lab 9 solves the steady equations anew at Re = 100, 400, 1000, 3200, 5000, 7500 and 10000. No CFD velocity values or Ghia samples enter fitting. Each run has its own Reynolds number; this does not test one model generalizing to an unseen Re.'),
         body('Coordinates x = X/W and y = Y/W lie in the unit square. Velocities use lid speed U; pressure uses density times U squared. Reynolds number Re = UW/nu uses kinematic viscosity nu. Subscripts below denote derivatives in these physical nondimensional coordinates.'),
         math(r'r_c=u_x+v_y=0'),
         math(r'r_x=u u_x+v u_y+p_x-Re^{-1}(u_{xx}+u_{yy})=0'),
         math(r'r_y=u v_x+v v_y+p_y-Re^{-1}(v_{xx}+v_{yy})=0'),
         body('The open top wall has u = 1 and v = 0; all other open walls have u = v = 0. A streamfunction gives u = psi_y and v = -psi_x, so continuity follows by mixed-partial symmetry. Signed vorticity is v_x - u_y = -Laplacian(psi). Manufactured derivative and pressure-sign tests verify the implementation.'),
         table([['Experiment','Information and question'],['Existing POD-DeepONet','Learns CFD fields; whole-Re tests at 3200 and 7500.'],['New Ghia PINN baseline','Learns the case-specific PDE; references are opened after freezing.']], [145,330]),
         small('The 42-run design is seven Re values, three seeds (7, 23, 61), and two corner collars. Submission, successful execution and validation are separate milestones. Fresh accuracy conclusions require the completed evidence ledger.')),
    page('The singular lid corners need their own test',
         body('A discontinuous constant lid is part of the Ghia problem. Replacing it by a Gaussian or polynomial makes a different benchmark. The two upper corner points have incompatible velocity traces; their exported values are explicitly masked rather than fabricated.'),
         body('The admitted representation includes the leading Stokes corner streamfunction and pressure. For the upper-left corner use X = x, Z = 1 - y, theta = atan2(Z,X), r squared = X squared + Z squared and nu = 1/Re.'),
         math(r'A=-\frac{\pi^2}{\pi^2-4},\quad B=\frac{2\pi}{\pi^2-4},\quad D=\frac{4}{\pi^2-4}'),
         math(r'\psi_L=(A+B\theta)Z+D\theta X,\qquad p_L=-\frac{2\nu(BZ+DX)}{r^2}'),
         body('At the right corner use X = 1 - x and reverse the pressure sign. The isolated corner fields cancel the leading Stokes momentum defect; the protocol records global blending and independent sign/derivative checks. Nonlinearity and global blending can still leave subleading singular residuals.'),
         body('Training points exclude top-corner quarter-disks of radius epsilon = 0.02W; the paired experiment uses 0.01W with the same initial network. The open-wall lid remains constant. No artificial boundary condition is applied on the collar arcs.'),
         body('This is a punctured-collocation approximation, not demonstrated full-domain strong-residual convergence. Audit original residuals in shells both outside and inside the collars. Compare actual fields on common coordinates and preserve any weak-vortex or near-wall failure.'),
         small('Corner analysis: Burda, Novotny and Sistek, ICCFD7 (2012), Section 4. The local sign convention and coefficients were independently checked; the exact implementation is in docs/CAVITY_PINN_CAMPAIGN_PROTOCOL.md.')),
    page('Read the recorded loss and endpoint correctly',
         body('The baseline uses four tanh hidden layers of width 64, float64, and 2048 fixed uniform collocation points outside each declared collar. Adam runs 2000 updates, followed by 500 L-BFGS outer updates with strong-Wolfe searches. Samples and objective remain fixed through line searches.'),
         math(r'J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\left[r_x(\mathbf{x}_i;\theta)^2+r_y(\mathbf{x}_i;\theta)^2\right]'),
         body('The last complete fixed-budget model is the endpoint. The independent outer and shell audits are computed at the end and do not select a checkpoint. There is no development-audit history in this baseline.'),
         table([['Recorded quantity','What it means'],['Training momentum RMS','Square root of the last recorded closure loss outside the collar.'],['Last closure','Not necessarily a freshly recomputed accepted-endpoint loss.'],['Final independent audit','Raw residuals on fixed outer points and corner shells; not a training curve.'],['Objective evaluations','Includes line-search closures; differs from outer-update count.'],['Time','Optimizer-step time; admission, checkpoint I/O and final scoring are separate.']], [140,335]),
         body('Do not label a training curve as validation loss. Small loss does not establish reference accuracy. Residual-guided sampling, alternative optimizers and deeper networks are future controlled variants; their mechanisms are studied in the new notebook but no gain is implied.'),
         small('A complete recovery preserves model, optimizer history, samples, RNG, phase, counters, configuration/source identities and frozen endpoint. Interrupt/restart equivalence is tested within the admitted runtime.')),
    page('Freeze first, then inspect the physical solution',
         body('The postprocessor verifies the completed model identity, freezes its evaluation at exact reference coordinates and hashes that prediction before loading reference values. Ghia predictions are direct network evaluations at the original tabulated positions, not interpolated dense predictions.'),
         body('The full-field comparison uses exactly the 511-by-511 retained CFD interior nodes. Velocity-vector errors retain u and v signs. Frozen vorticity scores compare native PINN automatic derivatives with retained CFD transported vorticity. Separate plots apply the same centered curl to both velocity fields on the 509-by-509 common inner grid. There is no CFD pressure truth.'),
         math(r'E_{\mathbf{u},R}=\sqrt{\frac{\sum_{i\in R}w_i[(\hat u_i-u_i)^2+(\hat v_i-v_i)^2]}{\sum_{i\in R}w_i(u_i^2+v_i^2)}}'),
         body('Here R is a declared physical region and w_i a quadrature weight. Report absolute error and reference RMS beside the relative ratio, especially for weak lower vortices. Keep both screened and unfiltered Ghia scores under the existing printed-entry policy.'),
         table([['Predeclared numerical target','Necessary additional interpretation'],['Outer momentum RMS <= 0.01','Raw shell distributions still need inspection.'],['Ghia u and v relative L2 <= 2%','Published and screened values are both retained.'],['Interior vector error <= 5%','CFD has its own unresolved grid sensitivity.'],['Paired collar difference <= 2%','Sensitivity evidence, not full corner convergence.']], [220,255]),
         body('The retained 256-to-512 CFD vector sensitivities are 2.03%, 3.14%, 4.72% and 6.32% at Re3200, 5000, 7500 and 10000. These are not exact-error bounds. A steady PINN and a short perturbation experiment do not establish asymptotic stability. All seeds and failures remain visible.')),
    page('A measured GPU cost improvement without changing the samples',
         body('All four chunk settings start from the same network and use the same 2048-point objective on a Quadro RTX 8000 using PyTorch 2.5.1+cu118. Only computational chunk size changes. Gradients are accumulated over the full sample set; this is not a different stochastic minibatch experiment.'),
         {'type':'figure','path':figure.relative_to(ROOT).as_posix(),'width':440,'max_height':280,
          'caption':'Discrete measured chunk settings. Gray points show five consecutive Adam updates; the black curve joins medians of the last three. These are not independent timing replicates. Lines guide the eye and do not certify untested chunk sizes.'},
         table([['Chunk points','Median last 3 (s)','Peak allocated (MB)']]+[[str(r['chunk_size']),f"{r['median_last3_seconds']:.3f}",f"{r['peak_allocated_bytes']/1e6:.1f}"] for r in rows],[145,165,165]),
         body(f'At chunk 2048 the measured median update is {ratio:.2f} times faster than chunk 256, with about 393 MB peak allocated GPU memory. The largest initial gradient difference is 2.22 x 10<super>-16</super>; all initial losses agree. This checks one admitted state, not every possible trajectory or device.'),
         small('Why it helps: fewer small launches and better GPU work aggregation can reduce overhead. This is a computational cost observation, not a solution-accuracy result or a change in the PDE. Preserve precision, samples and objective; remeasure memory and derivatives when changing hardware. Source: results/cavity_pinn_campaign/cost_study/chunk_cost_probe.json.')),
    page('Lab 9 deliverables: make the evidence auditable',
         body('<b>Table A - problem identity.</b> For Re100 and one high-Re case, record lid, physical units, collar, network, seed, sample count and audit roles. Identify a mismatch before any ranking.'),
         body('<b>Table B - sampling exercise.</b> Change the notebook toy proposal mixture from 0.2 to 0.1 and 0.5. Verify the exact expected importance-weighted finite-pool objective. Explain why a plain adaptive sample mean or clipped weights change the objective. This exercise is not an adaptive PINN result.'),
         body('<b>Figure 1/Table C - cost and training history.</b> Overlay paired collars against closure count and elapsed optimizer time. Use the recorded training-RMS label. Compare final independent audits separately; do not invent an audit trajectory or select a checkpoint from reference error.'),
         body('<b>Figure 2/Table D - collar sensitivity.</b> At matched Re/seed/budget, compare epsilon 0.02 and 0.01. Show original shell residuals, signed native profiles and common-grid differences. State where a weak vortex or near-wall mismatch remains.'),
         body('<b>Figure 3/Table E - accuracy and seeds.</b> Report whole-interior and lower-quarter reference RMS, absolute and relative vector errors, Ghia scores and signed vorticity. Show each predeclared seed and failure, not just a best run.'),
         body('<b>Table F - restart.</b> In a separate small exercise, interrupt Adam and L-BFGS once each. Compare next accepted update, full state, counters and predictions with an uninterrupted control.'),
         small('Conclude with observable-specific gains/losses and unresolved reference/collar sensitivity. All-seven case-specific neural coverage already exists in the literature. Our exercise tests reproducibility and controlled physical assessment, not priority or unseen-Re operator generalization.'))]
    d5=[
    page('Fresh D5 training must preserve the author formulation',
         body('Use x = X/W in [0,1] and y = Y/W in [0,5], measured upward from the floor. Velocities use U and pressure uses density times U squared. Re = UW/nu uses width, not depth. The original network receives y_source = y - 4; shifting plotted coordinates must not silently change network inputs.'),
         math(r'\eta=y/5:\qquad\partial_y=\frac{1}{5}\partial_\eta,\quad\partial_{yy}=\frac{1}{25}\partial_{\eta\eta}'),
         math(r'\psi=\psi_{\rm lid}+B^2E(d)N_\psi,\qquad p=E(d)N_p,\qquad d=5-y'),
         body('For this width-one, depth-five geometry the analytical wall factor and Gaussian-regularized lid lifting are:'),
         math(r'B=16x(1-x)\eta(1-\eta)'),
         math(r'\psi_{\rm lid}=(y-5)\eta^2G(x)\exp[-(5-y)^2/\delta_y^2]'),
         math(r'G(x)=(1-e^{-(x-1)^2/\delta_x^2})(1-e^{-x^2/\delta_x^2})'),
         small('The retained source uses delta_x = sqrt(0.001), delta_y = 0.1. N_psi and N_p are network outputs. Differentiate the complete wall/lid/envelope products; no velocity-boundary penalty is required.'),
         math(r'E(d)=\exp\{-4.212[\operatorname{softplus}_{10}(d-0.25)-\operatorname{softplus}_{10}(-0.25)]\}'),
         math(r'\operatorname{softplus}_{10}(z)=\frac{1}{10}\log(1+e^{10z})'),
         body('The source multiplies each raw momentum residual by M/E. Here r_L and r_R are distances to the upper corners and a = 0.01. This smooth mask is separate from the sampler exclusion radius.'),
         math(r'M=(1-e^{-r_L^2/a^2})(1-e^{-r_R^2/a^2}),\qquad J=\operatorname{mean}[(Mr_x/E)^2+(Mr_y/E)^2]'),
         body('Fresh trials use the authored balanced sampler: 50% full domain, 30% upper region, 15% floor and 5% logarithmic lower-corner points. Only top corners are excluded; lower corners are sampled down to radius 0.0001. This differs from the old all-four-corner uniform exclusion. Audit raw unmasked residuals separately.'),
         small('Source: Roohi and McDevitt, arXiv:2610.38814v2 and byte-verified cavity.py. Fresh matched trials are distinct from historical transfer/refinement histories.')),
    page('Compare actual optimizer variants at matched information',
         body('The intended routes are Adam followed by L-BFGS, and the authentic SOAP followed by self-scaled Broyden. Pin the implementation and full state. A dense Broyden alternative cannot silently be described as the same compact author implementation.'),
         table([['Mechanism','What to measure'],['Adam moment scaling','Gradient norms, accepted steps and early residual reduction.'],['L-BFGS secant curvature','Closure count, line-search failures, achieved physical audit and memory.'],['SOAP preconditioning','Component gradient alignment, setup cost, matrix/eigenbasis state.'],['Self-scaled Broyden','Secant/scaling behavior, generator state, line searches and time.']], [170,305]),
         body('Preconditioning or curvature information can help ill-conditioning, but may cost more per step and cannot repair wrong boundary conditions. State a mechanism as a hypothesis until its diagnostics and physical error support it.'),
         body('Within each seed, keep the same initial parameters, architecture, physics, exact samples, weights and endpoint rule. Freeze the objective for every quasi-Newton line search. Account for all function/gradient evaluations and prior training if a warm start is used.'),
         body('Never select the better optimizer or seed using the reference field later called the test. Freeze candidates first, then compare them. Keep failed or nonfinite runs as outcomes, including their consumed work. One historical trajectory per workflow is not an independent-seed study.'),
         small('Related mechanism studies: Rathore et al., ICML 2024 (loss landscape and staged optimization); SOAP gradient-alignment study, arXiv:2502.00604. These sources motivate local tests rather than establishing a winner here.')),
    page('Weak lower flow needs both absolute and relative error',
         body('For physical region R and quadrature weights w_i, show reference vector RMS A_R and absolute vector-error RMS e_R before the relative ratio E_R. A very small reference amplitude can make a relative percentage large.'),
         math(r'A_R^2=\frac{\sum_{i\in R}w_i(u_i^2+v_i^2)}{\sum_{i\in R}w_i}'),
         math(r'e_R^2=\frac{\sum_{i\in R}w_i[(\hat u_i-u_i)^2+(\hat v_i-v_i)^2]}{\sum_{i\in R}w_i},\qquad E_R=e_R/A_R'),
         body('Use exactly declared interior, upper-left and floor masks for a paper reproduction. A teaching box is allowed only when labeled as a different mask. Do not infer a continuous lower-vortex profile from a few sampled points.'),
         body('Pressure is defined up to one constant. Align the whole-interior weighted mean once, then retain that shift in every region. Show signed regional offsets and RMS errors separately. Fitting a new offset in each floor region would conceal a physical mismatch.'),
         math(r'\omega=v_x-u_y'),
         body('Signed vorticity and velocity profiles reveal weak circulation that speed magnitude can hide. Use one stated derivative operator and sample coverage when comparing CFD and PINN. Inspect actual native heights near y/W = 0.25, 2.5 and 4.75; report the chosen row coordinates.'),
         small('Velocity, pressure and vorticity may rank methods differently. Regional errors from the retained 121x605 author fields and the separate 161x801 historical workflow table are distinct evidence and must not be pooled as one fresh experiment.')),
    page('Work and recovery are part of the experiment',
         body('An iteration count is not a cost measure when line searches or matrix preconditioners perform different work. Record first-order and curvature-stage evaluations separately, plus total residual-point work, memory and elapsed training time. Name whether setup, compilation, checkpoint I/O and final audits are included.'),
         table([['Checkpoint component','Why weights alone are insufficient'],['Optimizer state','Moments, SOAP basis/preconditioners or Broyden generators determine the next step.'],['Points and weights','A changed objective invalidates an apparent continuation.'],['RNG and phase','Preserve sample sequence, stage and reproducible initialization.'],['Counters and selection','A restart must not erase compute cost or become another seed.'],['Configuration and source hashes','Reject mismatched physics or implementation identities.']], [160,315]),
         body('Save atomically after a complete accepted update. Test interrupted versus uninterrupted continuation through each stage, including the next accepted step and audit. Record device and precision; exact same-runtime replay does not automatically imply bitwise cross-hardware identity.'),
         body('The square-cavity GPU chunk probe shows a separate useful lesson: aggregate the same full-batch work efficiently before claiming an algorithmic gain. Its measured timing is not transferable to D5 without a new matched measurement; depth, derivatives and optimizer state change cost.'),
         small('The teaching notebook reads completed evidence on CPU and never launches a cluster run from Run All. A missing fresh manifest remains an explicit missing result. Existing historical evidence stays accessible in the companion notebook.')),
    page('D5 student deliverables: isolate causes, retain trade-offs',
         body('<b>Table A - exact formulation.</b> Record Reynolds number, D/W, lid widths, wall factor, envelope, derivative scaling, network and raw/scaled residual definitions. Show why using y/D derivatives without conversion changes the PDE.'),
         body('<b>Table B - paired controls.</b> Verify initial-model/sample hashes, implementation variant, seed, work budget and endpoint rule. Identify a mismatch before ranking optimizers.'),
         body('<b>Figure 1/Table C - convergence and time.</b> Plot the actually recorded metric using its declared label. Separate final independent audits when no audit trajectory exists. Count matrix setup and failed line searches. State whether a ranking changes with the cost axis.'),
         body('<b>Figures 2-3/Table D - physical error.</b> Overlay actual native velocity, pressure and signed vorticity profiles. Report A_R, e_R and E_R in the whole interior and exact floor masks. Explain a disagreement between observable rankings.'),
         body('<b>Table E - seed robustness.</b> Give every predeclared paired difference and failure, followed by median and full range. Three seeds do not automatically yield a confidence interval or universal optimizer ranking.'),
         body('<b>Table F - recovery.</b> Interrupt a separate small demonstration in each optimizer stage. Compare complete resumed state and the next accepted update with an uninterrupted control.'),
         small('Finish with observations, plausible mechanisms and untested ideas in separate sentences. Discuss deterioration as carefully as improvement. Fresh controlled results can differ from historical transferred workflows because the information and compute histories differ.'))]
    return {NAMES[0]:ghia,NAMES[1]:d5}


def result_pages():
    """Append only hash-verified scored evidence; never fabricate placeholder plots."""
    folder=ROOT/'results/cavity_pinn_campaign'
    fpath=folder/'figures/figure_manifest.json'
    if not fpath.exists():return []
    figures=json.loads(fpath.read_text(encoding='utf-8'))
    def checked(item):
        p=(ROOT/item['path']).resolve()
        if not p.is_relative_to(ROOT.resolve()) or digest(p.read_bytes())!=item['sha256']:
            raise ValueError('Result evidence changed: '+item['path'])
        return p
    data=json.loads(checked(figures['source_evaluation']).read_text(encoding='utf-8'))
    for item in figures['figures']:checked(item)
    sampling=json.loads(checked(figures['sampling_coverage']).read_text(encoding='utf-8')) if 'sampling_coverage' in figures else None
    scored=[r for r in data['records'] if r['outcome']=='scored']
    summary=figures['summary'];counts=summary['numeric_pass_counts']
    labels=[('outer_momentum_rms_le_1e_2','Independent outer momentum RMS <= 0.01'),
            ('divergence_rms_le_1e_10','Divergence RMS <= 1e-10'),
            ('both_screened_ghia_relative_l2_le_0_02','Both screened Ghia errors <= 2%'),
            ('interior_velocity_relative_l2_le_0_05','Interior velocity-vector error <= 5%')]
    pages=[page('A completed optimizer budget can still fail the physics',
        body(f"The frozen baseline contains {len(scored)} scored endpoints: seven Reynolds numbers, three seeds and two collars. Only {summary['all_four_numeric_targets']} endpoints meet all four recorded numerical targets. This finite-budget baseline is not a validated all-Re PINN result."),
        table([['Recorded target','Pass count']]+[[label,f"{counts.get(key,0)} / {len(scored)}"] for key,label in labels],[365,110]),
        body('Read the signed centerline and physical-field figures before summarizing accuracy. Low recorded training loss can coexist with a large independent residual and wrong interior flow. Exact continuity from a streamfunction is useful, but does not by itself solve momentum.'),
        body('Do not infer a universal PINN failure. This controlled baseline tests one fixed network, one sparse collocation realization per seed and one optimizer schedule. Better sampling, representation or optimization are separate hypotheses requiring newly declared controls and evidence.'),
        body('A zero collar difference is not enough: sparse random sampling may leave the removed annulus empty, so changing the collar does not change the objective. Inspect the sample hashes and annulus counts before calling this a sensitivity experiment.'),
        body('The earlier supervised POD-DeepONet learned retained CFD fields. Its much smaller field discrepancy is a different information regime, and cannot be reported as success of this label-free baseline.'),
        small('The following figures preserve all three seeds and both collars. Representative physical fields use predeclared seed 7, epsilon 0.02, at Re1000 and Re7500. No reference-based best seed or endpoint was chosen.'))]
    if sampling is not None:
        c=sampling['summary']
        pages.append(page('A failed sensitivity test can be a sampling lesson',
            body(f"The checkpoint-bound audit reconstructs all 42 frozen point tensors. It finds {c['identical_objective_pairs']} of {c['pairs']} collar pairs with identical points and identical discrete momentum objectives. Thus most pairs did not change the equation samples when the collar changed."),
            math(r'N_{\rm annulus}^{\rm expected}=N\frac{(\pi/2)(0.02^2-0.01^2)}{1-(\pi/2)0.01^2}\simeq0.965'),
            body('N = 2048 is the fixed uniform point count. The numerator is the combined area of the two upper quarter-annuli; the denominator is the available area for epsilon 0.01. A nominally large point count can barely sample a thin region that controls the intended test.'),
            table([['Observed coverage','Consequence'],['Seeds 7 and 61: zero extra annulus points','Identical sampled objectives; no independent collar perturbation.'],['Seed 23: one extra annulus point','Very weak coverage of the changed region.'],['13 of 14 identical-objective pairs: same model','Repeated-objective agreement, not continuum convergence.']], [235,240]),
            body('Re1000, seed61 has different final weights despite identical sampled objectives. That difference cannot be assigned to the collar. Floating-point or execution sensitivity is possible, but the cause is not established by these receipts.'),
            body('A useful next study must require explicit annulus coverage and independent spatial refinement, while keeping the actual objective and compute accounting clear. Adding points after viewing the failures is a new development experiment, not retroactive validation of this baseline.'),
            small('The independent outer audit had 2048 points, only seven between corner radius 0.02 and 0.05. Raw shell audits are separate. Source: results/cavity_pinn_campaign/sampling_coverage.json; hashes bind checkpoint, source, configuration, model and point tensors.')))
    for re in (100,400,1000,3200,5000,7500,10000):
        row=[r for r in scored if r['Re']==re]
        item=next((f for f in figures['figures'] if f.get('kind')=='signed_ghia_centerlines' and f.get('Re')==re),None)
        if item is None:continue
        values=[r['scores']['regions']['interior']['velocity_relative_l2'] for r in row]
        statement=(f"Across the six endpoints, the native interior velocity-vector relative L2 error ranges from {100*min(values):.2f}% to {100*max(values):.2f}%." if values else 'No scored endpoint is available.')
        pages.append(page(f'Re {re}: preserve signed flow and every seed',
            body(statement+' Ghia-error markers use direct frozen predictions at the printed positions. Continuous field curves use exactly retained interior coordinates.'),
            {'type':'figure','path':item['path'],'width':475,'max_height':430,
             'caption':'All three seeds; epsilon 0.02 solid and 0.01 dashed. Flagged printed Ghia entries remain visible. Lines between sparse error markers guide the eye; they do not create new samples.'},
            small('The discrete CFD field also has grid sensitivity. The neural discrepancy and the CFD grid difference are distinct quantities. A good value at one centerline sample does not establish correct secondary vortices or wall gradients.')))
    for kind,title in [('paired_speed_fields','The high-Re speed field shows the missing solution'),('paired_centered_curl_fields','Vorticity exposes circulation and gradient errors')]:
        item=next((f for f in figures['figures'] if f.get('kind')==kind),None)
        if item is None:continue
        pages.append(page(title,
            body('The same color and spatial scales are used for both Reynolds numbers and both methods. The run is the predeclared seed 7, epsilon 0.02. It is not a selected best prediction.'),
            {'type':'figure','path':item['path'],'width':475,'max_height':430,
             'caption':'Velocity is shown at 511-by-511 matched interior nodes. The additional vorticity diagnostic uses the same centered curl of both velocity fields on 509-by-509 inner nodes.'},
            small('The curl diagnostic does not replace frozen native PINN-AD versus CFD-transported-vorticity scores. Its symmetric-log color scales retain extrema while exposing weaker circulation. Signed velocity errors and streamline overlays are available in the linked notebook; no CFD pressure truth is claimed.')))
    return pages



def d5_cost_page():
    folder=ROOT/'results/deep_cavity_ablation/cost_study'
    source=folder/'evaluation_manifest.json'
    if not source.exists():return []
    data=json.loads(source.read_text(encoding='utf-8'))
    if data['status']!='cost_smoke_complete' or data['reference_values_accessed']:
        raise ValueError('This page is only the unscored cost gate')
    for pair in data['pair_checks']:
        for key in ('identical_initial_model','identical_samples','identical_numerical_contract_except_method','same_single_allocation_and_hardware'):
            if not pair[key]:raise ValueError('Cost pair is not controlled')
    item=next(f for f in data['figures'] if f['path'].endswith('.png'))
    image=folder/item['path']
    if digest(image.read_bytes())!=item['sha256']:raise ValueError('D5 cost plot hash changed')
    rows=data['records']
    return [page('A short D5 cost gate is not an optimizer accuracy ranking',
        body('Both routes start from the same fresh model and 8192 frozen balanced points on the same Quadro RTX 8000 allocation. Each has 50 first-order plus ten quasi-Newton function/gradient evaluations. No CFD reference values were accessed.'),
        {'type':'figure','path':image.relative_to(ROOT).as_posix(),'width':475,'max_height':270,
         'caption':'Every point is an actual evaluated closure. Orange quasi-Newton excursions are line-search trials, not accepted SSB endpoint losses. The wall-time axis includes independent audits and checkpoint I/O.'},
        table([['Route','Pure optimizer time (s)','Function / gradient calls']]+[[r['method'],f"{r['training_seconds']:.3f}",f"{r['function_evaluations']} / {r['gradient_evaluations']}"] for r in rows],[180,140,155]),
        body('The ten-call SSB refinement budget expires before the first accepted SSB step. The final model correctly remains at the SOAP endpoint. This cost gate cannot establish whether a sufficiently budgeted SSB stage improves or deteriorates physical accuracy.'),
        body('The scientific pilot starts fresh with three paired seeds, 2000 first-order and 8000 quasi-Newton evaluations per route. Its frozen field errors are a separate result. Short-history cost does not establish the memory or time of a mature compact Broyden history.'),
        small('Source: results/deep_cavity_ablation/cost_study/evaluation_manifest.json. Keep pure optimizer time, active wall time, trial objectives and accepted endpoints distinct. This 8192-point D5 study is separate from the 2048-point square-cavity computational-chunk probe.'))]



def checked_evidence(item, base=ROOT):
    path=(Path(base)/item['path']).resolve()
    if not path.is_relative_to(ROOT.resolve()) or digest(path.read_bytes())!=item['sha256']:
        raise ValueError('Lecture evidence changed: '+item['path'])
    return path


def d5_result_pages():
    """Append the actual six-run outcome; keep frozen scorer assets unchanged."""
    source=ROOT/'results/deep_cavity_ablation/evaluation_manifest.json'
    if not source.exists():return []
    data=json.loads(source.read_text(encoding='utf-8'));folder=source.parent
    assert data['status']=='controlled_pilot_scored' and data['paired_seeds']==[12345,23456,34567]
    records=data['records'];methods=['adam_lbfgs','soap_ssb'];seeds=data['paired_seeds']
    assert len(records)==6 and {(r['seed'],r['method']) for r in records}=={(seed,method) for seed in seeds for method in methods}
    for row in records:
        assert row['outcome']=='controlled_pilot_scored'
        assert row['prediction']['sha256']==row['freeze']['prediction_sha256']==row['score']['prediction_sha256']
        assert row['function_evaluations']==row['gradient_evaluations']==10000
        for key in ('prediction','comparison_prediction','reference','history','training_manifest','training_contract'):
            checked_evidence(row[key],folder)
    for pair in data['pair_checks']:
        assert all(pair[key] for key in ('identical_initial_model','identical_samples',
                                        'identical_numerical_contract_except_method','same_single_allocation_and_hardware'))
    checked_evidence(data['metrics_file'],folder)
    # Keep the hash-bound originals. Presentation copies remove only the fully
    # opaque alpha channel, preserving every RGB pixel of the scientific figure.
    from PIL import Image
    rgb_folder=folder/'lesson_figures';rgb_folder.mkdir(exist_ok=True)
    gallery={};rgb_entries=[]
    for item in data['figures']:
        original=checked_evidence(item,folder)
        if original.suffix!='.png':continue
        with Image.open(original) as im:
            if im.mode=='RGBA':assert im.getextrema()[3]==(255,255)
            pixels=im.convert('RGB');destination=rgb_folder/original.name;pixels.save(destination)
            with Image.open(destination) as check:
                assert check.mode=='RGB' and np.array_equal(np.asarray(check),np.asarray(pixels))
        copied={'path':destination.relative_to(ROOT).as_posix(),'sha256':digest(destination.read_bytes()),
                'source':{'path':original.relative_to(ROOT).as_posix(),'sha256':item['sha256']},
                'conversion':'RGB channels preserved exactly; removed only fully opaque alpha channel'}
        gallery[original.name]=copied;rgb_entries.append(copied)
    from io import BytesIO
    from pypdf import PdfReader, PdfWriter, PageObject, Transformation
    from reportlab.pdfgen import canvas
    from lecture_typography import register_reportlab
    footer_font=register_reportlab('D5StandaloneFooter')['regular']
    combined=PdfWriter();pdf_sources=[];page_index=[]
    for item in data['figures']:
        if item['path'].endswith('.pdf'):
            original=checked_evidence(item,folder)
            original_pdf=PdfReader(original)
            assert len(original_pdf.pages)==1, 'Each retained D5 figure must be one page.'
            page_number=len(combined.pages)
            if original.stem=='optimizer_objective_history':
                title='Re1000, D/W=5: all-seed objective histories'
                page_label='All-seed history'
            else:
                stem,seed=original.stem.rsplit('_s',1)
                region={'profiles_upper':'upper profiles','profiles_floor':'floor profiles',
                        'streamlines':'streamlines'}[stem]
                title=f'Re1000, D/W=5: seed {seed}, {region}'
                page_label=f's{seed} {region}'
            # Add an external 24-point footer margin. Original scientific page
            # contents remain vector objects, translated without resampling.
            original_page=original_pdf.pages[0]
            width=float(original_page.mediabox.width)
            height=float(original_page.mediabox.height)
            presented=PageObject.create_blank_page(width=width,height=height+24)
            presented.merge_transformed_page(original_page,Transformation().translate(0,24))
            footer=BytesIO()
            footer_canvas=canvas.Canvas(footer,pagesize=(width,height+24))
            footer_canvas.setFillColorRGB(0,0,0)
            footer_canvas.setFont(footer_font,8)
            footer_canvas.drawString(12,8,title+f' | page {page_number+1}')
            footer_canvas.save();footer.seek(0)
            presented.merge_page(PdfReader(footer).pages[0])
            combined.add_page(presented)
            combined.add_outline_item(title,page_number)
            combined.set_page_label(page_number,page_number,prefix=f'{page_number+1}: {page_label}')
            page_index.append({'page':page_number+1,'label':title,
                               'source':original.relative_to(ROOT).as_posix()})
            pdf_sources.append({'path':original.relative_to(ROOT).as_posix(),'sha256':item['sha256']})
    combined_path=rgb_folder/'controlled_pilot_figures.pdf'
    with combined_path.open('wb') as stream:combined.write(stream)
    rgb_entries.append({'path':combined_path.relative_to(ROOT).as_posix(),
                        'sha256':digest(combined_path.read_bytes()),'source_pdfs':pdf_sources,
                        'assembly':'Original scientific vector page contents translated upward by 24 points without resampling; external black Times New Roman seed/region footers, outline bookmarks and page labels added; no scientific recomputation',
                        'page_index':page_index})
    rgb_manifest={'schema_version':1,'campaign_kind':'d5_gaussian_lid',
                  'source_evaluation':{'path':source.relative_to(ROOT).as_posix(),'sha256':digest(source.read_bytes())},
                  'figures':rgb_entries,'visual_review':'pending individual-page inspection'}
    (rgb_folder/'figure_manifest.json').write_text(json.dumps(rgb_manifest,indent=2)+'\n',encoding='utf-8')
    metrics={(r['seed'],r['method'],r['region']):r for r in data['metrics']}
    config=records[0]['configuration']
    errors=[metrics[(seed,method,'Interior')]['velocity_NRMSE_percent'] for seed in seeds for method in methods]
    labels={'adam_lbfgs':'Adam to L-BFGS','soap_ssb':'SOAP to SSB'}
    pages=[page('The fresh D5 trial finished, but did not solve the flow accurately',
        body(f"All six predeclared runs at Re1000 and D/W = 5 are frozen and scored: seeds {', '.join(map(str,seeds))}, two optimizer pipelines per seed. Interior velocity-vector NRMSE ranges from {min(errors):.2f}% to {max(errors):.2f}%. Successful execution is not accurate recovery of the CFD field."),
        body(f"The network has {config['depth']} hidden layers of width {config['width']}, float64, and {config['points']} frozen authored balanced points. Each pipeline starts fresh with identical paired initial parameters and samples. Both receive 2000 first-order and 8000 quasi-Newton function/gradient calls; these are not 10000 accepted updates."),
        table([['Seed','Pipeline','Velocity NRMSE (%)','Pressure RMS','Vorticity RMS']]+[
            [str(seed),labels[method],f"{metrics[(seed,method,'Interior')]['velocity_NRMSE_percent']:.2f}",
             f"{metrics[(seed,method,'Interior')]['pressure_RMS_rhoU2']:.4f}",
             f"{metrics[(seed,method,'Interior')]['vorticity_RMS_UoverW']:.4f}"]
            for seed in seeds for method in methods],[45,110,110,105,105]),
        body('Pressure RMS uses density times lid speed squared; vorticity RMS uses lid speed divided by width. Vorticity is computed by the same sampled derivative on both fields. These figures are a fresh controlled pilot, not the historical transferred workflow or a successful reproduction of its accuracy.'),
        small('All model endpoints froze before the CFD scoring read. The two-pipeline design does not isolate SOAP from SSB or Adam from L-BFGS separately. No model or seed was chosen by its CFD error.'))]
    regions=['Interior','Upper-left','Floor_0.4','Floor_0.65']
    summary={(r['region'],r['metric']):r for r in data['descriptive_seed_summary']}
    pages.append(page('An optimizer ranking depends on the physical observable',
        body('Count the paired seeds with lower SOAP-to-SSB error while retaining all failures. This is a descriptive comparison of three fresh realizations, not a significance test or a universal optimizer ranking.'),
        table([['Region','Lower velocity error','Lower pressure error','Lower vorticity error']]+[
            [region]+[f"{summary[(region,metric)]['candidate_lower_error_seeds']} / 3" for metric in
                      ['velocity_RMS_U','pressure_RMS_rhoU2','vorticity_RMS_UoverW']] for region in regions],[115,120,120,120]),
        body('In the whole interior, SOAP-to-SSB reduces vorticity error in all three seeds, reduces velocity error in only one, and increases pressure error in all three. Near the upper-left corner its velocity and vorticity errors are lower in all three seeds, but their absolute errors remain large.'),
        body('The floor ratios stay close to 100% because both pipelines largely miss the weak reference circulation. A small improvement in an already inaccurate field does not make it validated. The next pages show the signed physical profiles that explain these numbers.'),
        body('Mechanisms such as preconditioning and secant curvature can change the optimization trajectory. This experiment does not prove which internal mechanism caused each regional change. Fixed representation, sampling and limited optimization can still obstruct the physical solution.'),
        small('The two complete pipelines differ in both stages. Attribute a result to the tested route, not exclusively to one optimizer. Keep the existing accurate supervised POD-DeepONet study separate from these label-free PDE fits.')))
    floor_rows=[['Region','Seed','CFD vector RMS','Adam-route RMS','SOAP-route RMS','Adam (%)','SOAP (%)']]
    for region in ('Floor_0.4','Floor_0.65'):
        for seed in seeds:
            a=metrics[(seed,'adam_lbfgs',region)];b=metrics[(seed,'soap_ssb',region)]
            assert a['CFD_velocity_RMS_U']==b['CFD_velocity_RMS_U']
            floor_rows.append([region,str(seed),f"{a['CFD_velocity_RMS_U']:.3e}",f"{a['velocity_RMS_U']:.3e}",
                               f"{b['velocity_RMS_U']:.3e}",f"{a['velocity_NRMSE_percent']:.2f}",f"{b['velocity_NRMSE_percent']:.2f}"])
    pages.append(page('Weak flow: report the reference amplitude beside relative error',
        body('The floor masks retain interior x/W between 0.02 and 0.98 and physical height h/W below 0.4 or 0.65. Velocities and errors below are scaled by lid speed U. RMS includes both velocity components and uses the actual common sampled mask.'),
        table(floor_rows,[70,40,85,85,85,55,55]),
        body('The absolute floor error is tiny on the lid-speed scale, but comparable to the tiny reference flow itself. A nearly zero prediction therefore has about 100% regional error. Reporting only its absolute error would hide failure to recover the lower circulation.'),
        body('Conversely, a large relative value in a very weak region is not by itself a large absolute energy defect. The correct interpretation needs reference amplitude, absolute error, signed profiles and the intended observable together.'),
        small('These are the exact declared masks and retained CFD samples. They are not grid-convergence bounds or measurements of a separately refitted floor model.')))
    pressure_rows=[['Seed','Pipeline','Floor signed mean','Floor RMS','Local shape RMS']]
    for seed in seeds:
        for method in methods:
            row=metrics[(seed,method,'Floor_0.4')]
            pressure_rows.append([str(seed),labels[method],f"{row['pressure_signed_mean_offset_rhoU2']:+.5e}",
                                  f"{row['pressure_RMS_rhoU2']:.5e}",f"{row['pressure_local_shape_RMS_rhoU2']:.3e}"])
    pages.append(page('A local pressure recentering can hide the main discrepancy',
        body('A single common-interior mean pressure difference is removed before any regional score. The same gauge is retained everywhere. The table is for Floor_0.4; all pressure quantities use density times U squared.'),
        table(pressure_rows,[45,110,115,105,100]),
        body('For these outcomes, the floor pressure RMS is dominated by a signed offset of order 0.008, while the local shape residual is near 1e-10. Removing a separate floor offset would hide the principal regional discrepancy.'),
        body('The physical profile panels explicitly show pressure minus each horizontal cut mean as an additional shape diagnostic. Those curves must not be interpreted as the signed pressure error in the shared global gauge. Their small amplitude does not replace the table above.'),
        small('Use the complete regional_metrics.csv for both floor masks and the upper-left region. The comparison arrays preserve the one global gauge; original frozen native predictions remain separately hash-bound.')))
    history=gallery['optimizer_objective_history.png']
    pages.append(page('Same evaluation budget: inspect the trajectory and count actual work',
        {'type':'figure','path':history['path'],'width':475,'max_height':255,
         'caption':'Every plotted point is an evaluated closure. Quasi-Newton trial values are not automatically accepted endpoint losses. All three seeds and both pipelines are retained; no curve is selected by its final reference score.'},
        table([['Seed','Pipeline','Optimizer minutes','Active minutes','Function / gradient calls']]+[
            [str(r['seed']),labels[r['method']],f"{r['training_seconds']/60:.2f}",f"{r['active_seconds']/60:.2f}",
             f"{r['function_evaluations']} / {r['gradient_evaluations']}"] for r in records],[45,115,95,95,125]),
        body('Pure optimizer-step time excludes audits and checkpoint I/O. Active wall time includes those operations but excludes downtime. Each paired run used the same allocation and hardware; the timings describe these measured trajectories and are not a universal speed advantage.'),
        small('The earlier 50+10-call cost gate remains a separate admission experiment. These scientific trials used fresh 2000+8000-call budgets. A lower training objective does not establish better independent physical accuracy.')))
    for seed in seeds:
        for region in ('upper','floor'):
            item=gallery[f'profiles_{region}_s{seed}.png']
            pages.append(page(f'D5 signed {region} profiles: paired seed {seed}',
                body('All three physical observables compare the same CFD reference with both frozen optimizer endpoints. The selected horizontal heights are stated above the columns; values use linear interpolation only between the two adjacent retained height rows.'),
                {'type':'figure','path':item['path'],'width':475,'max_height':430,
                 'caption':('Upper profiles retain the shared interior pressure gauge. Vorticity uses the same sampled derivative for CFD and both PINNs.' if region=='upper' else
                            'Floor pressure is a cut-mean-centered shape diagnostic. Signed global-gauge offsets remain in the regional metrics; the small centered scale must not conceal them.')},
                small('Curves preserve signs and native horizontal sample coverage. No extrapolation beyond the stored domain and no reference-driven seed selection are used.')))
    item=gallery['streamlines_s12345.png']
    pages.append(page('D5 circulation: the predeclared first seed, with physical zooms',
        body('Seed 12345 was declared before scoring. Whole-height, upper and lower views compare velocity-integrated streamlines for CFD and both frozen pipelines. This example supplements, rather than replaces, the three-seed regional audit.'),
        {'type':'figure','path':item['path'],'width':475,'max_height':470,
         'caption':'Streamlines are drawn from stored velocity fields on their matched physical coordinates. Line colors and styles identify the methods; the lower-flow amplitudes must be read alongside the signed profiles and RMS values.'},
        small('These fresh pilots do not reproduce the reference solution accurately. A fully converged author workflow and this finite-budget comparison have different histories and must not be conflated.')))
    return pages


def stratified_result_pages():
    import csv
    folder=ROOT/'results/cavity_pinn_stratified'
    source=folder/'comparison/comparison_manifest.json';fpath=folder/'figures/figure_manifest.json'
    if not source.exists() or not fpath.exists():return []
    data=json.loads(source.read_text(encoding='utf-8'));figures=json.loads(fpath.read_text(encoding='utf-8'))
    assert data['campaign_kind']=='ghia_uniform_vs_stratified_matched_work' and len(data['records'])==12
    assert checked_evidence(figures['source_comparison'])==source.resolve()
    for item in data['inputs'].values():checked_evidence(item)
    for item in data.get('evaluation_provenance',{}).values():checked_evidence(item)
    for item in figures['figures']:checked_evidence(item)
    with checked_evidence(data['outputs']['paired_metrics']).open(encoding='utf-8',newline='') as stream:
        metrics=list(csv.DictReader(stream))
    velocity=[r for r in metrics if r['metric']=='field.interior.velocity_relative_l2']
    assert len(velocity)==12
    improved=sum(r['outcome']=='improved' for r in velocity);worsened=sum(r['outcome']=='worsened' for r in velocity)
    pages=[page('One bounded intervention: guarantee coverage without changing area weights',
        body('The follow-up tests Re100 and Re7500 with seeds 7, 23, 61 and collars 0.01 and 0.02. Only the frozen collocation distribution and its correct area weights change. The network, physical lid, initial model and optimizer settings remain paired with the original controls.'),
        table([['Stratum','Support','Points'],['Upper-corner annuli','epsilon < r < R; R = 0.05','512'],
               ['Remaining wall band','Outside the inner 0.9 x 0.9 square and the two upper radius-R disks','512'],
               ['Interior','R < x,y < 1-R','1024']],[125,285,65]),
        math(r'\widehat J=\sum_s\frac{A_s}{A n_s}\sum_{i\in s}(r_{x,i}^2+r_{y,i}^2),\qquad A=1-\frac{\pi\epsilon^2}{2}'),
        body('A_s is stratum area and n_s its point count. The positive weights sum to one and estimate the same normalized area objective as uniform sampling on the punctured square. Equal sample counts would not mean equal physical area weights.'),
        body('Both methods in each pair consume the same recorded total closures and residual-point evaluations. The new arm stops at that work cap; it need not complete exactly 500 accepted L-BFGS updates. Different seeds or collars can have different original caps.'),
        small('This is a prespecified development experiment after the baseline failure, not a blind claim that arbitrary stratification helps. Points are not chosen from residuals and are not redrawn during training. It is not full-domain corner convergence.'))]
    pages.append(page('The coverage intervention mostly worsened the velocity field',
        body(f"Interior velocity error improves in {improved} of 12 paired cases and worsens in {worsened}. The two gains occur at Re100 with collar 0.02, seeds 23 and 61. Every Re7500 case is worse. More explicit spatial coverage did not solve the high-Re flow at this matched budget."),
        table([['Re','Seed','Collar','Uniform error (%)','Stratified error (%)','Outcome']]+[
            [r['Re'],r['seed'],r['collar'],f"{100*float(r['baseline']):.2f}",f"{100*float(r['stratified']):.2f}",r['outcome']]
            for r in velocity],[45,45,50,115,115,105]),
        body('The independent outer residual can decrease while reference field error increases. A better residual number on a particular audit set is not sufficient evidence of a better velocity distribution. The signed centreline, curl and streamline figures below preserve that disagreement.'),
        small('All twelve paired outcomes remain. No best seed is substituted into the representative figures, and the original 42-case baseline gallery and scientific scores remain unchanged.')))
    chronology=data.get('evaluation_chronology',{})
    pages.append(page('The saved point tensor is part of reproducible scientific state',
        body(chronology.get('disclosure','See the exact completed-pass chronology in the comparison manifest.')),
        body('Regenerating annular coordinates on another runtime shifted four to six values in affected cases by at most about 1.11 x 10<super>-16</super>. Initial models, saved points and weights, optimizer/RNG state and model identities remained intact. The exact saved samples, not fresh approximations, were authoritative.'),
        body('The repair uses a separate inference-only loader that restores the stored trained state exactly. It does not retrain, resample, select a model or relax the physical criteria. The specific low-level platform cause of the final-bit difference was not isolated.'),
        table([['Verified statement','Statement not established'],['All twelve trained endpoints preceded earlier target access.','No earlier evaluation had opened references.'],
               ['The completed pass froze its native projections before its own scoring reads.','Cross-platform optimizer continuation is bitwise certified.'],
               ['Restored frozen predictions are bound to the original model and samples.','Tiny regeneration differences may simply be ignored in every workflow.']],[240,235]),
        small('Keep the earlier four target reads, failed-attempt receipt, exact-state admission and repaired-pass barrier in provenance. A successful retry must not rewrite its chronology.')))
    for re in (100,7500):
        for collar in (.02,.01):
            item=next(f for f in figures['figures'] if f.get('kind')=='signed_profiles_all_paired_seeds' and f['Re']==re and f['collar']==collar)
            pages.append(page(f'Re{re}, collar {collar:g}: original and stratified physical profiles',
                body('All three paired seeds are shown against the same retained CFD and printed Ghia positions. Dashed curves are the original uniform sample set; solid colored curves use area-weighted stratification. Both collars share axis limits for this Reynolds number.'),
                {'type':'figure','path':item['path'],'width':475,'max_height':450,
                 'caption':'Error markers are direct frozen predictions at actual printed positions. Flagged entries remain visible. Different pairs can have different work caps; equality is guaranteed within each pair.'},
                small('No sampled error curve is extrapolated. Inspect signed flow and near-wall behavior instead of reducing the comparison to a single aggregate winner.')))
    for variable in ('speed','curl'):
        item=next(f for f in figures['figures'] if f.get('kind')=='matched_'+variable+'_fields')
        pages.append(page('Matched speed fields after the sampling change' if variable=='speed' else 'Matched vorticity fields after the sampling change',
            body('The predeclared seed 7, collar 0.02 is shown at Re100 and Re7500. CFD, uniform PINN and stratified PINN share spatial limits and one color scale. A visually weak or missing primary circulation is not a successful high-Re solution.'),
            {'type':'figure','path':item['path'],'width':475,'max_height':440,
             'caption':('Speed uses native 511-by-511 interiors and a linear scale.' if variable=='speed' else
                        'Vorticity uses one centered curl on all stored velocity fields, giving 509-by-509 nodes. The symmetric-log scale retains the extrema; original native vorticity scores are unchanged.')},
            small('No CFD pressure truth is available in this square-cavity experiment. CFD is itself a numerical reference with grid sensitivity.')))
    for re in (100,7500):
        for kind,title in [('signed_native_and_common_curl_errors','Signed errors'),('three_method_streamfunction_overlay','Streamline overlays')]:
            item=next(f for f in figures['figures'] if f.get('kind')==kind and f['Re']==re)
            pages.append(page(f'{title}: Re{re}, predeclared seed 7',
                body('These views compare both frozen PINNs with the same retained CFD. The example is fixed at collar 0.02 and is not the best-scoring seed.'),
                {'type':'figure','path':item['path'],'width':475,'max_height':480,
                 'caption':('Signed errors are PINN minus CFD. The linear scales match across methods and both Reynolds numbers; the curl diagnostic uses the same derivative on u and v.' if kind.startswith('signed') else
                            'All three methods use the same streamfunction levels. Lower-corner zooms preserve weak circulation; no wall rows or field extrapolation are invented.')},
                small('Report improvements and deterioration by observable. This single finite-budget intervention does not establish optimizer superiority or full-domain convergence.')))
    pages.append(page('Follow-up exercises: distinguish coverage, objective and physical accuracy',
        body('<b>Table G - paired work.</b> For all twelve records, verify closures, residual-point evaluations, initial-state identity and each stratum weight sum. Explain why collar-to-collar comparisons can have unequal budgets even though every method pair is matched.'),
        body('<b>Figure 4/Table H - physical change.</b> At Re100 and Re7500, report each seed and collar. Overlay signed centreline velocities and Ghia errors. Compare whole-interior and lower-quarter errors with independent outer and shell audits. Find an example where an audit residual decreases but a physical error increases.'),
        body('<b>Figure 5/Table I - field interpretation.</b> Keep seed 7, collar 0.02. Inspect the common-scale speed, signed velocity error and common-operator curl. Identify missing or displaced circulation and distinguish native AD-vorticity scores from the additional sampled-curl diagnostic.'),
        body('<b>Table J - provenance.</b> Draw the chronology of training freeze, the earlier four scoring reads, exact-state repair, repaired projection freeze and final scoring. State why using exact saved samples is legitimate and why this does not prove cross-platform optimizer continuation.'),
        body('Conclude with two observations supported by these fields, one plausible mechanism and one untested proposal. Do not launch an undeclared search or relabel a failed baseline as a validated method.'),
        small('The complete paired CSVs and source hashes are linked from results/cavity_pinn_stratified/README.md. The original baseline and its failed criteria remain available for comparison.')))
    return pages

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cost-probe',type=Path,required=True)
    args=parser.parse_args();prior_receipt()
    data,retained,figure=cost_figure(args.cost_probe)
    source=ROOT/'lectures/source/cavity_expansion.json';content=json.loads(source.read_text(encoding='utf-8'))
    prepared=make_pages(data,retained,figure)
    prepared[NAMES[0]]+=result_pages()+stratified_result_pages()
    prepared[NAMES[1]]+=d5_cost_page()+d5_result_pages()
    for name,pages in prepared.items():
        content[name]=[p for p in content[name] if p.get('campaign_extension_id')!=TAG]+pages
    source.write_text(json.dumps(content,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Prepared extensions:',{name:len(content[name]) for name in NAMES})

if __name__=='__main__':main()
