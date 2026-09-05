"""Build original CPU labs and PDF lecture notes; execute with --execute.

Authoring tool: intentionally writes the two named notebooks and teaching assets.
Student notebook execution never writes retained evidence. PDF copies are staged
in output/pdf and copied into lectures for the course navigation.
"""
from pathlib import Path
import argparse
import base64
import html
import shutil
import sys
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qa'))
from add_colab_entrypoints import badge, bootstrap


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = '''
from pathlib import Path
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "flowmllab" / "feature_reconstruction.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from flowmllab.feature_reconstruction import *
plt.rcParams.update({'figure.dpi': 125, 'font.size': 10})
print('Original CPU teaching analog; no research checkpoint or CFD/DSMC run.')
'''


def week11():
    return [md('''
    # Week 11 - Shock and vortex identification
    <!-- MIE690A article-aligned validation v4 -->
    **Scope:** original manufactured-field laboratory inspired by Ehsan Roohi's
    *Physics-audited joint neural segmentation of shocks and vortex cores:
    cross-solver transfer and controlled airfoil--cylinder studies* (author-supplied
    research manuscript, 2026). Research code: https://github.com/Ehsan-Roohi/ShockVortexML.
    This lab does not reproduce that network, checkpoint, CFD fields or reported scores.
    The compression layer is not a Rankine--Hugoniot-consistent shock solution.

    **Prerequisites:** velocity gradients (Week 1), supervised splits (Week 2),
    cylinder wakes (Week 7) and gas dynamics (Week 8). Allow 75-90 minutes.
    CPU only; NumPy, SciPy, scikit-learn and Matplotlib. Use Restart and Run All.
    All generated fields remain in memory; no files under results/ are rewritten.

    **Outcomes:** reject shear as a vortex; permit label overlap; fit preprocessing
    on training cases only; choose thresholds on validation cases; report failures.
    Predict first: can a large vorticity magnitude alone identify a vortex core?
    '''), code(SETUP), md(r'''
    ## 1. Derivatives before detection
    With $A=\nabla\mathbf{u}$, compute divergence and planar vorticity.
    Define $Q_d=-\frac14(u_x-v_y)^2-u_yv_x$ for the trace-free in-plane tensor.
    In two dimensions $\lambda_{ci}=\sqrt{\max(Q_d,0)}$; these are not independent votes.
    Do not confuse this planar diagnostic with a universal 3-D vortex definition.
    Pure shear has vorticity but no swirling strength. Pure compression is not rotation.
    '''), code('''
    x = np.linspace(-1, 1, 41)
    X, Y = np.meshgrid(x, x)
    h = x[1]-x[0]
    controls = {'rotation': (-Y, X), 'shear': (Y, 0*X), 'compression': (-X, -Y)}
    control_rows = []
    for name, (u, v) in controls.items():
        d = diagnostics(u, v, h, h)
        control_rows.append({'case': name, **{k: float(d[k].mean())
                              for k in ['omega', 'swirl', 'compression']}})
    display(pd.DataFrame(control_rows))
    assert np.isclose(control_rows[1]['swirl'], 0)
    assert np.isclose(control_rows[0]['swirl'], 1)
    '''), md('''
    ## 2. Declare entire-case splits before fitting
    Each seed defines a complete analytic field: a compression-layer position and
    thickness, a vortex centre and radius, a rotation sign, and a shear component.
    Labels come from this construction, not thresholding the diagnostic inputs.
    Training pixels may be subsampled, but no validation or test field supplies
    training pixels. This is a controlled toy distribution, not solver transfer.
    Shock-like layer and core labels can overlap, so no exclusive softmax is used.
    '''), code('''
    split = {'train': list(range(10, 18)), 'validation': [30, 31], 'test': [50, 51]}
    assert not (set(split['train']) & set(split['validation']) |
                set(split['train']) & set(split['test']) |
                set(split['validation']) & set(split['test']))
    data = {k: [manufactured_case(i) for i in ids] for k, ids in split.items()}
    print(split)
    '''), md('''
    ## 3. Fit two independent foreground decisions with a small MLP
    Eight local primitive/derivative channels enter a 32-by-24 tanh MLP.
    Its two logistic outputs allow overlap. This is a pixel classifier, not the
    research dual-decoder U-Net, not a spatial segmentation network, and not a PINN.
    All channels include deterministic physical information; call it a learned
    diagnostic, not a physics-free model. Scaling is fitted on training pixels only.
    '''), code('''
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    features = np.concatenate([c['features'].reshape(-1, 8)[::3] for c in data['train']])
    labels = np.concatenate([c['labels'].reshape(-1, 2)[::3] for c in data['train']])
    model = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(32, 24), activation='tanh', solver='adam',
        max_iter=180, random_state=11, early_stopping=False))
    with threadpool_limits(limits=1):
        model.fit(features, labels)
    print('Optimizer iterations:', model[-1].n_iter_, 'training loss:', model[-1].loss_)
    def scores(c):
        return model.predict_proba(c['features'].reshape(-1, 8)).reshape(*c['p'].shape, 2)
    '''), md('''
    ## 4. Validation-only operating points; equal treatment of the baseline
    The physical baseline thresholds compression and swirling strength separately.
    Both methods select one threshold per task by mean case-wise validation Dice.
    Freeze those values before opening test scores; do not choose a pixel quantile
    separately for every test field. A threshold sweep is not a confidence interval.
    '''), code('''
    val = data['validation']
    neural_threshold = choose_thresholds(val, [scores(c) for c in val], np.linspace(.1, .9, 9))
    physical_threshold = choose_thresholds(val, [physical_scores(c) for c in val],
                                          np.linspace(.2, 5, 25))
    print('Frozen MLP thresholds:', neural_threshold)
    print('Frozen physical thresholds:', physical_threshold)
    '''), md('''
    ## 5. Once-only held-out evaluation
    Report each complete case, not thousands of pixels as independent replicates.
    Dice is overlap, not front-location error or vortex-object recovery. We also
    report connected-component counts to expose fragmentation; counts alone do not
    establish one-to-one object matches. No result is required to favour the MLP.
    '''), code('''
    from scipy.ndimage import label
    rows = []
    for seed, c in zip(split['test'], data['test']):
        for method, pred in [('physical', physical_scores(c) >= physical_threshold),
                             ('MLP', scores(c) >= neural_threshold)]:
            for k, task in enumerate(['compression layer', 'vortex core']):
                rows.append({'case': seed, 'method': method, 'task': task,
                    'Dice': dice(c['labels'][..., k], pred[..., k]),
                    'components': label(pred[..., k], structure=np.ones((3,3)))[1]})
    metrics = pd.DataFrame(rows)
    display(metrics)
    c = data['test'][0]
    masks = [c['labels'], physical_scores(c) >= physical_threshold, scores(c) >= neural_threshold]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(['#edf2f4', '#e76f51', '#168aad', '#7b2cbf'])
    for ax, mask, title in zip(axes, masks, ['Construction labels', 'Physical baseline', 'Trained local MLP']):
        image = ax.imshow(mask[..., 0].astype(int)+2*mask[..., 1], origin='lower',
            extent=[-1,1,-1,1], cmap=cmap, norm=BoundaryNorm(np.arange(5)-.5, 4))
        ax.set(title=title, xlabel='x', ylabel='y')
    bar = fig.colorbar(image, ax=axes, ticks=[0,1,2,3], shrink=.8)
    bar.ax.set_yticklabels(['background', 'layer', 'core', 'overlap'])
    fig.suptitle('Week 11: manufactured controls, not research CFD results')
    plt.show()
    '''), md('''
    ## 6. Research transfer and assignment
    The research manuscript uses independent heads, geometry masks, grouped
    trajectories and separately reported learned/physical/hybrid products.
    Its physics-derived reference masks are weak labels, not exact truth.
    Supporting a hybrid output with the same proposal used to judge it creates
    reference-conditioned agreement. A development-test set is not an untouched
    external validation set. Do not quote this lab's Dice as article reproduction.

    **Assignments:** (1) Add noisy velocities and recompute derivatives; never add
    noise only to already-derived channels. (2) With new train/validation/test case
    IDs, remove rotational features and report both tasks. (3) Add pure shear and
    expansion controls to held-out evaluation. (4) Propose a distance-tolerant front
    metric and one-to-one core matching rule. (5) Explain why freezing only the
    vortex decoder does not preserve its output if the shared encoder is updated.
    Preserve all failures; do not tune on the test cases above.

    **Reading:** [Week 11 lecture](../../lectures/week11_shock_vortex_identification.pdf),
    [ShockVortexML research repository](https://github.com/Ehsan-Roohi/ShockVortexML).
    Original FlowMLLab teaching code and analytic fields; AI-assisted authoring
    and checks. Existing research data were not generated by this notebook.
    ''')]


def week12():
    return [md('''
    # Week 12 - Observation-conditioned reconstruction of DSMC moments
    <!-- MIE690A article-aligned validation v4 -->
    Research source: Ehsan Roohi, *Geometry-native machine learning reconstruction
    of DSMC moment fields with support monitoring*, 2026,
    [arXiv:2609.01637](https://doi.org/10.48550/arXiv.2609.01637).
    Submitted to JCP according to the author; not described here as a published JCP article.

    **Scope:** an original CPU teaching analog, not MambaIR training or reproduction
    of the paper's nine-field cylinder estimator. Synthetic Gaussian blocks are
    NOT DSMC simulations and are deliberately labelled as such. The particle
    exercise checks moment algebra; the field exercise checks reconstruction logic.
    Prerequisites: Weeks 3, 4, 10; allow 90 minutes. Restart and Run All.
    No research checkpoint, raw archive or retained results file is modified.

    Predict: can a perfectly smooth heat-flux field have the correct energy
    residual and still be wrong? Can a historical prior replace the measured mean?
    '''), code(SETUP), md(r'''
    ## 1. Pool additive statistics before centralising
    For unit-weight particles store count $C_0$, first moment $C_i$, second
    moment $C_{ij}$, squared-speed sum $E_2$ and energy-flux sum $F_i$.
    After pooling, $u_i=C_i/C_0$ and
    $J_i=F_i-u_iE_2-2\sum_j C_{ij}u_j+2C_0u_i|u|^2$.
    Physical heat flux also requires mass, cell volume and sampling-time factors.
    The next example uses unequal blocks and different means to make the
    non-commutation visible. It does not model a DSMC collision process.
    '''), code('''
    rng = np.random.default_rng(12)
    blocks_v = [rng.normal(size=(300, 3))+[2,0,0], rng.normal(size=(190, 3))+[-1,1,0]]
    raw = [additive_moments(v) for v in blocks_v]
    pooled = {k: sum(r[k] for r in raw) for k in raw[0]}
    all_v = np.concatenate(blocks_v)
    c = all_v-all_v.mean(axis=0)
    direct = (np.sum(c*c, axis=1)[:,None]*c).sum(axis=0)
    correct = central_heat_numerator(pooled)
    wrong = sum(central_heat_numerator(r) for r in raw)
    np.testing.assert_allclose(correct, direct, atol=1e-9)
    display(pd.DataFrame({'pooled': correct, 'separately centralised': wrong}, index=['x','y','z']))
    '''), md('''
    ## 2. Freeze independent development, validation and test units
    Each case has a known analytic scalar pattern, ten noisy observation blocks,
    and a separate noisy 80-block-equivalent reference. A shared reference is used
    for every method in a case. Raw(3) is a subset of Raw(10), reflecting nested
    sampling budgets; neither overlaps the reference. Gaussian independent blocks
    cannot establish how a real correlated DSMC sampler behaves.
    '''), code('''
    development = [sampling_case(i) for i in range(100,108)]
    validation = [sampling_case(i) for i in range(200,204)]
    tests = [sampling_case(i) for i in range(300,306)]
    prior, gain = fit_spectral_prior(development, budget=3)
    assert gain.min() >= 0 and gain.max() <= 1 and gain[0,0] == 1
    print('Prior and gains fitted on development only; frozen before test.')
    '''), md(r'''
    ## 3. Prior + bounded observation residual
    In an orthonormal DCT, $\hat z=z_{prior}+G(z_{obs}-z_{prior})$, with
    $0\leq G\leq1$. This lab estimates a mean prior and signal/noise powers from
    development cases. It is NOT the article's trained MambaIR prior or its
    cylinder two-component transfer matrices. Set $G_{00}=1$ and restore the
    measured mean. Preserving the mean does not make the noisy mean exact.
    Compare Raw(3), Raw(10), a Gaussian filter, prior-only, and corrected prior.
    Filter width is selected on validation only; no winner is assumed.
    '''), code('''
    from scipy.ndimage import gaussian_filter
    widths = [.5, 1., 1.5, 2.]
    width = min(widths, key=lambda s: np.mean([nrmse(
        gaussian_filter(c['blocks'][:3].mean(axis=0), s, mode='reflect'),
        c['reference']) for c in validation]))
    def estimates(case):
        observation = case['blocks'][:3].mean(axis=0)
        return {'Raw(3)': observation, 'Raw(10)': case['blocks'].mean(axis=0),
                'Gaussian': gaussian_filter(observation, width, mode='reflect'),
                'Prior only': prior, 'Prior + observation': reconstruct(observation, prior, gain)}
    print('Frozen Gaussian width:', width)
    '''), md('''
    ## 4. Paired errors, gradients and measured zero-frequency content
    NRMSE is sqrt(sum(A*(prediction-reference)^2)/sum(A*reference^2)).
    This grid has equal cell areas. Spatial cells are not independent replicates.
    We show reference-based error and analytic-truth error separately, because an
    independent high-budget average still has noise. Gradient error catches a
    failure that a smooth-looking contour may hide. All seed-level errors remain.
    '''), code('''
    rows = []
    for seed, case in enumerate(tests, start=300):
        for name, field in estimates(case).items():
            grad = np.stack(np.gradient(field, case['x'][1]-case['x'][0]))
            true_grad = np.stack(np.gradient(case['truth'], case['x'][1]-case['x'][0]))
            rows.append({'seed': seed, 'method': name,
                'reference NRMSE': nrmse(field, case['reference']),
                'truth NRMSE': nrmse(field, case['truth']),
                'gradient NRMSE': nrmse(grad, true_grad),
                'observed mean change': field.mean()-case['blocks'][:3].mean()})
    metrics = pd.DataFrame(rows)
    display(metrics)
    display(metrics.groupby('method').mean(numeric_only=True).drop(columns='seed'))
    corrected = metrics[metrics['method']=='Prior + observation']
    assert corrected['observed mean change'].abs().max() < 1e-12
    case = tests[0]
    fields = {'Analytic truth': case['truth'], **estimates(case)}
    fig, axes = plt.subplots(2,3,figsize=(11,6.4),layout='constrained')
    for ax, (name, field) in zip(axes.flat, fields.items()):
        image = ax.imshow(field, origin='lower', extent=[0,1,0,1], cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set(title=name, xlabel='x/L', ylabel='y/L')
    fig.colorbar(image, ax=axes, shrink=.8, label='Dimensionless scalar heat-flux analog')
    fig.suptitle('Week 12: synthetic sampling analog - common color range')
    plt.show()
    print('Values outside display range, by method:',
          {k: int(np.count_nonzero(abs(v)>2)) for k,v in fields.items()})
    '''), md('''
    ## 5. Support monitoring without looking at the test reference
    A deliberately simple field-level residual/noise score is calibrated by the
    maximum validation score, then frozen. A new shifted observation can trigger
    abstention. This is not the article's 27 component-zone gain-envelope score
    and carries no calibrated coverage guarantee. The fallback is to report the
    raw observation and request more sampling or new in-condition development data.
    A low score alone does not prove the reconstruction is accurate.
    '''), code('''
    noise_variance = float(np.mean([np.var(c['blocks'],axis=0,ddof=1).mean()/3
                                   for c in development]))
    score = lambda c: support_score(c['blocks'][:3].mean(axis=0), prior, noise_variance)
    threshold = max(score(c) for c in validation)
    shifted = sampling_case(900, offset=2.)
    display(pd.DataFrame([{'case': name, 'score': score(c), 'threshold': threshold,
                          'decision': 'abstain' if score(c)>threshold else 'within heuristic envelope'}
                         for name,c in [('new in-condition draw',tests[0]),('shifted draw',shifted)]]))
    assert score(shifted)>threshold
    '''), md(r'''
    ## 6. Why energy-residual matching is insufficient
    For $\delta q=(\partial_y\psi,-\partial_x\psi)$, divergence is zero.
    Adding such a field changes heat flux without changing its divergence.
    The next finite-difference check uses commuting derivatives on one grid;
    it is an algebraic control, not a boundary-value uniqueness theorem.
    '''), code('''
    x = tests[0]['x']; X,Y = np.meshgrid(x,x); h=x[1]-x[0]
    psi = np.sin(np.pi*X)**2*np.sin(np.pi*Y)**2
    py,px = np.gradient(psi,h,h)
    delta_qx,delta_qy=py,-px
    divergence=np.gradient(delta_qx,h,axis=1)+np.gradient(delta_qy,h,axis=0)
    print('Maximum divergence:', abs(divergence).max(), 'nonzero flux norm:', np.linalg.norm([delta_qx,delta_qy]))
    assert abs(divergence).max()<1e-10
    '''), md('''
    ## Research bridge and assignment
    The paper retains nine moment fields. Its cavity prior is a trained MambaIR
    restoration model. Its cylinder method rotates Cartesian heat flux to normal
    and tangential components, uses a geometry-adapted transform and bounded
    two-component residual transfer, then restores native-area Cartesian means.
    This single-scalar Cartesian lab does not implement that complete estimator.

    **Assignment:** (1) Introduce temporal correlation and measure the failure of
    the 1/B noise law. (2) Replace the scalar pattern with two heat-flux components
    and test normal/tangential rotation and its inverse. (3) Add a narrow layer:
    measure peak/width bias as well as NRMSE after filtering. (4) Design an
    independent-seed reference and matched-budget comparison for your DSMC case.
    (5) Explain why mean preservation is not positivity or full conservation.
    Do not force Fourier's law in a rarefied non-equilibrium flow.

    [Lecture 12](../../lectures/week12_dsmc_moment_reconstruction.pdf).
    Source: [Roohi, arXiv:2609.01637](https://doi.org/10.48550/arXiv.2609.01637).
    Original teaching implementation and synthetic observations, AI-assisted
    authoring and verification; no unpublished solver or checkpoint redistributed.
    ''')]


NAMES = {11: ('W11_Shock_Vortex_Identification', 'week11_shock_vortex_identification'),
         12: ('W12_DSMC_Moment_Reconstruction', 'week12_dsmc_moment_reconstruction')}


def build_notebook(week, execute):
    name, _ = NAMES[week]
    relative = f'notebooks/week{week}/{name}.ipynb'
    cells = week11() if week == 11 else week12()
    cells[0].source += badge(relative)
    cells[1].source = bootstrap(f'notebooks/week{week}') + cells[1].source
    for i, c in enumerate(cells):
        c.id = f'w{week}-{i:02d}'
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
        'language_info': {'name':'python', 'version':'3.12'},
        'flowmllab': {'scope':'manufactured teaching analog, not research reproduction'}})
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=240, kernel_name='python3',
                       resources={'metadata': {'path': str(path.parent)}}).execute()
        figures = [o['data']['image/png'] for c in nb.cells if c.cell_type == 'code'
                   for o in c.outputs if 'image/png' in o.get('data', {})]
        if len(figures) != 1:
            raise ValueError(f'Expected one teaching figure for Week {week}, got {len(figures)}')
        dest = ROOT / 'results' / 'week11_12_teaching'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f'week{week}_teaching.png').write_bytes(base64.b64decode(figures[0]))
    nbf.validate(nb)
    nbf.write(nb, path)
    print(path.relative_to(ROOT), 'executed' if execute else 'built')


def build_pdf(week):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.enums import TA_LEFT
    _, stem = NAMES[week]
    source = ROOT / 'lectures' / 'source' / (stem+'.md')
    text = source.read_text(encoding='utf-8')
    title, *sections = text.split('\n## ')
    out = ROOT / 'output' / 'pdf' / (stem+'.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)
    body = ParagraphStyle('body', fontName='Helvetica', fontSize=11.5, leading=17,
                          spaceAfter=13, textColor=colors.HexColor('#243746'), alignment=TA_LEFT)
    heading = ParagraphStyle('heading', parent=body, fontName='Helvetica-Bold',
                             fontSize=23, leading=29, spaceAfter=22)
    small = ParagraphStyle('small', parent=body, fontSize=9, leading=13)
    story = []
    for i, section in enumerate(sections):
        name, content = section.split('\n', 1)
        if i:
            story.append(PageBreak())
        story.append(Paragraph(f'FLOWMLLAB / WEEK {week} / {i+1:02d}', small))
        story.append(Paragraph(html.escape(name), heading))
        for para in content.strip().split('\n\n'):
            if para.strip() == '[TEACHING_FIGURE]':
                image = ROOT / 'results' / 'week11_12_teaching' / f'week{week}_teaching.png'
                if image.exists():
                    from PIL import Image as PILImage
                    with PILImage.open(image) as im:
                        w, h = im.size
                    story.append(Image(str(image), width=475, height=475*h/w))
                continue
            story.append(Paragraph(html.escape(para.replace('\n',' ')), body))
        story.append(Spacer(1, 8))
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#168aad'))
        canvas.line(50, 47, 545, 47)
        canvas.setFont('Helvetica', 9)
        canvas.drawString(50, 31, f'Ehsan Roohi | FlowMLLab | Week {week} | Teaching analogs clearly labelled')
        canvas.drawRightString(545, 31, str(doc.page))
    SimpleDocTemplate(str(out), pagesize=(595,842), leftMargin=60, rightMargin=60,
                      topMargin=55, bottomMargin=65, title=title.strip('# \n'),
                      author='Ehsan Roohi').build(story, onFirstPage=footer, onLaterPages=footer)
    shutil.copy2(out, ROOT / 'lectures' / out.name)
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for week in (11,12):
        build_notebook(week, args.execute)
        build_pdf(week)
