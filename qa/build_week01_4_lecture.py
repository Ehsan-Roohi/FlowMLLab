"""Build the Week 1.4 lecture from freshly executed notebook evidence.

Text: licensed Times New Roman, black. Equations are typeset with mathtext.
Run the notebook first; this builder never invents results or timings.
"""
from pathlib import Path
from io import BytesIO
import json, os, sys, re
from xml.sax.saxutils import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import mathtext, font_manager
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qa'))
from lecture_typography import register_reportlab
from week01_4_cavity_lecture import cavity_pages

EVIDENCE = ROOT / 'results/week01_4_spectral_dg'
RESULTS = json.loads((EVIDENCE / 'execution.json').read_text(encoding='utf-8'))
if os.name == 'nt':
    os.environ.setdefault('FLOWMLLAB_TNR_DIR', 'C:/Windows/Fonts')
FONTS = register_reportlab('W14TNR')
matplotlib.rcParams.update({'mathtext.fontset':'stix', 'font.family':'Times New Roman'})
WIDTH, HEIGHT = A4
BODY = WIDTH - 104
styles = {
    'body': ParagraphStyle('body', fontName=FONTS['regular'], fontSize=11, leading=15.2, spaceAfter=8, textColor=colors.black),
    'title': ParagraphStyle('title', fontName=FONTS['bold'], fontSize=19, leading=23, spaceAfter=13, textColor=colors.black),
    'sub': ParagraphStyle('sub', fontName=FONTS['bold'], fontSize=12, leading=15, spaceBefore=7, spaceAfter=6, textColor=colors.black),
    'small': ParagraphStyle('small', fontName=FONTS['regular'], fontSize=9, leading=12, spaceAfter=6, textColor=colors.black),
    'caption': ParagraphStyle('caption', fontName=FONTS['regular'], fontSize=9.5, leading=12.5, spaceAfter=10, textColor=colors.black),
}
story = []
sections = []


def para(text, style='body'):
    story.append(Paragraph(text, styles[style]))


def page(title):
    if story:
        story.append(PageBreak())
    sections.append(title)
    para(title, 'title')


def eq(expression):
    expression = expression.replace(r'\bmod', r'\;\mathrm{mod}\;')
    expression = re.sub(r'\\frac([0-9])([0-9])', r'\\frac{\1}{\2}', expression)
    expression = re.sub(r'\\mathcal\s+([A-Za-z])', r'\\mathcal{\1}', expression)
    stream = BytesIO()
    mathtext.math_to_image('$' + expression + '$', stream,
                          prop=font_manager.FontProperties(size=13), dpi=300, format='png', color='black')
    stream.seek(0)
    from PIL import Image as PILImage
    im = PILImage.open(stream)
    w, h = im.size
    scale = min(72 / 300, BODY / w)
    stream.seek(0)
    story.extend([Spacer(1, 4), Image(stream, width=w*scale, height=h*scale), Spacer(1, 8)])


def figure(name, caption, width=BODY):
    from PIL import Image as PILImage
    path = EVIDENCE / (name + '.png')
    with PILImage.open(path) as im:
        w, h = im.size
    story.append(Image(str(path), width=width, height=width*h/w))
    para(caption, 'caption')


def table(headers, rows, widths=None):
    text = [[Paragraph(str(v), styles['small']) for v in row] for row in [headers] + rows]
    t = Table(text, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'), ('LINEABOVE',(0,0),(-1,0),0.7,colors.black),
        ('LINEBELOW',(0,0),(-1,0),0.5,colors.black), ('LINEBELOW',(0,-1),(-1,-1),0.7,colors.black),
        ('LEFTPADDING',(0,0),(-1,-1),5), ('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t, Spacer(1, 9)])


page('Week 1.4\nSpectral and discontinuous Galerkin methods')
para('A CPU laboratory following SIMPLE, PISO and PIMPLE', 'sub')
para('FlowMLLab · Ehsan Roohi · University of Massachusetts Amherst')
para('A higher-order method gives a richer approximation inside a region of the domain. '
     'It does not automatically produce a more accurate answer: the solution must be resolved, '
     'the boundary treatment must be consistent, and the discrete equations must be solved accurately. '
     'This unit lets you change those choices and observe their consequences.')
para('The first part constructs a global Chebyshev polynomial and solves a two-dimensional '
     'Poisson equation. The second constructs a discontinuous Galerkin (DG) transport solver '
     'from local Legendre polynomials. The final part solves an actual lid-driven cavity at Re=100 and 400, '
     'including velocity, pressure and vorticity. Every displayed experiment is recomputed by the companion notebook.')
table(['Part', 'Learning evidence'], [
    ['Chebyshev approximation', 'A differentiation matrix, smooth/corner interpolation and a verified Poisson solve.'],
    ['Discontinuous Galerkin', 'A weak-form residual, conservative flux, time integrator and transported profiles.'],
    ['Actual driven cavity', 'Chebyshev Navier-Stokes solve, pressure gauge, flow fields and Ghia profile comparisons.'],
    ['Engineering comparison', 'Accuracy, work, invariants, refinement and a justified method recommendation.']
], [140, BODY-140])
para('<b>Prerequisites:</b> Week 1 NumPy and derivatives; Week 1.2 continuity, momentum and pressure correction. '
     'Week 1.3 neural-network optimization is not required. The numbering preserves the existing course addresses.')
para('<b>Compute:</b> a conventional CPU and Python 3.10-3.13 with NumPy, SciPy, pandas and Matplotlib. '
     'The core needs no GPU, MPI, external CFD package, downloaded data or training run. '
     'Start with the supplied small settings; refinement increases work.')
para('<b>Suggested teaching allocation:</b> two sessions plus the investigations at the end. '
     'This is a teaching plan, not a measured student completion time.')
para('<b>Companion:</b> notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb. '
     'Tables A, P, D, F, T, H, C and G have the same meaning in the lecture and notebook.', 'small')

page('1. Which part of a CFD algorithm are we changing?')
para('A flow computation has several distinct choices: the governing equations, the spatial approximation, '
     'the time integration, and the algebraic or pressure-velocity solution procedure. '
     'A useful experiment varies one layer while holding the others fixed.')
table(['Layer', 'Examples', 'Primary question'], [
    ['Spatial approximation', 'Finite volume, finite difference, Chebyshev collocation, DG', 'How is a field represented and differentiated?'],
    ['Pressure-velocity coupling', 'SIMPLE, PISO, PIMPLE, projection', 'How are momentum and continuity satisfied together?'],
    ['Time integration', 'Euler, Runge-Kutta, BDF', 'How is the state advanced, and with what temporal error?'],
    ['Algebraic solve', 'Direct factorization, Krylov iteration, preconditioning', 'How accurately and efficiently are the discrete equations solved?']
], [112, 169, BODY-281])
para('Week 1.2 focused on pressure-velocity coupling. This unit focuses on spatial approximation. '
     'DG and PISO are therefore not two names for competing versions of the same algorithm. '
     'A compatible high-order spatial method can be combined with a pressure-correction strategy.')
para('We start with scalar equations whose exact solutions are known. A verified spatial operator '
     'is a building block for a flow solver, but a scalar success does not verify a complete '
     'Navier-Stokes implementation. This distinction prevents an attractive contour from becoming an unsupported CFD claim.')
para('Verification, convergence and validation', 'sub')
para('<b>Verification:</b> compare with a known analytic solution or an exact discrete identity. '
     '<b>Convergence:</b> show what happens when spatial and temporal resolution are increased. '
     '<b>Validation:</b> compare the complete physical problem with appropriate independent evidence, '
     'such as Ghia velocity samples for the standard driven cavity.')
para('Before running, write the domain, equation, boundary data, exact solution, evaluation grid and error norm. '
     'A comparison is only interpretable when these are identical or their differences are explicitly explained.')

page('2. Global interpolation and wall-clustered nodes')
para('An interpolating polynomial represents a field through its values at nodes. On an equally spaced grid, '
     'high-degree global interpolation can behave poorly near the endpoints. Chebyshev-Lobatto nodes '
     'cluster near the boundaries and provide a well-conditioned route to polynomial interpolation when evaluated barycentrically.')
eq(r'x_j=\frac{1-\cos(j\pi/N)}{2},\qquad j=0,\ldots,N')
eq(r'q_N(x)=\sum_{j=0}^{N}q_j\ell_j(x),\qquad \ell_j(x_i)=\delta_{ij}')
para('Here N is the global degree, N+1 is the number of nodes, q<sub>j</sub> is the value at node j, '
     'and the cardinal polynomial l<sub>j</sub> is one at its own node and zero at the others. '
     'These coordinates already lie on the physical unit interval.')
figure('interpolation_smoothness', 'Figure 1. The corner target and its interpolant; sampled interpolation error for smooth and corner targets. '
       'All errors use the same 2001 evaluation points. N controls the approximation, not the number of plotted pixels.')
para('The smooth target has rapidly decaying polynomial coefficients and can converge very quickly. '
     'The derivative jump in |x-0.37| limits this behavior. Adding degree does not remove the corner. '
     'Once the smooth error approaches floating-point roundoff, additional degree need not improve it.')
para('<b>Notebook connection:</b> Table A varies N. Exercise 1 adds polynomial and derivative tests. '
     'A finite-resolution sampled maximum is not a rigorous bound on the continuous maximum.')

page('3. Build the differentiation matrix')
para('Differentiate the interpolant at the nodes. This turns differentiation into a matrix-vector product. '
     'Unlike a short finite-difference stencil, a row of the Chebyshev matrix generally uses every nodal value.')
eq(r'q_N^{\prime}(x_i)=\sum_{j=0}^{N}D_{ij}q_j,\qquad D_{ij}=\ell_j^{\prime}(x_i)')
eq(r'D_{ij}=\frac{w_j}{w_i(x_i-x_j)}\ (i\ne j),\qquad D_{ii}=-\sum_{j\ne i}D_{ij}')
para('Use weights w<sub>j</sub>=(-1)<super>j</super>, with each endpoint weight divided by two. '
     'The diagonal is set by the negative row sum. Consequently D applied to a constant is zero to rounding accuracy. '
     'Multiplying D by itself gives the nodal second-derivative operator for the interpolating polynomial.')
para('A small calculation you can check by hand', 'sub')
para('For N=2 the nodes are 0, 1/2 and 1. The three rows of D are (-3,4,-1), (-1,0,1), '
     'and (1,-4,3). Applying this matrix to the nodal vector of x gives (1,1,1). '
     'Applying it to x<super>2</super> gives (0,1,2). Verify both before solving any PDE.')
para('The coordinate map matters', 'sub')
eq(r'x=\frac{1+\xi}{2},\qquad \frac{d}{dx}=2\frac{d}{d\xi},\qquad \frac{d^2}{dx^2}=4\frac{d^2}{d\xi^2}')
para('A differentiation matrix built on [-1,1] cannot be reused on [0,1] without these factors. '
     'The notebook constructs D directly from the mapped x coordinates, so the factors are already included. '
     'Applying them again would introduce a scaling error.')
para('Algorithm: chebyshev_grid', 'sub')
para('1. Generate increasing Lobatto nodes.<br/>2. Form the pairwise coordinate differences.<br/>'
     '3. Fill off-diagonal entries from barycentric weights.<br/>4. Set the diagonal from row sums.<br/>'
     '5. Test constants and several polynomials before using D or D squared.')
para('Differentiation amplifies unresolved content. A visually good interpolant can still have an inaccurate '
     'derivative, especially near a corner or discontinuity. Plot the derivative error as well as the field.')

page('4. A manufactured pressure Poisson problem')
para('Use a prescribed nondimensional pressure on the unit square. This choice gives an exact reference '
     'for the spatial operator and isolates discretization error from unknown physical data.')
eq(r'p^*(x,y)=\sin(\pi x)\sin(2\pi y),\qquad (x,y)\in[0,1]^2')
eq(r'\nabla^2p^*=f,\qquad f=-5\pi^2\sin(\pi x)\sin(2\pi y)')
eq(r'p^*|_{\partial\Omega}=0')
para('The field has positive and negative regions. The source f and every boundary value are known. '
     'These are deliberately Dirichlet pressure conditions for verification; they are not asserted to be '
     'the pressure boundary conditions of a physical lid-driven cavity.')
para('A tensor-product solve', 'sub')
eq(r'A=(D^2)_{I,I},\quad I=1,\ldots,N-1,\qquad AP+PA^T=F')
para('P contains interior values, with rows indexing y and columns indexing x. The product AP differentiates '
     'in y; PA<super>T</super> differentiates in x. Zero boundary values make the boundary contribution zero. '
     'For nonzero boundary data, subtract their contributions from F before solving the interior system.')
para('The Sylvester solver uses the two one-dimensional matrices. A naive dense two-dimensional '
     'matrix would have (N-1)<super>4</super> entries; exploiting tensor structure avoids storing that matrix. '
     'The finite-difference comparator instead builds a sparse Kronecker sum from a tridiagonal second derivative.')
para('Algorithm: poisson_square', 'sub')
para('Choose resolution and method → assemble the 1-D second derivative → restrict to interior nodes → '
     'sample the source on that method\'s nodes → solve the tensor equation or sparse system → '
     'insert wall values → measure the equation residual → interpolate to the common evaluation grid.')
para('<b>Equal-count comparison:</b> both methods have (N-1)<super>2</super> interior unknowns, '
     'but their node locations, sparsity and solution procedures differ. Equal unknown count is a useful '
     'control; it is not equal floating-point work.')

page('5. Compare the field, not only a residual')
figure('pressure_poisson_fields', 'Figure 2. Manufactured analytic pressure on the unit square, with Chebyshev and finite-difference '
       'contours at the same levels (-0.8, -0.4, 0, 0.4, 0.8). Right: pressure along x=0.5. Both numerical '
       'fields use N=12, or 121 interior unknowns. This is a scalar verification field, not a Ghia cavity result.')
eq(r'\epsilon_2=\frac{\left\|p_h-p^*\right\|_2}{\left\|p^*\right\|_2},\qquad \epsilon_\infty=\max|p_h-p^*|')
para('Both norms are evaluated on the same uniform 151 by 151 points. The Chebyshev nodal field is '
     'evaluated as its global polynomial; the finite-difference nodal field is reconstructed bilinearly. '
     'The reported error therefore includes the stated reconstruction. It is not a comparison of only nodal values.')
selected = [r for r in RESULTS['tables']['P'] if r['N'] in (12, 32)]
table(['Method', 'N', 'Unknowns', 'Relative L2', 'Max error'],
      [[r['method'].replace('_',' '), r['N'], r['unknowns'], f"{r['relative_L2']:.3e}", f"{r['max_error']:.3e}"] for r in selected],
      [145, 35, 65, 100, BODY-345])
para('These are measured values from the accompanying notebook execution. Inspect signed profiles: '
     'a global norm can hide where a discrepancy occurs. Changing the colour range separately for '
     'each method would obscure the comparison, so all overlays use identical physical levels.')

page('6. Accuracy, residual and cost tell different stories')
figure('poisson_accuracy_cost', 'Figure 3. Common-grid pressure error versus interior unknown count and measured assembly-plus-solve time. '
       'Each timing is the median of three repetitions after a warm-up; Table P also reports minimum and maximum.')
para('The algebraic residual tests whether the discrete equations were solved. The exact-solution '
     'error tests whether those equations approximate the continuous problem at the requested resolution. '
     'A coarse finite-difference solve can have an almost machine-precision residual and a visible field error.')
eq(r'r_h=A_h p_h-f_h')
para('Second-order finite differences should reduce error by about four when the grid spacing is halved '
     'in an appropriate asymptotic regime. The smooth analytic target can converge much faster with '
     'Chebyshev degree, until roundoff and conditioning matter. That favorable case does not establish '
     'the same rate for a corner-singular cavity.')
para('Timing contract', 'sub')
para('The recorded time includes operator assembly and the linear solve. It excludes common-grid '
     'error interpolation, plotting and notebook display. Tiny problems are sensitive to timer overhead, '
     'BLAS threading and process warm-up. Keep all repeats and avoid claiming a universal speedup '
     'from a single point. Compare the cost of meeting a stated error tolerance.')
para('<b>Exercise 2 decision:</b> choose the smallest tested system with relative field error below 10<super>-3</super>. '
     'Report its residual, accuracy, repeated timings and implementation assumptions. '
     'If no tested system passes, report that fact and refine; do not change the threshold afterward.')

page('7. DG begins with a local weak form')
para('We now solve periodic linear advection. A scalar profile moves without changing shape. '
     'Its known exact translation makes amplitude, phase, diffusion and oscillation errors directly observable.')
eq(r'q_t+a q_x=0,\qquad q(x,t)=q_0((x-at)\bmod 1)')
para('Here q is a transported scalar, a is a prescribed nonzero speed, and t is time. The domain is '
     'the unit interval with periodic boundaries. Divide it into K equal cells of width h=1/K. '
     'There is no Reynolds or Mach number in this model problem.')
eq(r'q_h|_{I_e}=\sum_{n=0}^{p}c_{e,n}(t)P_n(\xi),\qquad \xi=2(x-x_{e,L})/h-1')
para('P<sub>n</sub> is a Legendre polynomial. The degree p controls how many local coefficients are stored: '
     'K(p+1) coefficients in total. The polynomial on one cell need not equal its neighbour at an interface. '
     'That discontinuity supplies two traces, which are coupled by a numerical flux.')
eq(r'\int_{I_e}P_m q_t\,dx-\int_{I_e}a q_h\frac{dP_m}{dx}\,dx+\widehat f_R P_m(1)-\widehat f_L P_m(-1)=0')
para('The derivative has been moved to the test function by integration by parts. The volume term '
     'describes within-cell transport; the two boundary terms describe exchange with neighbouring cells. '
     'A single flux value at each interface is essential: computing incompatible values independently '
     'on the two sides would destroy conservation.')
para('Notation', 'sub')
para('e labels the cell; n labels a trial basis function; m labels the test function; c<sub>e,n</sub> '
     'is its time-dependent coefficient; x<sub>e,L</sub> is the left cell boundary; R and L denote '
     'the cell faces. In this section p is polynomial degree; p* in the Poisson section was pressure.')

page('8. From integrals to a short residual routine')
eq(r'\int_{I_e}P_mP_n\,dx=\frac{h}{2m+1}\delta_{mn},\qquad S_{mn}=\int_{-1}^{1}P_m^{\prime}P_n\,d\xi')
para('The diagonal mass matrix follows from Legendre orthogonality. The derivative matrix S is '
     'strictly lower triangular: S<sub>mn</sub>=2 when m is larger than n and m-n is odd, and zero otherwise. '
     'This exact expression avoids repeatedly approximating a polynomial volume integral.')
eq(r'\dot c_{e,m}=\frac{2m+1}{h}\left(a\sum_n S_{mn}c_{e,n}-\widehat f_R+(-1)^m\widehat f_L\right)')
para('The factors P<sub>m</sub>(1)=1 and P<sub>m</sub>(-1)=(-1)<super>m</super> explain the two signs. '
     'For p=0 the volume derivative is zero, giving the familiar finite-volume update for a cell average:')
eq(r'\dot c_{e,0}=-\frac{\widehat f_R-\widehat f_L}{h}')
figure('spatial_representation_schematic', 'Spatial representation schematic. Left: equal-count uniform and Chebyshev nodes. '
       'Right: independent cell polynomials exchange information through a shared interface flux. '
       'The right-hand values illustrate the representation; they are not simulation results.')
para('Algorithm: dg_rhs', 'sub')
para('1. Sum modal coefficients to obtain right traces.<br/>'
     '2. Sum alternating-sign coefficients to obtain left traces.<br/>'
     '3. Pair each right trace with the next cell\'s left trace, wrapping periodically.<br/>'
     '4. Compute one interface flux.<br/>'
     '5. Shift that same flux to the neighbouring left face.<br/>'
     '6. Add the volume derivative and multiply by the inverse diagonal mass.')
para('Projection of the initial field', 'sub')
eq(r'c_{e,m}(0)=\frac{2m+1}{h}\int_{I_e}q_0(x)P_m(\xi(x))\,dx')
para('The notebook evaluates this integral with Gauss quadrature. For the square pulse, it first '
     'splits a cell at each discontinuity. A centre-point sample is insufficient to initialize the higher modes. '
     'The initial representation error is real and must not be confused with time-stepping error.')

page('9. Numerical flux, mass and energy')
eq(r'\widehat f=\frac{a}{2}(q^-+q^+)-\frac{\alpha|a|}{2}(q^+-q^-)')
para('At an interface, q<super>-</super> is the trace from the left cell and q<super>+</super> '
     'from the right cell. With alpha=1 this reduces to the upwind flux: aq<super>-</super> '
     'for positive speed, aq<super>+</super> for negative speed. Alpha=0 is central; intermediate '
     'values adjust the dissipation for this teaching experiment.')
eq(r'\mathcal M=h\sum_e c_{e,0},\qquad E=\frac12\sum_{e,m}\frac{h c_{e,m}^2}{2m+1}')
eq(r'\frac{d\mathcal M}{dt}=0,\qquad \frac{dE}{dt}=-\frac{\alpha|a|}{2}\sum_{\rm interfaces}(q^+-q^-)^2')
para('The mass identity comes from cancellation of the same face flux between neighbours. '
     'The energy identity holds for the semidiscrete periodic linear problem. It explains what '
     'upwinding does: jumps between local polynomials remove energy. With central flux the semidiscrete '
     'energy derivative vanishes, but the time integrator can still introduce dissipation.')
para('Three distinct checks', 'sub')
para('<b>Constant preservation:</b> a constant profile should remain constant for either speed sign.<br/>'
     '<b>Mass conservation:</b> sum the cell averages; the result should be unchanged to rounding error.<br/>'
     '<b>Energy behavior:</b> compare the computed semidiscrete derivative with the jump expression, '
     'then distinguish it from the energy change after a finite time step.')
para('Low dissipation is not always low error. A sharp front contains unresolved high-frequency content. '
     'Central flux can retain substantial oscillation; upwind flux damps it but also changes the profile. '
     'An unlimited high-order DG method is not automatically monotone or positivity-preserving. '
     'A nonlinear conservation law with shocks would need further stability and limiting analysis.')

page('10. Advance in time and record the work')
para('The spatial residual L(c) defines an ordinary differential equation for the coefficient array. '
     'We use three-stage, third-order strong-stability-preserving Runge-Kutta (SSPRK3).')
eq(r'c^{(1)}=c^n+\Delta t L(c^n)')
eq(r'c^{(2)}=\frac34c^n+\frac14\left[c^{(1)}+\Delta t L(c^{(1)})\right]')
eq(r'c^{n+1}=\frac13c^n+\frac23\left[c^{(2)}+\Delta t L(c^{(2)})\right]')
para('Each step requires three residual evaluations. SSP describes a conditional inheritance '
     'of a suitable forward-Euler stability property; it does not make arbitrary high-order DG '
     'data monotone, and it does not remove the need for a stable time step.')
eq(r'\Delta t\leq\frac{C h}{|a|(2p+1)},\qquad C=0.12\ \mathrm{in\ the\ default\ lab}')
para('This is the tested default controller for the notebook, not a universal CFL theorem. '
     'The implementation chooses an integer number of steps and reduces the step size to hit '
     'the requested final time exactly. Increasing degree changes both the spatial operator '
     'and the allowable explicit step. Equal coefficient counts can therefore have different costs.')
para('Algorithm: solve_dg', 'sub')
para('Build cell/basis operators → project the initial profile → choose the step count → '
     'repeat the three Runge-Kutta stages → check finite values → record mass and energy → '
     'evaluate the final piecewise polynomial on fixed points → compare with exact translation.')
para('The routine rejects an interactive request above 100,000 steps and stops a nonfinite or '
     'grossly amplified trajectory. These guards protect the session, but passing them is not '
     'proof of accuracy. Preserve the controls of a failed run and diagnose its cause.')
para('Report K, p, the flux parameter, C, actual time step, final time, coefficient count, '
     'residual evaluations and elapsed time alongside every error. A small error with an undisclosed '
     'work budget is not a useful engineering comparison.')

page('11. Equal unknown count: where should resolution go?')
figure('dg_transport_equal_dofs', 'Figure 4. A localized periodic profile transported at a=1 to t=0.35 with upwind flux. '
       'Every curve uses 64 coefficients: (K,p)=(64,0), (32,1), (16,3), (8,7). '
       'Right: full signed error. The exact profile is translated, not refitted.')
table(['K', 'p', 'RHS calls', 'Relative L2', 'Max error'],
      [[r['K'], r['p'], r['rhs_calls'], f"{r['relative_L2']:.3e}", f"{r['max_error']:.3e}"] for r in RESULTS['tables']['D']],
      [45, 35, 95, 130, BODY-305])
para('A degree-zero calculation transports cell averages and tends to diffuse the narrow peak. '
     'Higher degree can resolve within-cell structure, but very wide cells can still be a poor '
     'allocation for a localized feature. The numerical result decides the ranking; do not assume '
     'that the largest degree wins for every target.')
para('Table D also gives measured time and mass drift. Those times are one execution per setting, '
     'unlike the repeated Poisson timings. For a performance claim, repeat the DG runs, report '
     'their spread and keep the accuracy target fixed. The residual-call count is a useful work '
     'indicator, though a call at higher degree is itself more expensive.')
para('<b>Exercise 3:</b> repeat for the smooth and localized targets, first at 64 and then at '
     '128 coefficients. Explain both the global error and the position of the largest signed error.')

page('12. Sharp fronts reveal different losses')
figure('dg_flux_pulse_energy', 'Figure 5. A unit square pulse initially occupying [0.25,0.55), transported periodically with '
       'a=1 to t=0.35. K=24, p=3 and C=0.12 are identical. Right: energy relative to '
       'the initial projected state. Overshoots and undershoots are retained.')
table(['Flux parameter', 'Minimum', 'Maximum', 'Mass drift', 'Relative L2'],
      [[r['alpha'], f"{r['minimum']:.4f}", f"{r['maximum']:.4f}", f"{r['mass_drift']:.2e}", f"{r['relative_L2']:.3e}"] for r in RESULTS['tables']['F']],
      [105, 85, 85, 100, BODY-375])
para('The exact scalar mass is 0.3. The initial DG projection already has a finite representation '
     'error. Normalizing energy by the projected initial energy isolates subsequent evolution '
     'from that initial projection difference.')
para('A less dissipative solution can preserve oscillations that should not be interpreted as '
     'physical waves. Upwind flux generally reduces unresolved inter-cell content, but it is not a '
     'limiter and need not eliminate negative values. Clipping the display would conceal the issue.')
para('This is a discontinuous passive scalar, not a compressible shock. It has no Mach number, '
     'equation of state or nonlinear shock speed. The purpose is to isolate the role of the numerical flux.')
para('<b>Exercise 4:</b> add alpha=0.5 with every other setting unchanged. Report the extrema, '
     'mass and energy changes, and the complete profile. Explain the tradeoff before choosing a flux.')

page('13. Space and time need separate refinement')
figure('dg_refinement_time', 'Figure 6. Left: fixed p=2 and C=0.06, with 8,16,32 cells for the smooth periodic target. '
       'Right: signed error at fixed K=8, p=5 and t=0.35 while halving C.')
table(['C', 'dt', 'Relative L2', 'RHS calls'],
      [[r['CFL'], f"{r['dt']:.3e}", f"{r['relative_L2']:.3e}", r['rhs_calls']] for r in RESULTS['tables']['T']],
      [70, 135, 145, BODY-350])
para('The successive coefficient-difference ratio in this execution is '
     f"<b>{RESULTS['time_difference_ratio']:.3f}</b>, giving an observed temporal order of "
     f"<b>{np.log2(RESULTS['time_difference_ratio']):.3f}</b>. A ratio near eight is consistent "
     'with third-order time integration when the difference is dominated by temporal error.')
eq(r'r_t=\frac{\|c_{\Delta t}-c_{\Delta t/2}\|_2}{\|c_{\Delta t/2}-c_{\Delta t/4}\|_2},\qquad s_t=\log_2 r_t')
para('This difference test keeps the spatial discretization fixed. The total error against the exact PDE '
     'can plateau at a spatial or projection-error floor even while time differences converge. '
     'Do not interpret that plateau as proof that the Runge-Kutta method has lost its formal order.')
para('The integer step count makes the actual step-size ratios only approximately two; Table T reports '
     'the actual values. Treat the displayed order as an approximate diagnostic of the nominal halving study.', 'small')
para('For h-refinement, hold p, the PDE and final time fixed and make temporal error sufficiently small. '
     'For p-refinement, hold the cell partition fixed. Table D asks a third question: how to allocate '
     'a fixed total number of coefficients. Keep these comparisons separate.')

cavity_pages(page, para, eq, table, figure, RESULTS, BODY)

page('20. Student investigations: approximation and flux')
para('For every investigation record a prediction, one controlled change, the measured outcome '
     'and a numerical explanation. Preserve the supplied default tables and add your changed cases.')
para('Exercise 1 - Differentiate what you can verify', 'sub')
para('In Table A add x<super>5</super> and its derivative 5x<super>4</super>. Test N=4,6,12. '
     'Evaluate Dq at the solver nodes, then study the corner target away from x=0.37. '
     'Submit interpolation and derivative errors; explain polynomial exactness, nonsmoothness '
     'and why a denser evaluation grid does not improve the approximation.')
para('Exercise 2 - An accuracy target for Poisson', 'sub')
para('Extend finite differences to N=48,64 and Chebyshev to N=20. Using Table P\'s exact '
     'boundary/source data and common-grid norm, find the smallest tested unknown count reaching '
     'relative error below 10<super>-3</super>. Report median/min/max assembly-plus-solve time, '
     'equation residual and field error. State whether the ranking is robust to timing spread.')
para('Exercise 3 - Allocate the same coefficient budget', 'sub')
para('Repeat Table D for smooth and localized data at 64 and 128 coefficients. At 128 use '
     '(K,p)=(128,0),(64,1),(32,3),(16,7). Keep the speed, final time and flux fixed. '
     'Plot the physical profiles and signed errors. Compare RHS calls and repeat timings '
     'before making a cost claim. Explain why the most effective degree can depend on the target.')
para('Exercise 4 - Between central and upwind', 'sub')
para('Add alpha=0.5 to Table F at K=24,p=3,C=0.12,t=0.35. Retain the full profile, including '
     'all extrema. Report mass drift and energy change, and distinguish projection oscillation '
     'from evolution error. State whether your preferred flux minimizes norm error, overshoot '
     'or dissipation; those objectives need not agree.')
para('Deliverable for this group', 'sub')
para('Submit the executed cells, revised Tables A/P/D/F, and two figures that show the decisive '
     'physical differences. A table of numbers alone is insufficient; a screenshot without '
     'settings and norms is also insufficient. Keep failed or inconclusive cases visible.')

page('21. Student investigations: time, conservation and CFD')
para('Exercise 5 - Check the time integrator', 'sub')
para('Repeat Table T at K=16,p=5 using C=0.12,0.06,0.03. Compare successive coefficient '
     'differences and exact-PDE errors at the same final time. Explain whether time error or '
     'spatial error controls the final accuracy. Change the final time only in a separate experiment.')
para('Exercise 6 - Reverse the transport', 'sub')
para('Set a=-1 for the smooth target. Confirm that the exact and numerical profiles move '
     'in the opposite direction. Measure mass using the cell averages and verify constant '
     'preservation. Explain why the upwind state switches when a changes sign.')
para('Exercise 7 - Compute and refine the cavity', 'sub')
para('At Re=100 add N=12,20 to the existing N=16,24,32 sequence. Compare both computed '
     'centreline velocities with Ghia. Plot pressure at y=0.5 and signed vorticity at x=0.5, '
     'always using the same coordinates and centre pressure gauge. Report the velocity errors '
     'and both continuity diagnostics from Tables C and G. Identify which quantities have converged '
     'and which still change; do not use residual convergence as a spatial-error estimate.')
table(['Symptom', 'First diagnosis'], [
    ['Small residual, inaccurate field', 'Check resolution, boundary data and the common-grid error; the algebraic solve can be correct.'],
    ['Oscillating sharp front', 'Inspect projection and flux; do not clip. SSPRK3 alone is not a limiter.'],
    ['Mass changes', 'Check interface pairing, periodic wrap, weights and coefficient normalization.'],
    ['Error stops improving', 'Separate temporal, spatial, quadrature and roundoff contributions.'],
    ['A slow high-degree run', 'Inspect step count, work per residual and the linear-solve implementation.']
], [162, BODY-162])

page('22. Cavity investigations and submission')
para('Exercise 8 - Reynolds number and boundary regularity', 'sub')
para('Compare standard-lid Re=100 and 400 at matched N. Use flow fields and signed profiles '
     'to explain changes in recirculation, pressure and wall gradients. Retain the flagged Re=400 '
     'printed reference point and report both all-sample and screened scores. Describe what the '
     'reference supports and what remains unvalidated, particularly for pressure.')
para('In a separate experiment choose the smooth lid at Re=100 and N=12,20,28. Its maximum '
     'speed remains one but its driving profile differs. Compare its own fields and off-grid '
     'continuity under refinement. Explain why the notebook disables Ghia scoring for this change. '
     'Do not combine the smooth-lid and standard-lid results in a single benchmark-error column.')
para('Exercise 9 - Tolerance is not resolution', 'sub')
para('For Re=400,N=24 use tolerances 10<super>-5</super>, 10<super>-7</super> and '
     '10<super>-9</super>. Compare Newton iterations, residual histories, Ghia errors and '
     'off-grid continuity. Then limit the solve to one Newton update. Inspect the failed-convergence '
     'record and explain why it must not be reported as a qualified steady solution.')
para('Identify the role of the Newton damping parameter. Explain why its update is neither '
     'a physical time step nor proof of physical flow stability. Return to Week 1.2 to contrast '
     'this coupled nonlinear solve with pressure-correction iterations.')
para('Final submission', 'sub')
para('Provide the executed notebook, revised Tables A/P/D/F/T/H/C/G as relevant, three decisive '
     'figures including actual cavity fields and CFD-versus-Ghia profiles, and a 400-600 word '
     'recommendation. Include settings, error definitions, computational work and limitations. '
     'Numerical correctness, controlled comparison and interpretation matter more than a named method winning.')

page('23. Reproducibility, checks and reading')
para('Restart the kernel and Run All before submitting. The notebook saves Tables A/P/D/F/T/H/C/G, '
     'the figures and a compact execution record containing controls, package versions and '
     'timings. Every displayed result in this lecture comes from the retained execution record.')
para('Independent checks in tests/test_high_order.py', 'sub')
para('The checks cover exact differentiation of polynomials, a manufactured Poisson solution '
     'and finite-difference refinement, constant-state preservation, mass conservation, '
     'the semidiscrete DG energy identity, both transport directions, a split discontinuous '
     'initial projection and third-order temporal differences. Passing these tests supports '
     'the stated scalar problems.')
para('Tests in tests/test_high_order_cavity.py independently check a manufactured full '
     'Navier-Stokes velocity/pressure solution, the analytic Jacobian, wall and gauge conditions, '
     'Ghia error bounds, spatial continuity refinement and an explicit failed-iteration budget. '
     'All nine tests in the two modules passed. This is a teaching-level qualification of the '
     'documented settings, not a high-Re or full-field reference validation.')
para('Implementation notes', 'sub')
para('The notebook contains the actual solver functions, copied from the tested teaching module '
     'during authoring. Students can edit and rerun them directly. No result is fetched from '
     'a cluster. Core functions use NumPy/SciPy; pandas formats tables and Matplotlib plots them. '
     'A different machine or library can change timing and roundoff without changing the experimental question.')
para('Further reading', 'sub')
para('1. L. N. Trefethen, <i>Spectral Methods in MATLAB</i>, SIAM, 2000. '
     '<link href="https://people.maths.ox.ac.uk/trefethen/spectral.html">Author\'s programs and chapters</link>. '
     'Short spectral examples and the role of solution smoothness.', 'small')
para('2. J. S. Hesthaven and T. Warburton, <i>Nodal Discontinuous Galerkin Methods</i>, Springer, 2008. '
     '<link href="https://doi.org/10.1007/978-0-387-72067-8">doi:10.1007/978-0-387-72067-8</link>. '
     'The notebook uses modal Legendre coefficients; the numerical-flux and weak-form principles also underlie nodal DG.', 'small')
para('3. deal.II, <link href="https://dealii.org/9.7.0/doxygen/deal.II/step_12.html">Step 12: DG advection</link>. '
     'A further two-dimensional example with interface integrals and adaptive meshes.', 'small')
para('4. S. Gottlieb and C.-W. Shu, Total variation diminishing Runge-Kutta schemes, '
     '<i>Mathematics of Computation</i> 67 (1998), 73-85. '
     '<link href="https://doi.org/10.1090/S0025-5718-98-00913-2">doi:10.1090/S0025-5718-98-00913-2</link>.', 'small')
para('5. <link href="https://www.nektar.info/">Nektar++</link> and '
     '<link href="https://github.com/exadg/exadg">ExaDG</link>: optional external high-order flow solvers. '
     'Consult the case-specific documentation before comparing boundary conditions or costs.', 'small')
para('The educational algorithms are independently written from the equations shown here. '
     'The scalar tests and actual cavity comparisons support separate claims. No general '
     'speedup or high-Re validation is inferred from this small CPU study.', 'small')

page('24. Cavity sources and interpretation')
para('Ghia velocity reference', 'sub')
para('U. Ghia, K. N. Ghia and C. T. Shin, High-Re solutions for incompressible flow using '
     'the Navier-Stokes equations and a multigrid method, <i>Journal of Computational Physics</i> '
     '48 (1982), 387-411. <link href="https://doi.org/10.1016/0021-9991(82)90058-4">Original article</link>. '
     'Tables I and II provide the centreline velocity samples. The notebook embeds the repository '
     'transcription and records its SHA-256; the reference is not fitted or changed by the solver.')
para('Spectral cavity background', 'sub')
para('O. Botella and R. Peyret, Benchmark spectral results on the lid-driven cavity flow, '
     '<i>Computers &amp; Fluids</i> 27 (1998), 421-433. '
     '<link href="https://doi.org/10.1016/S0045-7930(98)00002-4">Article DOI</link>. '
     'Their treatment of corner singularities is beyond the small implementation in this unit.')
para('O. Botella and R. Peyret, Computing singular solutions of the Navier-Stokes equations '
     'with the Chebyshev-collocation method, <i>International Journal for Numerical Methods in Fluids</i> '
     '36 (2001), 125-163. <link href="https://doi.org/10.1002/fld.121">Article DOI</link>. '
     'Background for mixed velocity/pressure spaces and singular boundary data. The notebook uses '
     'a direct steady Newton solve, and does not reproduce the paper\'s singularity-subtraction algorithm.')
para('What each comparison can establish', 'sub')
table(['Evidence', 'Supported interpretation'],[
    ['Manufactured Navier-Stokes test','The discrete operators, forcing, Jacobian and pressure gauge reproduce the specified smooth solution.'],
    ['Small steady residual','The retained finite-dimensional steady equations are solved to tolerance.'],
    ['Ghia profiles','Agreement or disagreement at the published centreline velocity samples.'],
    ['Off-grid divergence and N-refinement','Remaining spatial error, particularly with a discontinuous lid.'],
    ['Pressure contours','A computed flow quantity that still requires an independent pressure reference or refinement evidence.'],
    ['Scalar DG transport','Flux, conservation and time-integration learning; not a DG cavity calculation.'],
], [153,BODY-153])


def furniture(canvas, doc):
    canvas.setFillColor(colors.black)
    canvas.setFont(FONTS['regular'], 9)
    canvas.drawString(52, HEIGHT-31, 'FlowMLLab | Week 1.4 | Spectral and discontinuous Galerkin methods')
    canvas.setLineWidth(0.4)
    canvas.line(52, HEIGHT-39, WIDTH-52, HEIGHT-39)
    canvas.drawString(52, 28, 'CPU teaching companion | November 2026')
    canvas.drawRightString(WIDTH-52, 28, str(doc.page))


out = ROOT / 'lectures/week01_4_spectral_dg.pdf'
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=52, rightMargin=52,
                  topMargin=57, bottomMargin=48, title='FlowMLLab Week 1.4: Spectral and discontinuous Galerkin methods',
                  author='Ehsan Roohi').build(story, onFirstPage=furniture, onLaterPages=furniture)
print(out, 'sections:', len(sections))
