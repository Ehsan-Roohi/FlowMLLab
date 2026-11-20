'Teaching source for the main Week 4 lecture; build with qa/build_week 04_lab_guide.py.\n\nAll measurements belong to named retained experiments, never to this PDF build.\nPage declarations are intentional teaching units, with notebook links and exact aims.\n'
from pathlib import Path

N = {
 1:'W4_Lab1_CFD_Data_Production_Student.ipynb',
 2:'W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb',
 3:'W4_Lab3_DeepONet_Cavity_Student.ipynb',
 4:'W4_Lab4_Stokes_to_Navier_Stokes.ipynb',
 5:'W4_Lab5_All_Re_Ghia_Validation.ipynb',
 6:'W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb',
 7:'W4_Lab7_High_Re_CFD_Ghia.ipynb',
 8:'W4_Lab8_High_Re_Field_DeepONet.ipynb',
 9:'W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb',
}
def B(text): return dict(type='body',text=text)
def S(text): return dict(type='sub',text=text)
def C(text): return dict(type='caption',text=text)
def E(latex): return dict(type='math',latex=latex)
def T(rows,widths=None): return dict(type='table',rows=rows,widths=widths)
def D(rows): return dict(type='diagram',rows=rows)
def F(path,caption,max_height=300): return dict(type='figure',path=path,caption=caption,max_height=max_height)
def link(path,label): return '<link color="black" href="https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/'+path+'"><u>'+label+'</u></link>'

def get_pages(root):
    pages=[]
    def P(title,blocks,lab=None,section=''):
        p=dict(title=title,blocks=blocks)
        if lab is not None:p.update(notebook=N[lab],section=section,lab_number=lab)
        pages.append(p)

    P('Week 4\nLearning cavity flows from numerical evidence',[
        S('A complete guide to Labs 1-9 and their companion studies'),
        B('FlowMLLab | Ehsan Roohi | University of Massachusetts Amherst'),
        B('A fast neural prediction is useful only when we understand the flow it approximates, the numerical evidence behind its labels, and the conditions under which it has been tested. This unit develops that understanding through a single physical problem: the lid-driven cavity. The same geometry makes it possible to see precisely what changes when we change the solver, the representation, the learning algorithm, or the information supplied to it.'),
        B('The laboratories form several related experiments, not one model with a single accuracy number. You will generate and qualify CFD data, learn primary-vortex quantities, predict complete velocity fields, construct a POD-DeepONet, learn a correction to Stokes flow, compare all seven Ghia Reynolds numbers, and audit a label-free PINN. Each section states its aim, explains its algorithm, identifies the notebook cells to run, and specifies physical evidence that a student report must contain.'),
        D([['<b>Qualify the flow</b><br/>CFD, mesh, time, walls','<b>Build a predictor</b><br/>baseline, network, POD','<b>Inspect the result</b><br/>profiles, vortices, errors']]),
        S('How to use these notes'),
        B('Read the route on the next page before opening a notebook. Complete the main supervised-learning route on a CPU. Return to advanced Lab 9 after learning PINN differentiation and residual construction in Week 13 Part A. The evidence appendix preserves earlier measured results, including unsuccessful experiments; it is indexed by laboratory and physical question rather than presented as a second course to finish before starting.'),
        B('The current notebooks are the executable source of settings and outputs. Retained figures show named completed experiments. A fresh run becomes your result only after its settings, data identity and execution are recorded. Published held-out cases remain useful regression tests, but they are no longer unseen to a student who has already inspected them.'),
    ])
    P('The laboratory route and reading map',[
        B('Use the page ranges below as a working map. Lab numbers are stable identifiers; Lab 4 is also taught as Week 4.2, and the classical ROM companion is Week 4.1. They are complementary branches, not missing labs.'),
        T([['Lab / route','Aim and main output','Guide pages'],
           ['1. CFD data','Produce a qualified numerical field and quality card.','7-9'],
           ['2. Scalar and field surrogates','Compare simple regression and neural predictions using physical outputs.','10-16'],
           ['3. POD-DeepONet','Learn parameter-dependent modal coefficients for velocity and pressure.','17-20'],
           ['4. Stokes correction','Learn nonlinear-flow corrections and test boundary-family transfer.','21-23'],
           ['5. Ghia and coverage','Separate our CFD, reference samples and different neural experiments.','24-26'],
           ['6. Profile learning','Compare depth, optimizer and weights with complete-Re splits.','27-29'],
           ['7. High-Re CFD','Inspect mesh/time sensitivity and the cost of a steady solver.','30-32'],
           ['8. Full-field learning','Compare frozen high-Re fields, interpolation and local flow structure.','33-36'],
           ['9. Advanced PINN audit','Test a label-free solution, independent residuals and failed runs.','37-40'],
           ['Optional Week 4.1','Derive a dynamical POD-Galerkin model and explain DEIM cost.','41'],
           ['Report / sources / evidence','Use the assignment checklist and indexed scientific appendix.','42-44'],
        ],[1.05,2.5,.65]),
        B('<b>Main route:</b> Labs 1 → 2 → 3 → 5 → 7 → 8. Lab 6 is a focused optimizer/depth branch after Lab 5. Lab 4 and the classical ROM companion explain two different ways of using physics or reduced structure. Lab 9 is an advanced return visit; its research-scale training is not a prerequisite for the CPU learning route.'),
    ])
    P('Start on a CPU; know what each result means',[
        S('Execution and interpretation'),
        B('Use each notebook from its first setup cell and run cells in order. A Colab link opens a notebook, while its bootstrap obtains the repository files that the lesson needs. Keep your edited notebook separately when updating a checkout. The introductory solver and compact regressors run on CPU. Later notebooks normally load and recompute diagnostics from retained numerical fields and frozen models; their Run All path does not require a high-end GPU or launch a cluster job.'),
        B('A switch such as RUN_NEW_CFD or TRAIN_NEW_PROFILE_MODEL changes the amount and meaning of the work. Leave it off for the first audit, then enable one controlled development experiment. Record total elapsed time and numerical-library versions. The 513-by-513 sparse CFD factorizations and the completed multi-seed PINN campaign are different workloads from a notebook that reads their results.'),
        T([['Term','Meaning in this course'],
           ['CFD','Computational fluid dynamics: numerical solution of the governing flow equations.'],
           ['Current CFD results / coverage','The retained numerical files available now, with method, Reynolds number, grid and provenance. Availability is not an accuracy certificate.'],
           ['MLP / DNN','A multilayer perceptron / deep neural network, built from affine layers and nonlinear activations.'],
           ['POD','Proper orthogonal decomposition: a data-derived spatial basis.'],
           ['RBF','Radial basis function: a distance-kernel model, used in the historical boundary-family pilot described in Lab 4.'],
           ['PINN','Physics-informed neural network: a model trained with governing-equation residuals; its exact loss and data usage must be stated.'],
           ['Retained / fresh','A saved completed experiment / a computation executed with the current declared settings.'],
        ],[1,3.2]),
        B('Do not read an unexplained abbreviation as a new solver. In particular, the Lab 5 coverage table is a file-and-qualification audit. A centreline network is not a full-field network, and a coloured field is not automatically a solved PDE. Read the labels, inputs and target variables before comparing methods.'),
    ])
    P('The physical problem and its variables',[
        B('A square cavity has width L and a lid moving in the positive x direction at speed U. The other walls are stationary. Density is ρ and kinematic viscosity is ν. Divide positions by L, velocities by U, time by L/U and pressure by ρU². Unless a section says otherwise, the plotted coordinates and fields below are nondimensional.'),
        E(r'Re=UL/\nu,\quad x=X/L,\quad y=Y/L,\quad t=t_{\rm dim}U/L,\quad p=P/(\rho U^2)'),
        E(r'u_x+v_y=0,\qquad u_t+uu_x+vu_y=-p_x+Re^{-1}(u_{xx}+u_{yy})'),
        E(r'v_t+uv_x+vv_y=-p_y+Re^{-1}(v_{xx}+v_{yy})'),
        B('Subscripts indicate partial derivatives. The first equation conserves mass. The two momentum equations balance acceleration and convection against pressure gradients and viscous diffusion. Increasing Re at fixed U and L reduces the relative importance of viscous diffusion; thinner wall layers and more demanding corner structures can appear. A steady solution removes the time derivatives, but a numerical steady root does not by itself establish that a perturbed physical flow returns to that root.'),
        D([['<b>Moving lid</b><br/>u = 1, v = 0','<b>Interior flow</b><br/>mass + momentum','<b>Stationary walls</b><br/>u = v = 0']]),
        B('The constant lid has incompatible limits at the two upper corners. The stored solver uses an explicit corner convention. A polynomial or Gaussian lid that vanishes smoothly at the ends is a different boundary condition, not a harmless plotting change. Match this convention before comparing a neural model with Ghia or with a CFD file.'),
        B('Pressure is determined only up to an additive constant in this closed incompressible problem. Fix and report one whole-domain gauge before comparing pressure fields. Removing a different offset from every local region can hide an error and is not the same comparison.'),
    ])
    P('The case is the unit of independence',[
        B('A CFD case contains a Reynolds number, boundary conditions, grid, solver settings and a complete spatial field. Thousands of locations from one case are strongly related. Turning a field into many training rows does not create thousands of independent physical experiments.'),
        D([['<b>Split complete cases</b><br/>training / validation / test','<b>Fit preprocessing</b><br/>training cases only','<b>Freeze the design</b><br/>then score all test cases']]),
        B('Training cases determine weights, scaling constants, means and POD modes. Validation cases compare candidate depth, width, rank, learning rate, loss weights and stopping rules. Test cases assess a frozen choice. Make this division before flattening fields. If Re = 275 appears in both training and test because you split individual points, the reported score is partly a spatial interpolation test inside a seen flow, not generalization to an unseen Reynolds number.'),
        E(r'\widetilde q=(q-\mu_{\rm tr})/s_{\rm tr}'),
        B('Here q is an input or target variable, μ<sub>tr</sub> its training mean and s<sub>tr</sub> its nonzero training scale. Fit these values separately in each selection fold. Likewise, a POD basis must exclude the validation field while choosing its rank. After choices are frozen, a declared final refit may use training plus validation cases; the test fields still stay out.'),
        S('Worked leakage check'),
        B('Suppose you hold out Re = 275 but use all eleven fields to compute the mean and POD modes. The branch has not fitted its coefficients on that case, yet the spatial representation has already seen it. That is leaked representation information. Recompute the mean and basis using allowed cases only. Conversely, using the prescribed lid speed at a held-out Re is not target leakage: it is a known boundary condition.'),
        B('The public course already shows many historical test results. Keep the original split for reproduction, but call them public held-out regression tests. If repeated inspection guides a new design, document that development use and obtain newly withheld cases for an untouched research assessment. A new random seed does not undo exposure to the same physical cases.'),
    ])
    P('What to measure and what to plot',[
        E(r'E_{uv}=\left[\frac{\sum_i w_i[(\widehat u_i-u_i)^2+(\widehat v_i-v_i)^2]}{\sum_i w_i(u_i^2+v_i^2)}\right]^{1/2}'),
        B('The index i runs over the declared comparison region; w<sub>i</sub> are physical quadrature weights, or identical weights for the stated uniform-grid metric. The denominator is a field norm, not the velocity at one point. Thus a zero crossing does not produce an artificial pointwise relative-error singularity. Multiply E by 100 to report percent, and state whether walls and corners were included.'),
        E(r'e_i^u=(\widehat u_i-u_i)/U,\quad e_i^v=(\widehat v_i-v_i)/U,\quad e_i^{\rm vec}=\sqrt{(e_i^u)^2+(e_i^v)^2}'),
        T([['Evidence','Question it answers'],
           ['u and v profiles; signed errors','Where is the flow too fast, too slow, reversed or displaced?'],
           ['Speed, streamlines and signed ψ contours','Is the primary circulation correct? Is a weak corner vortex resolved?'],
           ['Wall velocity and discrete divergence','Do imposed boundary conditions and mass balance hold?'],
           ['ω and momentum defects','Do small velocity differences become large gradient or PDE errors?'],
           ['Mesh / time comparisons','How much does the numerical reference itself still change?'],
           ['Training / validation histories, work and time','Did optimization stop, overfit or become too expensive?'],
        ],[1.7,2.9]),
        B('Keep physical limits, contour levels and sample coordinates equal between methods. Display all compared methods, not only the last network trained. Overlay reference and prediction lines with different styles, place legends outside data, and show signed discrepancies alongside any nonnegative vector-error field. A maximum error near a singular lid corner may dominate the colour range; report its location and a separately labelled interior diagnostic instead of clipping it away.'),
        B('There are three separate discrepancies: surrogate versus its CFD label, CFD versus another grid, and CFD versus a published benchmark. None can replace the others. A small global norm can also miss a weak vortex: report regional absolute error and the strength of the reference circulation before interpreting a regional percentage.'),
    ])
    P('Lab 1 | Aim: turn a CFD run into a qualified label',[
        B('<b>Aim.</b> Reproduce the common Re = 100 case, then generate one assigned flow under the shared protocol. The output is both a numerical field and a quality card that explains whether the field is suitable for learning. This establishes the physical data used by the following labs.'),
        B('<b>What you do.</b> Set STUDENT_ID and STUDENT_NAME; inspect the assigned case from CASE_MAP; run the common reproduction; inspect the diagnostic figure; save the reproduction; then run and save the assigned production case. The notebook writes NPZ arrays and a JSON card. The NPZ contains the spatial values needed by a model, while the card records how those values were produced.'),
        B('The common reproduction uses Re = 100, 65 nodes per direction, a time step of 0.001, a maximum of 12,000 steps and the stated vorticity-change tolerance. A fixed production grid lets cases be stacked without spatial interpolation. A separate refinement study may change the grid, but do not silently mix these fields into the original training matrix.'),
        D([['<b>Solver settings</b><br/>Re, grid, step, stopping rule','<b>Numerical field</b><br/>u, v, ψ, ω','<b>Quality card</b><br/>residuals, profiles, identity']]),
        S('Why this lab comes before machine learning'),
        B('A network approximates the supplied labels. It can reproduce an unconverged transient or a coarse-grid bias extremely accurately. Lowering the training loss cannot establish that the label is an adequate approximation of the intended steady flow. In the lab, “accepted” means the specified finite checks pass; it does not mean that all possible physical errors have disappeared.'),
        B('<b>Deliverable.</b> Submit the common reproduction figure, the assigned-case figure, and both quality cards. Report the final residual, primary-vortex location and strength, and both Ghia profile errors where a benchmark exists. If a gate fails, explain which requirement failed and preserve the failed result; do not relabel it as a qualified steady field.'),
    ],1,'Parts 1-2; common reproduction and assigned production')
    P('Lab 1 | How the streamfunction-vorticity solver works',[
        E(r'u=\psi_y,\quad v=-\psi_x,\quad \omega=v_x-u_y=-\nabla^2\psi'),
        E(r'\omega_t+u\omega_x+v\omega_y=Re^{-1}\nabla^2\omega'),
        B('The streamfunction ψ couples the two velocity components. Given vorticity ω, solve the Poisson equation for ψ with constant wall streamfunction, differentiate it to obtain u and v, apply wall conditions, and advance the transport equation for ω. The spatial discretization, Poisson stopping budget and explicit time step all affect the numerical label.'),
        D([['<b>Current ω</b><br/>interior + wall closure','<b>Poisson solve</b><br/>recover ψ from ω','<b>Velocity</b><br/>differentiate ψ'],['<b>Convection + diffusion</b><br/>transport of ω','<b>Advance one step</b><br/>reapply wall closure','<b>Check convergence</b><br/>continue or stop']]),
        B('The moving lid injects vorticity through the wall treatment. A wall-vorticity formula is part of the discretization, not an optional visual correction. Verify the sign convention: with u=ψ<sub>y</sub> and v=-ψ<sub>x</sub>, vorticity is minus the Laplacian of ψ. Reversing one sign can produce a smooth but physically wrong circulation.'),
        E(r'r_\omega^{(k)}=\frac{\|\omega^{(k)}-\omega^{(k-\Delta k)}\|}{\max(\|\omega^{(k)}\|,\epsilon)}'),
        B('This expression illustrates a relative-change stopping test: k is a step index, Δk is the checking interval and ε prevents a zero denominator. Read the implemented normalization in the quality card. A small change says the iteration is no longer moving much; an independent discrete PDE defect asks whether the field satisfies the intended equation. These tests need not fail or pass together.'),
        B('Mass conservation may be very small because matched discrete mixed derivatives cancel. That is a useful structural property, but it does not guarantee correct nonlinear momentum, a well-resolved boundary layer, or agreement with Ghia. Keep this distinction when you later see a POD model preserve discrete divergence almost to round-off.'),
    ],1,'run_cavity, diagnostic_figure and quality_gate')
    P('Lab 1 | Read the plots; design one useful refinement',[
        B('The two centreline panels compare u along x/L=0.5 and v along y/L=0.5. Reference markers are published numerical samples. The CFD curve comes from the computed field, with interpolation only to compare at the marker positions. The residual panel shows whether the run approaches its stopping rule. The field panel shows the primary circulation and whatever secondary structure this grid resolves.'),
        B('Find the minimum streamfunction and record its grid coordinates as the primary-vortex observable. On a fixed grid, this location is quantized; moving one node does not necessarily indicate a discontinuity in the physical vortex trajectory. A quadratic local fit is a possible additional diagnostic, but it must not be confused with a new independently resolved field.'),
        T([['Controlled change','Hold fixed','Inspect before accepting'],
           ['Increase maximum steps','Re, grid, time step, wall treatment','Residual plateau and changes in both profiles.'],
           ['Halve the time step','Re, grid, physical observation time','Matched-time field/profile changes; do not compare different end times.'],
           ['Refine the grid','Re, boundaries, a qualified stopping rule','Native fields, profiles, vortex positions and computational cost.'],
           ['Increase Poisson work','Other solver settings','Poisson defect and downstream velocity/vorticity changes.'],
        ],[1.2,1.4,2.0]),
        B('<b>Exercise 1A.</b> Identify the last three residual samples in the common run. Does the stopping rule reflect a sustained plateau, a downward trend, or a passed threshold? Quote the rule and observed values. Repeat only if a stated numerical question needs another run.'),
        B('<b>Exercise 1B.</b> Choose one refinement from the table, predict which feature it should affect, then compare signed centreline changes and the primary-vortex location. Report wall time alongside the achieved change. A finer grid that changes the answer is evidence of sensitivity; it is not by itself a demonstration of convergence to an exact solution.'),
        B('<b>Exercise 1C.</b> Inspect the saved NPZ and card. Check array dimensions, increasing coordinates, finite values, Reynolds number and boundary identity. Explain how a swapped x/y order could pass a shape check while corrupting a field-surrogate target. Carry case IDs with the arrays into Lab 2.'),
    ],1,'Checkpoint 1, quality cards and written response')
    P('Lab 2 | Aim: compare models at increasing output complexity',[
        B('<b>Aim.</b> Find out whether a neural network adds value beyond a simple interpolation model, first for three primary-vortex quantities and then for the complete velocity field. You should be able to explain what each model receives, what it predicts, how it is fitted, and where its errors occur. A neural method is not required to win; it is required to be evaluated fairly.'),
        T([['Method','Input → output','What the comparison tests'],
           ['Scalar interpolation','Re → ψ minimum, vortex x and y','Local dependence between neighbouring training cases.'],
           ['Quadratic scalar regression','Scaled Re → the same three quantities','A compact global trend with few fitted coefficients.'],
           ['Scalar MLP','Scaled Re → scaled vortex quantities','Whether learned nonlinear features help sparse case regression.'],
           ['Field interpolation','Re → every stored u and v value','A strong baseline that preserves matched grid structure.'],
           ['Coordinate MLP','Re, x, y → u, v','Whether a common pointwise network represents spatial and parameter dependence.'],
           ['Boundary-aware coordinate MLP','Re, x, y → an interior velocity correction','Whether encoding prescribed walls improves the same field-regression task.'],
        ],[1.2,1.4,1.9]),
        B('The revised experiment reserves complete Re = 225 and 350 fields for validation. The fitting cases are Re = 100, 150, 200, 250, 300, 400; the public held-out regression cases remain Re = 175, 275, 375. Do not count a grid point as an independent new case. Scalers are fitted on the fitting cases; the validation fields guide the declared checkpoint choice, and test predictions are scored after the fitting and checkpoint choices are frozen.'),
        B('The raw coordinate network remains visible as a baseline. The boundary-aware version adds known wall information, not hidden CFD targets. This is a controlled improvement in representation. Neither model is called a new CFD solver, and enforcing walls does not turn either one into a physics-informed solution of Navier-Stokes.'),
        B('<b>Expected outputs.</b> Scalar trends and errors for every scalar method; learning histories; a field comparison, both centreline profiles and wall diagnostics for every public held-out Re. A table without the physical trends is incomplete, as is a final picture containing only the last network trained.'),
    ],2,'Dataset audit through coordinate-field comparison')
    P('Lab 2 | Interpolation and polynomial regression',[
        B('Let q denote one of the three extracted quantities: the minimum streamfunction ψ<sub>min</sub>, its x coordinate, or its y coordinate. Each complete CFD case contributes one value of q. For a query Re between two available training cases Re<sub>a</sub> and Re<sub>b</sub>, linear interpolation is'),
        E(r'\widehat q(Re)=(1-w)q_a+wq_b,\qquad w=\frac{Re-Re_a}{Re_b-Re_a}'),
        B('The same formula can be applied to the complete u and v arrays because the two source fields share coordinates and boundary conditions. It uses both bracketing training cases and no test field. It is inexpensive and transparent: any change lies between the supplied endpoints. Extrapolation outside the training range is a different experiment and must be marked explicitly.'),
        E(r'\widehat q(z)=c_0+c_1z+c_2z^2,\qquad \min_c\sum_{j\in\mathcal T}[\widehat q(z_j)-q_j]^2'),
        B('Here z is a training-scaled Reynolds coordinate, c<sub>0</sub>, c<sub>1</sub>, c<sub>2</sub> are fitted coefficients and T is the set of training cases. The quadratic model learns one smooth trend from all training cases. Scaling z improves the numerical conditioning of the small least-squares problem. Increasing degree may fit every training point but produce oscillations between them; compare validation predictions before choosing complexity.'),
        S('Why these are serious baselines'),
        B('The geometry and lid stay fixed and Re is one scalar parameter. The family can therefore be smooth and low-dimensional even though each velocity field contains thousands of numbers. A neural network with many weights does not automatically possess more useful information. Compare identical training cases and the same error measure before attributing a gain to machine learning.'),
        B('Plot ψ<sub>min</sub>, x<sub>v</sub> and y<sub>v</sub> separately against Re. Mark training, validation and test cases differently. For each target, report signed errors rather than averaging the three unlike quantities into an unexplained scalar. The grid-extracted vortex coordinates are quantized; a stair-step reference curve can reflect the extraction grid rather than a discontinuous physical motion.'),
        B('<b>Implementation check.</b> Recompute a training-point prediction and an in-between prediction by hand for the linear baseline. Then compare the fitted quadratic coefficients with its displayed trend. This small check connects the numerical output to the algorithm and catches a mistaken case ordering before a more complicated network is trained.'),
    ],2,'Primary-vortex extraction; linear and polynomial baselines')
    P('Lab 2 | What a multilayer neural network actually learns',[
        E(r'h^{(0)}=z,\qquad h^{(\ell+1)}=\sigma(W^{(\ell)}h^{(\ell)}+b^{(\ell)})'),
        E(r'\widehat{\mathbf q}=\boldsymbol\mu_q+\boldsymbol s_q\odot(W^{(L)}h^{(L)}+b^{(L)})'),
        B('The input z is scaled Re; h is a hidden feature vector; W is a matrix of weights; b is a vector of biases; σ is a componentwise activation; L counts hidden layers; and the circled-dot operation is componentwise multiplication. The last layer is linear because the three real-valued outputs should not be clipped to an activation range. Training-only mean μ<sub>q</sub> and scale s<sub>q</sub> undo output standardization.'),
        B('The scalar demonstration uses two hidden layers of 32 tanh neurons and three predeclared seeds. Tanh is smooth and has bounded hidden activations. Its derivative becomes small when its argument saturates, so poorly scaled inputs can slow learning. ReLU is piecewise linear and can be useful, but changing activation changes the approximation and optimization behaviour; it is not a guaranteed improvement.'),
        E(r'J(\theta)=\frac{1}{n_{\rm tr}m}\sum_{j\in\mathcal T}\sum_{k=1}^{m}(\widehat{\widetilde q}_{jk}-\widetilde q_{jk})^2'),
        B('θ contains all weights and biases, n<sub>tr</sub> is the number of training cases and m=3 is the number of scalar outputs. The tildes denote standardized targets. Standardization prevents one target scale from dominating the objective. It also changes the relative emphasis of outputs, so always return to dimensional or nondimensional physical units in the final error plots.'),
        D([['<b>One case</b><br/>scaled Re','<b>Hidden features</b><br/>32 tanh → 32 tanh','<b>Three outputs</b><br/>ψ minimum, x, y']]),
        B('Depth adds successive nonlinear transformations; width adds features within a layer. More parameters can reduce training error while worsening generalization because only a few complete cases constrain the fit. Change one factor at a time, count parameters, and compare both validation error and CPU time. Three random starts reveal sensitivity to initialization; selecting the most attractive test seed would hide that sensitivity.'),
        B('Training loss and validation loss answer different questions. The first measures fit to the examples whose gradients update θ. The second measures performance on reserved cases. Their separation can indicate overfitting, but a noisy curve can also come from minibatches, checkpoint frequency or optimizer stage changes. Read how each curve was computed before diagnosing a jump.'),
    ],2,'Compact scalar DNN and three-seed comparison')
    P('Lab 2 | Optimizer steps, loss curves and fair stopping',[
        B('Backpropagation computes the objective gradient g<sub>t</sub> with respect to the network parameters θ at optimizer step t. Adam uses moving averages of the gradient and its square. With zero initial moment arrays, learning rate α, constants β<sub>1</sub>, β<sub>2</sub> in (0,1), and a small positive ε, its update is'),
        E(r'm_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2'),
        E(r'\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad \widehat v_t=\frac{v_t}{1-\beta_2^t},\quad \theta_{t+1}=\theta_t-\alpha\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}'),
        B('Squares, square roots and division are componentwise. The first moment smooths the direction; the second scales coordinates with different gradient magnitudes. Adam can make useful early progress, but its name is not evidence of superior final physical accuracy. A large learning rate may cause oscillation; a small one may waste the finite budget.'),
        E(r'd_k=-H_k g_k,\qquad \theta_{k+1}=\theta_k+\alpha_k d_k'),
        B('L-BFGS approximates an inverse curvature matrix H<sub>k</sub> using a limited history of parameter differences and gradient differences. It does not store a full dense Hessian. A line search chooses α<sub>k</sub> by evaluating candidate steps. It needs a fixed deterministic objective during the line search: changing samples or weights inside those evaluations invalidates the comparison.'),
        B('The hybrid schedule uses Adam to obtain a useful initial fit, then L-BFGS to refine the same full-batch objective. Count function/gradient evaluations and elapsed time, not just nominal iterations: one L-BFGS iteration may contain several closures. Check the saved validation-loss history and restore the declared validation-selected model rather than returning whichever weights happened to be evaluated last.'),
        S('Explain a loss jump before changing the network'),
        B('A jump at the optimizer transition may reflect a new scale, a different reduction of the loss, a line-search trial or a real deterioration. Plot one consistently defined training MSE and the same validation MSE after accepted updates. Keep wall/field errors separate. Do not smooth a history so aggressively that rejected steps or a failing stage disappear. If validation rises while training falls, the evidence supports less capacity, stronger regularization or earlier stopping; it does not justify tuning on the test fields.'),
    ],2,'Scalar/field loss histories and validation checkpoint rule')
    P('Lab 2 | Coordinate fields and prescribed-wall lifting',[
        E(r'(Re,x,y)\ \longmapsto\ (\widehat u,\widehat v)'),
        B('The coordinate network receives three values for each spatial query and predicts two velocity components. Both field models use three hidden layers of 48 tanh neurons, seed 690, 400 full-batch Adam updates and a maximum of 2400 accepted L-BFGS iterations. The saved metadata reports actual iterations and function evaluations. Training rows sample only the allowed Reynolds cases. Unlike a scalar vortex model, it must represent the entire spatial field, including rapidly changing wall layers. An unconstrained MLP can produce visually plausible circulation while leaking velocity through a wall.'),
        B('The boundary-aware version predicts an interior correction around a known lift. Let s=x(1-x). For interior coordinates, define'),
        E(r'g(x,y)=\frac{ys^2}{s^2+(1-y)^2},\qquad b(x,y)=\frac{4ys(1-y)}{s^2+(1-y)^2}'),
        E(r'\widehat u=g+bN_u(Re,x,y),\qquad \widehat v=bN_v(Re,x,y)'),
        B('N<sub>u</sub> and N<sub>v</sub> are the raw network outputs. On the top wall away from the corner nodes, g=1 and b=0; on the stationary walls away from the corners, g=0 and b=0. Thus known velocity boundaries hold independently of trained weights. At the two upper corner nodes, use the solver’s explicit zero-corner convention instead of evaluating 0/0. The singular limiting behaviour is part of the problem, not an instruction to smooth the lid.'),
        D([['<b>Coordinate MLP</b><br/>learns Nᵤ and Nᵥ','<b>Known boundary lift</b><br/>g + bNᵤ, bNᵥ','<b>Velocity prediction</b><br/>inspect interior + walls']]),
        B('This transform changes the function class and embeds known boundary information. It does not enforce u<sub>x</sub>+v<sub>y</sub>=0 or the momentum equations. Its derivatives near the lid corners can remain challenging. Evaluate the same CFD targets, the same case split and an independent physical diagnostic for both raw and lifted networks; do not claim that exact walls alone establish an accurate flow.'),
        B('A failed raw baseline is useful evidence. Retain it and show where the constrained representation helps or hurts. If a new choice is motivated by published test pictures, describe the repair as development on disclosed cases; its validation-only fitting rule does not retroactively make those pictures unseen.'),
    ],2,'Raw and boundary-aware coordinate DNN')
    P('Lab 2 | How to read a complete field comparison',[
        B('For each public held-out Re, compare CFD, field interpolation, the raw coordinate network and the boundary-aware network. Each method should produce a field and profile comparison. Model quality cannot be assessed from a single attractive streamline panel. A speed field can look similar even when u and v have the wrong signs, so keep vector and component errors.'),
        T([['Panel or curve','What to inspect','What a defect suggests'],
           ['Matched speed and streamline views','Main vortex position, secondary circulation and wall behaviour','Spatial underfitting, wrong coordinates or a representation problem.'],
           ['u at x/L=0.5 and v at y/L=0.5','Zero crossings, extrema and signed departures from CFD','A global norm is hiding local bias or a displaced flow structure.'],
           ['Linear vector-error field','Peak error and its location; interior and wall regions','Small total error does not make a local defect unimportant.'],
           ['Wall-normal and tangential velocity','Each open wall and the corner convention','The model may violate prescribed data even if streamlines look plausible.'],
           ['Training/validation curves','Stage changes, convergence and chosen checkpoint','A finite training budget may be insufficient or overfit.'],
        ],[1.2,1.65,1.8]),
        B('Re = 100-400 in this introductory experiment does not promise the high-Re upper-left vortex seen in a different CFD family. First ask whether that structure exists in the matching reference field. Automatic streamplot seeding may also omit a weak recirculation. Inspect signed streamfunction contours or a deliberately seeded close-up before calling a vortex absent.'),
        B('Near a zero reference velocity, a pointwise percentage can become arbitrarily large. Use the field-normalized vector error defined earlier, and add absolute regional RMS values in units of U. For a weak vortex, report its signed streamfunction strength and location; do not replace a poor percentage by a different denominator without explaining the change.'),
        B('<b>Required interpretation.</b> For every tested method, name one location where it agrees and one where it differs. State which diagnostic supports the claim. If interpolation is better, report that outcome. The purpose of the lab is to identify which additional structure improves this learning problem, not to guarantee a neural winner.'),
    ],2,'All held-out field, profile and wall-diagnostic figures')
    P('Lab 2 | A controlled student investigation',[
        B('<b>Exercise 2A: establish a complete baseline.</b> Run all cells once, retain the settings and runtime, and produce a table for all three public test Re values. Include every scalar method, field interpolation, raw coordinate MLP and boundary-aware MLP. Report scalar-target errors separately and velocity vector error, wall error and a signed-profile maximum for each field model.'),
        B('<b>Exercise 2B: change capacity.</b> Compare two chosen depths at the same activation, data split and declared optimization budget. Either approximately match parameter counts by changing width or state explicitly that capacity changes too. Plot validation loss versus evaluations and elapsed time. Select using validation only, then freeze your report configuration.'),
        B('<b>Exercise 2C: change activation.</b> Compare tanh with another supported activation at fixed widths. Record whether training stalls, oscillates or fits more rapidly. Explain the role of smoothness and scaling. A lower training loss without better validation is not a successful generalization improvement.'),
        B('<b>Exercise 2D: separate optimization and information.</b> Compare Adam alone with Adam-to-L-BFGS while keeping training cases and representation fixed. Separately investigate adding a complete development case. The first changes optimization; the second changes physical information. Do not attribute both gains to the optimizer.'),
        B('<b>Exercise 2E: boundaries versus interior.</b> Compare raw and lifted networks on identical queries. Use wall traces to verify what the lift guarantees, then use centreline profiles, vector-error fields and discrete divergence to test what it does not. Inspect the largest error near the lid and report the corner convention.'),
        B('<b>Exercise 2F: feature-based acceptance.</b> Write a requirement before calculating results: for example, an allowed centreline velocity discrepancy or a primary-vortex position tolerance in units of grid spacing. Evaluate it independently of the training objective. Explain why a model can satisfy a global L2 requirement and fail a local feature requirement.'),
        S('What to submit'),
        B('Submit the complete model table, scalar-trend figure, training/validation history, one matched field comparison for each held-out Re, and the signed profiles and wall check. Include the actual seed list and model-selection rule. Finish with an evidence-based recommendation for this dataset: retain a baseline, use a constrained coordinate network, or move to the low-dimensional field representation introduced in Lab 3. Do not replace original exercises with only a screenshot of the last result.'),
    ],2,'Required analysis; controlled architecture/optimizer studies')
    P('Lab 2 | Scalar outputs: every baseline stays visible',[
        F('results/week04_lecture/lab2_scalar_trends.png','Executed scalar predictions. Markers distinguish fitting, validation and public held-out cases. The shaded interval is the minimum-to-maximum range of three neural initializations, not a calibrated confidence interval.',365),
        B('Read the three observables separately. Streamfunction strength and the two vortex coordinates have different physical roles and scales. A network can track the smooth strength trend while missing a one-node location change. That discrepancy should remain visible rather than disappear in a combined normalized score.'),
        B('The neural curve uses the declared three-seed aggregation. Interpolation and the quadratic baseline receive the same fitting cases. The final notebook retains their individual target errors at all three public held-out Re values. Use this figure with the training/validation histories to determine whether a missed trend comes from capacity, optimization or limited case information.'),
        B('When changing the network, regenerate the scalar curves from your current model and preserve the baseline curves. Do not compare a fresh network table with a historical image whose predictions belong to different weights.'),
    ],2,'Scalar trends, public test errors and seed spread')
    P('Lab 2 | Physical field outputs at Re = 275',[
        F('results/week04_lecture/lab2_re275_fields.png','Native 65-by-65 fields at public held-out Re = 275. The upper-left panel fills the CFD speed and overlays model contours at identical speed levels. The other panels show every model’s vector error on one full-range linear scale.',430),
        B('This comparison separates field similarity from local error. A bright corner maximum is retained; it is not clipped to improve the appearance of the bulk. Interpolation is a genuine baseline, and exact wall values do not certify that the interior velocity field or its derivatives are accurate.'),
        B('At Re = 275, velocity-vector relative L2 errors are 0.212% for interpolation, 1.986% for the raw DNN and 1.910% for the boundary-aware DNN. Interpolation wins at all three test Re values. The boundary-aware model also loses to the raw DNN at Re = 375 and has larger peak errors and divergence. Inspect the complete three-case figure suites before recommending a method.'),
    ],2,'Matched fields and full-extrema vector-error maps')
    P('Lab 2 | Signed profiles explain the field error',[
        F('results/week04_lecture/lab2_re275_profiles.png','The same Re = 275 fields on the two cavity centrelines. Lower panels retain signed neural/interpolation departures from CFD at the native query coordinates; no Ghia measurements exist at this particular Re.',395),
        B('Find the largest signed discrepancy and its position for each method, then relate it to the two-dimensional comparison on the preceding page. A small speed difference can hide opposing component errors, while similar centreline curves can still miss a corner feature outside these two lines.'),
        B('Use the separate wall-diagnostic plots to assess boundary enforcement. Use validation curves to judge the selected checkpoint. The earlier Lab 3 retained model uses eight development cases after refitting, while this revised Lab 2 fit uses six cases and reserves two for validation; their headline errors are not a matched-information architecture comparison.'),
    ],2,'Physical profiles, signed discrepancies and wall checks')
    P('Lab 3 | Aim: factor a field into spatial modes and coefficients',[
        B('<b>Aim.</b> Build a compact parameter-to-field predictor whose spatial structure is explicit. Instead of asking a coordinate MLP to discover everything at once, derive reusable patterns from CFD snapshots, then learn how their amplitudes change with Reynolds number. Predict velocity and zero-mean pressure with separate representations.'),
        E(r'\widehat{\mathbf u}(Re)=\overline{\mathbf u}+\sum_{k=1}^{r_u}b_k^u(Re)\,\mathbf t_k^u'),
        E(r'\widehat p(Re)=\overline p+\sum_{k=1}^{r_p}b_k^p(\log Re)\,t_k^p'),
        B('The overbar is a development-derived mean, t<sub>k</sub> is a fixed spatial POD mode, b<sub>k</sub> is its predicted coefficient, and r is the number of retained modes. The velocity state stacks both components. Pressure has its own mean, basis and coefficient scales, so pressure units cannot distort the velocity modes.'),
        D([['<b>Allowed CFD snapshots</b><br/>means and two SVDs','<b>Fixed POD trunks</b><br/>velocity / pressure'],['<b>Reynolds input</b><br/>Re or log Re','<b>Small branch MLPs</b><br/>modal coefficients','<b>Reconstructed fields</b><br/>u, v and zero-mean p']]),
        B('In the original DeepONet formulation, a branch encodes sampled input-function values and a trunk represents query locations. Here the input is only Re and the spatial trunk is a fixed data-derived basis. “Restricted scalar-parameter POD-DeepONet” states that scope accurately. Success on this fixed geometry does not establish an operator for arbitrary lid profiles or new shapes.'),
        B('<b>Notebook sequence.</b> Audit the branch-trunk-target pairing, qualify the CFD, construct selection-only POD trunks, compare branch/rank candidates, perform the declared refit and finally assess every public held-out field. The first metadata check deliberately permutes coordinates to verify that a wrong pairing is caught. Matching array shapes alone cannot establish matching physics.'),
    ],3,'Concept map; data contract; construction of two POD trunks')
    P('Lab 3 | POD: representation error before prediction error',[
        B('Let each column of Q contain one flattened training field. Subtract the training mean to form S. A singular-value decomposition identifies orthogonal spatial patterns ranked by the variance they represent:'),
        E(r'S=U\Sigma V^T,\qquad \Phi=U_{[:,1:r]},\qquad \mathbf a_j=\Phi^T(\mathbf q_j-\overline{\mathbf q})'),
        B('U contains orthonormal spatial vectors, Σ contains nonnegative singular values σ<sub>k</sub>, and V contains snapshot coefficients. Φ retains r modes. The stated equations use the uniform-grid Euclidean inner product; a nonuniform grid needs an explicitly weighted POD. Sign changes of a mode and its coefficient together leave the reconstructed field unchanged.'),
        E(r'\eta_r=\frac{\sum_{k=1}^{r}\sigma_k^2}{\sum_k\sigma_k^2},\qquad \mathbf q_{\rm proj}=\overline{\mathbf q}+\Phi\Phi^T(\mathbf q-\overline{\mathbf q})'),
        B('η<sub>r</sub> is the retained snapshot variance. It is not a guarantee of a correct local vortex, a new-Re prediction or a time-evolved ROM. The projection q<sub>proj</sub> uses the true field and gives the best approximation within this fixed orthonormal span. It is an oracle diagnostic, not a predictor that can be queried without the answer.'),
        E(r'\|\widehat{\mathbf q}-\mathbf q\|_2^2=\|\mathbf q_{\rm proj}-\mathbf q\|_2^2+\|\widehat{\mathbf a}-\mathbf a\|_2^2'),
        B('For the same mean, orthonormal basis and complete Euclidean field norm, the spatial residual is orthogonal to coefficient error. This identity separates an insufficient basis from a weak branch regressor. It does not permit subtracting unrelated relative norms or errors measured in different regions.'),
        S('A two-component worked example'),
        B('Take a rank-one basis φ=(1,0)ᵀ and a mean-subtracted target q=(2,0.5)ᵀ. Its exact coefficient is 2, but its projection is (2,0)ᵀ. Even a perfect optimizer leaves an absolute error of 0.5. If the branch predicts coefficient 1.8, the total error is √(0.2²+0.5²)=√0.29. Improving only the branch cannot recover the missing second component.'),
        B('Before increasing network depth, inspect a validation projection and the first few modes. A low-energy mode may carry the small corner circulation that a global variance criterion barely notices. Choose rank using whole-case validation and physical diagnostics, not a visually attractive energy percentage selected after viewing test data.'),
    ],3,'Development-only SVD; energy and reconstruction diagnostics')
    P('Lab 3 | Selection, refitting and pressure meaning',[
        B('The development Re values are 100, 150, 200, 225, 250, 300, 350, 400. The complete Re = 225 field is reserved during architecture selection; Re = 175, 275, 375 are historical held-out cases now exposed in the public course. During selection, every mean, scale and basis is built without Re = 225 or the public tests.'),
        B('The velocity candidates compare ranks 3 and 4 with two-layer branch widths 16, 32 and 64. Pressure compares ranks 2, 3, 4, several smaller branches and declared Re/log-Re transforms. Seeds 690, 691, 692 are specified before assessment. L-BFGS fits a deterministic coefficient objective; complete-case validation chooses the representation and branch.'),
        T([['Retained author selection','Velocity head','Pressure head'],
           ['POD rank','3','3'],
           ['Branch hidden layers','32, 32 tanh units','8 tanh units'],
           ['Input transform','Re','log Re'],
           ['Mean validation field error','0.240%','0.185%'],
        ],[1.8,1.3,1.3]),
        B('After this choice, the declared refit rebuilds each trunk using all eight allowed development cases and trains all three seeds. No public test snapshot enters the refit. The mean of the three predictions is reported, together with seed variability; choosing the best test seed would change the assessment rule.'),
        B('The pressure head directly predicts stored zero-mean pressure labels. Those labels were recovered from velocity using a momentum-gradient procedure. Small pressure prediction error therefore means accurate reproduction of that procedure’s numerical labels. Ghia’s u and v samples do not independently validate pressure, and the existence of a separate pressure head does not remove label uncertainty.'),
        B('Velocity boundary values are restored using prescribed walls. A fixed linear combination of compatible snapshots can retain their discrete mass balance; this is a structural benefit, not proof of nonlinear momentum balance at a new Re. Re-evaluate wall, divergence and pressure-gauge diagnostics after reconstruction.'),
        B('<b>Experiment.</b> Include rank 2 as a new declared validation candidate. Compare its projection error with branch-prediction error and identify whether representation or optimization limits the result. Keep the test cases out of that choice.'),
    ],3,'Selection table; final refit; separate velocity and pressure heads')
    P('Lab 3 | What the reported accuracy and speed establish',[
        T([['Public held-out Re','Velocity Euv','Zero-mean pressure Ep'],
           ['175','0.4455%','0.2506%'],['275','0.0727%','0.1245%'],['375','0.0947%','0.1073%'],
        ],[1.5,1.3,1.7]),
        B('These are the retained author ensemble results, not values generated by opening this PDF. They measure agreement with the educational 65-by-65 CFD fields. The notebook distinguishes a newly executed fit from the historical evidence. Compare current in-memory predictions when running a fresh experiment; do not relabel a stored figure as its output.'),
        B('At Re = 100 and 400, the neural centreline errors remain close to the errors of their source CFD. That is appropriate for a surrogate: the model largely preserves the fidelity of its labels. The nonzero Ghia discrepancy, particularly the original Re = 400 v score, cannot be removed simply by quoting a much smaller neural-to-CFD error.'),
        E(r'T_{\rm surrogate}(M)=T_{\rm labels}+T_{\rm train}+M\,t_{\rm query}'),
        E(r'M_{\rm break-even}>\frac{T_{\rm labels}+T_{\rm train}}{t_{\rm CFD}-t_{\rm query}}\quad(t_{\rm CFD}>t_{\rm query})'),
        B('M is the number of future field queries. T<sub>labels</sub> and T<sub>train</sub> are offline costs; t<sub>query</sub> includes full field reconstruction. The retained CPU record reports about 1 ms for the three-field ensemble query and about 5.3 s for a qualified CFD solve including pressure recovery. This is a measured amortized query advantage on that setup, not a universal machine-independent speedup or a claim that the first query is cheaper after including all offline work.'),
        S('Required lab evidence'),
        B('Supply a complete selection table, all three held-out results, seed spread, field/error comparisons, u/v centreline curves, wall/divergence/gauge checks and a matched timing ledger. Plot the first three modes and explain which structure each carries. Compare a one-seed result with the ensemble without selecting a seed from the test errors.'),
        B('For a new operator-learning extension, vary an actual lid-velocity function and feed its values at fixed sensors to the branch. Preserve the sensor order and case identity. This is a different dataset and information budget; the current scalar-Re results cannot serve as its completed validation.'),
    ],3,'Public held-out results, physical checks and timing')
    P('Lab 4 | Aim: learn the nonlinear correction to Stokes flow',[
        B('<b>Aim.</b> Use a cheap physical solution as an informative input, then learn the correction needed for the more complete Navier-Stokes solution. This is not transfer of network weights from a Stokes model. The baseline is an actual computed Stokes flow with the same geometry, lid and grid as its nonlinear partner.'),
        E(r'\psi_{NS}=\psi_S+\Delta\psi_\theta(\psi_S,Re),\qquad u=\psi_y,\quad v=-\psi_x'),
        B('ψ<sub>S</sub> is the Stokes streamfunction, ψ<sub>NS</sub> is the nonlinear target, and Δψ<sub>θ</sub> is the predicted correction. Stokes flow removes convection from momentum. The learned term recovers what that simplification omits within the training family. Differentiating a streamfunction correction couples u and v instead of predicting unrelated components.'),
        D([['<b>Matched Stokes solve</b><br/>boundary shape + Re','<b>Low-dimensional features</b><br/>POD coefficients','<b>Learn correction</b><br/>Δψ then derivatives']]),
        B('The experiment uses L=1 and ν=0.0025, with reference speed U<sub>ref</sub>=Re ν/L. Re and dimensional lid speed therefore cannot be changed independently. A constant-lid family and a spatially diverse family test whether shape information is needed. The diverse lid is a positive, maximum-normalized Fourier profile with three sine modes in the in-distribution family.'),
        E(r'U_{\rm lid}(x)=U_{\rm ref}\frac{1+\sum_{k=1}^{3}a_k\sin(k\pi x/L)}{\max_\xi|1+\sum_{k=1}^{3}a_k\sin(k\pi \xi/L)|}'),
        B('The coefficients a<sub>k</sub> set lid shape, and ξ ranges over positions along the lid. A shape-out-of-distribution group activates five modes with larger total amplitude; a Reynolds-out-of-distribution group uses 450&lt;Re&lt;600 instead of the 100&lt;Re&lt;400 training interval. These two extrapolations ask different questions and should be reported separately.'),
        B('Read Sections 1-3 in the notebook before interpreting its four train/test boundary-family combinations. A constant-only training set does not teach dependence on lid shape. Better optimization cannot create the missing family information.'),
    ],4,'Reynolds/boundary families and physical correction target')
    P('Lab 4 | A derivative-aware correction model',[
        B('The numerical labels use a 25-by-25 streamfunction-vorticity discretization. A midpoint RK2 method advances vorticity and a discrete sine transform solves the Poisson problem. The steady PDE residual must pass twice before a label is accepted. The Stokes solver uses matching walls and grid. A later companion repeats the study on 51-by-51 nodes.'),
        E(r'\mathbf z=[Re/400,\ \log(Re/400),\ \widehat{\mathbf a}_S]'),
        B('The vector â<sub>S</sub> contains scaled POD coefficients of the Stokes input. A second training-only POD basis represents Δψ. The MLP predicts scaled correction coefficients, which are reconstructed into a streamfunction correction. This converts a field-to-field task within the sampled family into a tractable coefficient regression.'),
        E(r'J=\|e_\psi\|_2^2+\lambda_u\|\nabla e_\psi\|_2^2+\lambda_\omega\|\nabla^2e_\psi\|_2^2'),
        B('Here e<sub>ψ</sub> is the predicted-minus-reference correction; λ<sub>u</sub> and λ<sub>ω</sub> are declared nonnegative weights. The gradient term emphasizes velocity, while the Laplacian term emphasizes vorticity. A field can have small streamfunction error but larger derivative error, so these terms connect the objective to the physical outputs students inspect.'),
        B('The candidate pairs of correction rank and hidden width are (12,64), (24,64) and (24,96). Validation velocity error selects rank 24 and two 96-neuron tanh layers, with a linear coefficient output. Adam runs for 1800 epochs with cosine learning-rate decay; L-BFGS begins from the validation-selected Adam checkpoint. Seeds 7, 17, 27 are averaged after selection.'),
        B('Derivative weighting is not free accuracy. A larger λ<sub>ω</sub> can improve vorticity at the cost of velocity or parameter transfer. Fit every preprocessing step on the allowed training family; choose weights on validation; report each physical metric in its original scale. The final weighted training objective cannot substitute for an independent velocity or vorticity error.'),
        B('<b>Exercise.</b> Change only one derivative weight in a separately saved development run. Before fitting, predict which plotted feature should improve and which may deteriorate. Compare the same velocity profiles, vorticity distributions and validation cases, and retain both outcomes even when the result is worse.'),
    ],4,'POD features, architecture and Sobolev objective')
    P('Lab 4 | Family transfer, pressure and grid sensitivity',[
        B('Compare constant-to-constant, constant-to-diverse, diverse-to-constant and diverse-to-diverse predictions. The first term identifies training boundary information; the second identifies the tested family. A model trained only on constant lids can fail badly on diverse lids even when it fits its own family. The diverse-lid model has seen the relevant variation and can learn that dependence.'),
        B('The retained best mean velocity discrepancies are 0.107% for constant-to-constant and 0.917% for diverse-to-diverse. These are comparisons with same-grid numerical labels. The expanded training stage retains the original inspected test cases, so its improvement is a public regression comparison, not a newly blind benchmark.'),
        S('Pressure is a derived quantity in this lab'),
        B('Pressure is recovered from each velocity field using the same least-squares momentum-gradient procedure and a zero-global-mean gauge. It is not a network output here. A gradient-compatibility residual measures how well an integrated pressure matches supplied gradients; it is not a direct pressure-error norm. Pressure agreement and gradient compatibility can rank candidates differently.'),
        S('The 51-by-51 companion is a separate numerical check'),
        B('The companion recomputes 184 matched Stokes/Navier-Stokes cases on the finer grid and repeats the selected correction model. Compare two quantities separately: model-to-label error on each fixed grid, and the change in the CFD label between grids. The latter remains substantial; neither grid establishes mesh-independent physical truth.'),
        T([['Student comparison','Evidence to provide'],
           ['Four boundary-family combinations','A table of velocity errors and matched field/profile examples, including failure.'],
           ['Shape OOD versus Re OOD','Separate results with the actual ranges and lid definitions.'],
           ['Pressure and vorticity','One shared-gauge pressure comparison; same-operator signed vorticity.'],
           ['25 versus 51 nodes','Same-grid surrogate error and inter-grid CFD change in distinct columns.'],
        ],[1.45,3]),
        B('For vortex capture, inspect native streamfunction extrema and the lower-right recirculation candidate. The current gallery displays predictions on the native 25-by-25 grid; plotting them at a larger size does not add resolved degrees of freedom. Distinguish a grid-resolved feature from an independently validated vortex. The dedicated Week 4.2 lecture and grid 51 addendum remain available for the complete correction study.'),
    ],4,'Velocity results, pressure audit and grid 51 companion')
    P('Lab 4 | Why the seven historical pilot methods differ',[
        B('The notebook also opens an earlier boundary-family pilot containing five neural methods and two non-neural baselines. These are separate retained fits, not the later selected Stokes-correction model. The pilot uses a 25-by-25 grid, 16 training, 4 validation and 6 test cases per in-distribution family, 120 Adam epochs and one seed 7. Its purpose is to reveal architectural behaviour under a small budget; it cannot establish a universal model ranking.'),
        T([['Method','Representation and computation'],
           ['Coordinate MLP','Concatenates Re/lid descriptors with (x, y); two 48-unit tanh layers predict streamfunction at each query.'],
           ['POD-DeepONet','Fits a 48, 48 tanh branch to at most 12 training-derived streamfunction modes. The spatial trunk is fixed.'],
           ['Small U-Net','Local 3-by-3 convolutions form features, average pooling coarsens them, and an upsampled branch joins high-resolution features through a skip connection.'],
           ['FNO','Three Fourier layers combine learned low-frequency spectral mixing with a local 1-by-1 path; a small decoder predicts streamfunction.'],
           ['FNO plus physics','The same FNO adds a discrete vorticity-equation penalty and wall-derivative consistency to the supervised objective.'],
           ['Nearest descriptor','Returns the training field closest in the supplied Re/lid descriptor space.'],
           ['RBF','Interpolates flattened streamfunction fields from active Re/lid descriptors using radial basis functions.'],
        ],[1.05,3.7]),
        B('The grid-input networks receive four channels: the lid profile broadcast through the domain, scaled Re, x and y. The descriptor models receive Re/400 and five lid coefficients. The pilot Fourier block retains four modes per direction and uses reflected padding before Fourier operations. The numerical cavity is not a periodic-flow problem merely because the architecture uses Fourier transforms.'),
        B('All neural methods fit streamfunction plus a derivative-aware data loss; the physics variant adds its stated residual and wall terms. Prescribing ψ=0 on walls enforces no penetration but does not by itself impose every tangential derivative. The pilot retains a lid-derivative consistency diagnostic even when the velocity reconstruction explicitly restores wall velocities.'),
        B('In “Historical seven-method pilot: inspect every algorithm”, compare all seven fields, signed profiles and errors at the same case ID. Read its actual Re and lid shape. A short training budget, different parameter counts and a coarse label grid limit the conclusion; a more sophisticated architecture may simply be undertrained.'),
    ],4,'Historical seven-method pilot: inspect every algorithm')
    P('Lab 4 | RBF: a kernel baseline, not another CFD solver',[
        B('<b>RBF means radial basis function.</b> Let z be the active case descriptor and z<sub>j</sub> a training descriptor. The prediction is a weighted sum of distance-based kernels plus a low-order polynomial. One common kernel system is solved for all flattened target components; each target column has its own coefficient vector.'),
        E(r'\widehat{\boldsymbol\psi}(\mathbf z)=\sum_j\mathbf a_j\,\phi(\|\mathbf z-\mathbf z_j\|)+P(\mathbf z)\mathbf b'),
        E(r'(K+\lambda I)A+PB=Y,\qquad P^TA=0'),
        B('K contains kernel evaluations between training descriptors, A contains radial coefficients, P contains polynomial features, B contains their coefficients and Y contains training streamfunction snapshots. λ is a nonnegative smoothing parameter. These are linear equations for the coefficients, not gradient descent through a deep network. Exact interpolation at λ=0 is possible only when the system is well posed and the required polynomial constraints are satisfied.'),
        B('The retained pilot uses SciPy’s thin-plate-spline kernel φ(r)=r²log r with its degree-one polynomial default and smoothing λ=10⁻⁷. The value at r=0 is defined by its zero limit. Only descriptor dimensions that vary in the training family enter this RBF fit. In constant-lid training, lid coefficients do not vary and the kernel cannot learn their influence on a diverse-lid query.'),
        B('A kernel method can be a strong baseline when a small parameter family varies smoothly. For example, the retained constant-to-constant pilot mean velocity error is about 0.120% for RBF, versus about 2.766% for its short-budget POD-DeepONet and 45.75% for its coordinate MLP. These are pilot means against coarse numerical labels, not the improved main Lab 2 model or the high-Re Lab 8 result.'),
        B('RBF smoothing is regularization in the interpolation system. It is not graphical smoothing of a displayed field. Descriptor scaling changes distance and therefore changes the model. If selecting a kernel, smoothing value or shape parameter, use complete validation cases and retain the same training information as the neural candidates.'),
        B('<b>Exercise.</b> Contrast constant-to-diverse with diverse-to-diverse predictions for RBF and one network. Explain which inputs contain lid-shape information in each fit. Use signed velocity profiles and the actual streamfunction field, not only the mean table. A smooth-looking interpolant can still violate nonlinear momentum; neither RBF nor a neural regressor becomes a CFD solution merely by reproducing stored data.'),
    ],4,'Historical pilot RBF definition and information budget')
    P('Lab 5 | Aim: distinguish our solutions from benchmark samples',[
        B('<b>Aim.</b> Establish what CFD and neural evidence actually exists for all seven Ghia Reynolds numbers, and compare the numerical velocity profiles with the published samples. A list of reference points is not a CFD result, and a file-coverage table is not a validation certificate.'),
        E(r'Re\in\{100,400,1000,3200, 5000, 7500, 10000\}'),
        B('Ghia, Ghia and Shin report 17 u values along the vertical centreline and 17 v values along the horizontal centreline for each Re. Those two profiles are valuable independent benchmarks, but they do not determine a complete two-dimensional field, a corner-vortex map or pressure. The notebook starts with actual retained FlowMLLab fields and overlays their computed profiles on the benchmark markers.'),
        T([['Notebook table','What it contains','What it does not prove'],
           ['A: measured CFD errors','Re = 100 SIMPLE grids and Re = 400/1000 streamfunction-vorticity grids; recomputed Ghia errors and file hashes.','An algorithm ranking across different Re or a monotone mesh-convergence result.'],
           ['B: neural profile comparison','Frozen Adam, selected profile MLP and interpolation at held-out Re = 3200/7500.','A full-field CFD solution or a learned corner vortex.'],
           ['C: current coverage','Local presence of the retained CFD and reference records for all seven Re.','That every present field is mesh-independent or physically stable.'],
        ],[1,2.1,1.7]),
        B('Table A uses existing MAC/SIMPLE fields at Re = 100 and an independent streamfunction-vorticity formulation at Re = 400 and 1000. Interpolate each native field according to its own variable locations. A MAC face velocity is not a cell-centred sample. Likewise, native momentum and vorticity defects correspond to different equations; their numerical magnitudes cannot rank methods directly.'),
        B('Retained 512-cell fields now exist at all seven Re. Lab 7 supplies their finer-grid and time checks. “Current CFD results” refers to these actual numerical records and their qualification, not a new neural model or an unexplained abbreviation. Verify the file identity before comparing old and new pictures.'),
    ],5,'Tables A, B and C; current coverage and numerical qualification')
    P('Lab 5 | Benchmark comparison without hiding printed anomalies',[
        E(r'e_{{\rm RMS},u}/U=\sqrt{N^{-1}\sum_{i=1}^{N}[(\widehat u_i-u_i)/U]^2}'),
        E(r'E_u=\frac{\sqrt{\sum_i(\widehat u_i-u_i)^2}}{\sqrt{\sum_i u_i^2}}'),
        B('N is the number of retained reference samples. The first measure gives an absolute velocity scale relative to U; the second normalizes by the benchmark profile norm. Use the analogous formulas for v. Interpolate the actual CFD field to each published coordinate, then preserve signed differences at those discrete locations.'),
        B('Three suspicious printed entries remain in the canonical raw reference: Re = 400 v at x/L=0.9063 is -0.23827; Re = 3200 u at y/L=0.4531 is -0.86636; Re = 10000 u at y/L=0.5 is +0.03111. A predeclared sensitivity mask reports a screened score alongside the all-point score. It is not a claim that the course has established corrected values.'),
        B('Keep these markers visible in the figures. Do not select a mask after discovering which points hurt your model most. For Re = 400, the isolated v discrepancy can dominate the all-point norm; for Re = 3200, the printed u value can dominate a profile score. A changed score after screening reflects a changed comparison set, not automatically a better CFD solution.'),
        S('Read both curves before quoting one number'),
        B('On the retained Re = 100 SIMPLE grids, refinement does not monotonically reduce the Ghia error. Report that outcome and inspect the signed spatial discrepancy. A sparse historical benchmark has its own discretization and precision; the grid with the smallest benchmark norm is not automatically the most accurate approximation of the continuum flow.'),
        B('<b>Exercise 5A.</b> Rebuild the Table A rows for every stored grid. Give u/v RMSE, all-point and screened relative errors, sample counts, method and field hash. Locate the largest signed discrepancy and discuss whether it is isolated or part of a spatial trend.'),
        B('<b>Exercise 5B.</b> Rebuild Table C and select one high-Re record. Follow its Lab 7 evidence and write an admission decision separating file presence, equation convergence, mesh sensitivity, short-time behaviour and Ghia agreement. If a file is missing locally, report missing local evidence rather than asserting the scientific run never occurred.'),
    ],5,'CFD figures, raw/screened score policy and tasks 1-2/5')
    P('Lab 5 | A corner vortex needs full-field evidence',[
        B('Lab 5’s profile MLP sees only centreline samples, so it cannot establish the upper-left vortex. The full-field bridge uses the frozen Lab 8 CFD and POD-DeepONet fields at the same native coordinates. It is a post-hoc physical audit of published predictions, not a new fit or a model-selection step.'),
        F('results/week04_corner_audit/re7500_corner.png','Re = 7500, native 513-by-513 fields. Compare matched signed-streamfunction levels in the upper-left corner. The values and predicted field belong to the frozen high-Re model, not the Lab 2 coordinate network.',280),
        B('In the declared upper-left window, the retained CFD positive streamfunction maximum is about 0.00210361 at (0.06640625,0.9140625). The frozen neural maximum is about 0.00210011 at (0.068359375,0.9140625): one grid cell in x and about -0.166% in strength. This structure is present in that model; an unseeded whole-cavity picture can conceal it.'),
        B('At Re = 1000 the same CFD window has no resolved positive island under this diagnostic. Do not force a high-Re vortex into a lower-Re reference, or treat the introductory Re = 100-400 coordinate-MLP picture as the high-Re experiment. Inspect the matching case and field before diagnosing the model.'),
        B('<b>Exercise 5C.</b> Compare signed contour levels, extremum locations and off-centre velocity profiles for CFD, neural and interpolation predictions. Report absolute and relative regional error together because circulation is weak. If you change the diagnostic window after viewing a test field, label that analysis as exploratory; do not use it to retune the frozen model and retain the same untouched-test claim.'),
    ],5,'Full-field bridge to Lab 8; corner contours and off-centre profiles')
    P('Lab 6 | Aim: isolate depth, optimizer and data weighting',[
        B('<b>Aim.</b> Learn the seven-Re centreline family directly from Ghia samples and determine whether depth, an optimizer sequence or boundary emphasis improves generalization over a basic MLP and a strong interpolation baseline. This is supervised profile regression, not a CFD solve or a two-dimensional PINN.'),
        E(r'\mathbf z=[t,\log_{10}Re-3,c],\qquad \widehat u(t)=t+t(1-t)N_\theta,\quad\widehat v(t)=t(1-t)N_\theta'),
        B('t is the normalized coordinate along the selected centreline and c identifies the u or v component. For u, t=y/L; for v, t=x/L. The endpoint lift imposes the known profile endpoints. N<sub>θ</sub> is the scalar network output selected for that component. These endpoint identities do not constrain a two-dimensional divergence or momentum balance.'),
        T([['Case role','Re values','Use'],
           ['Training','100, 1000, 5000, 10000','Fit model and training scales.'],
           ['Validation','400','Select configuration and weighting.'],
           ['Public held-out tests','3200, 7500','Evaluate frozen choices, not tune them.'],
        ],[1.1,1.6,1.8]),
        B('The retained comparison uses seeds 7, 23, 61, tanh activations, Xavier initialization, width 24 and a maximum of 1200 joint function/gradient evaluations per fit. It compares two versus five hidden layers, Adam, L-BFGS and their sequence. Two layers have 721 parameters; five have 2521, so the original depth comparison is not parameter-matched.'),
        B('The wall-weighted variant multiplies training weights by three at t&lt;0.12 or t&gt;0.88, then renormalizes each case/component to equal total weight. These are data-regression weights. They are not weights on a PINN PDE residual or a boundary-condition loss. The final audit remains in physical units and includes both boundary and core regions.'),
        B('<b>Notebook sequence.</b> Verify the frozen models, read the split and validation choice, recompute the per-seed table, inspect all seven physical-profile figures and only then run a separately named optional training experiment. The same Ghia reference and anomaly policy used in Lab 5 apply here.'),
    ],6,'Frozen comparison, profile plots and optional CPU training')
    P('Lab 6 | Why an optimizer can help and still lose to interpolation',[
        B('Adam uses moment-scaled first-order updates; L-BFGS uses recent secant information to approximate curvature of a fixed objective. A hybrid can improve a poorly conditioned coefficient fit, but it cannot supply missing Reynolds cases. More layers can describe sharper profiles while making optimization harder or increasing overfit. A wall-weighted objective can improve a boundary layer and worsen the centre.'),
        T([['Retained model','Re = 3200 mean RMSE/U','Re = 7500 mean RMSE/U'],
           ['Two-layer Adam','0.03395','0.03800'],
           ['Selected five-layer wall-weighted hybrid','0.01974','0.00563'],
           ['Piecewise linear interpolation in log Re','0.01629','0.00518'],
        ],[2.1,1.25,1.25]),
        B('The selected neural configuration improves strongly on the neural baseline, especially at Re = 7500. Interpolation still wins both test cases by the reported mean u/v RMSE. The useful result is therefore two conclusions: optimization/representation matter within the neural family, and this sparse parametric profile problem does not justify replacing the simpler method on these tests.'),
        B('The table first averages component errors within each seed and then averages seeds. The plotted neural curves are means of predictions. The error of the mean prediction is generally not the mean of the individual errors, because the norm is nonlinear. Keep these two aggregation rules distinct and retain seed spread.'),
        S('A fair claim needs paired changes'),
        B('Compare Adam-2 with hybrid-2 to isolate the optimizer schedule at fixed depth. Compare hybrid-2 with hybrid-5 to examine depth and increased capacity together. Compare hybrid-5 with hybrid-5-wall to examine weighting. A single baseline-to-final comparison changes several factors and cannot identify which caused the gain.'),
        B('Elapsed time and evaluations are complementary budgets. One evaluation on a larger network costs more work, and a quasi-Newton line search may need several evaluations per accepted step. Explain any improvement using the physical profiles and convergence behaviour rather than a general statement that one optimizer is “best.” All retained outcomes, including losses, belong in the appendix and result tables.'),
    ],6,'Retained experiment; optimizer/depth/weight comparisons')
    P('Lab 6 | Experiments that test an engineering hypothesis',[
        B('<b>Exercise 6A: reproduce the aggregation.</b> From scores.csv, compute u/v mean error per seed, then mean and standard deviation across seeds. Verify the reported test table. Overlay the physical profiles and signed discrepancies at the original reference positions; do not turn a three-seed mean curve into three independent measurements.'),
        B('<b>Exercise 6B: depth without a large capacity change.</b> Choose two depths and widths giving approximately equal trainable parameter counts. Fix training cases, activation, seed list and evaluation budget. Report validation error, time and parameter count. Contrast this with the retained 721-versus-2521 comparison and identify the capacity confound.'),
        B('<b>Exercise 6C: wall weighting.</b> Compare declared weights 1, 3 and 5 with each case/component normalized to the same total weight. Report unweighted wall and core errors separately. Select on the reserved validation Re = 400. A lower weighted training objective is not evidence of a lower unweighted field error.'),
        B('<b>Exercise 6D: the parameter coordinate.</b> Compare interpolation in Re with interpolation in log Re using identical training cases. Select the coordinate convention on validation. Explain why logarithmic spacing may be appropriate for a broad parameter range but is still a modelling choice rather than a theorem about this particular dataset.'),
        B('<b>Exercise 6E: diagnose a local failure.</b> Find the largest signed discrepancy in each held-out profile. Identify whether it is at a flagged printed point, a wall layer, an extremum or a zero crossing. Explain the effect on RMSE and relative L2 without editing the reference values.'),
        S('A useful report and the next step'),
        B('Submit one reproducible CSV, paired physical-profile figures, a training/validation history for any new fit, and a paragraph explaining which hypothesis the evidence supports. State whether the new fit is validation-only development or an exploratory use of disclosed tests. Preserve a new independent test family for a research claim after repeated redesign.'),
        B('To learn a complete flow field, obtain independently qualified CFD snapshots, keep every point of a Reynolds case together, and fit a training-only spatial representation. A dense line drawn through sparse Ghia samples is still a profile prediction. It does not provide the two-dimensional labels needed to learn a corner recirculation. Lab 7 qualifies those fields, and Lab 8 performs the actual full-field comparison.'),
    ],6,'Specific student tasks and transition to full-field learning')
    P('Lab 7 | Aim: qualify the actual higher-Reynolds CFD',[
        B('<b>Aim.</b> Study Re = 3200, 5000, 7500, 10000 as numerical solutions of the governing equations, then decide what they support as learning labels. The retained dataset contains 45 fields: 15 Re values on 128, 256, 512 cells per side. A 512-cell nodal grid has 513 nodes per side; keep this terminology clear.'),
        E(r'\nabla_h^2\psi+\omega=0,\qquad Re^{-1}\nabla_h^2\omega-uD_x\omega-vD_y\omega=0'),
        B('The subscript h identifies the discrete grid operator, and D<sub>x</sub>, D<sub>y</sub> are its central derivative matrices. Thom wall vorticity closes the boundary. A damped Newton method uses the analytical Jacobian of the coupled steady equations, with continuation in Re supplying a usable initial state. No Ghia values enter the PDE solve.'),
        E(r'J(\mathbf q_k)\Delta\mathbf q_k=-F(\mathbf q_k),\qquad\mathbf q_{k+1}=\mathbf q_k+\alpha_k\Delta\mathbf q_k'),
        B('q stacks the interior ψ and ω unknowns, F is the discrete residual, J its Jacobian, Δq the correction, and 0&lt;α≤1 a damping factor. Newton iteration count is not physical time. Solving F=0 establishes a discrete steady root; time stability needs a separate perturbation experiment.'),
        B('Run All reads the retained fields and evaluates their evidence on CPU. It does not submit cluster jobs or reproduce every expensive factorization. The production command is optional and must use a deliberate output location. The protocol preserves the failed direct jump from Re = 1000 to 3200 as well as the successful continuation route.'),
        B('Inspect the highest-resolution streamlines first, then both centreline profiles on all three grids. A sparse Ghia norm can improve or worsen nonmonotonically. In particular, the flagged printed entries stay visible. A small residual and an attractive picture are insufficient reasons to declare the finest field a converged continuum reference.'),
        B('<b>Deliverable.</b> For each of the four high-Re cases, give method, grid, equation defect, benchmark norms, signed spatial discrepancies and a mesh/time admission decision. Keep the primary circulation and at least one corner close-up in the evidence, with the actual Re in every caption.'),
    ],7,'Physical problem, three-grid CFD and Ghia comparisons')
    P('Lab 7 | Mesh sensitivity and physical time are distinct',[
        T([['Re','256-to-512 vector discrepancy','What this means'],
           ['3200','2.03%','Change between two finite CFD discretizations.'],
           ['5000','3.14%','Not an exact continuum-error estimate.'],
           ['7500','4.72%','Can exceed the neural-to-finest-grid error.'],
           ['10000','6.32%','Corner and gradient resolution remain demanding.'],
        ],[.75,1.7,2.1]),
        B('These comparisons use matched spatial locations and a declared field norm. Dense interpolation to compare profiles does not create new benchmark samples. A three-grid sensitivity table is useful, but a formal Richardson extrapolation requires an appropriate asymptotic error model and consistent convergence; do not infer an exact order from arbitrary finite-grid differences.'),
        E(r'\delta\psi=10^{-4}\sin^2(\pi x)\sin^2(\pi y),\qquad \delta\omega=-\nabla_h^2\delta\psi'),
        B('The retained temporal study starts both step sizes from the same perturbed 256-cell steady field, including a consistent vorticity perturbation. SSPRK3 advances with a new Poisson and wall-vorticity solve at each stage. Compare Δt=0.002 and 0.001 at matching observation times through tU/L=10.'),
        B('The paired time-step checks pass, but that conclusion concerns discretization of this finite perturbation trajectory. Re = 10000 shows about 8.7% growth of the perturbation over the short interval, whereas the other three cases decay. One perturbation and one finite time window neither prove nor disprove asymptotic stability.'),
        S('Exercise: a bounded admission decision'),
        B('Use the mesh table to test a declared 1% of U maximum-profile-change requirement for the 256-cell grid. Plot signed 256-minus-512 profiles. Then use the time table to distinguish step-size discrepancy from perturbation amplification. If a requirement fails, describe the finite-grid evidence honestly and propose one follow-up with a longer interval, a finer grid or a different perturbation.'),
        B('The high-Re surrogate later learns the stated 512-cell numerical family. This is still a meaningful learning target when its limitations are explicit. What is not justified is describing a 0.1% surrogate-to-label error as proof that the physical flow is accurate to 0.1%.'),
    ],7,'Mesh table, matched-time check and exercises B-D')
    P('Lab 7 | Measured solver cost: fewer iterations can take longer',[
        B('The solver experiment changes the Newton linear solve while preserving the cavity equations, walls, grids, initial fields, Jacobian and final acceptance checks. The baseline factors the sparse Jacobian directly. The candidate uses GMRES with a geometric Galerkin multigrid preconditioner and incomplete-LU smoothing. Preconditioning can make iterations easier, but constructing the preconditioner is also work.'),
        B('Both methods take four outer Newton steps in the warm cases. The successful candidate needs only eight GMRES iterations in total, yet its strong hierarchy is rebuilt at every Newton step. Setup accounts for 89.3-95.2% of measured solve time, and the candidate takes 1.95-2.17 times the direct solver’s warm-case time. The decision for these cases is to retain the direct solver.'),
        T([['Controlled item','Recorded scope'],
           ['Hardware / software','One thread on Intel Xeon Gold 5220; Python 3.12.3, NumPy 2.2.6, SciPy 1.15.3.'],
           ['Warm cases','Re = 6500→7500 and 9000→10000; same seed per pair; three repetitions in AB, BA, AB order.'],
           ['Included time','Jacobian assembly and linear solution, including preconditioner construction.'],
           ['Excluded time','Queue, imports, initial I/O, seed generation, verification and checkpoint writing.'],
           ['Acceptance','Same final equation and field-agreement tests; candidate has an additional stricter inner-defect check.'],
        ],[1.2,3.5]),
        B('The early lighter preconditioners failed through singular factorizations or original-equation defect checks. Stronger pilot settings were selected before fine-grid timing. Preserve these failed attempts; a robust final configuration can still be too expensive. The linear-total timer already includes setup, so adding setup again double-counts it.'),
        B('<b>Exercise.</b> Recalculate the warm-case medians and ratios from the run records. Draw residual versus cumulative solve time and compare signed physical profiles. Explain why identical final fields do not imply identical cost, and why this one candidate implementation is not a verdict on multigrid or on Python as a language.'),
        B('Historical Ghia timing, hardware prices and current Unity wall timers are retained in the evidence appendix. Their grids, algorithms, initialization and timing scopes differ. They provide historical context; dividing them does not produce a controlled speedup or a monetary invoice for these runs.'),
    ],7,'Measured solver appendix; setup cost and failed trials')
    P('Lab 8 | Aim: predict complete high-Re fields fairly',[
        B('<b>Aim.</b> Learn the full velocity field over Re = 100-10000 and compare a frozen POD-DeepONet with interpolation using exactly the same training information. This is a supervised scalar-parameter field model. It does not use Ghia labels for fitting, does not solve the PDE without data, and does not directly predict pressure.'),
        T([['Role','Re values'],
           ['Training','100, 250, 500, 750, 1000, 2500, 4000, 5000, 6000, 9000, 10000'],
           ['Validation','400, 6500'],
           ['Public held-out tests','3200, 7500'],
        ],[1.1,3.6]),
        E(r'\widehat{\mathbf q}(Re)=\overline{\mathbf q}+\Phi\mathbf a(Re),\quad \mathbf a(Re)=\mathbf s\odot f_\theta(\log_{10}Re-3)'),
        B('q stacks both velocity components on the native 513-by-513 grid. The training-only mean and POD matrix Φ form the fixed trunk; the tanh branch f<sub>θ</sub> predicts modal coefficients. The coefficient scale s is derived only from training data. One global RMS scale and separate per-mode RMS scales emphasize different aspects of the coefficient fit.'),
        E(r'J(\theta)=\frac{1}{n_{\rm tr}r}\sum_{j=1}^{n_{\rm tr}}\sum_{k=1}^{r}[f_{\theta,k}(z_j)-a_{jk}/s_k]^2'),
        B('n<sub>tr</sub> is the number of training cases, r is the retained rank and z=log<sub>10</sub>Re-3. Per-mode scaling raises the relative importance of low-energy coefficients. That can help a small feature or amplify a poorly constrained mode; validation decides, not an assumption that more equal modal weights must be better.'),
        B('The retained experiment compares five configurations and three seeds 7, 23, 61 with the same maximum function/gradient-evaluation budget. A capacity-matched five-layer control helps distinguish depth from parameter count. Mean validation vector error selects the two-layer hybrid configuration. The delivered field averages its three predeclared seeds.'),
        B('The notebook’s ordinary route verifies the frozen evidence and plots on CPU. Its optional training cell creates a separately named development model and scores validation. Do not replace the published frozen model or silently reuse already inspected tests for selection.'),
    ],8,'Training-only POD, branch architecture and frozen split')
    P('Lab 8 | Diagnose the basis before changing the optimizer',[
        B('The initial training round had nine CFD cases. At validation Re = 400, even the best orthogonal projection onto its POD span had 3.251% velocity error. A branch network restricted to this basis cannot beat that representation floor, whatever its depth or optimizer. This identifies missing spatial information before spending more optimization work.'),
        B('Two independently computed training cases, Re = 500 and 750, were added before neural test scoring. They improve coverage of the low-Re change in field shape. Validation/test roles and candidate budget remain unchanged. This is a documented validation-driven amendment to the training set, not an experiment whose original sample set was fixed prospectively forever.'),
        D([['<b>Large projection error?</b><br/>check training basis','<b>Add justified information</b><br/>new development fields','<b>Refit on allowed cases</b><br/>then validate'],['<b>Small projection error?</b><br/>basis can represent target','<b>Large prediction error?</b><br/>inspect branch fit','<b>Controlled model change</b><br/>depth, optimizer, weights']]),
        B('A projection using a test target is allowed only as a post-freeze diagnostic, and its oracle status must remain clear. It is not a prediction and cannot be used to choose a mode count after inspecting that test while retaining an untouched-test claim. The stored projection table records this distinction.'),
        B('<b>Exercise 8A.</b> Compare each configuration at both validation Re values and all seeds. Pair Adam with hybrid at fixed depth, then compare depth and scaling controls. Report wins and losses, parameter counts, objective evaluations and time. Select only with validation evidence.'),
        B('<b>Exercise 8B.</b> Compare the original and amended validation projection floors. Explain what changed in the physical information. Then compare oracle projection, fitted branch prediction and interpolation at the frozen tests. Keep representation, optimization and generalization as distinct mechanisms rather than attributing every improvement to a “better neural network.”'),
        B('A weak corner structure may occupy little area and carry little global snapshot variance. Evaluate physical regional diagnostics as well as full-field error. If a new local diagnostic changes the design after public test exposure, call it development and reserve fresh cases before making a new generalization claim.'),
    ],8,'Projection lower bound; validation-driven data amendment')
    P('Lab 8 | Measured wins and remaining physical limitations',[
        T([['Held-out Re','Selected neural mean','Linear in Re','Linear in log Re'],
           ['3200','0.0951%','0.8400%','0.5394%'],
           ['7500','0.0844%','0.4703%','0.3139%'],
        ],[1,1.2,1.2,1.3]),
        B('These velocity-vector relative L2 errors compare all methods with the same retained 512-cell CFD field. Both interpolation baselines receive exactly the same training snapshots as the neural model. The selected ensemble wins these two full-field tests. This is a different experiment from Lab 6, where log-Re interpolation wins the Ghia-profile comparison.'),
        B('Exact matched wall values and very small discrete divergence follow naturally from a compatible linear basis. Nonlinear momentum generally does not. The selected ensemble’s vorticity-transport maximum residual is about 798 at Re = 3200 and 296 at Re = 7500 under the stated whole-domain derivative diagnostic, with sharp-corner sensitivity. Report this limitation even though velocity-to-label discrepancy is small.'),
        B('The 256-to 512 CFD vector changes are 2.03% and 4.72% at these Re values, exceeding the surrogate discrepancy. They are finite-grid sensitivities, not confidence intervals or rigorous bounds on the continuum error. The network is accurate relative to the specified discretization; that statement should accompany the attractive field comparisons.'),
        S('Flow structure matters beyond the global norm'),
        B('Compare Re = 1000, which is a training-case reconstruction, with Re = 7500, a held-out field. Use matching streamline seeds and identical signed streamfunction contours in each corner. Locate the extrema on native nodes, then inspect off-centre u/v profiles across the circulation. A missing automatic streamline can be a display issue; a missing signed island or wrong local profile is a stronger model defect.'),
        B('The current upper-left audit in Lab 5 shows the frozen Re = 7500 neural vortex within one native grid cell of the CFD extremum and about 0.166% different in streamfunction strength. The log-Re interpolation baseline is closer on this one peak-strength metric (about 0.101%), even though the neural model has lower full-field velocity error. This does not prove all corner derivatives accurate, and the local audit remains post-freeze. Use the signed profiles and regional measures rather than replacing them with one favourable number.'),
        B('<b>Required conclusion.</b> State where the model beats interpolation, where numerical reference uncertainty dominates, and which PDE or local quantities still fail a chosen requirement. The results support a fast surrogate of this CFD family, not a data-free or universally valid cavity solver.'),
    ],8,'Held-out fields, interpolation, corner structures and equation defects')
    P('Lab 8 | Vorticity and pressure are demanding derived outputs',[
        E(r'\omega=v_x-u_y,\qquad E_\omega=\frac{\|\widehat\omega-\omega_{\rm CFD}\|_2}{\|\omega_{\rm CFD}\|_2}'),
        B('Compute both curls with the same centred operator at common interior nodes. A small-scale velocity discrepancy can have a much larger derivative. At Re = 7500 the retained common-curl discrepancy is 0.2931%, compared with 0.0844% for velocity. Vorticity is derived here, not a separately trained target. Do not compare a transported CFD vorticity with a differentiated neural field without explaining that operator difference.'),
        E(r'p_x=-uu_x-vu_y+Re^{-1}\nabla^2u,\qquad p_y=-uv_x-vv_y+Re^{-1}\nabla^2v'),
        B('The steady momentum equations supply a candidate pressure-gradient vector from a velocity field. Numerical noise and velocity defects can make it nonintegrable. A least-squares pressure recovery chooses the scalar whose discrete gradient best matches that vector, with one whole-interior mean set to zero. It does not create an independently validated pressure solution.'),
        B('The retained Re = 7500 neural/CFD reconstructed-pressure discrepancy is 0.05373%, while the 256-to 512 CFD pressure sensitivity is 8.90%. Both pressures are recovered by the same procedure. Good agreement between them can coexist with substantial grid sensitivity. Ghia’s two velocity tables provide no direct pressure reference.'),
        B('The physical comparison gallery uses shared full-extrema symmetric-log scales where needed for signed vorticity and pressure, with stated linear thresholds of 1 and 0.02. Signed difference panels use linear scales. Read the ticks and units, not the visual fraction of a colour interval. A symmetric-log scale is a display transform; it does not change the raw field or its numerical error norm.'),
        B('<b>Exercise 8C.</b> Report the maximum signed vorticity discrepancy and its coordinates, the relative norm and the shared comparison domain. Compare the pressure difference with the CFD mesh sensitivity and explain which sources of uncertainty are measured.'),
        B('<b>Exercise 8D.</b> Build a cost ledger with label generation, selection/training and repeated full-field reconstruction. Use the measured CFD solver appendix to explain why an apparently more advanced linear solver did not reduce label cost. Mark missing timing terms explicitly instead of inventing a numerical break-even query count.'),
    ],8,'Physical comparison gallery; vorticity, pressure and cost exercises')
    P('Lab 9 | Aim: audit a solution learned from equations',[
        B('<b>Aim.</b> Determine whether a finite-budget, label-free neural calculation actually solves the cavity problem, using independent numerical and physical evidence. Complete Week 13 Part A first: you need automatic differentiation, streamfunction networks, equation residuals and a small PINN example before studying this research audit.'),
        B('The retained baseline has 42 endpoints: seven Re values, three seeds 7, 23, 61, and two declared corner collars. Each network solves its own Re. This is not one network generalizing across Reynolds number. It sees no CFD or Ghia labels during fitting; those references are accessed after the prediction is frozen.'),
        E(r'r_c=u_x+v_y,\quad r_x=uu_x+vu_y+p_x-Re^{-1}(u_{xx}+u_{yy})'),
        E(r'r_y=uv_x+vv_y+p_y-Re^{-1}(v_{xx}+v_{yy})'),
        B('The residuals test continuity and the two steady momentum equations at collocation points. A streamfunction representation, u=ψ<sub>y</sub> and v=-ψ<sub>x</sub>, enforces continuity by symmetry of mixed derivatives. Pressure must still be represented with a fixed gauge. Exact kinematic identities leave the nonlinear momentum problem to be solved.'),
        D([['<b>Known equations and walls</b><br/>no CFD targets in fit','<b>Automatic derivatives</b><br/>momentum residuals','<b>Finite optimizer budget</b><br/>freeze final state'],['<b>Independent residual samples</b><br/>outside training set','<b>Frozen CFD/Ghia comparison</b><br/>fields and profiles','<b>Scientific conclusion</b><br/>all seeds, including failure']]),
        B('The baseline uses four tanh hidden layers of width 64, float64 arithmetic, 2048 fixed collocation points, 2000 Adam updates and 500 L-BFGS outer updates. Actual closure counts differ from nominal updates because line searches evaluate trial steps. The last fixed-budget model is the declared endpoint; post-freeze physical errors do not select a more favourable checkpoint.'),
        B('Run All is an evidence audit on CPU and needs no GPU, Torch or TensorFlow training. It verifies saved identities and displays actual recorded outcomes. The full campaign’s computational budget is not the same as this notebook’s audit cost. A successful program run is not yet a successful neural PDE solution.'),
    ],9,'Advanced prerequisites, equations and fixed-budget experiment')
    P('Lab 9 | Corners, collocation and the objective being minimized',[
        B('The constant lid is discontinuous at its upper corners. A smooth-lid replacement changes the benchmark. The retained formulation subtracts leading Stokes corner velocity and pressure terms before learning the remaining correction; its exact formula and derivative checks are in the campaign protocol. An earlier generic rational lift had an uncorrectable leading defect and was rejected before training.'),
        B('The declared collocation collars are ε=0.02 L and 0.01 L. Points nearer to a top corner are excluded from the training objective, while the lid velocity away from the corners remains constant. Even after leading corner subtraction, blending and nonlinear terms can leave singular residual behaviour. Low loss outside a collar does not certify the excluded region or full-domain strong-residual convergence.'),
        E(r'J_N=\frac{1}{N}\sum_{i=1}^{N}f_i,\qquad f_i=r_x(\mathbf x_i)^2+r_y(\mathbf x_i)^2'),
        B('N is the number of candidate locations and f<sub>i</sub> is their squared momentum residual. If locations are sampled adaptively with probabilities q<sub>i</sub>&gt;0, an unweighted sample mean generally optimizes a different spatial objective. With m independent draws I<sub>j</sub>, an unbiased estimator of this fixed finite sum is'),
        E(r'\widehat J_N=\frac{1}{m}\sum_{j=1}^{m}\frac{f_{I_j}}{Nq_{I_j}}'),
        B('The inverse-probability factors correct the sampling distribution. A uniform component keeps every q positive. Clipping or self-normalizing weights changes the estimator. The notebook’s small sampling exercise demonstrates this algebra; it is not evidence that an adaptive cavity PINN has been trained.'),
        B('A paired collar experiment is informative only if the actual sample sets differ. With sparse uniform points, the annulus between two collars can contain no point. Compare saved sample tensors, minimum distances and stratum counts, not just requested collar widths. This distinction explains why identical losses can fail to test the intended corner sensitivity.'),
        B('<b>Exercise 9A.</b> Recompute the expected weighted and unweighted toy objectives after changing the proposal mixture. Then inspect the real campaign’s saved point hashes and collar coverage. Explain why a density curve or a requested sample count cannot replace an audit of the actual numerical samples used by the optimizer.'),
    ],9,'Corner subtraction, collars and sampling-estimator exercise')
    P('Lab 9 | Loss, physical error and complete scientific state',[
        E(r'A_R^2=\frac{\sum_{i\in R}w_i(u_i^2+v_i^2)}{\sum_{i\in R}w_i},\quad e_R^2=\frac{\sum_{i\in R}w_i[(\widehat u_i-u_i)^2+(\widehat v_i-v_i)^2]}{\sum_{i\in R}w_i}'),
        E(r'E_R=e_R/A_R'),
        B('R is a declared region, A<sub>R</sub> is its reference velocity RMS, e<sub>R</sub> is absolute vector-error RMS and E<sub>R</sub> is the relative error. In a weak vortex A<sub>R</sub> can be very small; preserve the absolute and relative values together. A very large percentage may reveal weak reference amplitude, a poor prediction or both. Changing the denominator to make the number attractive is not a repair.'),
        B('The recorded training curve is the RMS momentum objective of a closure outside the collar. It is not an independent audit history, and the last closure need not be the last accepted iterate. Independent outer-region and excluded-shell residuals are final measurements. Do not invent a continuous independent-validation curve or use final audits to choose a checkpoint.'),
        T([['State or check','Why it matters'],
           ['Weights and optimizer/history','Restoring weights alone does not restore Adam moments or L-BFGS curvature history.'],
           ['RNG and exact sample/weight tensors','Regenerated random or annular points may differ across hosts.'],
           ['Stage, counters and endpoint rule','A restart must continue the same budget and trajectory.'],
           ['Configuration/source/model identities','A matching filename is insufficient to establish a matching experiment.'],
           ['Freeze and target-access ledger','Evaluation follows frozen predictions; disclose earlier reference access.'],
        ],[1.7,2.8]),
        B('The GPU chunk-size probe aggregates the same 2048 points in a full gradient. It is a computational batching experiment, not a change in stochastic minibatch information. Its initial loss/gradient equivalence checks support that tested state, not bitwise equivalence of every future trajectory on every device.'),
        B('<b>Exercise 9B.</b> Overlay recorded objective histories against both closures and time. Compare independent endpoint residuals and physical profiles separately. In a small new development experiment, test interrupted/uninterrupted continuation through one accepted update of each optimizer stage; save the full state and report discrepancies.'),
    ],9,'Work accounting, endpoint audits and restart evidence')
    P('Lab 9 | Retain failed runs and interpret the intervention',[
        B('The completed 42-run baseline fails its field-accuracy gates, particularly at high Re. Small training loss on fixed collocation points coexists with poor interior velocities and failed independent residual checks. This is evidence about this architecture, sampling strategy, boundary treatment and finite budget; it does not establish that all PINNs fail.'),
        B('The appendix shows all seven Reynolds numbers and all predeclared seeds rather than a best-seed picture. Compare signed u/v profiles, circulation, vorticity and regional amplitudes with CFD. Keep native transported-vorticity comparisons separate from the additional common finite-difference curl gallery; different derivative operators answer different questions.'),
        B('A bounded follow-up changes fixed point distribution and global area weights at Re = 100 and 7500. It preserves each paired original state and its closure/point-work cap. The caps can differ between runs; report those differences. The intervention adds no CFD labels, does not smooth the lid and does not select a best seed.'),
        B('It improves interior velocity error in 2 of 12 paired cases and worsens it in 10. This is a useful negative result: guaranteed sampling coverage alone did not repair the physical solution. A lower independent residual can still coexist with worse field error, so neither the objective nor a single derivative diagnostic should replace the physical comparison.'),
        S('The repaired inference path and its limits'),
        B('Saved annular coordinates can differ in the last bits if regenerated on another host. The repaired evaluation loads the exact saved tensors and trained state rather than resampling or refitting. It also discloses earlier partial target access. This verifies the stated inference evidence; it does not certify cross-platform optimizer continuation or erase earlier test exposure.'),
        B('<b>Exercise 9C.</b> Reproduce the paired work/coverage ledger and identify one run where field error worsens despite a residual improvement. Show its signed profiles, common-scale fields and reference amplitude. Explain which part of the scientific hypothesis was tested and which remains unresolved.'),
        B('<b>Exercise 9D.</b> Write a proposed next development experiment that changes only one of representation, sample coverage, weighting or optimizer budget. Define a stopping and scoring rule before running. Keep the existing supervised POD-DeepONet success separate: it learns CFD labels and cannot be cited as evidence that this label-free PINN succeeded.'),
    ],9,'Completed baseline, bounded sampling change and required reports')
    P('Companion Week 4.1 | From POD reconstruction to a dynamical ROM',[
        B('<b>Aim.</b> Explain why a small state vector does not automatically give a fast solver. Lab 3 regresses steady modal coefficients from Re. The classical companion instead evolves coefficients by projecting the discrete vorticity equations. The full-order model, walls and streamfunction recovery remain explicit.'),
        E(r'\mathbf q(t)\approx\overline{\mathbf q}+\Phi\mathbf a(t),\qquad \dot{\mathbf a}=\Phi^T F(\overline{\mathbf q}+\Phi\mathbf a;Re)'),
        B('q contains interior vorticity, a is the reduced state, Φ is a training-derived orthonormal basis and F is the discrete governing operator. A direct Galerkin evaluation may still reconstruct every spatial value, solve the full Poisson problem, compute full convection and then project. Fewer state variables do not remove that work.'),
        E(r'\Phi^Tg(\mathbf q)\approx\Phi^T\overline g+\Phi^TU_g(P^TU_g)^{-1}P^T[g(\mathbf q)-\overline g]'),
        B('DEIM approximates nonlinear convection g using its own basis U<sub>g</sub> and selected spatial rows P. Precomputed affine maps evaluate velocities and vorticity gradients only at those selected rows. The inverse selected-basis matrix controls amplification; a small rank alone does not guarantee a stable nonlinear approximation.'),
        B('The retained rank 12 basis contains 99.926% of vorticity variance yet fails the 1% maximum-in-time velocity gate at validation Re = 300: Galerkin gives 1.829% and DEIM 3.516%. The oracle projection’s maximum velocity error is 0.557%, locating much of the deterioration in dynamics and nonlinear reduction. These maxima occur over trajectories and must not be subtracted as if they were additive terms at one instant.'),
        B('Rank 16 passes the declared gate. On the 33-by-33 grid and t=0-5 interval, all three held-out velocity maxima stay below 0.64%; early-time vorticity relative error can be larger when its reference norm is small. The 33-to 65-grid CFD difference at Re = 275 is about 10.16%, a different and larger quantity. The retained DEIM timing is about nine times faster than the full model with break-even near eight queries on that setup; plain Galerkin is slightly slower.'),
        B('Run '+link('notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb','the classical ROM notebook')+' to reproduce both reduced trajectories and a fresh physical comparison. Submit representation, trajectory, nonlinear-approximation, grid and timing evidence separately. The complete Week 4.1 lecture and its original exercises remain available.'),
    ])
    P('A coherent student report: compare like with like',[
        B('Each lab has its own exercises. This final structure connects them without replacing those tasks. A convincing report explains a numerical decision through reproducible physical evidence; it does not collect unrelated low errors from different experiments.'),
        T([['Report part','Required content'],
           ['1. Aim and problem','State the lab, Re, geometry, lid, grid and output being predicted. Distinguish scalar, profile, full-field, correction and PDE-solution tasks.'],
           ['2. Information and algorithm','Identify labels, training/validation/test cases, preprocessing, architecture, optimizer and baseline. Define every changed loss weight.'],
           ['3. Numerical controls','State solver qualification, refinement/time evidence, seeds, work budget, stopping rule and source/model identities.'],
           ['4. Physical outputs','Show every compared method, signed profiles, errors, walls and appropriate vortex/derivative diagnostics with matched axes and scales.'],
           ['5. Decision and limitations','Report wins and losses, including the baseline that wins. Separate surrogate, discretization and benchmark discrepancies.'],
        ],[1.1,3.6]),
        S('A suggested sequence for a two-week supervised study'),
        B('First reproduce Lab 1 qualification and Lab 2 baselines, then investigate one controlled coordinate-network change. In the second stage, compare the POD representation in Lab 3 with direct coordinate learning, audit all-Re CFD coverage in Lab 5 and study one frozen full-field result in Lab 8 after reading its Lab 7 qualification. Use Lab 6 if the question is optimizer/depth rather than full-field structure. Take Lab 9 only after its prerequisites.'),
        B('For any new experiment, save the settings before running, then retain the executed notebook, measured tables and figures. Record whether results are historical, freshly reproduced or newly trained. Do not claim a complete restart test merely because a weight file loads, or a fresh blind test merely because its fields were excluded from this one call to fit.'),
        B('<b>Final self-check.</b> Can another student locate the cells and figures supporting each numerical statement? Can the reader tell which model was trained on CFD, which on Ghia and which on equations? Can the reader identify the actual reference grid and the worst local discrepancy? If not, the report needs clarification even if its best error is small.'),
    ])
    P('Sources and reproducible reading',[
        B('The algorithms below are established methods. The course-specific configurations, wins, failures and numerical limits come from the linked notebooks, frozen manifests and retained numerical records. A published method motivates a controlled experiment; it does not replace local validation.'),
        T([['Topic','Primary reading / reproducible source'],
           ['Cavity benchmark','Ghia, Ghia and Shin (1982), High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. <link color="black" href="https://doi.org/10.1016/0021-9991(82)90058-4"><u>Journal of Computational Physics 48, 387-411</u></link>.'],
           ['DeepONet','Lu, Jin, Pang, Zhang and Karniadakis (2021), Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. <link color="black" href="https://doi.org/10.1038/s42256-021-00302-5"><u>Nature Machine Intelligence 3, 218-229</u></link>.'],
           ['POD','Berkooz, Holmes and Lumley (1993), The proper orthogonal decomposition in the analysis of turbulent flows. <link color="black" href="https://doi.org/10.1146/annurev.fl.25.010193.002543"><u>Annual Review of Fluid Mechanics 25, 539-575</u></link>.'],
           ['DEIM','Chaturantabut and Sorensen (2010), Nonlinear model reduction via discrete empirical interpolation. <link color="black" href="https://doi.org/10.1137/090766498"><u>SIAM Journal on Scientific Computing 32, 2737-2764</u></link>.'],
           ['Adam','Kingma and Ba (2015), Adam: A method for stochastic optimization. <link color="black" href="https://arxiv.org/abs/1412.6980"><u>ICLR / arXiv 1412.6980</u></link>.'],
           ['PINN optimization','Rathore et al. (2024), Challenges in training PINNs: A loss landscape perspective. <link color="black" href="https://proceedings.mlr.press/v235/rathore24a.html"><u>Proceedings of Machine Learning Research 235</u></link>.'],
        ],[1,3.65]),
        B('Local evidence: '+link('references/ghia/README.md','Ghia data and printed-anomaly policy')+'; '+link('results/pod_deeponet/README.md','low-Re POD-DeepONet')+'; '+link('results/cavity_ghia_allre/README.md','seven-Re profile learning')+'; '+link('results/cavity_highre_ghia/README.md','high-Re CFD')+'; '+link('results/cavity_field_deeponet/README.md','full-field model')+'; '+link('results/cavity_solver_benchmark/README.md','solver-cost experiment')+'; '+link('results/cavity_pinn_campaign/README.md','label-free PINN audit')+'.'),
        B('Use '+link('docs/STUDENT_EXECUTION_GUIDE.md','the student execution guide')+' for environment and evidence terminology. Use the current notebook’s actual cells and saved configuration for numerical settings. Historical public test labels in retained evidence describe their original protocol; they do not create a newly unseen student assessment.'),
    ])
    P('Evidence appendix | Find the result behind the lesson',[
        B('The following 54 pages retain the previously published scientific evidence and exercises. They have been renumbered after the coherent lab guide. PDF bookmarks identify every evidence page. Numerical results remain tied to their original methods, grids, seeds and protocols; reading this appendix does not rerun or reselect those experiments.'),
        T([['Evidence ID','Laboratory and physical question','PDF pages'],
           ['E01-E04','Lab 6: profile learning, optimizer/depth, held-out Re = 3200 and engineering tasks.','{E01}-{E04}'],
           ['E05-E06','Lab 7: four high-Re CFD cases, mesh/time sensitivity and native circulation.','{E05}-{E06}'],
           ['E07-E10','Lab 8: full-field POD model, representation floor, matched tests and exercises.','{E07}-{E10}'],
           ['E11-E16','Lab 8: overlaid streamlines, corners, signed vorticity, pressure and limits.','{E11}-{E16}'],
           ['E17','Labs 7/8: Ghia’s computer and timing context; hardware prices versus run cost.','{E17}'],
           ['E18-E23','Labs 7/8: measured direct-versus-Krylov/multigrid cost, failures and exercises.','{E18}-{E23}'],
           ['E24-E29','Lab 9: label-free formulation, collars, endpoint rules, work and state.','{E24}-{E29}'],
           ['E30-E40','Lab 9: all 42 baseline outcomes, all seven Re, matched speed and vorticity.','{E30}-{E40}'],
           ['E41-E54','Lab 9: bounded sampling intervention, paired failures, profiles, fields and follow-up work.','{E41}-{E54}'],
        ],[1,3.1,.75]),
        B('For the Stokes-correction branch use '+link('lectures/week04_2_pinn_cavity.pdf','the dedicated Week 4.2 lecture')+' and '+link('lectures/week04_2_grid51_validation.pdf','the 51-by-51 numerical addendum')+'. For the dynamical reduction branch use '+link('notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb','the classical ROM notebook')+'. These companions supplement the nine-lab route without changing its original experiments.'),
        B('When citing a retained number, name its lab and evidence ID, distinguish raw from screened reference metrics and state whether it is an individual-seed error or an ensemble error. For a new local fit, use its newly executed outputs; the historical appendix remains a comparison record.'),
    ])
    # Reading-map ranges are derived from actual page declarations, not hand-kept offsets.
    route_table = next(b for b in pages[1]['blocks'] if b['type']=='table')
    for lab in range(1,10):
        ids=[i+1 for i,p in enumerate(pages) if p.get('lab_number')==lab]
        route_table['rows'][lab][2]=f'{min(ids)}-{max(ids)}'
    route_table['rows'][10][2]=str(next(i+1 for i,p in enumerate(pages) if p['title'].startswith('Companion Week')))
    route_table['rows'][11][2]=f'{len(pages)-2}-{len(pages)}'
    for block in pages[-1]['blocks']:
        if block['type']=='table':
            for row in block['rows']:
                for j,s in enumerate(row):
                    for i in range(1,55):s=s.replace('{E'+str(i).zfill(2)+'}',str(len(pages)+i))
                    row[j]=s
    return pages
