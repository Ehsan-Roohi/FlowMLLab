"""Explanatory cavity pages, generated from the notebook's actual evidence."""


def cavity_pages(page, para, eq, table, figure, results, width):
    rows = results['tables']['C']
    page('14. The actual driven cavity: equations and walls')
    para('We now solve the complete steady, two-dimensional incompressible Navier-Stokes equations '
         'on the unit square. The notebook calculates both velocity components and pressure from '
         'scratch. Ghia data enter only after the solve, as an independent profile comparison.')
    eq(r'u u_x+v u_y=-p_x+Re^{-1}(u_{xx}+u_{yy})')
    eq(r'u v_x+v v_y=-p_y+Re^{-1}(v_{xx}+v_{yy}),\qquad u_x+v_y=0')
    para('Here x and y are position divided by cavity width L, u and v are velocities divided by '
         'lid speed, and p is pressure divided by density times lid speed squared. Subscripts denote '
         'derivatives. Re=U<sub>lid</sub>L/nu and nu is kinematic viscosity. In these nondimensional '
         'equations L=U<sub>lid</sub>=rho=1, so nu=1/Re. There is no body force in the cavity runs.')
    table(['Location', 'Prescribed velocity'],[
        ['Open top edge: y=1, 0&lt;x&lt;1','u=1, v=0; lid moves to the right.'],
        ['Left, right and bottom walls','u=v=0.'],
        ['The two upper corner nodes','u=v=0, following the stationary side walls.'],
        ['Pressure gauge at the centre','p(0.5,0.5)=0; no zero-pressure wall condition is imposed.'],
    ],[width*.48,width*.52])
    para('The standard lid jumps at its two endpoints. This is the Ghia boundary convention, '
         'not a hidden smooth-lid approximation. Finite-degree polynomials cannot represent this '
         'jump without local error. Pressure and vorticity near these corners require particular care.')
    para('The supplied calculations use Re=100 and 400, each with N=16,24,32. Re is a physical '
         'control; N is numerical resolution. Keep those changes separate when interpreting a result. '
         'There is no Mach or Knudsen number in this incompressible continuum model.')
    para('Connection to Week 1.2', 'sub')
    para('SIMPLE/PISO/PIMPLE use repeated pressure-velocity corrections. Here a direct Newton '
         'step updates velocity and pressure together. The coupling strategy and spatial approximation '
         'both differ, so this comparison cannot isolate a universal speed advantage of one component.')

    page('15. From spectral operators to a coupled Newton solve')
    para('Velocity is interpolated through N+1 Chebyshev points per direction, including prescribed '
         'wall values. Each interior component has (N-1)<super>2</super> unknowns. Pressure is a '
         'degree-(N-2) polynomial through the interior N-1 points; it has its own derivative matrix. '
         'The pressure derivative is not a slice of the velocity derivative.')
    eq(r'z=(u_I,v_I,p_I),\qquad F(z)=0,\qquad J(z_k)\,\delta z=-F(z_k)')
    eq(r'z_{k+1}=z_k+\lambda\,\delta z,\qquad 0<\lambda\leq1')
    para('I identifies the interior nodes; z collects the unknowns. F contains both momentum '
         'equations and continuity. J is the exact derivative of F with respect to z. The Newton '
         'iteration index is k, and lambda is reduced by backtracking until the maximum residual decreases.')
    table(['Step', 'Notebook operation'],[
        ['1. Operators','Build velocity derivatives, the Laplacian, pressure gradients and contributions from known walls.'],
        ['2. Residual','Evaluate diffusion, nonlinear advection, pressure gradients and interior divergence.'],
        ['3. Pressure gauge','Replace one redundant continuity row by centre pressure; check the omitted equation separately.'],
        ['4. Newton update','Solve the full Jacobian system, backtrack the update and retain the iteration history.'],
        ['5. Qualification','Require every checked steady equation to meet tolerance; record failures rather than relabel them.'],
    ],[100,width-100])
    para('The default tolerance is 10<super>-9</super>. Starting with zero interior values is an '
         'algebraic initial guess. <b>Newton iterations are not physical time steps:</b> this notebook '
         'does not model startup transients or demonstrate dynamical stability of the steady state.')
    para('At N=32 the system has 2883 coupled unknowns. Dense matrices keep the derivation and code '
         'transparent, but increase memory and factorization cost rapidly. Even N=8 through 32 are '
         'supported here. A normal CPU suffices; larger cases need scalable linear algebra. '
         'No GPU, MPI or external solver is required.')

    for section,re in [(16,100),(17,400)]:
        chosen = max((r for r in rows if r['Re']==re),key=lambda r:r['N'])
        page(f'{section}. Computed cavity flow at Re = {re}')
        figure(f'cavity_fields_re{re}',
               f'Fresh Chebyshev cavity, Re={re}, N={chosen["N"]}: {chosen["N"]+1} x {chosen["N"]+1} '
               'velocity nodes. Top left: speed and streamlines. Top right: centre-gauged pressure. '
               'Bottom left: signed vorticity. Bottom right: signed centreline velocities.')
        para('Speed uses a linear scale. Pressure and vorticity use symmetric-log scales with '
             'linear intervals |p| &lt;= 0.01 and |omega| &lt;= 1, retaining native-node extrema. '
             'Pressure is displayed on its own interior nodes without extrapolation to the singular corners. '
             'Streamlines use a separate uniform plotting grid; it does not add CFD resolution.', 'small')
        para(f'This solve took {chosen["seconds"]:.2f} s on the recorded CPU and '
             f'{chosen["Newton_iterations"]} Newton updates. Its maximum discrete steady residual was '
             f'{chosen["steady_residual_max"]:.2e}. These describe this execution and the discrete '
             'equations; they do not establish mesh independence or pressure accuracy.', 'small')

    page('18. Our velocity profiles and Ghia samples')
    figure('cavity_ghia_profiles',
           'Computed profiles at N=16,24,32 and the original Ghia sample positions. Re=100 is the top row; '
           'Re=400 is the bottom row. Left: u at x=0.5. Right: v at y=0.5. The Re=400 printed '
           'entry at x=0.9063 is retained as a cross and flagged by the existing reference policy.')
    table(['Re','N','u RMS','v RMS: all','v RMS: screened'],[
        [r['Re'],r['N'],f'{r["u_RMS_all"]:.4f}',f'{r["v_RMS_all"]:.4f}',f'{r["v_RMS_screened"]:.4f}']
        for r in rows],[50,40,100,145,width-335])
    para('RMS values are in lid-speed units; all-sample scores use 17 points. Screening omits the '
         'single predeclared Re=400 v entry, leaving 16; no replacement value is invented. '
         'Table G retains every signed difference. The profile norm does not divide by velocities '
         'near zero. Reference data have finite reported precision and are not exact full fields.', 'small')
    para('The Re=100 discrepancies do not decrease monotonically with N. Report the measured '
         'agreement, rather than assuming spectral refinement must improve every reference score. '
         'Ghia velocity agreement alone cannot validate the computed pressure.', 'small')

    page('19. Small algebraic residual, remaining spatial defect')
    figure('cavity_convergence_continuity',
           'Re=100: left, the Newton residual against iteration count; right, signed off-grid '
           'divergence along y=0.95 on x in [0.05,0.95]. These measure different errors.')
    table(['Re','N','Node continuity max','Off-grid core RMS'],[
        [r['Re'],r['N'],f'{r["continuity_nodes_max"]:.2e}',f'{r["continuity_core_RMS"]:.4f}'] for r in rows
    ],[55,50,180,width-285])
    para('Node continuity includes the equation replaced by the pressure gauge. The independent '
         'core diagnostic evaluates the differentiated velocity polynomial at 101 x 101 fixed points '
         'in [0.05,0.95]<super>2</super>. This is a disclosed interior diagnostic, not a full-domain '
         'norm. It excludes the immediate corner zones and remains nonzero after Newton has converged.')
    para('Do not describe this discrete convergence as exact pointwise incompressibility. The '
         'standard lid has corner singularities and the pressure/velocity spaces are truncated. '
         'More spatial resolution, compatible boundary treatment and a better corner representation '
         'may be needed; tightening Newton tolerance alone does not remove these effects.')
    para('A useful contrast is the optional smooth lid 16x<super>2</super>(1-x)<super>2</super>, '
         'with maximum speed one. Its boundary driving differs, so Ghia scoring is disabled. '
         'Use its own refinement sequence to study boundary regularity. The preceding scalar DG '
         'solver remains a transport lesson: the actual cavity solver is Chebyshev collocation, '
         'and is not labelled as a DG Navier-Stokes calculation.')
