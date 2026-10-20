"""Student-facing pressure-coupling lesson shared by notebook and lecture.

The algebra follows pressure_velocity.coupling_step, including its absolute-
pressure PISO corrections. No solver settings or numerical results are changed.
"""

LESSONS = [
    ("## 2. Start with the physical equations", r"""
The unknowns are horizontal velocity $u(x,y,t)$, vertical velocity $v(x,y,t)$,
and pressure $p(x,y,t)$. We solve **continuity and both momentum equations
together**. For constant-density, constant-viscosity incompressible flow, this
is the incompressible Navier–Stokes system.

The following equations are dimensionless:

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0.$$

$$\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}
=-\frac{\partial p}{\partial x}
+\frac{1}{\mathrm{Re}}\left(\frac{\partial^2u}{\partial x^2}
+\frac{\partial^2u}{\partial y^2}\right).$$

$$\frac{\partial v}{\partial t}
+u\frac{\partial v}{\partial x}+v\frac{\partial v}{\partial y}
=-\frac{\partial p}{\partial y}
+\frac{1}{\mathrm{Re}}\left(\frac{\partial^2v}{\partial x^2}
+\frac{\partial^2v}{\partial y^2}\right).$$

Here $x,y$ are coordinates divided by cavity side $L$; $t$ is physical time
divided by $L/U$; $u,v$ are velocities divided by lid speed $U$; and
$p=(p_{\rm dimensional}-p_{\rm reference})/(\rho U^2)$. Density is $\rho$,
kinematic viscosity is $\nu$, and $\mathrm{Re}=UL/\nu$.
From left to right, momentum contains local acceleration, convection,
pressure force, and viscous diffusion. Continuity requires zero net volume
flux from each fluid element. At steady state only the local time derivative
vanishes; convection and diffusion remain.

The equivalent compact form uses $\boldsymbol{u}=(u,v)$:
$$\partial_t\boldsymbol{u}+(\boldsymbol{u}\cdot\nabla)\boldsymbol{u}
=-\nabla p+\mathrm{Re}^{-1}\nabla^2\boldsymbol{u},
\qquad\nabla\cdot\boldsymbol{u}=0.$$
The unit-square cavity has no penetration on every wall, zero tangential
speed on three walls, and $u=1$ on the lid. The ideal lid corners have a
velocity discontinuity; their extreme gradients need special care in plots.
"""),
    ("## 2.1 Why pressure must be corrected", r"""
Solving momentum using a guessed pressure generally gives a velocity that
violates continuity. A pressure solve adjusts the shared face velocities until
their cell fluxes balance. Pressure is therefore determined by the mass
constraint, even though continuity contains no explicit pressure term.

On the staggered (MAC) grid, pressure is at cell centers, $u$ on vertical faces,
and $v$ on horizontal faces. The cell divergence is
$$ (Dq)_{i,j}=\frac{u_{i+1/2,j}-u_{i-1/2,j}}{h}
+\frac{v_{i,j+1/2}-v_{i,j-1/2}}{h}. $$
Here $h=1/N$ for $N$ cells along each side. The vector $q$ collects all
unknown interior face velocities; it is a vector of velocity values, **not**
volumetric flow rate. $D$ is this discrete divergence; $G$ is the pressure
gradient at the velocity faces. Interior face fluxes cancel between neighbors.

After freezing the advecting velocities, each component momentum equation is
a linear system:
$$Aq=b-Gp.$$
The symbol $A$ denotes the two component matrices together; $b$ contains wall
sources and, for a transient step, the fixed old-time velocity source. The
diagonal entries $a_i=A_{ii}$ describe the local momentum coefficient.
Define diagonal inverse $d=\operatorname{diag}(1/a_i)$ and off-diagonal
matrix $R=A-\operatorname{diag}(a_i)$. Then
$$H(q)=b-Rq,\qquad q=d(H(q)-Gp).$$
$H$ is the current source plus neighbor-velocity contribution. It is not a
new physical variable. Since $Dq=0$, the pressure equation becomes
$$L=-DdG,\qquad Lp=-D(dH).$$
**Derivation:** apply $D$ to $q=dH-dGp$, set the result to zero, and move
$DdGp$ to the other side. The minus sign in $L$ follows the implementation's
positive-diagonal pressure operator. Pressure is unique only up to a constant;
one cell is pinned for the solve and the reported mean is removed. Velocity
depends on pressure differences, so this gauge does not affect the flow.
"""),
    ("## 3. SIMPLE: one steady correction per iteration", r"""
**SIMPLE** means Semi-Implicit Method for Pressure-Linked Equations.
Its index $k$ counts attempts to solve the steady equations; it is not time.

1. Start from $q^k,p^k$. Freeze convection using $q^k$ and assemble the
   **steady** momentum matrix and wall source.
2. Apply momentum equation relaxation. With unrelaxed diagonal $a_i$, set
   $a_i\leftarrow a_i/\alpha_u$ and add
   $(1/\alpha_u-1)a_i q_i^k$ to the original source, using the original $a_i$
   in that added term. Here $0<\alpha_u\leq1$.
3. Solve $Aq^*=b-Gp^k$. The star means tentative velocity: momentum has
   been solved for the guessed pressure, but $Dq^*$ need not vanish.
4. Approximate the velocity change by
   $q'= -dGp'$. This uses the inverse diagonal instead of the full inverse
   momentum matrix. Enforce $D(q^*+q')=0$:
   $$Lp'=-Dq^*,\qquad q^{k+1}=q^*-dGp'.$$
5. Update pressure with $p^{k+1}=p^k+\alpha_p p'$, where
   $0<\alpha_p\leq1$, and remove its mean. Reassemble from the corrected
   velocity and repeat.

The prime is a **correction**, not a spatial derivative. Apply the velocity
correction fully: multiplying it by $\alpha_p$ would leave mass imbalance.
Pressure relaxation damps the coupling iteration; momentum relaxation damps
the nonlinear equation update. Neither changes the target steady equations.

**Stop only when both** the unrelaxed nonlinear momentum residual and cell
divergence meet their tolerances. A small velocity change can result from
strong relaxation and does not prove that momentum is satisfied.

**Flow:** initialize → assemble/relax momentum → predict → pressure correction
→ correct velocity / relax pressure → check both residuals → repeat or finish.
"""),
    ("## 4. PISO: several corrections inside one physical time step", r"""
**PISO** means Pressure-Implicit with Splitting of Operators. Physical time
levels are $n$ and $n+1$; $m$ counts corrections **within** a time step.
For backward Euler, the local acceleration is
$$\partial_t q\ \approx\ \frac{q^{n+1}-q^n}{\Delta t}.$$
Thus $1/\Delta t$ is added to the momentum diagonal and $q^n/\Delta t$ to
the source. The old-time vector $q^n$ remains fixed throughout the step.

1. Assemble $A,b$ from the current advecting velocity and old-time source.
   Solve the momentum predictor $Aq^*=b-Gp^{\rm guess}$.
2. Set $q^{(0)}=q^*$ and keep $A,b,d,L$ fixed.
3. For each inner correction $m$, recompute the neighbor contribution,
   solve pressure, and reconstruct velocity:
   $$H^{(m)}=b-Rq^{(m)},$$
   $$Lp^{(m+1)}=-D(dH^{(m)}),$$
   $$q^{(m+1)}=d(H^{(m)}-Gp^{(m+1)}).$$
4. After the requested corrections, accept $q^{n+1},p^{n+1}$ and advance
   physical time by $\Delta t$.

In **this code**, each inner solve computes absolute pressure for the current
$H$, rather than accumulating a pressure increment. Both descriptions are
used in CFD literature; do not add this solved pressure to the previous one.
The second correction updates $H$ using corrected neighbor velocities. It is
not merely another iteration of the same linear pressure solve.

**What improves?** Pressure and velocity become more consistent with the frozen
momentum matrix. **What remains?** Nonlinear convection coefficients are still
those of the predictor, and backward-Euler time error remains first order.
Continuity can already be tiny while momentum coupling remains imperfect.

**Flow:** save $q^n$ → assemble/predict once → [update $H$ → solve pressure
→ reconstruct velocity] repeated → advance time. The retained default uses
two inner corrections. More corrections cost more pressure solves.
"""),
    ("## 5. PIMPLE: rebuild momentum around the PISO inner loop", r"""
PIMPLE combines SIMPLE-like outer coefficient updates with PISO inner
corrections. Use $\ell$ for the **outer** loop and $m$ for the **inner** loop;
neither advances physical time. Save $q^n$ once at the start of the step.

1. Outer loop $\ell$: assemble $A^{(\ell)},b^{(\ell)}$ using the latest
   velocity to update nonlinear convection coefficients. The transient source
   is always the same $q^n/\Delta t$.
2. Solve a new momentum predictor using the latest pressure.
3. Inner loop $m$: hold this outer loop's matrix fixed; recompute $H$, solve
   absolute pressure and reconstruct velocity exactly as in PISO.
4. Start the next outer loop from those corrected fields. After all outer
   loops, accept the step and advance time **once**.

**Flow:** save $q^n$ → [assemble/predict → [update $H$ → pressure → velocity]
inner loop] outer loop → advance time.

| Operation per accepted step | PISO | PIMPLE |
|---|---:|---:|
| Momentum assemblies / predictors | 1 | $N_o$ |
| Component momentum linear solves | 2 | $2N_o$ |
| Pressure solves | $N_i$ | $N_oN_i$ |
| Physical time advances | 1 | 1 |

$N_o$ is `outer_correctors`; $N_i$ is `pressure_correctors`. With three outer
and two inner loops, there are six pressure solves and six component momentum
solves per accepted step. With one outer loop, this implementation reproduces
PISO. These are fixed loop counts, not adaptive convergence-controlled loops.

**What improves?** Outer loops reduce the error caused by lagged nonlinear
coefficients; inner loops improve pressure–velocity consistency for each
frozen matrix. **Cost:** repeated momentum assembly, factorization and
pressure solves. Extra loops do not guarantee faster convergence or accurate
large time steps. For the steady cavity they may be unnecessary overhead.
"""),
    ("## 5.1 Read the convergence and timing plots correctly", r"""
**Accuracy** is the difference from independent reference velocities (Ghia
centerline data here); spatial refinement is essential. **Convergence** means
the chosen discrete equations and mass constraint are satisfied. **Cost** is
elapsed solver time plus meaningful work counts, not iteration count alone.

| Method | What one recorded iteration means | Native stopping quantity |
|---|---|---|
| SIMPLE | One steady coupling iteration | FV momentum residual and divergence |
| PISO | One physical step with inner corrections | Steady FV momentum residual and divergence for these steady-target runs |
| PIMPLE | One physical step with outer and inner loops | Same steady FV checks |
| Streamfunction–vorticity | One vorticity time update and streamfunction solve | Vorticity equation/update criterion reported by that solver |

Do not overlay distinct native residuals as though their numerical magnitudes
were the same physical error. Compare velocity errors, final fields and
runtime alongside separately labeled convergence histories. A converged
coarse-grid result can still be inaccurate. A transient trajectory additionally
requires time-step refinement at matched physical times.

For timed comparisons, use the same machine, no concurrent solver cases, the
same resolution spacing and clearly stated tolerances. Meshes use different
storage conventions: $N$ FV cells correspond to $N+1$ streamfunction nodes.
Higher Reynolds numbers generally need finer meshes; memory, iteration counts
and runtime can grow substantially. Start with the supplied coarse study
before selecting a finer mesh on a laptop or a limited Colab session.

**Check your understanding:** (1) Why does relaxing the velocity correction
leave mass error? (2) Which quantities change in a PISO inner loop?
(3) Which additional quantity changes in a PIMPLE outer loop?
(4) Why can a near-zero divergence coexist with a large momentum defect?
"""),
]


def improve_notebook(nb):
    """Replace short algorithm introductions; preserve code/output cells."""
    import nbformat
    prefixes = ('## 2.', '## 3. SIMPLE', '## 4. PISO', '## 5. PIMPLE')
    index = next((i for i,c in enumerate(nb.cells)
                  if c.cell_type == 'markdown' and
                  (c.metadata.get('pressure_coupling_lesson') or
                   c.source.lstrip().startswith('## 2.'))), 4)
    nb.cells = [c for c in nb.cells
                if not (c.cell_type == 'markdown' and
                        (c.metadata.get('pressure_coupling_lesson') or
                         c.source.lstrip().startswith(prefixes)))]
    cells = []
    for title, body in LESSONS:
        cell = nbformat.v4.new_markdown_cell(title + '\n\n' + body.strip())
        cell.metadata['pressure_coupling_lesson'] = True
        cells.append(cell)
    nb.cells[index:index] = cells
    return nb


def lecture_lesson(heading, add, eq, box):
    """Detailed supplemental pages using the lecture's native math flowable."""
    heading('Pressure coupling: equations before algorithms')
    add('The incompressible Navier–Stokes system consists of continuity and two momentum equations. All quantities below are dimensionless, scaled by cavity side L, lid speed U and density rho. Pressure is divided by rho U squared; Re = UL/nu.')
    eq(r'\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0')
    eq(r'\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}=-\frac{\partial p}{\partial x}+\frac{1}{\mathrm{Re}}\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}\right)')
    eq(r'\frac{\partial v}{\partial t}+u\frac{\partial v}{\partial x}+v\frac{\partial v}{\partial y}=-\frac{\partial p}{\partial y}+\frac{1}{\mathrm{Re}}\left(\frac{\partial^2v}{\partial x^2}+\frac{\partial^2v}{\partial y^2}\right)')
    add('Here x and y are coordinates, t is physical time, u and v are velocity components, and p is pressure. Momentum balances local acceleration and convection against pressure force and viscous diffusion. At steady state the time derivative vanishes; convection remains. Continuity requires zero net flux.')
    add('Pressure is stored at cell centers, u on vertical faces and v on horizontal faces. The grid spacing is h = 1/N for N cells per side. Shared face fluxes cancel between neighbors.')
    eq(r'(Dq)_{i,j}=\frac{u_{i+1/2,j}-u_{i-1/2,j}}{h}+\frac{v_{i,j+1/2}-v_{i,j-1/2}}{h}')
    heading('Decode every matrix symbol')
    add('q collects the unknown interior face velocities of both components; it does not denote flow rate. A denotes their two momentum matrices. b collects wall and old-time sources. G maps cell pressure to face gradients; D maps face velocity to cell divergence.')
    eq(r'Aq=b-Gp,\qquad R=A-\mathrm{diag}(A),\qquad d=\mathrm{diag}(1/A_{ii})')
    add('R is the off-diagonal neighbor-coupling matrix. d is an inverse diagonal operator, not grid spacing or a derivative. H is a computed source plus neighbor contribution, not an extra physical field.')
    eq(r'H(q)=b-Rq,\qquad q=d(H-Gp)')
    add('Apply divergence to this velocity relation and impose continuity. This derives the pressure equation; its right-hand side is the mass imbalance of the pressure-free trial velocity.')
    eq(r'0=D(dH)-DdGp,\qquad L=-DdG,\qquad Lp=-D(dH)')
    box('Pressure gauge','Only pressure gradients affect velocity. Pin one pressure cell to solve the singular closed-cavity system, then remove the mean for reporting. A constant pressure shift leaves velocity unchanged.')
    heading('SIMPLE: derive the mass correction')
    add('SIMPLE means Semi-Implicit Method for Pressure-Linked Equations. k counts steady coupling iterations, not physical time. Predict velocity using the current pressure and relaxed momentum system.')
    eq(r'Aq^*=b-Gp^k')
    add('A star means tentative velocity. A prime denotes a correction, not differentiation. Approximate the velocity correction using the momentum inverse diagonal; then demand zero corrected divergence.')
    eq(r'q^{\prime}=-dGp^{\prime},\qquad 0=Dq^*-DdGp^{\prime}')
    eq(r'Lp^{\prime}=-Dq^*,\qquad q^{k+1}=q^*-dGp^{\prime}')
    eq(r'p^{k+1}=p^k+\alpha_p p^{\prime}')
    add('alpha_p is pressure relaxation. alpha_u is momentum equation relaxation: divide the original diagonal by alpha_u and add (1/alpha_u - 1) times the original diagonal times the previous velocity to the source. Both lie between zero and one. Reassemble convection after correcting velocity.')
    box('Why the full velocity correction?','Its pressure equation was derived to cancel the complete mass defect. Scaling the face correction by alpha_p leaves part of that defect. Relax pressure, but apply the conservative velocity correction fully. Check the original nonlinear momentum equations, not merely the size of a relaxed update.')
    heading('PISO: fixed matrix, changing neighbor contribution')
    add('PISO means Pressure-Implicit with Splitting of Operators. n labels physical time; m labels inner corrections. Backward Euler adds 1/Delta t to the diagonal and q at the old time divided by Delta t to the source. Save the old-time velocity once.')
    eq(r'\frac{q^{n+1}-q^n}{\Delta t}+\mathcal{C}(q^{n+1})=-Gp^{n+1}+\mathcal{V}(q^{n+1})')
    add('C denotes discretized convection; V denotes viscous diffusion. First freeze convection coefficients, assemble A and b, and solve one momentum predictor. Hold A, b, d and L fixed through all inner corrections.')
    eq(r'H^{(m)}=b-Rq^{(m)}')
    eq(r'Lp^{(m+1)}=-D(dH^{(m)})')
    eq(r'q^{(m+1)}=d(H^{(m)}-Gp^{(m+1)})')
    add('This implementation solves absolute pressure in each correction: replace the pressure; do not add it to the previous pressure. Updated neighbor velocities change H at the next correction. Two sparse-solver iterations on an unchanged pressure system do not constitute two PISO corrections.')
    box('Scope of improvement','Inner corrections improve consistency with the frozen momentum matrix. They do not refresh nonlinear convection coefficients or remove first-order backward-Euler time error. Advance physical time once after all corrections.')
    heading('PIMPLE: two loops, one physical time advance')
    add('The outer index labels momentum coefficient updates; the inner index labels pressure/velocity corrections. Save old-time q once. Each outer loop rebuilds convection using the latest corrected velocity, assembles a new backward-Euler momentum system and solves a predictor. Its inner loop then performs the PISO equations on that frozen matrix.')
    box('Follow one step','Save old-time velocity → outer loop: rebuild momentum and predict → inner loop: update H, solve pressure, reconstruct velocity → repeat inner corrections → repeat outer momentum loops → accept fields and advance time once.')
    add('N_o = outer_correctors and N_i = pressure_correctors. One accepted PIMPLE step uses N_o momentum assemblies, 2 N_o component momentum solves, and N_o N_i pressure solves. Three outer and two inner loops therefore require six pressure solves and six component momentum solves. Old-time sources remain unchanged throughout all loops.')
    box('Interpretation and cost','One outer loop reproduces this PISO implementation. Additional outer loops address lagged nonlinear coefficients; additional inner corrections address coupling for a fixed matrix. More loops cost more work and need not be economical for a steady cavity. A larger stable step does not imply an accurate transient trajectory.')
    heading('A student checklist for a fair comparison')
    add('Separate accuracy, convergence and cost. Accuracy uses independent velocity reference data and mesh refinement. Convergence checks each method\'s equations and mass balance. Cost includes solver seconds, momentum solves and pressure solves. An iteration has a different meaning in SIMPLE, PISO, PIMPLE and streamfunction–vorticity.')
    add('Keep native residuals separately labeled: a vorticity update norm is not an FV momentum residual. Equal residual numbers across formulations do not ensure equal accuracy. Compare common centerline errors, final velocity fields and timings at stated resolution spacing and tolerances.')
    add('Time cases sequentially on the same machine. N FV cells correspond to N+1 streamfunction nodes. Higher Re and finer grids can require substantially more memory and runtime; begin with the coarse case before using a laptop or limited Colab runtime for refinement.')
    box('Questions to answer','Which quantities change in an inner correction? Which additional quantity changes in an outer loop? Why can divergence be tiny while momentum error remains large? Why must old-time velocity stay fixed inside both loops? Why is a steady benchmark insufficient to validate a transient time step?')

