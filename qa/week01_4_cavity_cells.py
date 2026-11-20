"""Author the actual CPU cavity portion of the standalone W1.4 notebook."""
import csv
import hashlib


def append_cavity_cells(md, code, algorithms, root):
    reference_path = root / 'references/ghia/published_samples.csv'
    rows = list(csv.DictReader(reference_path.read_text().splitlines()))
    references = {}
    for re in (100, 400):
        refs = references[re] = {}
        for component, coordinate in [('u','y'),('v','x')]:
            selection = [r for r in rows if int(r['Re']) == re and r['component'] == component]
            refs[coordinate] = [float(r['coordinate']) for r in selection]
            refs[component] = [float(r['value']) for r in selection]
            refs[component+'_flagged'] = [r['flagged'].lower() == 'true' for r in selection]
    md(r'''
## 10. Solve the actual lid-driven cavity

The next cells solve the **steady, two-dimensional incompressible Navier–Stokes equations**,
including pressure. They recompute the flow; Ghia's values are used only afterwards for comparison.
The unit square has $L=1$, density $\rho=1$, open-edge lid speed $U_{lid}=1$ in the $+x$ direction,
and kinematic viscosity $\nu=1/Re$. The other walls are stationary and impermeable.
The two lid-corner values are set to zero, following the stationary side walls; no smoothing is hidden.

$$u u_x+v u_y=-p_x+Re^{-1}(u_{xx}+u_{yy}),$$
$$u v_x+v v_y=-p_y+Re^{-1}(v_{xx}+v_{yy}),\qquad u_x+v_y=0.$$

Here $u,v$ are velocity components, $x,y$ are position, $p$ is pressure divided by
$\rho U_{lid}^2$, and subscripts indicate derivatives. Pressure is determined up to a constant;
we impose $p(1/2,1/2)=0$. There is no imposed zero-pressure wall condition.
The standard cavity has neither a Mach number nor a Knudsen number in this incompressible model.

### 10.1 Spatial operators and pressure–velocity coupling

Velocity uses the same $N+1$ Chebyshev points introduced above. Its boundary values are prescribed;
the two interior velocity arrays each have $(N-1)^2$ unknowns. Pressure uses a degree-$N-2$
polynomial through the $N-1$ interior points per direction, so its derivative matrix is built
from **those nodes**, rather than slicing the velocity derivative matrix.
This mixed approximation reduces the spurious pressure modes of an equal-degree construction.

Let $z=(u_I,v_I,p_I)$ collect the interior unknowns, and let $F(z)$ contain both steady momentum
residuals and the discrete divergence. One redundant continuity row is replaced by the pressure gauge.
The omitted continuity equation is still checked when deciding whether the calculation converged.

$$J(z_k)\,\delta z=-F(z_k),\qquad z_{k+1}=z_k+\lambda\delta z,\qquad 0<\lambda\leq1.$$

$J$ is the exact Jacobian of the discrete equations; $k$ counts Newton iterations and $\lambda$
is a backtracking damping factor. Each iteration solves velocity and pressure together.
This differs from SIMPLE/PISO/PIMPLE's correction loops, but solves the same steady physical equations.
**An iteration is not a physical time step.** Starting the algebraic solve with zero interior values
does not produce a time-resolved startup history.

The dense implementation deliberately supports even $8\leq N\leq32$: at $N=32$ it has 2883
coupled unknowns and needs memory for dense matrices. That keeps the code inspectable on a normal
CPU. Larger research calculations need scalable linear algebra, not a GPU requirement for this lab.

**Read the solver in this order:** `cavity_operators` → `cavity_residual` → `cavity_jacobian` →
`solve_spectral_cavity`. The remaining routines differentiate pressure and evaluate saved polynomials.
''')
    algorithms('nodal_derivative','cavity_operators','cavity_residual','cavity_jacobian',
               'cavity_sample','solve_spectral_cavity')
    md(r'''
### 10.2 Run and qualify two Reynolds numbers

Start with Re=100, then observe Re=400. The supplied Run All uses N=16,24,32 for each.
For a shorter first pass set `CAVITY_REYNOLDS = [100]`; the tables and plots adapt.
Do not tighten a residual tolerance and call that mesh refinement.

**Table C** reports actual solve time, Newton iterations, momentum/gauge residual, all interior-node
continuity equations, and an independent off-grid continuity diagnostic.
The latter is RMS of the differentiated velocity polynomial on 101×101 fixed locations in
$[0.05,0.95]^2$. It is **not** a full-domain norm and excludes the immediate corner zones.
A collocation residual can be tiny while this spatial defect remains visible.

Ghia's 17 samples per component are embedded from the repository's original table, with their
reported precision and provenance. In Re=400 the previously flagged printed $v$ value at x=0.9063
is retained and marked; both all-sample and screened scores are reported. No replacement value is invented.
RMS errors are in lid-speed units. Relative L2 uses one whole-profile reference norm, never division
by a pointwise velocity near a zero crossing. Ghia supplies velocity profiles, not a pressure reference.
''')
    code(r'''
# Actual CFD controls. Standard-lid results are compared with Ghia.
CAVITY_REYNOLDS = [100, 400]
CAVITY_DEGREES = [16, 24, 32]
CAVITY_TOLERANCE = 1e-9
CAVITY_MAX_ITERATIONS = 25
CAVITY_LID = "standard"  # "smooth" is a DIFFERENT physical boundary condition.
GHIA_REFERENCE = __GHIA__
GHIA_SOURCE_SHA256 = "__GHIA_HASH__"
GHIA_DOI = "https://doi.org/10.1016/0021-9991(82)90058-4"
cavity_solutions, cavity_rows, ghia_rows = {}, [], []
for re in CAVITY_REYNOLDS:
    for n in CAVITY_DEGREES:
        sol = solve_spectral_cavity(n, re, CAVITY_LID, CAVITY_TOLERANCE, CAVITY_MAX_ITERATIONS)
        cavity_solutions[(re,n)] = sol
        np.savez_compressed(OUTPUT_DIR / f"cavity_re{re}_N{n}.npz",
                            **{k:v for k,v in sol.items() if k != "state"})
        row = {"Re":re, "N":n, "velocity_nodes":(n+1)**2,
               "coupled_unknowns":3*(n-1)**2, "lid":CAVITY_LID,
               "converged":sol["converged"], "stop_reason":sol["stop_reason"],
               "Newton_iterations":sol["iterations"], "seconds":sol["solve_seconds"],
               "steady_residual_max":sol["residual_max"],
               "continuity_nodes_max":sol["divergence_nodes_max"],
               "continuity_core_RMS":sol["divergence_core_rms"],
               "continuity_core_max":sol["divergence_core_max"]}
        if CAVITY_LID == "standard" and re in GHIA_REFERENCE:
            ref = GHIA_REFERENCE[re]
            for component, coordinate in [("u","y"),("v","x")]:
                positions = np.asarray(ref[coordinate]); observed = np.asarray(ref[component])
                predicted = (cavity_sample(sol,"u",[.5],positions)[:,0] if component == "u"
                             else cavity_sample(sol,"v",positions,[.5])[0])
                error = predicted-observed
                flagged = np.asarray(ref[component+"_flagged"], dtype=bool)
                row[component+"_RMS_all"] = np.sqrt(np.mean(error**2))
                row[component+"_relative_L2_all"] = np.linalg.norm(error)/np.linalg.norm(observed)
                row[component+"_RMS_screened"] = np.sqrt(np.mean(error[~flagged]**2))
                row[component+"_relative_L2_screened"] = np.linalg.norm(error[~flagged])/np.linalg.norm(observed[~flagged])
                row[component+"_n_all"], row[component+"_n_screened"] = len(error), int((~flagged).sum())
                ghia_rows.extend({"Re":re,"N":n,"component":component,"coordinate":float(s),
                                  "CFD":float(a),"Ghia":float(b),"signed_error":float(e),"flagged":bool(f)}
                                 for s,a,b,e,f in zip(positions,predicted,observed,error,flagged))
        cavity_rows.append(row)
        print(f"Re={re}, N={n}: {sol['stop_reason']}; {sol['solve_seconds']:.2f} s")
table_c = pd.DataFrame(cavity_rows)
table_g = pd.DataFrame(ghia_rows)
display(table_c.rename_axis("Table C: freshly solved cavities"))
if not table_c["converged"].all():
    print("ATTENTION: a case failed its algebraic criterion; keep the record but do not call it validated.")
'''.replace('__GHIA__',repr(references)).replace('__GHIA_HASH__',hashlib.sha256(reference_path.read_bytes()).hexdigest()))
    md(r'''
### 10.3 Inspect the flow, pressure, vorticity and recirculation

Each figure below shows one Reynolds number at the largest requested N. The contour fields are
evaluated on the native solver nodes, retaining the computed extrema. Streamlines integrate the
interpolated velocity on a uniform plotting grid; that plotting grid does not add CFD resolution.
Vorticity is $\omega=v_x-u_y$, including its sign. Pressure uses the centre gauge above.
Pressure is displayed on its native interior nodes, without extrapolating values to the corner singularities.

Speed uses a linear scale. Pressure uses a symmetric-log scale, linear for $|p|\leq0.01$;
signed vorticity uses a symmetric-log scale, linear for $|\omega|\leq1$. Both retain the native-node
extrema. The upper-corner singularities can dominate their full ranges. The fourth panel is a
second view of the same solved velocity field: positive and negative centreline velocities reveal
recirculation that speed magnitude alone hides. Legends are outside the data axes.
''')
    code(r'''
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import MaxNLocator
for re in CAVITY_REYNOLDS:
    n = max(CAVITY_DEGREES); sol = cavity_solutions[(re,n)]
    if not sol["converged"]:
        print(f"Skipping publication-style fields for unconverged Re={re}, N={n}"); continue
    x = sol["x"]; speed = np.hypot(sol["u"],sol["v"])
    pressure = sol["p"]
    omega = sol["omega"]
    fig, axes = plt.subplots(2,2,figsize=(9.4,7.9),layout="constrained")
    specifications = [(speed,x,r"$|\mathbf{u}|/U_{lid}$",None),
                      (pressure,x[1:-1],r"$(p-p_c)/(\rho U_{lid}^2)$",SymLogNorm(linthresh=.01,vmin=pressure.min(),vmax=pressure.max())),
                      (omega,x,r"$\omega L/U_{lid}$",SymLogNorm(linthresh=1,vmin=omega.min(),vmax=omega.max()))]
    for ax,(field,nodes,label,norm) in zip(axes.flat,specifications):
        if norm is None:
            levels = np.linspace(field.min(),field.max(),25)
        else:
            levels = norm.inverse(np.linspace(0,1,25))
        filled = ax.contourf(nodes,nodes,field,levels=levels,cmap="jet",norm=norm)
        cb = fig.colorbar(filled,ax=ax,shrink=.88,pad=.025)
        cb.set_label(label)
        if norm is None: cb.locator=MaxNLocator(nbins=5); cb.update_ticks()
        ax.set(xlabel="x/L",ylabel="y/L",xlim=(0,1),ylim=(0,1),aspect="equal")
    q = np.linspace(0,1,181)
    uq, vq = cavity_sample(sol,"u",q,q), cavity_sample(sol,"v",q,q)
    axes[0,0].streamplot(q,q,uq,vq,color="black",density=1.35,linewidth=.55,arrowsize=.65)
    ax=axes[1,1]
    ax.plot(q,cavity_sample(sol,"u",[.5],q)[:,0],color=COLORS[0],label="u(0.5,s)")
    ax.plot(q,cavity_sample(sol,"v",q,[.5])[0],color=COLORS[2],ls="--",label="v(s,0.5)")
    ax.axhline(0,color=".65",lw=.7); ax.grid(alpha=.18)
    ax.set(xlabel="Centreline coordinate s/L",ylabel="Signed velocity / lid speed",xlim=(0,1))
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
    display(__import__('IPython').display.Markdown(
        f"**Actual cavity: Re={re}, N={n}, {(n+1)}×{(n+1)} velocity nodes, lid `{CAVITY_LID}`.** "
        "Top left: speed and streamlines. Top right: pressure. Bottom left: signed vorticity. Bottom right: centreline velocities."))
    save_figure(fig,f"cavity_fields_re{re}")
''')
    md(r'''
### 10.4 Compare with Ghia, then separate algebraic and spatial error

All curves below come from the just-computed fields. Reference markers remain at the original
Ghia coordinates; lines connect CFD evaluations on a dense centreline. The full per-sample signed
differences are in **Table G**. A flagged reference point is drawn as a cross rather than silently removed.
If the lid is changed to `smooth`, the Ghia comparison is disabled because its boundary condition differs.

The final diagnostic pairs the Newton residual history with the **signed off-grid divergence** along
y=0.95, x in [0.05,0.95]. The residual history is algebraic, and the line is a spatial check.
Reducing the former does not guarantee reducing the latter. Increasing N can help, but corner
singularities prevent us from promising exponential convergence for the standard discontinuous lid.
''')
    code(r'''
comparison_re = [re for re in CAVITY_REYNOLDS if re in GHIA_REFERENCE and CAVITY_LID == "standard"]
if comparison_re:
    fig,axes = plt.subplots(len(comparison_re),2,figsize=(10,3.8*len(comparison_re)),
                            squeeze=False,layout="constrained")
    q=np.linspace(0,1,501)
    for row,re in enumerate(comparison_re):
        ref=GHIA_REFERENCE[re]
        for j,component in enumerate(("u","v")):
            ax=axes[row,j]; coordinate="y" if component=="u" else "x"
            for k,n in enumerate(CAVITY_DEGREES):
                sol=cavity_solutions[(re,n)]
                pred=(cavity_sample(sol,"u",[.5],q)[:,0] if component=="u" else cavity_sample(sol,"v",q,[.5])[0])
                ax.plot(q,pred,color=COLORS[k%len(COLORS)],ls=[":","--","-"][k%3],label=f"N={n}")
            pos,val=np.asarray(ref[coordinate]),np.asarray(ref[component])
            flag=np.asarray(ref[component+"_flagged"],dtype=bool)
            ax.plot(pos[~flag],val[~flag],"ko",mfc="white",ms=4,label="Ghia (1982)")
            if flag.any(): ax.plot(pos[flag],val[flag],"x",color="black",ms=7,label="Flagged printed entry")
            ax.set(xlabel=f"{coordinate}/L",ylabel=f"{component}/lid speed",xlim=(0,1))
            ax.text(0,1.18,f"Re = {re}",transform=ax.transAxes,ha="left")
            ax.grid(alpha=.18)
            ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncol=3,fontsize=9,frameon=False)
    save_figure(fig,"cavity_ghia_profiles")

fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
re=CAVITY_REYNOLDS[0]; q=np.linspace(.05,.95,501)
for k,n in enumerate(CAVITY_DEGREES):
    sol=cavity_solutions[(re,n)]; history=sol["history"]; color=COLORS[k%len(COLORS)]
    axes[0].semilogy(history[:,0],history[:,1],"o-",color=color,ms=3,label=f"N={n}")
    axes[1].plot(q,cavity_sample(sol,"divergence",q,[.95])[0],color=color,label=f"N={n}")
axes[0].axhline(CAVITY_TOLERANCE,color="black",ls="--",lw=.8,label="Algebraic target")
axes[0].set(xlabel="Newton iteration (not time)",ylabel="Maximum discrete steady residual")
axes[1].set(xlabel="x/L at y/L = 0.95",ylabel="Signed off-grid divergence × L/U")
for ax in axes:
    ax.grid(alpha=.18); ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
display(__import__('IPython').display.Markdown(f"**Continuity and convergence diagnostics: Re={re}.**"))
save_figure(fig,"cavity_convergence_continuity")
''')
    md(r'''
### 10.5 What the comparison establishes

This is an actual steady cavity computation, with no body force, fresh velocity and pressure fields,
resolved wall values, a pressure gauge and explicit qualification criteria. The Ghia figures test
centreline velocities; they do not validate the pressure field or prove mesh independence.
Table C deliberately reports the disagreement and off-grid continuity defect rather than hiding them.

The DG implementation earlier in this notebook still solves the scalar transport equation.
It teaches numerical fluxes; **the cavity solver here is Chebyshev collocation, not a DG
Navier–Stokes solver**. Comparing a full DG cavity requires its viscous flux and compatible
incompressibility treatment. The separate Nektar++/ExaDG research runs are not used as notebook data.

The smooth lid $u_{lid}(x)=16x^2(1-x)^2$ is available for a controlled boundary-regularity exercise.
Its maximum speed is one, but its driving profile and lid-average speed differ from Ghia's.
Keep its results labelled and compare its own refinement sequence, never score it against the
standard-lid Ghia samples. No new high-Re or time-dependent-flow validation is claimed here.
''')
