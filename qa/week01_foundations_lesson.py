"""Repair the Week 1 mathematical introduction without running its solver.

The Markdown source is kept here so mathematical teaching updates can be
reapplied while retaining executed cells, outputs, and scientific evidence.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

GOVERNING_EQUATIONS = r"""## 2. From Navier–Stokes to streamfunction and vorticity

### 2.1 The coupled incompressible Navier–Stokes problem

We solve conservation of **mass (continuity)** and **both components of momentum together**. These form the incompressible Navier–Stokes system used in this course. For a constant-density Newtonian fluid, constant kinematic viscosity, and no body force, the dimensional equations are

$$
\frac{\partial u_d}{\partial x_d}+\frac{\partial v_d}{\partial y_d}=0,
$$
$$
\frac{\partial u_d}{\partial t_d}
+u_d\frac{\partial u_d}{\partial x_d}
+v_d\frac{\partial u_d}{\partial y_d}
=-\frac{1}{\rho}\frac{\partial p_d}{\partial x_d}
+\nu\left(\frac{\partial^2u_d}{\partial x_d^2}+\frac{\partial^2u_d}{\partial y_d^2}\right),
$$
$$
\frac{\partial v_d}{\partial t_d}
+u_d\frac{\partial v_d}{\partial x_d}
+v_d\frac{\partial v_d}{\partial y_d}
=-\frac{1}{\rho}\frac{\partial p_d}{\partial y_d}
+\nu\left(\frac{\partial^2v_d}{\partial x_d^2}+\frac{\partial^2v_d}{\partial y_d^2}\right).
$$

Here $(x_d,y_d)$ is position [m], $t_d$ is time [s], $(u_d,v_d)$ is velocity [m/s], $p_d$ is pressure [Pa], $\rho$ is density [kg/m³], and $\nu=\mu/\rho$ is kinematic viscosity [m²/s]. The left side of each momentum equation is local acceleration plus convective acceleration; the right side is pressure acceleration plus viscous diffusion. Pressure is an unknown that helps enforce continuity, not an independently prescribed field throughout the cavity.

Scale position by cavity length $L$, velocity by lid speed $U$, time by $L/U$, and pressure difference by $\rho U^2$. From here onward, variables without a subscript are dimensionless, $Re=UL/\nu$, and a subscript denotes differentiation: for example $u_x=\partial u/\partial x$. The coupled system becomes

$$
\boxed{\begin{aligned}
u_x+v_y&=0,\\
u_t+uu_x+vu_y&=-p_x+Re^{-1}(u_{xx}+u_{yy}),\\
v_t+uv_x+vv_y&=-p_y+Re^{-1}(v_{xx}+v_{yy}).
\end{aligned}}
$$

### 2.2 Satisfy continuity through a streamfunction

For this two-dimensional, simply connected cavity, define

$$u=\psi_y,\qquad v=-\psi_x.$$

Smooth mixed derivatives commute, so $u_x+v_y=\psi_{yx}-\psi_{xy}=0$. Define out-of-plane vorticity by

$$\omega=v_x-u_y=-\psi_{xx}-\psi_{yy},\qquad \boxed{\nabla^2\psi=-\omega}.$$

The streamfunction $\psi$ gives velocity and streamlines; vorticity $\omega$ measures local rotation, with positive values counterclockwise. A region of high speed is not necessarily a region of high vorticity.

### 2.3 Remove pressure by taking the curl of momentum

Apply $\partial_x$ to vertical momentum and subtract $\partial_y$ of horizontal momentum. The time term becomes $\omega_t$; pressure cancels because $-p_{yx}+p_{xy}=0$. For convection, use the product rule before applying continuity:

$$
\begin{aligned}
&\partial_x(uv_x+vv_y)-\partial_y(uu_x+vu_y)\\
&=u(v_{xx}-u_{xy})+v(v_{xy}-u_{yy})
  +u_xv_x+v_xv_y-u_yu_x-v_yu_y\\
&=u\omega_x+v\omega_y
  +\underbrace{(u_x+v_y)}_{0}\underbrace{(v_x-u_y)}_{\omega}.
\end{aligned}
$$

Thus the **sum** of the final four product-rule terms is zero. Individual terms need not vanish. The viscous curl, in contrast, remains:

$$Re^{-1}[\partial_x\nabla^2v-\partial_y\nabla^2u]
=Re^{-1}\nabla^2\omega.$$

Therefore we advance

$$\boxed{\omega_t+u\omega_x+v\omega_y
=Re^{-1}(\omega_{xx}+\omega_{yy}).}$$

In three dimensions, vortex stretching contains $(\boldsymbol\omega\cdot\nabla)\boldsymbol u$. Here $\boldsymbol\omega=(0,0,\omega)$ and the strictly two-dimensional velocity is independent of $z$, so it is $\omega\,\partial_z\boldsymbol u=0$. This is a separate cancellation from $\omega\nabla\cdot\boldsymbol u=0$.

### 2.4 Why the viscous term vanishes in the pressure equation

This is a **different operation**: apply $\partial_x$ to horizontal momentum and **add** $\partial_y$ of vertical momentum. To see why the divergence of viscous diffusion is zero, keep all four terms:

$$
\begin{aligned}
\partial_x\nabla^2u+\partial_y\nabla^2v
&=\partial_xu_{xx}+\partial_xu_{yy}
  +\partial_yv_{xx}+\partial_yv_{yy}\\
&=\partial_{xx}u_x+\partial_{yy}u_x
  +\partial_{xx}v_y+\partial_{yy}v_y\\
&=\partial_{xx}(u_x+v_y)+\partial_{yy}(u_x+v_y)\\
&=\nabla^2\underbrace{(u_x+v_y)}_{0}=0.
\end{aligned}
$$

For example, $\partial_xu_{yy}=\partial_{yy}u_x$ and $\partial_yv_{xx}=\partial_{xx}v_y$ require smooth commuting mixed derivatives. **Neither diagonal pair alone is generally zero.** For $u=x^3y$, $v=-\tfrac32x^2y^2$, continuity holds, but $\partial_{xx}u_x+\partial_{yy}v_y=6y$ and the other pair is $-6y$; only the complete sum cancels.

The divergence of the time term is also zero, because $\partial_t(u_x+v_y)=0$. Expanding convection then yields

$$\boxed{\nabla^2p=-[u_x^2+2u_yv_x+v_y^2].}$$

This does **not** make the flow inviscid: viscous diffusion remains in momentum and vorticity, and pressure boundary conditions still depend on viscosity. Pressure recovery also needs boundary data and a pressure reference. The identities above hold for sufficiently smooth interior fields; the moving-lid corners are singular and require an explicit numerical convention. A discrete divergence or residual is small, not identically zero by algebra unless the discrete operators ensure that identity.

**Check your understanding:** Which operation gives $Re^{-1}\nabla^2\omega$? Which gives zero? Explain why setting each third derivative to zero would be incorrect.

The solver below advances vorticity, solves $\nabla^2\psi=-\omega$, recovers velocity, and updates wall vorticity until the flow becomes approximately steady.
"""


STARTING_SYSTEM = '## Start with conservation of mass and momentum\n\nWe seek horizontal velocity $u(x,y,t)$, vertical velocity $v(x,y,t)$ and\npressure $p(x,y,t)$. For a Newtonian fluid with constant density $\\rho$ and\ndynamic viscosity $\\mu$, the two-dimensional incompressible system is\n\n$$\\frac{\\partial u}{\\partial x}+\\frac{\\partial v}{\\partial y}=0,$$\n$$\\rho\\left(\\frac{\\partial u}{\\partial t}+u\\frac{\\partial u}{\\partial x}\n+v\\frac{\\partial u}{\\partial y}\\right)\n=-\\frac{\\partial p}{\\partial x}\n+\\mu\\left(\\frac{\\partial^2u}{\\partial x^2}+\\frac{\\partial^2u}{\\partial y^2}\\right),$$\n$$\\rho\\left(\\frac{\\partial v}{\\partial t}+u\\frac{\\partial v}{\\partial x}\n+v\\frac{\\partial v}{\\partial y}\\right)\n=-\\frac{\\partial p}{\\partial y}\n+\\mu\\left(\\frac{\\partial^2v}{\\partial x^2}+\\frac{\\partial^2v}{\\partial y^2}\\right).$$\n\nThe first equation conserves mass. The next two conserve momentum in $x$ and\n$y$. **Together they form the incompressible Navier–Stokes system.** Here\n$x,y$ are spatial coordinates, $t$ is time, $\\rho$ has units kg/m$^3$, $\\mu$\nhas units Pa s, velocity has units m/s and pressure has units Pa. The kinematic\nviscosity is $\\nu=\\mu/\\rho$. Scaling lengths by cavity side $L$ and velocities\nby lid speed $U$ gives $Re=UL/\\nu$.\n\nWe derive the streamfunction–vorticity form from this system below. The\nreference plots are evidence to inspect, not a replacement for that derivation.\n'

def update_notebook(path=None):
    path = Path(path or ROOT / "notebooks/week01/03_cavity_ghia.ipynb")
    nb = json.loads(path.read_text(encoding="utf-8"))
    matches = [c for c in nb["cells"] if c["cell_type"] == "markdown"
               and "## 2. Governing equations" in "".join(c["source"])]
    if not matches:
        matches = [c for c in nb["cells"] if c["cell_type"] == "markdown"
                   and "## 2. From Navier" in "".join(c["source"])]
    if len(matches) != 1:
        raise ValueError("Expected one Week 1 governing-equations lesson")
    matches[0]["source"] = GOVERNING_EQUATIONS.splitlines(keepends=True)
    if not any(c.get("metadata", {}).get("week1_starting_system") for c in nb["cells"]):
        nb["cells"].insert(1, {"cell_type":"markdown", "id":"week1-starting-system", "metadata":{"week1_starting_system":True}, "source":STARTING_SYSTEM.splitlines(keepends=True)})
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    update_notebook()
