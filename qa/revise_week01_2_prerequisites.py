"""Maintain a scalar-first teaching order in the Week 1.2 lecture."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'lectures/source'

SIMPLE=r'''\topic{8}{SIMPLE: assemble the steady coupling iteration}
SIMPLE means Semi-Implicit Method for Pressure-Linked Equations. It combines
the momentum solve, pressure correction and conservative velocity update
already derived. Begin iteration $k$ with velocity $q^k$ and pressure $p^k$.

\begin{enumerate}
\item \textbf{Build momentum.} Evaluate convection from $q^k$ and assemble
the original steady matrix $A$ and known source $b$. Apply the velocity
under-relaxation rule from Section 7 to obtain $A^{\rm rel}$ and $b^{\rm rel}$.
\item \textbf{Predict velocity.} Solve the relaxed momentum equations using
the current pressure. The result is $q^*$, which may violate continuity.
\item \textbf{Calculate the imbalance.} Apply $D$ to obtain one residual
$r_{c,P}^*$ per pressure cell. Use the relaxed inverse diagonal $d$ to build
the pressure matrix $L=-DdG$.
\item \textbf{Solve pressure correction.} Solve $Lp'=-Dq^*$ with the wall
correction conditions and one pressure reference, as derived in Section 6.
\item \textbf{Update the fields.} Apply the full face-velocity correction
and the under-relaxed pressure update shown below.
\item \textbf{Test both equations.} Rebuild the unrelaxed momentum coefficients
from the corrected velocity and check the momentum and continuity tolerances.
If either fails, use the corrected fields as the next estimates and repeat.
\end{enumerate}
In compact notation, the solve and updates are
\begin{align*}
A^{\rm rel}q^*&=b^{\rm rel}-Gp^k,\\
Lp'&=-Dq^*,\\
q^{k+1}&=q^*-dGp',\\
p^{k+1}&=p^k+\alpha_p p'.
\end{align*}
The stored pressure increment is $\alpha_p p'$, whereas the unrelaxed
correction is $\Delta p=p'$. Velocity receives its full correction so that
the cell balances are conservative. Prescribed wall velocities remain fixed;
a common pressure offset is removed without changing pressure gradients.

The retained baseline uses $\alpha_u=0.7$, $\alpha_p=0.3$,
$\varepsilon_m=10^{-6}$ and $\varepsilon_c=10^{-9}$. These numerical controls
do not change the Reynolds number. This SIMPLE branch solves a steady
problem; increasing $k$ does not advance physical time.
'''

PISO=r'''\topic{10}{PISO: several corrections in one physical time step}
PISO means Pressure-Implicit with Splitting of Operators. Start with the
accepted old-time velocity $q^n$ and pressure $p^n$. The next accepted fields
will belong to time $n+1$, one physical interval $\Delta t$ later.

First assemble the transient momentum equation from Section 9 and solve
\[
Aq^*=b-Gp^{\rm guess}.
\]
The pressure $p^{\rm guess}$ is the initial pressure estimate for this step,
normally the preceding accepted pressure. The source $b$ includes the fixed
old-time contribution $q^n/\Delta t$ and known wall contributions. The
predictor $q^*$ satisfies this momentum solve, but can still violate continuity.

The next operations improve pressure and velocity within this same step.
Write $q^{(0)}=q^*$ and use $m$ to count inner corrections. Parenthesized
superscripts identify a correction estimate, not a physical time or a power.

\begin{center}\begin{tabular}{lp{.72\linewidth}}\toprule
Quantity & Behavior during this PISO loop\\\midrule
$q^n$ & Accepted old-time velocity; fixed throughout the step.\\
$A,b$ & Transient momentum coefficients and source; assembled once.\\
$d,R$ & Reciprocal diagonal and off-diagonal part of this same $A$; fixed.\\
$L=-DdG$ & Pressure matrix from the fixed reciprocal diagonal; fixed.\\
$q^{(m)},p^{(m)}$ & Current estimates at the new time level; updated by corrections.\\
$H^{(m)}$ & Known source plus the latest neighbor-velocity contributions; recomputed.\\\bottomrule
\end{tabular}\end{center}
The key distinction is between a fixed momentum matrix and changing velocity
values within it. Updating the neighbor contributions is what makes another
pressure correction different from solving an identical equation again.
\newpage
\topic{10 continued}{Form each inner pressure solve and velocity update}
Given the current estimate $q^{(m)}$, collect its neighbor contributions:
\[
H^{(m)}=b-Rq^{(m)}.
\]
The momentum-row reconstruction for a new pressure estimate is
\[
q^{(m+1)}=d[H^{(m)}-Gp^{(m+1)}].
\]
Require its cell divergence to vanish. Applying $D$ gives
\begin{align*}
0&=D(dH^{(m)})-DdG p^{(m+1)},\\
Lp^{(m+1)}&=-D(dH^{(m)}),\qquad L=-DdG.
\end{align*}
Thus one inner correction consists of recomputing $H^{(m)}$, solving this
pressure equation and reconstructing all face velocities. The next
correction repeats these operations with the newly reconstructed velocity.

\textbf{Full pressure or increment?} In this implementation,
$p^{(m+1)}$ is the full pressure field in the selected reference convention.
It replaces the previous pressure estimate. It is not SIMPLE's additive
correction $p'$, and must not be added to the previous pressure.

Let $N_i$ denote the chosen number of inner corrections. After those
corrections, accept $q^{n+1}=q^{(N_i)}$ and $p^{n+1}=p^{(N_i)}$.
Only this acceptance advances the physical time. The retained baseline uses
$N_i=2$ and does not apply SIMPLE's under-relaxation in this transient branch.

The convection coefficients remain those assembled for the predictor.
Further inner corrections improve coupling for that fixed matrix. They do
not rebuild convection from the latest new-time velocity; PIMPLE adds an
outer iteration for that purpose.
'''

def write(path,text):
    if path.read_text(encoding='utf-8')!=text:path.write_text(text,encoding='utf-8',newline='\n')

def main():
    path=SOURCE/'week01_2_pressure_velocity.tex';s=path.read_text(encoding='utf-8')
    s=s.replace(r'\input{week01_2_reading_map.tex}',r'\input{week01_2_introduction.tex}')
    s=s.replace(r'\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black}',
        r'\definecolor{linkblue}{HTML}{0645AD}'+'\n'+r'\hypersetup{colorlinks=true,linkcolor=linkblue,urlcolor=linkblue}')
    write(path,s)
    path=SOURCE/'week01_2_pressure_velocity_core.tex';s=path.read_text(encoding='utf-8')
    start=s.find(r'\topic{4}{Momentum coefficients and matrix notation}')
    if start!=-1:
        end=s.index(r'\input{week01_2_shared_face_visual.tex}',start)
        s=s[:start]+r'\input{week01_2_momentum_notation.tex}'+'\n'+r'\newpage'+'\n'+r'\input{week01_2_pressure_correction_intro.tex}'+'\n'+r'\newpage'+'\n'+s[end:]
    start=s.find(r'\topic{8}{SIMPLE: steady pressure--velocity coupling}')
    if start!=-1:
        end=s.index(r'\topic{8 continued}',start)
        s=s[:start]+r'\input{week01_2_iteration_preparation.tex}'+'\n'+r'\newpage'+'\n'+SIMPLE+r'\newpage'+'\n'+s[end:]
    start=s.find(r'\topic{10}{PISO: pressure corrections for a frozen momentum matrix}')
    if start!=-1:
        end=s.index(r'\topic{10 continued}{PISO: follow the loop}',start)
        s=s[:start]+PISO+r'\newpage'+'\n'+s[end:]
    preamble=r'''Before writing the nested equations, distinguish their labels and stored fields:
\begin{center}\begin{tabular}{lp{.72\linewidth}}\toprule
Label or field & Meaning\\\midrule
$n,n+1$ & Old and new physical time levels.\\
$\ell$ & Outer pass: rebuild momentum coefficients using the latest velocity estimate.\\
$m$ & Inner correction within one outer pass: update pressure and face velocity.\\
$q^n$ & Accepted old-time velocity, saved once and kept fixed.\\
$q_{\rm lag}^{(\ell)},p_{\rm lag}^{(\ell)}$ & Velocity and pressure estimates entering outer pass $\ell$. The subscript lag identifies inputs carried from the preceding pass.\\
$q^{(\ell,0)}$ & Momentum predictor at the start of outer pass $\ell$.\\
$q^{(\ell,m)}$ & New-time velocity estimate after $m$ inner corrections in pass $\ell$.\\\bottomrule
\end{tabular}\end{center}
For example, $(\ell,m)=(2,1)$ means the first inner correction of the second
outer pass. It remains within the same physical step. The first outer pass
starts from the accepted old-time fields; subsequent passes start from the
latest corrected new-time fields.
\newpage
\topic{12 continued}{Assemble an outer pass and carry its fields forward}
'''
    anchor='At the start of outer pass $\\ell$, assemble convection'
    if 'distinguish their labels and stored fields' not in s:s=s.replace(anchor,preamble+anchor)
    s=s.replace('With $I$ denoting the identity matrix, this gives new matrices and sources',
        'Here $A_{\\rm space}$ contains the spatial convection and diffusion coefficients,\n'
        '$b_{\\rm wall}$ contains prescribed wall contributions, and $I$ is the identity\n'
        'matrix (ones on its diagonal, zeros elsewhere). Adding the backward-Euler terms gives')
    write(path,s)
    path=SOURCE/'week01_2_shared_face_visual.tex';s=path.read_text(encoding='utf-8')
    s=s.replace('Return to the reading-map mass-balance example,','Return to the predictor mass-balance example in Section 5,')
    write(path,s)

if __name__=='__main__':main()
