"""Build deterministic teaching diagrams and insert self-contained notebook lessons.

No retained CFD solution or neural-network training result is recomputed here.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nbformat

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'lectures/source/week01_2_assets'
ML = ROOT/'lectures/source/week02_assets'
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,
                     'mathtext.fontset':'stix','pdf.fonttype':42})
BLUE, GREEN, RED = '#1859a9', '#177548', '#b32b2b'

def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(folder/f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

def arrow(ax,a,b,color='black',lw=1.5):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=lw))

def checkerboard():
    h=.25;x=np.arange(7)*h;p=(-1.0)**np.arange(7)
    centered=(p[2:]-p[:-2])/(2*h);faces=(p[1:]-p[:-1])/h
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    axes[0].plot(x,p,'o--',color='black',label='Pressure samples')
    axes[0].set(xlabel='Local x coordinate',ylabel=r'$p_i$',title='Synthetic alternating pressure')
    axes[1].plot(x[1:-1],centered,'o',color=BLUE,label='Naive collocated central gradient')
    axes[1].plot((x[1:]+x[:-1])/2,faces,'x',color=RED,ms=8,label='Shared-face pressure gradient')
    axes[1].axhline(0,color='.7',lw=.8)
    axes[1].set(xlabel='Local x coordinate',ylabel='Discrete pressure gradient',title=r'$h=0.25$: zero versus $\pm8$')
    for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.18)
    fig.tight_layout();save(fig,ASSETS,'checkerboard_pressure')

def geometry():
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax in axes:
        ax.set_aspect('equal');ax.set_xlim(-.85,1.85);ax.set_ylim(-1.35,1.4)
        ax.set_xlabel(r'$x/h$');ax.set_ylabel(r'$y/h$')
        ax.set_xticks([-.5,0,.5,1,1.5]);ax.set_yticks([-1,-.5,0,.5,1])
        for x in [-.5,.5,1.5]:ax.plot([x,x],[-1.2,1.2],color='.8',lw=.8)
        for y in [-.5,.5]:ax.plot([-.7,1.7],[y,y],color='.8',lw=.8)
        ax.add_patch(Rectangle((0,-.5),1,1,facecolor=BLUE,alpha=.07,edgecolor='none'))
        ax.add_patch(Rectangle((0,-.5),1,1,fill=False,edgecolor=BLUE,lw=2))
        ax.scatter([0,1],[0,0],s=65,marker='s',facecolors='white',edgecolors='black',zorder=4)
        ax.text(0,-.19,r'$p_P$',ha='center');ax.text(1,-.19,r'$p_E$',ha='center')
        ax.text(.5,-1.25,'Blue boundary: horizontal-momentum volume',ha='center',fontsize=10)
    ax=axes[0];ax.set_title('(a) Stored samples')
    for x,y,s in [(.5,0,r'$u_e$'),(-.5,0,r'$u_{\rm W}$'),(1.5,0,r'$u_{\rm E}$'),(.5,1,r'$u_{\rm N}$'),(.5,-1,r'$u_{\rm S}$')]:
        ax.scatter(x,y,c=BLUE,s=55,zorder=5);ax.text(x+.09,y+.12,s,color=BLUE)
    for x,y,s in [(0,.5,r'$v_{n,P}$'),(1,.5,r'$v_{n,E}$'),(0,-.5,r'$v_{s,P}$'),(1,-.5,r'$v_{s,E}$')]:
        ax.scatter(x,y,c=GREEN,marker='^',s=50,zorder=5);ax.text(x+.08,y+.11,s,color=GREEN,fontsize=11)
    ax=axes[1];ax.set_title('(b) Interpolated horizontal component')
    for x,y in [(-.5,0),(1.5,0),(.5,1),(.5,-1)]:
        ax.plot([.5,x],[0,y],ls='--',color=BLUE,alpha=.5,lw=1)
        ax.scatter(x,y,s=40,facecolors='white',edgecolors=BLUE,zorder=4)

    for x,y,s,dx,dy in [(0,0,r'$\widehat u_{\rm W}$',-.25,.18),(1,0,r'$\widehat u_{\rm E}$',.17,.18),(.5,.5,r'$\widehat u_{\rm N}$',.15,.08),(.5,-.5,r'$\widehat u_{\rm S}$',.15,-.22)]:
        ax.scatter(x,y,c=RED,marker='x',s=85,zorder=6);ax.text(x+dx,y+dy,s,color=RED)
    ax.scatter(.5,0,c=BLUE,s=55);ax.text(.58,.08,r'$u_e$',color=BLUE)
    ax.text(.5,1.06,'A hat marks interpolation.\nIts position is a momentum-volume face.',ha='center',fontsize=11)
    fig.tight_layout();save(fig,ASSETS,'hat_locations')

    fig,ax=plt.subplots(figsize=(10,4.5));ax.set_xlim(-.6,1.6);ax.set_ylim(-.65,1.1);ax.axis('off')
    ax.add_patch(Rectangle((0,-.5),1,1,fill=False,edgecolor=BLUE,lw=2))
    ax.plot([0,1],[.5,.5],color=RED,lw=3)
    arrow(ax,(.5,.15),(.5,.88),GREEN,2.5)
    arrow(ax,(.2,.5),(.8,.5),BLUE,2.5)
    ax.text(.56,.87,r'$c_{\rm N}=0.02$: flow crossing the north face',color=GREEN,fontsize=12)
    ax.text(.13,.33,r'$\widehat u_{\rm N}=0.06$: horizontal component carried by that flow',color=BLUE,fontsize=12)
    ax.text(.5,-.22,r'North-face flux per unit density: $h c_{\rm N}\widehat u_{\rm N}$',ha='center')
    ax.text(.5,-.43,r'$0.25\times0.02\times0.06=0.00030$',ha='center')
    ax.text(.5,1.04,'Crossing direction and transported component are different concepts',ha='center',weight='bold')
    save(fig,ASSETS,'north_face_transport')

    fig,ax=plt.subplots(figsize=(10,3.7));ax.axis('off');ax.set_xlim(-.5,2.5);ax.set_ylim(-.9,.9)
    for x in [0,1]:ax.add_patch(Rectangle((x,-.45),1,.9,fill=False,edgecolor='.5'))
    for x,t in [(.5,r'$p_P^\prime$'),(1.5,r'$p_E^\prime$')]:ax.scatter(x,0,c='black',s=45);ax.text(x,-.25,t,ha='center')
    ax.scatter(1,0,c=BLUE,s=60);arrow(ax,(.95,.08),(1.3,.08),BLUE)
    arrow(ax,(1.2,-.1),(.8,-.1),RED)
    ax.text(1,-.31,r"$u_e^\prime<0$",ha='center',color=RED)
    ax.text(1,.5,r'One shared face: $u_e=u_e^*+u_e^\prime$',ha='center')
    ax.text(1,-.63,r'$u_e^\prime=-d_e(p_E^\prime-p_P^\prime)/h,\qquad d_e>0$',ha='center')
    ax.text(1,-.85,r'$h=0.25,\ d_e=0.5,\ p_E^\prime-p_P^\prime=0.01:\quad u_e^*=0.12\ \rightarrow\ u_e=0.10$',ha='center',fontsize=11)
    save(fig,ASSETS,'shared_face_correction')

def loops():
    # Geometry is fixed to keep arrows outside every text box.
    descriptions={
      'simple_visual':('SIMPLE: steady nonlinear iteration',[
        'Start with pressure and face velocities',
        'Freeze convection coefficients; solve momentum\nusing current pressure -> predictor velocity',
        'Form mass imbalance; solve pressure-correction equation',
        'Correct shared face velocities; relax pressure',
        'Check continuity AND momentum defects',
        'Converged: retain steady solution'],1,'Not converged: update coefficients'),
      'piso_visual':('PISO: corrections within one physical time step',[
        'Keep old-time velocity fixed at time n',
        'Assemble transient momentum; solve predictor\nfor the new time level n+1',
        'First pressure solve; correct pressure and face velocities',
        'Re-evaluate correction terms with latest velocity;\nsolve pressure again and correct velocities',
        'Check time-step solve; finish configured corrections',
        'Accept new time level; advance n -> n+1'],3,'Additional pressure corrections'),
      'pimple_visual':('PIMPLE: outer nonlinear loop around PISO',[
        'Begin physical step; keep old-time velocity fixed',
        'Outer iteration: rebuild convection coefficients\nfrom latest new-time estimate; solve momentum',
        'Inner pressure solve; correct pressure and face velocity',
        'Repeat inner corrections with updated estimate\nwhile the momentum coefficients remain frozen',
        'Check nonlinear momentum and continuity defects',
        'Accept step only after solve targets are satisfied'],1,'More outer iterations: rebuild momentum')}
    for name,(title,labels,target,looptext) in descriptions.items():
        fig,ax=plt.subplots(figsize=(10,7));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(-.1,7)
        ax.text(4.5,6.8,title,ha='center',weight='bold',fontsize=15)
        for i,label in enumerate(labels):
            y=5.8-i
            ax.add_patch(Rectangle((.6,y-.29),7.1,.64,facecolor='#f4f7fa',edgecolor=BLUE,lw=1.3))
            ax.text(4.15,y+.03,label,ha='center',va='center',fontsize=12)
            if i<5:arrow(ax,(4.15,y-.3),(4.15,y-.64))
        yfrom=5.8-4;yto=5.8-target
        ax.plot([7.7,8.35,8.35],[yfrom,yfrom,yto],c=RED,lw=1.4)
        arrow(ax,(8.35,yto),(7.7,yto),RED)
        ax.text(8.65,(yfrom+yto)/2,looptext,rotation=90,ha='center',va='center',color=RED,fontsize=11)
        if name=='pimple_visual':
            # Inner loop is distinct from the red coefficient-rebuild loop.
            ax.plot([.6,.17,.17],[2.8,2.8,3.8],c=GREEN,lw=1.4)
            arrow(ax,(.17,3.8),(.6,3.8),GREEN)
            ax.text(-.05,3.3,'Inner corrections',rotation=90,ha='center',va='center',color=GREEN,fontsize=10)
        if name!='simple_visual':
            ax.text(4.15,-.03,'Inner and outer iterations do not advance physical time.',ha='center',fontsize=11)
        else:ax.text(4.15,-.03,'A steady iteration is not a physical time step.',ha='center',fontsize=11)
        save(fig,ASSETS,name)

DEMO = '''# A controlled scalar example, not the fluid surrogate's training result.
import numpy as np
import matplotlib.pyplot as plt
eta_values = [0.05, 0.5, 1.5, 2.1]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for eta in eta_values:
    theta = 1.0
    path = [theta]
    for k in range(30):
        gradient = theta - 3.0
        theta -= eta * gradient
        path.append(theta)
    path = np.asarray(path)
    axes[0].plot(path, label=f"learning rate = {eta}")
    axes[1].semilogy(0.5 * (path - 3.0)**2, label=f"learning rate = {eta}")
axes[0].axhline(3.0, color="black", linestyle="--", label="minimum")
axes[0].set(xlabel="Parameter update k", ylabel="Parameter theta")
axes[1].set(xlabel="Parameter update k", ylabel="Loss C(theta)")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout()
plt.show()

# One Adam update, using the Keras epsilon-hat convention.
theta, m, v = 1.0, 0.0, 0.0
eta, beta1, beta2, epsilon = 0.1, 0.9, 0.999, 1e-7
k = 1
g = theta - 3.0
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g*g
m_hat, v_hat = m/(1-beta1**k), v/(1-beta2**k)
alpha = eta*np.sqrt(1-beta2**k)/(1-beta1**k)
theta -= alpha*m/(np.sqrt(v)+epsilon)
print(f"g={g:.3f}, m={m:.3f}, v={v:.6f}")
print(f"bias-corrected m={m_hat:.3f}, v={v_hat:.3f}")
print(f"theta after Adam update = {theta:.8f}")
assert np.isclose(theta, 1.099999841886367, rtol=0, atol=1e-12)
'''

ADAM_MD = r'''### Learning rate and Adam: what changes at each update?

A **trainable parameter** is a weight or bias learned from data. A **physical parameter**, such as Reynolds or Knudsen number, describes the flow. A **hyperparameter**, such as learning rate, batch size or network width, is chosen for training and is not updated by Adam. Collect all parameters in $\theta$. A **batch** is a subset of the training samples; its mean squared error is $C_B(\theta)$. The gradient $g_k=\nabla_\theta C_B(\theta_{k-1})$ gives the local uphill direction. An optimizer moves the parameters downhill. Here $k=1,2,\ldots$ counts **parameter updates**, not epochs.

For gradient descent, the update is

$$\theta_k=\theta_{k-1}-\eta g_k.$$

The learning rate $\eta$ sets the step scale. For the particular scalar loss $C(\theta)=\tfrac12(\theta-3)^2$, $g=\theta-3$. Starting from $\theta_0=1$, one update with $\eta=0.5$ gives $1-0.5(-2)=2$. The error obeys $e_k=(1-\eta)e_{k-1}$, so this **quadratic gradient-descent example** converges for $0<\eta<2$. That bound is not a general rule for Adam or a neural network.

Adam maintains two arrays, each with the same shape as the parameters. With $m_0=v_0=0$,

$$m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,$$
$$v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2.$$

All products, squares, square roots and divisions here act **component by component**. $m_k$ averages recent signed gradients; $v_k$ averages their squares. The name $v_k$ here is optimizer memory, not a fluid velocity. It averages squared gradients; it is not a statistical variance. Starting these averages at zero biases their early magnitudes downward. Bias correction gives

$$\widehat m_k=\frac{m_k}{1-\beta_1^k},\qquad \widehat v_k=\frac{v_k}{1-\beta_2^k}.$$

The familiar Adam form is $\theta_k=\theta_{k-1}-\eta\widehat m_k/(\sqrt{\widehat v_k}+\epsilon_{\mathrm{alg}})$. The **Keras implementation used below** places its small positive constant differently:

$$\alpha_k=\eta\frac{\sqrt{1-\beta_2^k}}{1-\beta_1^k},\qquad
\theta_k=\theta_{k-1}-\alpha_k\frac{m_k}{\sqrt{v_k}+\epsilon_K}.$$

Here $\beta_1=0.9$, $\beta_2=0.999$ and $\epsilon_K=10^{-7}$ are the default Keras values. The positive $\epsilon_K$ prevents division by zero. It is Keras's *epsilon hat*, not exactly the constant in the first bias-corrected formula; the equivalent denominator there contains $\epsilon_K/\sqrt{1-\beta_2^k}$. This distinction only matters for reproducing numerical updates exactly.

**One update by hand.** For the quadratic example, $g_1=-2$, $m_1=-0.2$, $v_1=0.004$, $\widehat m_1=-2$ and $\widehat v_1=4$. The teaching calculation uses $\eta=0.1$ and produces $\theta_1\approx1.09999984$. Its larger learning rate makes the arithmetic visible; the surrogate below retains $\eta=0.001$.

**Read the training call.** `Adam(learning_rate=1e-3)` sets the base learning rate to 0.001; Adam adapts individual step scales but does not remove this choice. `batch_size=64` means up to 64 training samples per update. This notebook has 3500 training samples and 750 samples in each of the validation and test sets. The 4-32-32-1 network has $4(32)+32+32(32)+32+32(1)+1=1249$ trainable parameters. An **epoch** visits the full training set once and contains approximately $\lceil N_{train}/64\rceil$ updates, including a smaller last batch. Here that is 55 updates: 54 full batches and a final batch of 44 samples. `epochs=500` is a maximum, giving at most 27,500 updates. `EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)` stops after 30 epochs without a lower validation loss (the default improvement threshold is zero) and restores the best weights. For example, if epoch 120 has the lowest validation loss and the next 30 epochs do not improve it, training ends after epoch 150 and restores epoch-120 weights. It does not lower the learning rate.

**Exercise.** Run the cell below. Explain the slow approach, fast approach, decaying oscillations and growing oscillations of gradient descent using the factor $1-\eta$. Explain why rates 0.5 and 1.5 have the same loss history but different parameter paths. Then change the toy Adam rate to 0.01 and compare its first step. For an independent neural-surrogate rate trial, call `keras.utils.set_random_seed(7)` immediately before constructing a fresh model, fresh Adam optimizer and fresh early-stopping callback. Change only `learning_rate` in the training call, keeping the split, training-only scaling, architecture and batch policy fixed. Continuing an already-trained model also carries optimizer memory into the next trial. Compare several matched seeds; select using validation loss and reserve the test data for the final evaluation. A small training loss alone does not establish physical accuracy.

Sources: [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980); [Keras Adam documentation](https://keras.io/api/optimizers/adam/); [Keras seed utility](https://keras.io/api/utils/python_utils/).
'''

HAT_MD = r'''### Locate a stored velocity before interpolating it

![Stored samples and interpolated momentum-face values](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/hat_locations.png)

Use local coordinates with the origin at pressure center $P$. The axes are measured in grid spacings $h$. Pressure centers are $P=(0,0)$ and $E=(h,0)$. The stored horizontal velocity $u_e$ lies at $(h/2,0)$, on their shared **pressure-cell face**. Its own **momentum control volume** is shifted: its boundaries are $x=0,h$ and $y=-h/2,h/2$. The blue boundary encloses this second volume.

The blue circles are stored horizontal velocity samples. The red crosses are the horizontal components needed at the **momentum-volume faces**. In these face-flux formulas, a hat means interpolation to a momentum-volume face; it does not mean pressure-cell center. The east/west hat locations coincide geometrically with pressure centers, while the north/south hat locations lie at pressure-cell corners. Pressure and horizontal velocity remain different fields even where their evaluation locations coincide.

For central interpolation, each red cross lies halfway between two blue samples:

$$\widehat u_{\rm E}=\frac{u_e+u_{\rm E}}2,\quad \widehat u_{\rm W}=\frac{u_{\rm W}+u_e}2,$$
$$\widehat u_{\rm N}=\frac{u_e+u_{\rm N}}2,\quad \widehat u_{\rm S}=\frac{u_{\rm S}+u_e}2.$$

Capital directional subscripts on $u$ identify neighboring stored **horizontal components**. Thus $u_{\rm N}$ is not a vertical velocity. The normal speed carrying fluid across the north face is instead $c_{\rm N}=(v_{n,P}+v_{n,E})/2$.

![North-face crossing velocity and transported horizontal component](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/north_face_transport.png)

For unit depth, the integrated north-face transport per unit density is $h c_{\rm N}\widehat u_{\rm N}$. North/east fluxes enter the outward balance with plus signs; south/west enter with minus signs. Positive $c_{\rm N}$ means motion in the positive $y$ direction; it carries the horizontal component $\widehat u_{\rm N}$. During a linear predictor solve, the advecting speeds $c$ are frozen at the current estimate, whereas the transported components belong to the momentum unknowns. Their east/west interpolation formulas look alike but play different roles.

The cell below is an arithmetic example at one interior volume, not a computed cavity solution. All its numbers are nondimensional.
'''

HAT_CODE='''import numpy as np
h = 0.25
u_e, u_E, u_W, u_N, u_S = 0.04, 0.12, 0.02, 0.08, -0.02
u_hat = np.array([(u_e+u_E)/2, (u_W+u_e)/2,
                  (u_e+u_N)/2, (u_S+u_e)/2])  # E, W, N, S
v_nP, v_nE, v_sP, v_sE = 0.03, 0.01, 0.05, 0.09
c = np.array([0.08, 0.03, (v_nP+v_nE)/2, (v_sP+v_sE)/2])
outward_sign = np.array([1, -1, 1, -1])
unsigned_flux = h*c*u_hat
print("Order: E, W, N, S")
print("Interpolated horizontal components:", u_hat)
print("Face transport h*c*u_hat:", unsigned_flux)
print("Net outward momentum transport:", outward_sign @ unsigned_flux)
print("Transport divided by volume h^2:", (outward_sign @ unsigned_flux)/h**2)
assert np.allclose(u_hat, [0.08, 0.03, 0.06, 0.01])
assert np.isclose(outward_sign @ c, 0.0)  # momentum-volume mass balance
assert np.isclose((u_e-u_W+v_nP-v_sP)/h, 0.0)  # pressure cell P
assert np.isclose((u_E-u_e+v_nE-v_sE)/h, 0.0)  # pressure cell E
assert np.isclose(outward_sign @ unsigned_flux, 0.0015)
'''

CHECKERBOARD_MD=r'''### Why stagger the variables?

![Alternating pressure and the gradients seen by two grid arrangements](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/checkerboard_pressure.png)

Consider a synthetic interior grid patch with $p_i=(-1)^i$: pressure alternates between +1 and -1. A naive collocated central gradient at the same point as pressure is

$$\frac{p_{i+1}-p_{i-1}}{2h}=0,$$

because both neighbors have the same value. This alternating pressure mode is invisible to that particular gradient, even though adjacent pressures differ. At the shared velocity face, the MAC grid instead uses

$$\frac{p_{i+1}-p_i}{h}=\pm\frac{2}{h}=\pm8\quad\text{for }h=0.25.$$

The face momentum equation therefore responds to the alternating pressure. This is one reason to stagger pressure and velocity. Properly stabilized collocated methods can also work; their interpolation treatment is a separate topic. This diagram is a discrete diagnostic pattern, not a cavity solution.
'''

CHECKERBOARD_CODE='''import numpy as np
h = 0.25
p_patch = (-1.0)**np.arange(7)
center_gradient = (p_patch[2:]-p_patch[:-2])/(2*h)
face_gradient = (p_patch[1:]-p_patch[:-1])/h
print("Naive pressure-center gradients:", center_gradient)
print("Shared-face gradients:", face_gradient)
assert np.all(center_gradient == 0.0)
assert np.all(np.abs(face_gradient) == 8.0)
'''

def insert(nb, anchor, cells):
    for c in cells:
        old=next((x for x in nb.cells if x.id==c.id),None)
        if old is not None and old.source==c.source: c.update(old)
    ids={c.id for c in cells}
    nb.cells=[c for c in nb.cells if c.id not in ids]
    idx=next(i for i,c in enumerate(nb.cells) if c.id==anchor)+1
    nb.cells[idx:idx]=cells

def md(id,text):return nbformat.v4.new_markdown_cell(text,id=id)
def code(id,text):return nbformat.v4.new_code_cell(text,id=id)

def notebooks():
    p=ROOT/'notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb';nb=nbformat.read(p,4)
    insert(nb,'4e4d456b',[md('visual-adam-lesson',ADAM_MD),code('visual-adam-experiment',DEMO)])
    insert(nb,'58990ed8',[md('visual-extrapolation-validation',r'''The validation subset in this experiment is also drawn from the low-Knudsen-number training range. Early stopping therefore selects weights for performance within that range; it cannot detect high-Knudsen-number extrapolation error. The high-range test is evaluated afterward. Do not repeatedly tune the model on those test results and still describe them as an untouched final test.
''')])
    p.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')
    p=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb';nb=nbformat.read(p,4)
    insert(nb,'pressure-lesson-2',[md('visual-checkerboard',CHECKERBOARD_MD),code('visual-checkerboard-check',CHECKERBOARD_CODE)])
    insert(nb,'pressure-lesson-2',[md('visual-reading-map',r'''### A first route through pressure-velocity coupling

1. Locate pressure and each velocity component on the staggered grid.
2. Add the four outward face fluxes of one pressure cell.
3. Solve momentum using a guessed pressure; call this velocity the predictor.
4. Compute its mass imbalance. Choose pressure corrections that remove this imbalance through shared-face velocity corrections.
5. Repeat the appropriate coupling loop and check both momentum and continuity.

For $h=0.25$ and predicted velocities $(u_e^*,u_w^*,v_n^*,v_s^*)=(0.12,0.08,0.01,0.03)$, the outward volume flux per unit depth is $Q_P^*=h(0.12-0.08+0.01-0.03)=0.005$. Dividing by area $h^2$ gives the divergence $r_{c,P}^*=0.08$. The correction must contribute $-0.005$ to the integrated outward flux. Neither quantity is a pressure correction.

Companion explanations: [CFD Direct: momentum -> face flux -> pressure equation](https://doc.cfd.direct/notes/cfd-general-principles/pressure-velocity-coupling), [OpenFOAM: SIMPLE and PISO correction patterns](https://www.openfoam.com/documentation/user-guide/6-solving/6.3-solution-and-algorithm-control), and [CFD Direct: the PIMPLE outer loop](https://doc.cfd.direct/notes/cfd-general-principles/the-pimple-algorithm). These include collocated software variants; this notebook derives its own staggered-grid balances.
''')])
    insert(nb,'pressure-lesson-3',[md('visual-hat-geometry',HAT_MD),code('visual-hat-arithmetic',HAT_CODE)])
    insert(nb,'pressure-lesson-5',[md('visual-shared-face',r'''### One face correction is shared by two pressure cells

![Pressure difference and shared-face velocity correction](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/shared_face_correction.png)

$u_e^*$ is the momentum predictor, the star marking a provisional solution. $p_P^\prime$ and $p_E^\prime$ are pressure corrections, the prime marking an increment rather than a derivative. In the volume-normalized momentum equation, $d_e=1/a_e>0$ is the reciprocal diagonal coefficient. The velocity correction is

$$u_e^\prime=-d_e\frac{p_E^\prime-p_P^\prime}{h},\qquad u_e=u_e^*+u_e^\prime.$$

Return to the reading-map example: $u_e^*=0.12$ and $r_{c,P}^*=0.08$. Prescribe the normalized diagonal $a_e=2$, hence $d_e=0.5$, and suppose the pressure solve gives $p_E^\prime-p_P^\prime=0.01$. Then $u_e^\prime=-0.02$ and $u_e=0.10$. Holding the other faces fixed reduces cell $P$'s outward flux by 0.005, removing its imbalance, but increases cell $E$'s outward flux by 0.005. The diagonal is prescribed for this illustration, independently of the earlier convection example.

An increasing pressure correction toward the east produces a westward velocity correction. This single face velocity leaves cell $P$ and enters cell $E$ with opposite outward signs; compute and store it once. The pressure solve chooses corrections whose face fluxes cancel each cell's predictor mass imbalance. In these face-flux formulas, a hat marks interpolation, a star marks a predictor and a prime marks a correction. The later PIMPLE section uses the subscript $\mathrm{lag}$ for latest available outer-iteration fields, keeping the hat for interpolation.
''')])
    for anchor,name in [('pressure-lesson-9','simple'),('pressure-lesson-11','piso'),('pressure-lesson-14','pimple')]:
        explanation={
          'simple':'The return arrow is a new steady nonlinear iteration. The pressure correction makes the current face fluxes conservative, but momentum must be checked again after the pressure and convection estimates change.',
          'piso':'All pressure corrections occur within the same physical time step. The old-time velocity in the transient term stays fixed. A further correction uses the latest velocity to improve the momentum coupling; solving the identical pressure equation repeatedly without updating its right-hand side is not an additional PISO correction.',
          'pimple':r'The fields $q_{\rm lag}^{(\ell)}$ and $p_{\rm lag}^{(\ell)}$ are the latest estimates entering outer pass $\ell$. They change between outer passes while the physical old-time field $q^n$ stays fixed. In this fixed-lid code, $b^{(\ell)}=b_{\rm wall}+q^n/\Delta t$ has the same value in every outer pass; only the convection-dependent matrix changes. The outer loop rebuilds the nonlinear momentum coefficients; the inner loop corrects pressure and velocity for those coefficients. Both loops stay at the same new time level. Only accepting the step advances physical time. The notebook also uses these correction patterns as steady iterations; that comparison does not establish transient accuracy.'}[name]
        insert(nb,anchor,[md('visual-loop-'+name,f'### Follow the {name.upper()} loop\n\n![{name.upper()} algorithm](https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/{name}_visual.png)\n\n'+explanation)])
    from integrate_student_review import COEFFICIENTS
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace(r'\widehat q',r'q_{\rm lag}').replace(r'\widehat p',r'p_{\rm lag}')
        if cell.id=='student-review-coefficient-normalization':cell.source=COEFFICIENTS
    p.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagrams',choices=['all','checkerboard','loops','none'],default='all')
    args=parser.parse_args()
    if args.diagrams=='all':geometry();loops();checkerboard()
    elif args.diagrams=='checkerboard':checkerboard()
    elif args.diagrams=='loops':loops()
    notebooks()
