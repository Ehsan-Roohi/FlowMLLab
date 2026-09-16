"""Visible, executable classroom experiments embedded by improve_classroom_notebooks.py."""
import nbformat as nbf
from textwrap import dedent

def md(s): return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s): return nbf.v4.new_code_cell(dedent(s).strip())

def cavity():
    return [md(r'''
## Build and train a small rectangular-cavity PINN on CPU
First follow a complete learning cycle: coordinates -> network -> streamfunction
lifting -> derivatives -> momentum residual -> gradient update -> held-out residual.
The two-output network represents streamfunction correction and pressure.
With $y=D\eta$, $u=\psi_\eta/D$, $v=-\psi_x$ and
$\Delta=\partial_{xx}+D^{-2}\partial_{\eta\eta}$.

The lifting imposes a smoothed lid, with $\delta_x=\sqrt{10^{-3}}$ and
$\delta_\eta=0.1$. Its correction vanishes quadratically at every wall. The
training mask is $m=(1-e^{-[x^2+(1-\eta)^2]/r^2})
(1-e^{-[(1-x)^2+(1-\eta)^2]/r^2})$, $r=0.01$.
It downweights corners; uniform collocation points are not removed. This differs
from an upstream exclusion radius 0.02. For D=2 the computational mask has
physical vertical extent 0.02, so changing D changes its physical footprint.
The short Adam experiment teaches the algorithm; use its printed residuals to
decide how much it learned. The longer archived experiment follows afterwards.
The default 200-step run is intentionally a budget-limited optimization example,
not a converged CFD substitute. Passing the continuity and wall checks only
verifies the representation; it does not establish momentum accuracy.
'''),code('''
import copy, torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
torch.set_num_threads(1)
torch.manual_seed(13)
dtype=torch.float64
RE_CPU, D_CPU, STEPS_CPU = 100., 1., 200
cpu_pinn=nn.Sequential(nn.Linear(2,20),nn.Tanh(),nn.Linear(20,20),
                       nn.Tanh(),nn.Linear(20,2)).to(dtype=dtype)
def derivative(value, xy):
    return torch.autograd.grad(value,xy,torch.ones_like(value),create_graph=True)[0]
def cavity_fields(xy):
    raw=cpu_pinn(xy); x,e=xy[:,0:1],xy[:,1:2]
    lid=(1-torch.exp(-x*x/.001))*(1-torch.exp(-(1-x)**2/.001))
    psi=D_CPU*(e-1)*e**2*lid*torch.exp(-(1-e)**2/.01)
    psi=psi+(16*x*(1-x)*e*(1-e))**2*raw[:,0:1]
    g=derivative(psi,xy)
    return g[:,1:2]/D_CPU,-g[:,0:1],raw[:,1:2],psi
def cavity_residual(points):
    xy=points.detach().clone().requires_grad_(True)
    u,v,p,_=cavity_fields(xy)
    gu,gv,gp=[derivative(z,xy) for z in (u,v,p)]
    def lap(g):
        return derivative(g[:,0:1],xy)[:,0:1]+derivative(g[:,1:2],xy)[:,1:2]/D_CPU**2
    rx=u*gu[:,0:1]+v*gu[:,1:2]/D_CPU+gp[:,0:1]-lap(gu)/RE_CPU
    ry=u*gv[:,0:1]+v*gv[:,1:2]/D_CPU+gp[:,1:2]/D_CPU-lap(gv)/RE_CPU
    return rx,ry,gu[:,0:1]+gv[:,1:2]/D_CPU
train_xy=torch.rand(192,2,dtype=dtype)
heldout_xy=torch.rand(384,2,dtype=dtype)
x,e=train_xy[:,0:1],train_xy[:,1:2]
mask=(1-torch.exp(-(x*x+(1-e)**2)/.0001))*(1-torch.exp(-((1-x)**2+(1-e)**2)/.0001))
optimizer=torch.optim.Adam(cpu_pinn.parameters(),lr=1e-3)
cpu_history=[]; best_loss=float('inf'); best_state=None
for epoch in range(STEPS_CPU):
    optimizer.zero_grad()
    rx,ry,_=cavity_residual(train_xy)
    # Pressure gauge removes an arbitrary additive constant.
    loss=((mask*rx)**2+(mask*ry)**2).mean()+cpu_pinn(torch.tensor([[.5,.5]],dtype=dtype))[0,1]**2
    if loss.item()<best_loss:
        best_loss=loss.item(); best_state=copy.deepcopy(cpu_pinn.state_dict())
    loss.backward(); optimizer.step()
    cpu_history.append(loss.item())
cpu_pinn.load_state_dict(best_state)
rx,ry,div=cavity_residual(heldout_xy)
print('Architecture:',cpu_pinn)
print({'initial_training_loss':cpu_history[0],'best_training_loss':best_loss,
       'heldout_vector_RMS':float(torch.sqrt((rx**2+ry**2).mean()).detach()),
       'continuity_max':float(div.detach().abs().max())})
assert np.isfinite(cpu_history).all() and best_loss<cpu_history[0]
assert float(div.detach().abs().max())<1e-9
fig,ax=plt.subplots(); ax.semilogy(cpu_history); ax.set(xlabel='Adam step',ylabel='Masked momentum loss + gauge'); plt.show()
'''),code('''
# Evaluate the learned field on an independent grid. Streamlines use u and v.
gx=np.linspace(0,1,65); ge=np.linspace(0,1,65); XX,EE=np.meshgrid(gx,ge)
xy=torch.tensor(np.c_[XX.ravel(),EE.ravel()],dtype=dtype,requires_grad=True)
uu,vv,pp,_=cavity_fields(xy)
uu,vv,pp=[q.detach().numpy().reshape(65,65) for q in (uu,vv,pp)]
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
im=axs[0].contourf(gx,D_CPU*ge,np.hypot(uu,vv),32,cmap='viridis')
axs[0].streamplot(gx,D_CPU*ge,uu,vv,color='white',density=.7,linewidth=.5)
fig.colorbar(im,ax=axs[0]); axs[0].set_title('Fresh CPU PINN: speed and streamlines')
im=axs[1].contourf(gx,D_CPU*ge,pp-pp.mean(),32,cmap='RdBu_r')
fig.colorbar(im,ax=axs[1]); axs[1].set_title('Fresh CPU PINN: centered pressure')
for ax in axs: ax.set(xlabel='x/L',ylabel='y/L',aspect='equal')
plt.show()
# Exact wall transform is checked independently of training quality.
expected=(1-np.exp(-gx**2/.001))*(1-np.exp(-(1-gx)**2/.001))
np.testing.assert_allclose(uu[-1],expected,atol=1e-10)
assert max(abs(vv).take([0,-1],axis=0).max(),abs(uu[0]).max(),abs(uu[:,[0,-1]]).max())<1e-9
'''),md(r'''
### Why the optimizer and residual convention matter
SSBroyden2 is a self-scaled Broyden quasi-Newton variant. With inverse Hessian
estimate $H_k$, its search step is $d_k=-H_k\nabla L_k$ and
$\theta_{k+1}=\theta_k+\alpha_k d_k$. A line search selects $\alpha_k$;
self-scaling modifies the full inverse-Hessian update, constrained by the
secant relation $H_{k+1}y_k=s_k$. Full storage costs $O(P^2)$ for P parameters.
Adam in the small experiment above is a separate, explicitly visible optimizer.
See [Urbán, Stefanou & Pons, JCP 523 (2025), 113656](https://arxiv.org/abs/2405.04230)
and [Kiyani et al.](https://arxiv.org/abs/2501.16371) for the self-scaling rules.

Both full and corner diagnostics below use
$R_S=\sqrt{\langle r_x^2+r_y^2\rangle_S}$. Historical corner-history values
stored as component RMS are multiplied by $\sqrt2$ for display. A narrow lid
transition increases second velocity derivatives, and the viscous term scales
as $1/Re$. This can make Re=100 residuals larger than Re=400 despite better
velocity agreement. Residual magnitude and field error answer different questions.
The 65x65 teaching CFD reference has discretization error and a discontinuous
lid; comparison with a smoothed-lid PINN is an approximate diagnostic.
''')]

def unet():
    return [md(r'''
## Train a two-level U-Net yourself
An encoder converts two velocity channels into features with 8, 16, and 32
channels. Pooling enlarges the receptive field; the decoder interpolates and
concatenates encoder features through skip connections. A linear 1x1 head
returns u and v. Reconstruction minimizes mean squared velocity error.
Segmentation instead uses a one-channel logit and BCEWithLogitsLoss; sigmoid
converts that logit to a probability. Here we train the reconstruction path
on small subsets of Re90/Re110, select the checkpoint on Re100, and evaluate
on Re105 after selection. The grid is the actual retained LBM grid.
'''),code('''
import copy, torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
torch.set_num_threads(1); torch.manual_seed(11)
class ClassroomUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
        self.e1=block(2,8); self.e2=block(8,16); self.bridge=block(16,32)
        self.d2=block(48,16); self.d1=block(24,8); self.head=nn.Conv2d(8,2,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(F.max_pool2d(a,2)); c=self.bridge(F.max_pool2d(b,2))
        c=self.d2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        c=self.d1(torch.cat((F.interpolate(c,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return self.head(c)
def velocity_batch(re):
    return torch.tensor(np.stack((cases[re]['u'][::12],cases[re]['v'][::12]),axis=1),dtype=torch.float32)
def coarse(a):
    return F.interpolate(F.interpolate(a,size=(8,19),mode='area'),size=a.shape[-2:],mode='bilinear',align_corners=False)
training=torch.cat((velocity_batch(90),velocity_batch(110)))
validation=velocity_batch(100)
mean=training.mean((0,2,3),keepdim=True); scale=training.std((0,2,3),keepdim=True).clamp_min(1e-8)
xt=(coarse(training)-mean)/scale; yt=(training-mean)/scale
xv=(coarse(validation)-mean)/scale; yv=(validation-mean)/scale
model=ClassroomUNet(); optimizer=torch.optim.Adam(model.parameters(),lr=.003)
training_loss=[]; validation_loss=[]; best=float('inf'); selected=None
for epoch in range(30):
    model.train(); total=0.
    for ids in torch.randperm(len(xt)).split(8):
        optimizer.zero_grad(); loss=F.mse_loss(model(xt[ids]),yt[ids]); loss.backward(); optimizer.step()
        total+=loss.item()*len(ids)
    model.eval()
    with torch.no_grad(): value=F.mse_loss(model(xv),yv).item()
    training_loss.append(total/len(xt)); validation_loss.append(value)
    if value<best: best=value; selected=copy.deepcopy(model.state_dict())
model.load_state_dict(selected)
evaluation=velocity_batch(105)
with torch.no_grad(): estimate=model((coarse(evaluation)-mean)/scale)*scale+mean
err=lambda a: float(torch.linalg.vector_norm(a-evaluation)/torch.linalg.vector_norm(evaluation))
print(model)
print({'selected_validation_MSE':best,'test_velocity_relative_L2':err(estimate),
       'interpolation_relative_L2':err(coarse(evaluation))})
assert np.isfinite(training_loss).all() and np.isfinite(validation_loss).all()
fig,ax=plt.subplots(); ax.semilogy(training_loss,label='train'); ax.semilogy(validation_loss,label='validation')
ax.set(xlabel='Epoch',ylabel='Standardized velocity MSE'); ax.legend(); plt.show()
fig,axs=plt.subplots(3,1,figsize=(10,7),layout='constrained')
arrays=[evaluation[0].numpy(),coarse(evaluation)[0].numpy(),estimate[0].numpy()]
limit=max(np.hypot(*a).max() for a in arrays)
for ax,a,title in zip(axs,arrays,['LBM reference','Interpolation','Fresh classroom U-Net']):
    ax.imshow(np.hypot(*a),origin='lower',cmap='viridis',vmin=0,vmax=limit)
    ax.set_title(title)
plt.show()
''')]

def inverse():
    return [md(r'''
## Learn the inverse PINN before reading the research audit
In the k equation, $0.09 C_k k\omega$ is modeled destruction: $C_k$ changes
how quickly turbulent kinetic energy is removed. The inverse stage first
infers diffusion from a known k profile, production P and dissipation epsilon:
$r=a'k'+(\nu+a)k''+P-\epsilon$. The following manufactured problem has known
$k(y)=y(1-y)$ and $a(y)=0.1+0.2y$. Thus $P-\epsilon=2\nu+0.8y$ makes r zero.
Training sees k and the forcing, not a labels; boundary values make the inverse
problem identifiable. This teaches the same differential operation used by
the research method, with an analytic answer for validation.
'''),code('''
import copy
from torch import nn
torch.manual_seed(14)
inverse_net=nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,16),nn.Tanh(),nn.Linear(16,1)).double()
yin=torch.linspace(0,1,96,dtype=torch.float64)[:,None].requires_grad_(True)
nu_demo=.01
def coefficient(y):
    # Exact boundary lifting a(0)=.1, a(1)=.3.
    return .1+.2*y+y*(1-y)*inverse_net(y)
opt_inverse=torch.optim.Adam(inverse_net.parameters(),lr=.003)
inverse_history=[]; best_inverse=float('inf'); best_weights=None
for epoch in range(1200):
    opt_inverse.zero_grad()
    a=coefficient(yin)
    da=torch.autograd.grad(a,yin,torch.ones_like(a),create_graph=True)[0]
    residual=da*(1-2*yin)-2*(nu_demo+a)+2*nu_demo+.8*yin
    loss=residual.square().mean()
    if loss.item()<best_inverse:
        best_inverse=loss.item(); best_weights=copy.deepcopy(inverse_net.state_dict())
    loss.backward(); opt_inverse.step(); inverse_history.append(loss.item())
inverse_net.load_state_dict(best_weights)
yg=torch.linspace(0,1,201,dtype=torch.float64)[:,None]
with torch.no_grad(): ahat=coefficient(yg).numpy().ravel()
atrue=.1+.2*yg.numpy().ravel()
inverse_error=np.linalg.norm(ahat-atrue)/np.linalg.norm(atrue)
print({'best_loss':best_inverse,'final_training_loss':inverse_history[-1],
       'loaded_checkpoint':'minimum training residual','independent_grid_relative_L2':inverse_error})
assert inverse_error<.03
fig,axs=plt.subplots(1,2,figsize=(10,3.4))
axs[0].semilogy(inverse_history); axs[0].set(xlabel='Adam step',ylabel='Inverse residual MSE')
axs[1].plot(yg,atrue,label='Analytic diffusion'); axs[1].plot(yg,ahat,'--',label='Inferred diffusion')
axs[1].set(xlabel='y',ylabel='a(y)'); axs[1].legend(); plt.show()
'''),md('''
### Reading the long research run
The archived inverse run reached about 1.24 at its best iteration and about
231 at the final iteration. That increase indicates optimizer instability or
oscillation after a better iterate; it does not demonstrate monotone convergence.
The source saves its final state and does not select the best checkpoint.
The corrected CFD profiles use the released correction tables; they are not
generated from the small manufactured fit above. Always name the checkpoint
used when reporting a loss or a downstream CFD curve.

A residual of 3e-13 can be numerically tiny while failing a requested 1e-14
stopping gate. Report both the residual and gate; a Boolean alone hides this.
One-step restarts agreeing with saved profiles check consistency, not independent
cold-start convergence. PCHIP tests interpolation of one known profile using y;
the local-feature NN tests a deployable closure map. Their error comparison
measures how demanding that map is, not a fair contest between identical inputs.
''')]
