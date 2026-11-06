from pathlib import Path
import os,re,html,json,hashlib,textwrap
ROOT=Path(__file__).resolve().parent
FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(ROOT/".matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
fontdir=Path(os.environ.get("FLOWMLLAB_FONT_DIR",r"C:\Windows\Fonts"))
for name in ("times.ttf","timesbd.ttf","timesi.ttf","timesbi.ttf"):
    fp=fontdir/name
    if not fp.is_file():
        raise FileNotFoundError(f"Times New Roman is required: {fp}")
    font_manager.fontManager.addfont(str(fp))
plt.rcParams.update({"font.family":"Times New Roman","text.color":"black","axes.labelcolor":"black","xtick.color":"black","ytick.color":"black","axes.prop_cycle":matplotlib.cycler(color=["black"]),"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white","figure.facecolor":"white"})
BLUE="#000000"
ORANGE="#000000"
GREEN="#000000"
def save(fig,name):
    fig.savefig(FIG/name,dpi=180,bbox_inches="tight",pad_inches=.16)
    plt.close(fig)
def box(ax,xy,w,h,label,color=BLUE):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.02,rounding_size=.035",edgecolor=color,facecolor="white",lw=1.3))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=11)
def arrow(ax,p,q,label=None):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=15,color="#000000",lw=1.2))
    if label:
        ax.text((p[0]+q[0])/2,(p[1]+q[1])/2+.07,label,ha="center",fontsize=9)
fig,ax=plt.subplots(figsize=(9,2.9))
ax.set(xlim=(0,10),ylim=(0,3));ax.axis("off")
for x,w,label in [(0.1,2.5,"One scalar field\n65 x 65"),(3.4,2.6,"Three channels\n65 x 65 x 3"),(6.85,2.9,"B snapshots\nB x 65 x 65 x 3")]:
    box(ax,(x,.85),w,1.35,label)
arrow(ax,(2.7,1.5),(3.35,1.5),"stack")
arrow(ax,(6.05,1.5),(6.8,1.5),"batch")
ax.text(5,.2,"Axis meanings travel with the data: batch, y, x, channel.",ha="center",fontsize=11)
save(fig,"tensor_pipeline.png")

fig,ax=plt.subplots(figsize=(9,3.4))
ax.set(xlim=(0,10),ylim=(0,4));ax.axis("off")
xs=[.8,3.2,6,8.6];ns=[1,5,5,1]
for layer,(x,n) in enumerate(zip(xs,ns)):
    ys=np.linspace(1.1,3,n) if n>1 else [2.05]
    if layer:
        last=np.linspace(1.1,3,ns[layer-1]) if ns[layer-1]>1 else [2.05]
        for yy in last:
            for y in ys:
                ax.plot([xs[layer-1]+.12,x-.12],[yy,y],color="#b0b0b0",lw=.6,zorder=0)
    ax.scatter([x]*len(ys),ys,s=200,facecolors="white",edgecolors=BLUE,linewidths=1.4,zorder=1)
for x,label in zip(xs,["Input x\n1 feature","Dense + tanh\n32 units","Dense + tanh\n32 units","Linear output\n1 value"]):
    ax.text(x,.55,label,ha="center",va="center",fontsize=10.5)
ax.text(5,3.65,"Representative units shown; all units are fully connected.",ha="center",fontsize=10)
ax.text(5,.04,"Parameter counts: 64 + 1056 + 33 = 1153 trainable values",ha="center",fontsize=11)
save(fig,"dense_network.png")

fig,axes=plt.subplots(3,1,figsize=(8.5,4.8),sharex=True,sharey=True)
rng=np.random.default_rng(42)
pool=rng.permutation(201)
x=np.linspace(0,1,201);xx=np.linspace(0,1,401)
for ax,ids,color,label,role in zip(axes,[pool[:121],pool[121:161],pool[161:]],[BLUE,ORANGE,GREEN],["Train (121)","Validation (40)","Test (40)"],["Update weights","Choose configuration / epoch","Evaluate frozen choice once"]):
    ax.plot(xx,np.sin(2*np.pi*xx),color="#b0b0b0",lw=1)
    ax.scatter(x[ids],np.sin(2*np.pi*x[ids]),s=8,color=color)
    ax.set_ylabel(label,fontsize=10)
    ax.text(1.015,.5,role,transform=ax.transAxes,va="center",fontsize=10)
    ax.grid(alpha=.15)
axes[-1].set_xlabel("Input x");fig.subplots_adjust(right=.70,hspace=.26)
save(fig,"split_roles.png")

fig,ax=plt.subplots(figsize=(9,4))
ax.set(xlim=(0,10),ylim=(0,5));ax.axis("off")
for xy,w,h,label in [((.2,3.2),2,1,"Training batch\n(x, target)"),((3.2,3.2),2.25,1,"Forward prediction\ncurrent weights"),((6.5,3.2),2.2,1,"MSE loss\none scalar"),((6.5,.8),2.2,1,"GradientTape\nparameter gradients"),((3.2,.8),2.25,1,"Optimizer update\nnew weights"),((.2,.8),2,1,"Validation pass\nno update")]:
    box(ax,xy,w,h,label,GREEN if xy[0]==.2 and xy[1]==.8 else BLUE)
arrow(ax,(2.25,3.7),(3.1,3.7));arrow(ax,(5.5,3.7),(6.4,3.7))
arrow(ax,(7.6,3.15),(7.6,1.85));arrow(ax,(6.45,1.3),(5.5,1.3))
arrow(ax,(4.3,1.85),(4.3,3.15),"repeat")
arrow(ax,(3.1,1.3),(2.25,1.3))
ax.text(5,.15,"At epoch end, evaluate train and validation with the same fixed weights.",ha="center",fontsize=10)
save(fig,"training_loop.png")

z=np.linspace(-5,5,500)
fig,ax=plt.subplots(figsize=(8,3.4))
for (y,name),ls in zip([(np.tanh(z),"tanh"),(np.maximum(z,0),"ReLU"),(1/(1+np.exp(-z)),"sigmoid"),(z,"linear")],["-","--","-.",":"]):
    ax.plot(z,y,lw=1.8,ls=ls,label=name,color="black")
ax.set(xlabel="Pre-activation z",ylabel="Activation",ylim=(-1.5,3),xlim=(-5,5))
ax.grid(alpha=.18);ax.legend(ncol=4,loc="upper center",bbox_to_anchor=(.5,1.22),frameon=False)
save(fig,"activations.png")

fig,ax=plt.subplots(figsize=(8,3.4))
epoch=np.arange(1,81)
base=.35*np.exp(-epoch/16)+.006
rng=np.random.default_rng(3)
ax.semilogy(epoch,np.clip(base*(1+.4*rng.normal(size=len(epoch))),.002,None),color="#b0b0b0",lw=.9,label="Illustrative batch-level variability")
ax.semilogy(epoch,base,color=BLUE,lw=2,label="Schematic epoch-end training")
ax.semilogy(epoch,.38*np.exp(-epoch/15)+.012+.00012*np.maximum(epoch-48,0),color=ORANGE,lw=2,ls="--",label="Schematic validation")
ax.set(xlabel="Epoch",ylabel="MSE (illustrative values)")
ax.legend(loc="upper right",fontsize=8.5);ax.grid(alpha=.18)
save(fig,"loss_diagnostics.png")

y=np.linspace(-1,1,301)
fig,ax=plt.subplots(figsize=(8,3.4))
ax.plot(y,1-y*y,color=BLUE,lw=2,label="u = 1 - y^2")
ax.plot(y,1-y*y+.3*y,"--",color=ORANGE,lw=2,label="u = 1 - y^2 + 0.3 y")
ax.scatter([-1,1],[0,0],color=BLUE,s=28,zorder=4)
ax.scatter([-1,1],[-.3,.3],color=ORANGE,s=28,zorder=4)
ax.set(xlabel="Wall-normal position y",ylabel="Velocity u",xlim=(-1.07,1.07))
ax.grid(alpha=.18);ax.legend(loc="lower center",frameon=False)
save(fig,"channel_boundary.png")

fig,ax=plt.subplots(figsize=(9,2.8))
ax.set(xlim=(0,11),ylim=(0,3));ax.axis("off")
for x,w,label in [(.1,2.1,"Input\n4 x 32 x 32 x 1"),(3,2.1,"Conv2D(16)\n4 x 32 x 32 x 16"),(5.9,2.1,"Conv2D(16)\n4 x 32 x 32 x 16"),(8.8,2.1,"Conv2D(1)\n4 x 32 x 32 x 1")]:
    box(ax,(x,.8),w,1.3,label)
for x in [2.25,5.15,8.05]:
    arrow(ax,(x,1.45),(x+.7,1.45))
ax.text(5.5,.2,"3 x 3 kernels, stride 1, same padding; 2625 trainable parameters.",ha="center",fontsize=10.5)
save(fig,"cnn_shapes.png")
print("Generated eight teaching figures")
