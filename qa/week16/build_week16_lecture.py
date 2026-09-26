"""Build the curated Week 16 Times-style lecture from retained evidence."""
from pathlib import Path
import json, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/"results/week16_lowboom/reference"
SRC=ROOT/"lectures/source"
OUT=SRC/"week16_generated"
OUT.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["Tinos","Times New Roman","DejaVu Serif"],
    "font.size":11,
    "axes.titlesize":12,
    "axes.labelsize":11,
    "legend.fontsize":9,
})

with np.load(R/"clean_dataset_v801.npz",allow_pickle=False) as z:
    data={k:z[k].copy() for k in z.files}

V=np.pi*0.06**2/2
s=np.linspace(0,1,2001)
def radius(s,a,b):
    f=np.sin(np.pi*s)*np.exp(a*(2*s-1)+b*np.cos(2*np.pi*s))
    c=np.sqrt(V/(np.pi*np.trapezoid(f*f,s)))
    return c*f

fig,ax=plt.subplots(figsize=(7.6,3.6))
for a,b,label in [(0,0,"baseline"),(.325,.122,"retained optimized"),(.3,-.15,"comparison")]:
    r=radius(s,a,b); line,=ax.plot(s,r,label=label); ax.plot(s,-r,color=line.get_color())
ax.set(xlabel="$x/L$",ylabel="$r/L$"); ax.set_aspect("equal",adjustable="box"); ax.grid(alpha=.2); ax.legend(ncol=3,loc="upper center")
fig.tight_layout(); fig.savefig(OUT/"geometry_profiles.pdf",bbox_inches="tight"); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.6,3.6))
r0=radius(s,0,0); r1=radius(s,.3249532654588658,.12237788693256269)
ax.plot(s,r0,label="baseline upper surface"); ax.plot(s,r1,label="optimized upper surface")
mu=np.arcsin(1/1.8)
for x0 in [0.12,0.42,0.72]:
    xx=np.array([x0,min(1.2,x0+.35)])
    yy=np.interp(x0,s,r0)+np.tan(mu)*(xx-x0)*.12
    ax.plot(xx,yy,ls=":",lw=1)
ax.set(xlim=(0,1.1),ylim=(0,.11),xlabel="$x/L$",ylabel="$r/L$"); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"geometry_wave_concept.pdf",bbox_inches="tight"); plt.close(fig)

markers={"train":"o","validation":"s","test":"^","extrapolation":"x"}
fig,ax=plt.subplots(figsize=(6.2,4.4))
for split in ["train","validation","test","extrapolation"]:
    m=data["splits"]==split
    ax.scatter(data["parameters"][m,0],data["parameters"][m,1],marker=markers[split],s=42,label=f"{split} ({m.sum()})")
ax.set(xlabel="shape parameter $a$",ylabel="shape parameter $b$"); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"split_map.pdf",bbox_inches="tight"); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.6,4.0))
for split in ["train","validation","test","extrapolation"]:
    idx=np.where(data["splits"]==split)[0]; i=idx[len(idx)//2]
    ax.plot(data["x"],data["waveforms"][i],label=f"{split}: {data['names'][i]}")
ax.axhline(0,lw=.7); ax.set(xlabel="$x/L$",ylabel="$C_p$"); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"representative_waveforms.pdf",bbox_inches="tight"); plt.close(fig)

train=data["splits"]=="train"; Y=data["waveforms"][train]; Yc=Y-Y.mean(axis=0)
_,S,_=np.linalg.svd(Yc,full_matrices=False); energy=np.cumsum(S**2)/np.sum(S**2)
fig,ax=plt.subplots(figsize=(6.4,4.0))
ax.plot(np.arange(1,len(S)+1),energy,"o-"); ax.axvline(12,ls="--",lw=1)
ax.set(xlabel="number of POD modes $K$",ylabel="cumulative retained energy",ylim=(0,1.01)); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT/"pod_energy.pdf",bbox_inches="tight"); plt.close(fig)

cone=json.loads((R/"cone_refinement_v801.json").read_text())
cells=np.array([row["mesh"]["cells"] for row in cone["levels"]])
err=100*np.array([row["analytical_comparison"]["relative_error"] for row in cone["levels"]])
fig,ax=plt.subplots(figsize=(6.4,4.0))
ax.plot(cells,err,"o-"); ax.set_xscale("log")
ax.set(xlabel="fluid cells",ylabel="Taylor-Maccoll wall-$C_p$ error [%]"); ax.grid(alpha=.2,which="both")
fig.tight_layout(); fig.savefig(OUT/"cone_refinement.pdf",bbox_inches="tight"); plt.close(fig)

ma=json.loads((R/"clean_model_audit_v801.json").read_text())
splits=["train","validation","test","extrapolation"]; x=np.arange(len(splits))
metrics=[("wave_relative_l2","waveform $L_2$"),("peak_mean_relative_error","mean peak"),("drag_mean_relative_error","mean drag"),("worst_case_wave_relative_l2","worst waveform")]
fig,ax=plt.subplots(figsize=(7.2,4.2))
for key,label in metrics:
    ax.plot(x,[100*ma["same_mesh"][sp][key] for sp in splits],"o-",label=label)
ax.set_xticks(x,splits); ax.set_ylabel("relative error [%]"); ax.set_yscale("log"); ax.grid(alpha=.2,which="both"); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(OUT/"model_errors.pdf",bbox_inches="tight"); plt.close(fig)

design=json.loads((R/"weakwall_design_audit.json").read_text())
with np.load(R/"weakwall_design_test.npz",allow_pickle=False) as ev:
    xv=ev["x"]; w=ev["waveforms"]
fig,ax=plt.subplots(figsize=(7.6,4.1))
ax.plot(xv,w[0],label="baseline, mesh level 1.5"); ax.plot(xv,w[1],label="optimized, mesh level 1.5")
ax.plot(xv,w[2],ls="--",label="baseline, mesh level 2"); ax.plot(xv,w[3],ls="--",label="optimized, mesh level 2")
ax.set(xlabel="$x/L$",ylabel="$C_p$"); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"design_waveforms.pdf",bbox_inches="tight"); plt.close(fig)

od=design["offdesign_pairs"]
M=np.array([1.7,1.8,1.9])
peak=np.array([100*od[0]["peak_reduction"],100*design["design_pairs"][0]["peak_reduction"],100*od[1]["peak_reduction"]])
drag=np.array([-100*od[0]["drag_change"],-100*design["design_pairs"][0]["drag_change"],-100*od[1]["drag_change"]])
fig,ax=plt.subplots(figsize=(6.4,4.0))
ax.plot(M,peak,"o-",label="peak-pressure reduction"); ax.plot(M,drag,"s-",label="pressure-drag reduction")
ax.set(xlabel="Mach number",ylabel="reduction relative to baseline [%]"); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"offdesign.pdf",bbox_inches="tight"); plt.close(fig)

tex=SRC/"week16_supersonic_shape_optimization.tex"
assert tex.is_file()
cmd=["xelatex","-interaction=nonstopmode","-halt-on-error","-output-directory=..",tex.name]
for _ in range(2):
    subprocess.run(cmd,cwd=SRC,check=True)
pdf=ROOT/"lectures/week16_supersonic_shape_optimization.pdf"
assert pdf.is_file() and pdf.stat().st_size>100_000
print("Built",pdf)
