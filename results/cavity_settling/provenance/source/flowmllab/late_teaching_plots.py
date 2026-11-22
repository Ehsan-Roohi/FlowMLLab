"""Native-grid field and signed-profile figures for the late-course CPU labs.

These functions visualize supplied arrays only. They do not fit, smooth, clip
errors, select a checkpoint or read/write any frozen research results.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLORS = ("#d62728", "#7928a1", "#007e87", "#dd7800", "#1845aa", "#7a4b20")
STYLES = ("--", ":", "-.", (0,(5,2,1,2)), (0,(1,1)), (0,(3,1)))


def field_profiles(x, y, reference, predictions, *, quantity, case, mask=None,
                   cut_y=0., reference_name="CFD", output=None, stem=None):
    """Compare each prediction with a filled reference and a full signed error.

    x/y are native one-dimensional coordinates. No interpolation is performed:
    the closest native row to cut_y is reported explicitly on the profile.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    truth = np.asarray(reference, float)
    if truth.shape != (len(y),len(x)) or min(truth.shape)<2:
        raise ValueError("Reference shape must match the native x/y grid")
    if np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):
        raise ValueError("Coordinates must be strictly increasing")
    valid = np.ones(truth.shape,bool) if mask is None else np.asarray(mask,bool)
    if valid.shape != truth.shape or not np.any(valid) or not predictions:
        raise ValueError("Supply a nonempty fluid mask and at least one prediction")
    if not np.isfinite(truth[valid]).all():
        raise ValueError("Nonfinite fluid reference")
    items=[]
    for name, value in predictions.items():
        value=np.asarray(value,float)
        if value.shape!=truth.shape or not np.isfinite(value[valid]).all():
            raise ValueError("Prediction shape/finiteness differs from the reference")
        items.append((str(name),value))
    reference_min,reference_max=float(truth[valid].min()),float(truth[valid].max())
    if reference_min==reference_max:
        pad=max(abs(reference_min)*.01,1e-12);reference_min-=pad;reference_max+=pad
    levels=np.linspace(reference_min,reference_max,25)
    lines=np.linspace(reference_min,reference_max,9)[1:-1]
    error_max=max(float(np.max(np.abs(value[valid]-truth[valid]))) for _,value in items)
    plot_error_max=max(error_max,1e-14)
    errors=np.linspace(-plot_error_max,plot_error_max,25)
    cmap=plt.get_cmap("jet").copy();cmap.set_bad("#e3e3e3")
    fields=[];stats=[]
    for i,(name,value) in enumerate(items):
        error=value-truth
        wide=(x[-1]-x[0])/(y[-1]-y[0])>3
        fig,(a,b)=plt.subplots(2 if wide else 1,1 if wide else 2,
                              figsize=(8.8,5.6 if wide else 3.6),constrained_layout=True)
        fields.append(fig)
        fill=a.contourf(x,y,np.ma.masked_where(~valid,truth),levels=levels,cmap=cmap)
        a.contour(x,y,np.ma.masked_where(~valid,value),levels=lines,
                  colors=COLORS[i%len(COLORS)],linestyles="--",linewidths=.8)
        a.set_title("Field and identical contour levels",fontsize=10)
        fig.legend(handles=[Patch(facecolor="#00bd96",label=reference_name),
                     Line2D([],[],color=COLORS[i%len(COLORS)],ls="--",label=name)],
                   loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=10)
        err=b.contourf(x,y,np.ma.masked_where(~valid,error),levels=errors,cmap=cmap)
        b.set_title(f"{name} minus {reference_name}",fontsize=10)
        fig.colorbar(fill,ax=a,label=quantity,shrink=.8)
        fig.colorbar(err,ax=b,label=f"signed error in {quantity}",shrink=.8)
        for ax in (a,b):
            ax.set(xlabel="x/D" if "wake" in case.lower() else "x/H",
                   ylabel="y/D" if "wake" in case.lower() else "y/H",
                   xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect="equal")
        fig.suptitle(case,fontsize=11,y=1.17)
        denom=np.linalg.norm(truth[valid])
        stats.append(dict(method=name,relative_l2=float(np.linalg.norm(error[valid])/denom) if denom else None,
                          signed_error_min=float(error[valid].min()),signed_error_max=float(error[valid].max())))
    j=int(np.argmin(np.abs(y-float(cut_y))))
    profile,pa=plt.subplots(2,1,figsize=(8.8,5.3),sharex=True,constrained_layout=True)
    rowmask=~valid[j]
    pa[0].plot(x,np.ma.masked_where(rowmask,truth[j]),color="black",lw=2,label=reference_name)
    for i,(name,value) in enumerate(items):
        kw=dict(color=COLORS[i%len(COLORS)],ls=STYLES[i%len(STYLES)],lw=1.7,label=name)
        pa[0].plot(x,np.ma.masked_where(rowmask,value[j]),**kw)
        pa[1].plot(x,np.ma.masked_where(rowmask,value[j]-truth[j]),**kw)
    pa[0].legend(loc="upper center",bbox_to_anchor=(.5,1.42),ncol=min(3,len(items)+1),frameon=False)
    pa[0].set(ylabel=quantity,title=f"{case}; native row y = {y[j]:.6g}")
    pa[1].axhline(0,color=".5",lw=.7)
    pa[1].set(ylabel=f"prediction minus {reference_name}",xlabel="x/D" if "wake" in case.lower() else "x/H")
    for ax in pa:ax.grid(alpha=.2)
    if output is not None:
        folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
        base=stem or re.sub(r"[^a-zA-Z0-9]+","_",case).strip("_").lower()
        for i,figure in enumerate(fields):
            figure.savefig(folder/(base+f"_fields_{i+1:02d}.png"),dpi=155,bbox_inches="tight")
        profile.savefig(folder/(base+"_profiles.png"),dpi=155,bbox_inches="tight")
    return (*fields,profile),dict(case=case,quantity=quantity,native_shape=list(truth.shape),
                             reference_range=[reference_min,reference_max],error_scale=[-plot_error_max,plot_error_max],
                             native_profile_y=float(y[j]),metrics=stats)
