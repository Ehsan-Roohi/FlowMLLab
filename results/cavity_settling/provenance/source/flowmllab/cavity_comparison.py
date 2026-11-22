"""Frozen-model CFD/POD-DeepONet overlays; no fitting or source-data mutation."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
import sys,json,hashlib,csv
import numpy as np
from scipy.fft import dctn,idctn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import SymLogNorm
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/cavity_field_deeponet/flow_comparison"
sys.path.insert(0,str(ROOT))
from flowmllab import cavity_field_operator as op
from flowmllab.cavity_highre import audit_field
ORANGE="#D55E00"
METHODS=("CFD","POD-DeepONet")
plt.rcParams.update({
    "font.family":"Times New Roman", "font.size":12,
    "mathtext.fontset":"stix","pdf.fonttype":42,"ps.fonttype":42,
    "axes.linewidth":.8,"xtick.direction":"in","ytick.direction":"in",
    "xtick.major.size":4,"ytick.major.size":4,"savefig.facecolor":"white",
    "axes.unicode_minus":True,"path.simplify":False,
})
SEEDS=np.array(
    [[.5,y] for y in (.025,.06,.12,.20,.30,.40,.48,.53)]
    +[[.025,.02],[.06,.04],[.12,.08],
      [.978,.015],[.95,.025],[.92,.05],[.88,.09],[.82,.14],
      [.025,.985],[.035,.96],[.055,.935],[.09,.90]])
TRACKED={}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(p,expected):
    actual=digest(p)
    if actual!=expected:raise ValueError("Source hash mismatch: "+str(p))
    TRACKED[str(p)]=actual
def read_npz(p):return dict(np.load(p,allow_pickle=False))
def load():
    folder=ROOT/"results/cavity_field_deeponet"
    selection=json.loads((folder/"frozen_selection.json").read_text())
    score=json.loads((folder/"scoring_receipt.json").read_text())
    checked(folder/"frozen_selection.json",score["frozen_selection_sha256"])
    checked(folder/"trunk.npz",selection["trunk_sha256"])
    checked(folder/"frozen_predictions.npz",selection["frozen_predictions_sha256"])
    trunk=read_npz(folder/"trunk.npz")
    frozen=read_npz(folder/"frozen_predictions.npz")
    training=[]
    for re in op.TRAIN:
        p=ROOT/f"results/cavity_highre_ghia/n512/re{re}.npz"
        checked(p,selection["data_read_during_training"][str(re)])
        training.append(read_npz(p))
    models=[]
    for i in selection["selected_indices"]:
        name=selection["candidate_models"][i]["name"]
        p=folder/"models"/(name+".npz")
        checked(p,{k.replace("\\","/"):v for k,v in selection["model_sha256"].items()}[p.relative_to(ROOT).as_posix()])
        z=read_npz(p)
        models.append(dict(theta=z["theta"],shapes=z["shapes"].tolist(),scale=z["scale"]))
    cases={}
    coeff_report={}
    for re in (1000,7500):
        p=ROOT/f"results/cavity_highre_ghia/n512/re{re}.npz"
        meta=json.loads(p.with_suffix(".json").read_text())
        checked(p,meta["field_sha256"])
        if re==7500:checked(p,score["heldout_field_sha256"][str(re)])
        truth=read_npz(p)
        if max(audit_field(truth,re).values())>=1e-6:raise ValueError("CFD audit failed")
        coefs=np.array([op.predict_coefficients(m,[re])[0] for m in models])
        coef=coefs.mean(axis=0)
        if re==7500:
            idx=list(frozen["test_re"]).index(re)
            fixed=frozen["candidate_coefficients"][selection["selected_indices"],idx]
            np.testing.assert_allclose(coefs,fixed,rtol=1e-12,atol=1e-12)
            coef=fixed.mean(axis=0)
        flat=op.reconstruct(trunk,coef)
        n=truth["u"].size
        prediction={k:truth[k] for k in ("x","y")}
        prediction.update(u=flat[:n].reshape(truth["u"].shape),v=flat[n:].reshape(truth["v"].shape))
        weights=op.mixing_weights(trunk,coef)
        for key in ("psi","omega"):
            prediction[key]=sum(w*f[key] for w,f in zip(weights,training))
        np.testing.assert_allclose(sum(weights),1.0,atol=1e-12)
        cases[re]={"CFD":truth,"POD-DeepONet":prediction}
        coeff_report[str(re)]={"role":"training case" if re==1000 else "held-out test",
                              "coefficients":coef.tolist(),"mixing_weights":weights.tolist()}
    return cases,selection,coeff_report
def scalar_from_edges(gx,gy,h):
    """Least squares on horizontal/vertical graph edges; one constant gauge."""
    ny,nx=gx.shape[0],gy.shape[1]
    assert gx.shape==(ny,nx-1) and gy.shape==(ny-1,nx)
    rhs=np.zeros((ny,nx))
    rhs[:,:-1]-=h*gx;rhs[:,1:]+=h*gx
    rhs[:-1,:]-=h*gy;rhs[1:,:]+=h*gy
    eig=2-2*np.cos(np.pi*np.arange(ny)/ny)
    eix=2-2*np.cos(np.pi*np.arange(nx)/nx)
    denom=eig[:,None]+eix[None,:]
    coeff=dctn(rhs,type=2,norm="ortho")
    denom[0,0]=1;coeff[0,0]=0
    p=idctn(coeff/denom,type=2,norm="ortho")
    p-=p.mean()
    residual_x=np.diff(p,axis=1)/h-gx
    residual_y=np.diff(p,axis=0)/h-gy
    return p,float(np.sqrt((np.sum(residual_x**2)+np.sum(residual_y**2))/(residual_x.size+residual_y.size)))
def pressure(f,re):
    """Recover interior p from the same centered steady momentum gradient for both methods."""
    u,v=f["u"],f["v"];h=float(f["x"][1]-f["x"][0])
    ux=(u[1:-1,2:]-u[1:-1,:-2])/(2*h)
    uy=(u[2:,1:-1]-u[:-2,1:-1])/(2*h)
    vx=(v[1:-1,2:]-v[1:-1,:-2])/(2*h)
    vy=(v[2:,1:-1]-v[:-2,1:-1])/(2*h)
    lap=lambda a:(a[1:-1,2:]+a[1:-1,:-2]+a[2:,1:-1]+a[:-2,1:-1]-4*a[1:-1,1:-1])/h**2
    ui,vi=u[1:-1,1:-1],v[1:-1,1:-1]
    gx=lap(u)/re-ui*ux-vi*uy
    gy=lap(v)/re-ui*vx-vi*vy
    ex=(gx[:,:-1]+gx[:,1:])/2
    ey=(gy[:-1,:]+gy[1:,:])/2
    p,res=scalar_from_edges(ex,ey,h)
    norm=float(np.sqrt((np.sum(ex**2)+np.sum(ey**2))/(ex.size+ey.size)))
    return dict(p=p,x=f["x"][1:-1],y=f["y"][1:-1],
                gradient_fit_rms=res,gradient_target_rms=norm,
                gradient_fit_relative=res/norm)
def pressure_verification():
    rows=[]
    for n in (64,128,256):
        x=np.linspace(0,1,n+1);xx,yy=np.meshgrid(x,x)
        re=40.;lam=re/2-np.sqrt(re**2/4+4*np.pi**2)
        u=1-np.exp(lam*xx)*np.cos(2*np.pi*yy)
        v=lam/(2*np.pi)*np.exp(lam*xx)*np.sin(2*np.pi*yy)
        exact=.5*(1-np.exp(2*lam*xx[1:-1,1:-1]));exact-=exact.mean()
        z=pressure(dict(x=x,y=x,u=u,v=v),re)
        err=np.linalg.norm(z["p"]-exact)/np.linalg.norm(exact)
        rows.append(dict(cells=n,relative_l2=float(err),gradient_fit_relative=z["gradient_fit_relative"]))
    orders=[np.log(rows[i-1]["relative_l2"]/rows[i]["relative_l2"])/np.log(2) for i in (1,2)]
    if min(orders)<1.8 or rows[-1]["relative_l2"]>1e-3:raise ValueError("Manufactured pressure convergence failed")
    p,r=scalar_from_edges(np.full((17,16),2.),np.full((16,17),-3.),1/16)
    x,y=np.meshgrid(np.linspace(0,1,17),np.linspace(0,1,17))
    exact=2*x-3*y;exact-=exact.mean()
    np.testing.assert_allclose(p,exact,atol=1e-12)
    return dict(test="Unforced exact Kovasznay solution, Re=40; plus affine-gradient gauge test",
                grids=rows,orders=orders,affine_linf=float(abs(p-exact).max()))
def combine_segments(collection):
    segments=collection.get_segments()
    if not segments or any(len(s)!=2 for s in segments):return
    paths=[];current=list(segments[0])
    for s in segments[1:]:
        if np.linalg.norm(np.asarray(current[-1])-s[0])<1e-12:current.append(s[1])
        else:paths.append(np.array(current));current=list(s)
    paths.append(np.array(current));collection.set_segments(paths)
def stream(ax,f,color,ls,bounds=(0,1,0,1),zoom=False):
    x0,x1,y0,y1=bounds
    ix=(f["x"]>=x0)&(f["x"]<=x1);iy=(f["y"]>=y0)&(f["y"]<=y1)
    x,y=f["x"][ix],f["y"][iy];u=f["u"][np.ix_(iy,ix)];v=f["v"][np.ix_(iy,ix)]
    if zoom:
        # A fixed staggered geometric set; the same points are used for both fields.
        frac=np.array([[.12,.12],[.22,.18],[.32,.25],[.42,.32],[.55,.42],
                       [.7,.55],[.85,.72],[.12,.7],[.32,.88],[.72,.15]])
        seeds=np.column_stack((x[0]+frac[:,0]*(x[-1]-x[0]),y[0]+frac[:,1]*(y[-1]-y[0])))
    else:seeds=SEEDS
    for seed in seeds:
        # Reset the occupancy mask for every prescribed seed in both methods.
        # Stop a path on its own return, rather than drawing repeated circuits.
        streamset=ax.streamplot(x,y,u,v,start_points=np.array([seed]),color=color,
            density=3.0,linewidth=1.0 if not zoom else .9,arrowsize=.65 if not zoom else .6,
            minlength=.035,maxlength=6.,integration_direction="both",broken_streamlines=True,
            zorder=2 if color=="black" else 3)
        combine_segments(streamset.lines)
        streamset.lines.set_linestyle(ls)
    return None
def axes_style(ax,bounds=(0,1,0,1)):
    x0,x1,y0,y1=bounds
    ax.set(xlim=(x0,x1),ylim=(y0,y1),aspect="equal",xlabel=r"$x/L$",ylabel=r"$y/L$")
    if bounds==(0,1,0,1):
        ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
    ax.tick_params(top=True,right=True,labelsize=12)
def save(fig,name):
    fig.savefig(OUT/(name+".pdf"),bbox_inches="tight",pad_inches=.07)
    fig.savefig(OUT/(name+".png"),dpi=200,bbox_inches="tight",pad_inches=.07)
    plt.close(fig)
    p=OUT/(name+".png")
    with Image.open(p) as im:
        if im.mode!="RGB":im.convert("RGB").save(p)
def legend(fig,y=.018):
    fig.legend([Line2D([0],[0],color="black",lw=1.15),
                Line2D([0],[0],color=ORANGE,lw=1.15,ls=(0,(4,2.5)))],
               ["CFD","POD-DeepONet"],loc="lower center",bbox_to_anchor=(.5,y),
               ncol=2,frameon=False,handlelength=3.2,columnspacing=3.)
def flow_plot(cases):
    fig,axs=plt.subplots(1,2,figsize=(10.3,5.5))
    for ax,re in zip(axs,(1000,7500)):
        for name,color,ls in [("CFD","black","-"),("POD-DeepONet",ORANGE,(0,(4,2.5)))]:
            stream(ax,cases[re][name],color,ls)
        axes_style(ax)
    fig.subplots_adjust(left=.07,right=.985,top=.98,bottom=.20,wspace=.26)
    for ax,label in zip(axs,("(a)  Re = 1000  |  Training case","(b)  Re = 7500  |  Held-out test")):
        pos=ax.get_position();fig.text(pos.x0+pos.width/2,.115,label,ha="center",fontsize=13)
    legend(fig)
    save(fig,"streamlines_overlay_Re1000_Re7500")
def zoom_plot(cases):
    bounds=[(0,.22,0,.22),(.72,1,0,.28),(0,.22,.78,1)]
    fig,axs=plt.subplots(2,3,figsize=(11.6,8.0))
    for row,re in enumerate((1000,7500)):
        for col,b in enumerate(bounds):
            ax=axs[row,col]
            for name,color,ls in [("CFD","black","-"),("POD-DeepONet",ORANGE,(0,(4,2.5)))]:
                stream(ax,cases[re][name],color,ls,b,zoom=True)
            axes_style(ax,b)
            ax.set_xticks(np.linspace(b[0],b[1],3));ax.set_yticks(np.linspace(b[2],b[3],3))
    fig.subplots_adjust(left=.075,right=.99,top=.98,bottom=.13,wspace=.32,hspace=.4)
    for row,re in enumerate((1000,7500)):
        for col,where in enumerate(("Lower left","Lower right","Upper left")):
            pos=axs[row,col].get_position()
            fig.text(pos.x0+pos.width/2,pos.y0-.075,f"({chr(97+3*row+col)})  Re = {re}: {where}",ha="center",fontsize=12)
    legend(fig,y=.005)
    save(fig,"corner_streamlines_overlay_Re1000_Re7500")
def pressure_plot(cases,pressures):
    fig,axs=plt.subplots(1,2,figsize=(10.3,5.5))
    chosen={}
    for ax,re in zip(axs,(1000,7500)):
        a,b=(pressures[re][m] for m in METHODS)
        # Values are dimensionalized once by rho*U^2 and share a global interior gauge.
        values=np.r_[a["p"].ravel(),b["p"].ravel()]
        low,high=float(values.min()),float(values.max())
        bound=max(abs(low),abs(high))
        # Full extrema remain in the retained arrays. Linear contours across the full range.
        levels=np.array([-.45,-.24,-.12,-.08,-.06,-.04,-.03,-.02,-.01,0.,.01,.02,.04,.08,.16,.32,.64,1.28,1.6])
        chosen[str(re)]={"levels":levels.tolist(),"extrema":[low,high]}
        for name,color,style in [("CFD","black","solid"),("POD-DeepONet",ORANGE,"dashed")]:
            z=pressures[re][name]
            ax.contour(z["x"],z["y"],z["p"],levels=levels,colors=color,
                       linewidths=1.0,linestyles=style,zorder=2 if name=="CFD" else 3)
        axes_style(ax)
    fig.subplots_adjust(left=.07,right=.985,top=.98,bottom=.20,wspace=.26)
    for ax,label in zip(axs,("(a)  Re = 1000  |  Reconstructed pressure","(b)  Re = 7500  |  Reconstructed pressure")):
        pos=ax.get_position();fig.text(pos.x0+pos.width/2,.115,label,ha="center",fontsize=12)
    legend(fig)
    save(fig,"pressure_contours_overlay_Re1000_Re7500")
    return chosen
def pressure_fields(pressures):
    # A common full-range pressure norm across both Re and methods; no clipping.
    pmin=min(z["p"].min() for methods in pressures.values() for z in methods.values())
    pmax=max(z["p"].max() for methods in pressures.values() for z in methods.values())
    norm=SymLogNorm(linthresh=.02,linscale=1.,vmin=pmin,vmax=pmax,base=10)
    fig=plt.figure(figsize=(11.5,9.1))
    for row,re in enumerate((1000,7500)):
        a,b=(pressures[re][m] for m in METHODS)
        p1,p2=a["p"],b["p"]
        err=p2-p1;eb=float(abs(err).max())
        y0=.64 if row==0 else .12
        for col,(p,label) in enumerate(((p1,"CFD-derived pressure"),(p2,"NN-derived pressure"),(err,"Signed difference"))):
            x0=.065+.318*col
            ax=fig.add_axes([x0,y0,.255,.322])
            z=ax.pcolormesh(a["x"],a["y"],p,shading="nearest",cmap="turbo",
                norm=norm if col<2 else None,vmin=None if col<2 else -eb,
                vmax=None if col<2 else eb,rasterized=True)
            axes_style(ax);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
            # Column identifiers sit above, outside each plotting region.
            fig.text(x0+.1275,y0+.337,f"({chr(97+row*3+col)}) Re = {re}: {label}",
                     ha="center",fontsize=10.5)
            cax=fig.add_axes([x0,y0-.071,.255,.012])
            cb=fig.colorbar(z,cax=cax,orientation="horizontal")
            cb.ax.tick_params(labelsize=9)
            cb.set_label(r"$p_g/(\rho U^2)$ (symmetric-log)" if col<2 else
                         r"$(p_{g,\mathrm{NN}}-p_{g,\mathrm{CFD}})/(\rho U^2)$ (linear)",fontsize=10)
            if col<2:cb.set_ticks([-.1,-.02,0,.02,.1,1.])
            else:
                cb.formatter=ScalarFormatter(useMathText=True);cb.formatter.set_powerlimits((-2,2));cb.update_ticks()
    save(fig,"pressure_fields_and_signed_error_Re1000_Re7500")


def vorticity(f):
    """Same centered curl for both velocity fields; interior nodes only."""
    hx=float(f['x'][1]-f['x'][0]);hy=float(f['y'][1]-f['y'][0])
    w=(f['v'][1:-1,2:]-f['v'][1:-1,:-2])/(2*hx)-(f['u'][2:,1:-1]-f['u'][:-2,1:-1])/(2*hy)
    return dict(w=w,x=f['x'][1:-1],y=f['y'][1:-1])

VORTICITY_LEVELS=np.array([-200,-100,-50,-20,-10,-5,-3,-2.2,-2,-1.8,-1,-.5,0,.5,1,2,5,10,20,50,100,200.])

def vorticity_plots(cases):
    vort={re:{name:vorticity(f) for name,f in methods.items()} for re,methods in cases.items()}
    fig,axs=plt.subplots(1,2,figsize=(10.3,5.5))
    fig.subplots_adjust(left=.055,right=.985,bottom=.22,top=.97,wspace=.26)
    for ax,re in zip(axs,(1000,7500)):
        for name,color,ls in zip(METHODS,('black',ORANGE),('solid','dashed')):
            z=vort[re][name]
            ax.contour(z['x'],z['y'],z['w'],levels=VORTICITY_LEVELS,colors=[color],linestyles=ls,linewidths=.9)
        axes_style(ax)
        ax.text(.5,-.105,rf'({"a" if re==1000 else "b"}) $\mathrm{{Re}}={re}$ | Velocity-derived vorticity',transform=ax.transAxes,ha='center',va='top',fontsize=11.5)
    legend(fig);save(fig,'vorticity_contours_overlay_Re1000_Re7500')
    bound=max(float(abs(z['w']).max()) for d in vort.values() for z in d.values())
    norm=SymLogNorm(linthresh=1.,vmin=-bound,vmax=bound,base=10)
    fig=plt.figure(figsize=(11.5,9.1));rows=[]
    for row,re in enumerate((1000,7500)):
        a,b=(vort[re][m] for m in METHODS);delta=b['w']-a['w'];eb=float(abs(delta).max())
        iy,ix=np.unravel_index(np.argmax(abs(delta)),delta.shape)
        rows.append(dict(Re=re,role='training' if re==1000 else 'held_out_test',
                         relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(a['w'])),
                         rms=float(np.sqrt(np.mean(delta**2))),linf=eb,
                         linf_signed=float(delta[iy,ix]),linf_x=float(a['x'][ix]),linf_y=float(a['y'][iy]),
                         cfd_min=float(a['w'].min()),cfd_max=float(a['w'].max()),
                         nn_min=float(b['w'].min()),nn_max=float(b['w'].max())))
        y0=.64 if row==0 else .12
        for col,(w,label) in enumerate(((a['w'],'CFD-derived vorticity'),(b['w'],'NN-derived vorticity'),(delta,'Signed difference'))):
            x0=.065+.318*col;ax=fig.add_axes([x0,y0,.255,.322])
            z=ax.pcolormesh(a['x'],a['y'],w,shading='nearest',cmap='turbo',norm=norm if col<2 else None,
                            vmin=None if col<2 else -eb,vmax=None if col<2 else eb,rasterized=True)
            axes_style(ax);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
            fig.text(x0+.1275,y0+.337,f'({chr(97+row*3+col)}) Re = {re}: {label}',ha='center',fontsize=10.5)
            cax=fig.add_axes([x0,y0-.071,.255,.012]);cb=fig.colorbar(z,cax=cax,orientation='horizontal')
            cb.ax.tick_params(labelsize=9)
            cb.set_label(r'$\omega L/U$ (symmetric-log)' if col<2 else r'$(\omega_{\mathrm{NN}}-\omega_{\mathrm{CFD}})L/U$ (linear)',fontsize=10)
            if col<2:cb.set_ticks([-100,-10,-1,0,1,10,100])
            else:
                cb.formatter=ScalarFormatter(useMathText=True);cb.formatter.set_powerlimits((-2,2));cb.update_ticks()
    save(fig,'vorticity_fields_and_signed_error_Re1000_Re7500')
    with (OUT/'vorticity_metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    x=np.linspace(0,1,17);xx,yy=np.meshgrid(x,x)
    np.testing.assert_allclose(vorticity(dict(x=x,y=x,u=-yy,v=xx))['w'],2.,atol=1e-14)
    return dict(definition='omega=dv/dx-du/dy; centered curl of each velocity field on the same 511x511 interior nodes; no extrapolated wall values',
                is_direct_neural_output=False,levels=VORTICITY_LEVELS.tolist(),
                common_color_bounds=[-bound,bound],color_scale='Symmetric-log, linthresh=1, full extrema retained; differences linear per Re',
                solid_body_rotation_test='passed: u=-y, v=x gives omega=+2',metrics=rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("Loading and verifying frozen sources...",flush=True)
    cases,selection,coeff_report=load()
    verification=pressure_verification()
    print("Pressure manufactured-solution verification:",json.dumps(verification),flush=True)
    pressures={};rows=[];arrays={}
    for re,methods in cases.items():
        pressures[re]={}
        truth=methods["CFD"]
        for name,f in methods.items():
            z=pressure(f,re);pressures[re][name]=z
            label="cfd" if name=="CFD" else "nn"
            arrays[f"re{re}_{label}_p"]=z["p"]
            ue=np.hypot(f["u"]-truth["u"],f["v"]-truth["v"])
            row=dict(Re=re,method=name,role="training" if re==1000 else "held_out_test",
                velocity_relative_l2=float(np.linalg.norm(ue)/np.linalg.norm(np.hypot(truth["u"],truth["v"]))),
                pressure_min=float(z["p"].min()),pressure_max=float(z["p"].max()),
                pressure_gradient_fit_rms=z["gradient_fit_rms"],
                pressure_gradient_fit_relative=z["gradient_fit_relative"])
            if name!="CFD":
                delta=z["p"]-pressures[re]["CFD"]["p"]
                row.update(pressure_relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(pressures[re]["CFD"]["p"])),
                           pressure_rms=float(np.sqrt(np.mean(delta**2))),pressure_linf=float(abs(delta).max()))
            else:row.update(pressure_relative_l2=0.,pressure_rms=0.,pressure_linf=0.)
            rows.append(row)
        arrays["pressure_x"]=z["x"];arrays["pressure_y"]=z["y"]
    # Same pressure recovery on the two coarser CFD fields, with a common gauge on matched nodes.
    mesh=[]
    for re in (1000,7500):
        ref=pressures[re]["CFD"]
        for cells in (128,256):
            path=ROOT/f"results/cavity_highre_ghia/n{cells}/re{re}.npz"
            meta=json.loads(path.with_suffix(".json").read_text())
            checked(path,meta["field_sha256"])
            f=read_npz(path);z=pressure(f,re)
            ind=np.rint(z["x"]*512).astype(int)-1
            finer=ref["p"][np.ix_(ind,ind)]
            offset=float(np.mean(z["p"]-finer))
            delta=z["p"]-finer-offset
            mesh.append(dict(Re=re,cells=cells,pressure_rms=float(np.sqrt(np.mean(delta**2))),
                pressure_relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(finer-finer.mean())),
                gauge_offset=offset))
    print("Metrics:",json.dumps(rows),flush=True)
    print("Pressure grid sensitivity:",json.dumps(mesh),flush=True)
    flow_plot(cases);zoom_plot(cases)
    levels=pressure_plot(cases,pressures);pressure_fields(pressures)
    vort_info=vorticity_plots(cases)
    for name,data in (("comparison_metrics.csv",rows),("pressure_mesh_sensitivity.csv",mesh)):
        with (OUT/name).open("w",newline="") as file:
            w=csv.DictWriter(file,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    changed=[p for p,v in TRACKED.items() if digest(p)!=v]
    if changed:raise RuntimeError("Scientific source changed: "+repr(changed))
    receipt=dict(source_hashes={Path(p).relative_to(ROOT).as_posix():v for p,v in TRACKED.items()},source_files_unchanged=True,
        vorticity=vort_info,
        model=selection["selected_configuration"],seeds=selection["seeds"],
        prediction_provenance=coeff_report,streamline_seeds=SEEDS.tolist(),
        streamlines="Same prescribed seeds, independent per-seed occupancy masks and integration settings for both methods; no other trajectory can suppress a seed; black solid and orange dashed.",
        pressure_method="Unweighted least-squares integration of centered steady momentum gradients on 511x511 interior nodes; uniform zero-interior-mean gauge for both fields; no local recentering.",
        generator_sha256_lf=hashlib.sha256(Path(__file__).read_text().replace("\r\n","\n").encode()).hexdigest(),
        pressure_is_model_output=False,pressure_is_postprocessed=True,
        pressure_reference_is_independent_pressure_solver=False,
        pressure_verification=verification,pressure_levels=levels,pressure_grid_sensitivity=mesh,
        pressure_color_scale="Same full-extrema symmetric-log norm across both Re and both methods; linthresh=0.02. Signed errors use linear scales.",
        limitations=["Re1000 was in training; Re7500 was held out of all fitting and selection.",
                     "Pressure is derived from velocity, not learned, and is not independently pressure-validated.",
                     "A scalar pressure cannot exactly balance a non-integrable sampled momentum gradient; fit residuals are reported.",
                     "The CFD reference retains significant mesh sensitivity.",
                     "No geometry/physics or frozen models changed; no refitting."],
        output_sha256={p.name:digest(p) for p in OUT.iterdir() if p.suffix in (".png",".pdf",".npz",".csv") and not p.name.startswith(("reference", "qa_"))})
    (OUT/"verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Completed",OUT,flush=True)
if __name__=="__main__":main()
