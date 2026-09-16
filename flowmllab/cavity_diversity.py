"""Week 4 Lab 4: boundary-conditioned cavity operators.

Independent educational extension of the Week-4 streamfunction/vorticity FOM.
Arrays are [case, y, x]; all fields use U_ref=max|U_lid| and L scaling.
No test field participates in normalization, POD, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

PROFILES = {
    "pilot": dict(n=25, train=16, val=4, test=6, epochs=120, seeds=[7]),
    "classroom": dict(n=33, train=64, val=12, test=20, epochs=400, seeds=[7, 17, 27]),
    "research": dict(n=65, train=256, val=40, test=60, epochs=1000, seeds=[7, 17, 27]),
}
MODELS = ["coordinate_mlp", "pod_deeponet", "unet", "fno", "fno_physics"]


def lid_profile(x, coefficients):
    """Positive variable lid, max-normalized on a fixed dense reference grid.

    All experiments use the conventional discontinuous top-corner lid;
    side-wall velocity takes precedence at the two corner nodes.
    """
    coefficients = np.asarray(coefficients)
    dense = np.linspace(0, 1, 8193)
    k = np.arange(1, len(coefficients) + 1)
    raw = 1 + np.sin(np.pi * np.outer(np.asarray(x), k)) @ coefficients
    reference = 1 + np.sin(np.pi * np.outer(dense, k)) @ coefficients
    if np.min(reference) <= 0:
        raise ValueError("This lesson restricts the lid to positive profiles")
    return raw / np.max(np.abs(reference))


def case_manifest(config, seed=20260916):
    rng = np.random.default_rng(seed)
    cases = []
    for split in ("train", "val", "test"):
        # Identical Re distributions in both families isolate the BC effect.
        re_values = rng.uniform(100., 400., config[split])
        for family in ("constant", "diverse"):
            for j, re in enumerate(re_values):
                coeff = np.zeros(5)
                if family == "diverse":
                    a = rng.normal(size=3) / np.arange(1, 4)
                    coeff[:3] = a * rng.uniform(.15, .65) / np.sum(np.abs(a))
                nu, length = .0025, 1.
                speed = re * nu / length
                cases.append(dict(id=f"{family}_{split}_{j:04d}", family=family,
                                  split=split, Re=float(re), U_ref=float(speed),
                                  nu=nu, L=length, coefficients=coeff.tolist()))
    for family in ("ood_shape", "ood_re"):
        for j in range(config["test"]):
            k = 5 if family == "ood_shape" else 3
            a = rng.normal(size=k) / np.sqrt(np.arange(1, k + 1))
            coeff = np.zeros(5)
            coeff[:k] = a * rng.uniform(.7, .85) / np.sum(np.abs(a)) if k == 5 else a * rng.uniform(.15, .65) / np.sum(np.abs(a))
            re = rng.uniform(100, 400) if family == "ood_shape" else rng.uniform(450, 600)
            cases.append(dict(id=f"{family}_{j:04d}", family=family, split="test",
                              Re=float(re), U_ref=float(re*.0025), nu=.0025, L=1.,
                              coefficients=coeff.tolist()))
    return cases


def poisson(omega, h):
    n = omega.shape[-1]
    lam = 2*(np.cos(np.pi*np.arange(1, n-1)/(n-1))-1)/h**2
    psi = np.zeros_like(omega)
    psi[..., 1:-1, 1:-1] = idstn(
        dstn(-omega[..., 1:-1, 1:-1], type=1, norm="ortho", axes=(-2, -1)) /
        (lam[:, None]+lam[None, :]), type=1, norm="ortho", axes=(-2, -1))
    return psi


def fields(psi, lid):
    h = 1/(psi.shape[-1]-1)
    u, v = np.zeros_like(psi), np.zeros_like(psi)
    u[..., 1:-1, 1:-1] = (psi[..., 2:, 1:-1]-psi[..., :-2, 1:-1])/(2*h)
    v[..., 1:-1, 1:-1] = -(psi[..., 1:-1, 2:]-psi[..., 1:-1, :-2])/(2*h)
    u[..., -1, 1:-1] = lid[..., 1:-1]
    return u, v


def vorticity_bc(omega, psi, lid, h):
    omega[..., 0, 1:-1] = -2*psi[..., 1, 1:-1]/h**2
    omega[..., -1, 1:-1] = -2*psi[..., -2, 1:-1]/h**2-2*lid[..., 1:-1]/h
    omega[..., 1:-1, 0] = -2*psi[..., 1:-1, 1]/h**2
    omega[..., 1:-1, -1] = -2*psi[..., 1:-1, -2]/h**2


def solve_cases(cases, n=33, tolerance=2e-5, max_steps=60000, dt_factor=1.):
    """Batched central differences, DST Poisson, explicit midpoint RK2.

    Acceptance uses the *steady PDE RHS*, not timestep-scaled iterate change.
    Fixed-point stopping is checked twice; unconverged labels are rejected.
    """
    started = time.perf_counter()
    h = 1/(n-1)
    x = np.linspace(0, 1, n)
    lid = np.array([lid_profile(x, c["coefficients"]) for c in cases])
    re = np.array([c["Re"] for c in cases])[:, None, None]
    dt = dt_factor * min(.18*h, .18*float(re.min())*h*h)
    w = np.zeros((len(cases), n, n))

    def rhs(q):
        p = poisson(q, h)
        q = q.copy()
        vorticity_bc(q, p, lid, h)
        u, v = fields(p, lid)
        out = np.zeros_like(q)
        dx = (q[:, 1:-1, 2:]-q[:, 1:-1, :-2])/(2*h)
        dy = (q[:, 2:, 1:-1]-q[:, :-2, 1:-1])/(2*h)
        lap = (q[:, 1:-1, 2:]+q[:, 1:-1, :-2]+q[:, 2:, 1:-1]+q[:, :-2, 1:-1]-4*q[:, 1:-1, 1:-1])/h**2
        out[:, 1:-1, 1:-1] = -u[:, 1:-1, 1:-1]*dx-v[:, 1:-1, 1:-1]*dy+lap/re
        return out

    passed = 0
    for step in range(1, max_steps+1):
        k1 = rhs(w)
        w += dt*rhs(w + .5*dt*k1)
        if step % 200 == 0:
            residual = np.max(np.abs(rhs(w)), axis=(1, 2))
            if not np.isfinite(w).all():
                raise FloatingPointError("CFD instability: reduce dt or refine grid")
            passed = passed+1 if np.all(residual < tolerance) else 0
            if passed >= 2:
                break
    psi = poisson(w, h)
    vorticity_bc(w, psi, lid, h)
    residual = np.max(np.abs(rhs(w)), axis=(1, 2))
    if np.any(residual >= tolerance):
        bad = [(cases[i]["id"], float(r)) for i, r in enumerate(residual) if r >= tolerance]
        raise RuntimeError(f"Unconverged cases; not admitted as labels: {bad}")
    u, v = fields(psi, lid)
    return dict(psi=psi, omega=w, u=u, v=v, lid=lid, residual=residual,
                steps=step, dt=dt, seconds=time.perf_counter()-started)


def generate_dataset(out, config):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cases = case_manifest(config)
    manifest = dict(config=config, cases=cases, convention="[case,y,x]; U_ref=max absolute lid speed; fixed nu=0.0025, L=1", source="new FD streamfunction-vorticity simulations; no synthetic flow labels")
    target = out/"dataset.npz"
    if target.exists():
        old = json.loads((out/"manifest.json").read_text())
        if old != manifest:
            raise ValueError("Dataset configuration changed: use a new output directory")
        return cases, dict(np.load(target))
    data = solve_cases(cases, config["n"])
    np.savez_compressed(target, **data)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return cases, data


def tensor_features(cases, n):
    x = np.linspace(0, 1, n)
    bc = np.array([lid_profile(x, c["coefficients"]) for c in cases])
    # Fixed physical scales, not fit to held-out fields; zero BC features retain
    # unit scale rather than amplifying unseen coefficients by near-zero std.
    descriptors = np.array([[c["Re"]/400, *c["coefficients"]] for c in cases], dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    images = np.stack([np.broadcast_to(bc[:, None, :], (len(cases), n, n)),
                       np.broadcast_to(descriptors[:, :1, None], (len(cases), n, n)),
                       np.broadcast_to(xx, (len(cases), n, n)),
                       np.broadcast_to(yy, (len(cases), n, n))], axis=1)
    return descriptors, images.astype(np.float32)


def build_model(name, n, train_psi, rank=12, descriptor_dim=6, input_channels=4):
    import torch
    from torch import nn
    import torch.nn.functional as F
    x = torch.linspace(0, 1, n)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    mask = (16*xx*(1-xx)*yy*(1-yy))[None, None]

    class Base(nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer("mask", mask)

    class Coordinate(Base):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(descriptor_dim+2, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))
        def forward(self, desc, img):
            coords = img[:, 2:4].permute(0, 2, 3, 1)
            d = desc[:, None, None, :].expand(-1, n, n, -1)
            return self.net(torch.cat([d, coords], -1)).permute(0, 3, 1, 2)*self.mask

    class POD(Base):
        def __init__(self):
            super().__init__()
            y = torch.as_tensor(train_psi/.1, dtype=torch.float32).flatten(1)
            mean = y.mean(0); _, s, vt = torch.linalg.svd(y-mean, full_matrices=False)
            r = min(rank, max(1, int((s > s[0]*1e-6).sum())))
            self.register_buffer("mean", mean)
            self.register_buffer("basis", vt[:r])
            self.net = nn.Sequential(nn.Linear(descriptor_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, r))
        def forward(self, desc, img):
            # Fixed POD trunk, train-only basis; no test snapshots in SVD.
            return (self.mean+self.net(desc)@self.basis).reshape(-1, 1, n, n)

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GELU())

    class UNet(Base):
        def __init__(self):
            super().__init__(); self.a=block(input_channels, 12); self.b=block(12, 24)
            self.c=block(36, 12); self.out=nn.Conv2d(12, 1, 1)
        def forward(self, desc, img):
            a=self.a(img); b=self.b(F.avg_pool2d(a, 2))
            c=self.c(torch.cat([a,F.interpolate(b,size=(n,n),mode="bilinear",align_corners=False)],1))
            return self.out(c)*self.mask

    class Spectral(nn.Module):
        def __init__(self, width=8, modes=4):
            super().__init__(); self.m=modes
            self.w=nn.Parameter(torch.randn(2,width,width,modes,modes,dtype=torch.cfloat)/width)
        def forward(self, x):
            z=torch.fft.rfft2(x); out=torch.zeros_like(z); m=self.m
            out[:,:,:m,:m]=torch.einsum("bixy,ioxy->boxy",z[:,:,:m,:m],self.w[0])
            out[:,:,-m:,:m]=torch.einsum("bixy,ioxy->boxy",z[:,:,-m:,:m],self.w[1])
            return torch.fft.irfft2(out,s=x.shape[-2:])

    class FNO(Base):
        def __init__(self):
            super().__init__(); self.lift=nn.Conv2d(input_channels,8,1)
            self.spec=nn.ModuleList([Spectral() for _ in range(3)])
            self.local=nn.ModuleList([nn.Conv2d(8,8,1) for _ in range(3)])
            self.out=nn.Sequential(nn.Conv2d(8,16,1),nn.GELU(),nn.Conv2d(16,1,1))
        def forward(self, desc, img):
            z=self.lift(F.pad(img,(3,3,3,3),mode="reflect"))
            for s,l in zip(self.spec,self.local): z=F.gelu(s(z)+l(z))
            return self.out(z)[...,3:-3,3:-3]*self.mask

    return {"coordinate_mlp":Coordinate,"pod_deeponet":POD,"unet":UNet,"fno":FNO,"fno_physics":FNO}[name]()


def derivative_loss(pred, target, h):
    import torch
    dy=lambda z:(z[...,2:,1:-1]-z[...,:-2,1:-1])/(2*h)
    dx=lambda z:(z[...,1:-1,2:]-z[...,1:-1,:-2])/(2*h)
    # Prediction is psi/0.1, restore physical nondimensional velocity scale.
    return .01*(torch.mean((dx(pred)-dx(target))**2)+torch.mean((dy(pred)-dy(target))**2))


def physics_loss(pred, desc, img, h):
    import torch
    p=.1*pred[:,0]
    dx=lambda z:(z[:,1:-1,2:]-z[:,1:-1,:-2])/(2*h)
    dy=lambda z:(z[:,2:,1:-1]-z[:,:-2,1:-1])/(2*h)
    lap=lambda z:(z[:,1:-1,2:]+z[:,1:-1,:-2]+z[:,2:,1:-1]+z[:,:-2,1:-1]-4*z[:,1:-1,1:-1])/h**2
    w=-lap(p); u=dy(p); v=-dx(p)
    residual=u[:,1:-1,1:-1]*dx(w)+v[:,1:-1,1:-1]*dy(w)-lap(w)/(400*desc[:,0,None,None])
    # One-sided wall derivative consistency, rather than a trivially imposed BC.
    top=(3*p[:,-1,2:-2]-4*p[:,-2,2:-2]+p[:,-3,2:-2])/(2*h)-img[:,0,-1,2:-2]
    bottom=(-3*p[:,0,2:-2]+4*p[:,1,2:-2]-p[:,2,2:-2])/(2*h)
    left=(-3*p[:,2:-2,0]+4*p[:,2:-2,1]-p[:,2:-2,2])/(2*h)
    right=(3*p[:,2:-2,-1]-4*p[:,2:-2,-2]+p[:,2:-2,-3])/(2*h)
    return 1e-4*torch.mean(residual**2)+.05*sum(torch.mean(z*z) for z in (top,bottom,left,right))


def evaluate(psi, ref, lid):
    h=1/(psi.shape[-1]-1)
    u,v=fields(psi,lid); ur,vr=fields(ref,lid)
    interior=(slice(None),slice(1,-1),slice(1,-1))
    rel=lambda a,b:np.linalg.norm((a-b).reshape(len(a),-1),axis=1)/np.maximum(np.linalg.norm(b.reshape(len(b),-1),axis=1),1e-12)
    uv=np.stack([u[interior],v[interior]],1); uvref=np.stack([ur[interior],vr[interior]],1)
    du=(u[:,2:-2,3:-1]-u[:,2:-2,1:-3])/(2*h)
    dv=(v[:,3:-1,2:-2]-v[:,1:-3,2:-2])/(2*h)
    wall=(3*psi[:,-1,2:-2]-4*psi[:,-2,2:-2]+psi[:,-3,2:-2])/(2*h)-lid[:,2:-2]
    loc=np.array([np.unravel_index(np.argmin(p),p.shape) for p in psi])*h
    loc_ref=np.array([np.unravel_index(np.argmin(p),p.shape) for p in ref])*h
    lap=lambda p:(p[:,1:-1,2:]+p[:,1:-1,:-2]+p[:,2:,1:-1]+p[:,:-2,1:-1]-4*p[:,1:-1,1:-1])/h**2
    return dict(psi_rel_l2=rel(psi,ref),velocity_rel_l2=rel(uv,uvref),
                interior_vorticity_rel_l2=rel(-lap(psi),-lap(ref)),
                divergence_core_rms=np.sqrt(np.mean((du+dv)**2,axis=(1,2))),
                lid_derivative_rmse=np.sqrt(np.mean(wall**2,axis=1)),
                primary_vortex_location_error=np.linalg.norm(loc-loc_ref,axis=1))


def benchmark(out, config, cases, data):
    import torch
    import pandas as pd
    torch.set_num_threads(2)
    out=Path(out); (out/"checkpoints").mkdir(exist_ok=True)
    desc,img=tensor_features(cases,config["n"])
    desc=torch.tensor(desc); img=torch.tensor(img)
    target=torch.tensor(data["psi"][:,None]/.1,dtype=torch.float32)
    rows=[]; histories=[]; predictions={}
    h=1/(config["n"]-1)
    for family in ("constant","diverse"):
        train=np.array([i for i,c in enumerate(cases) if c["family"]==family and c["split"]=="train"])
        val=np.array([i for i,c in enumerate(cases) if c["family"]==family and c["split"]=="val"])
        for seed in config["seeds"]:
            for name in MODELS:
                torch.manual_seed(seed)
                model=build_model(name,config["n"],data["psi"][train])
                optimizer=torch.optim.Adam(model.parameters(),lr=.003)
                best=float("inf"); started=time.perf_counter()
                for epoch in range(config["epochs"]):
                    model.train(); optimizer.zero_grad()
                    pred=model(desc[train],img[train])
                    loss=((pred-target[train])**2).mean()+derivative_loss(pred,target[train],h)
                    if name=="fno_physics":loss=loss+physics_loss(pred,desc[train],img[train],h)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        vp=model(desc[val],img[val]); vl=((vp-target[val])**2).mean()+derivative_loss(vp,target[val],h)
                    if float(vl)<best:best=float(vl); selected=copy.deepcopy(model.state_dict()); chosen=epoch+1
                    histories.append(dict(family=family,seed=seed,model=name,epoch=epoch+1,train=float(loss.detach()),validation=float(vl)))
                model.load_state_dict(selected); model.eval()
                elapsed=time.perf_counter()-started
                checkpoint=out/"checkpoints"/f"{family}_{name}_{seed}.pt"
                torch.save(dict(state_dict=selected,config=config,model=name,family=family,seed=seed,epoch=chosen),checkpoint)
                checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                params=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
                for tf in ("constant","diverse","ood_shape","ood_re"):
                    ids=np.array([i for i,c in enumerate(cases) if c["family"]==tf and c["split"]=="test"])
                    with torch.no_grad():
                        model(desc[ids],img[ids]) # warm-up
                        tick=time.perf_counter(); pp=.1*model(desc[ids],img[ids])[:,0].numpy(); infer=(time.perf_counter()-tick)/len(ids)
                    met=evaluate(pp,data["psi"][ids],data["lid"][ids])
                    for j,i in enumerate(ids):
                        rows.append(dict(train_family=family,test_family=tf,model=name,seed=seed,case=cases[i]["id"],Re=cases[i]["Re"],parameters=params,epoch=chosen,training_seconds=elapsed,inference_seconds=infer,checkpoint_sha256=checkpoint_hash,**{k:float(v[j]) for k,v in met.items()}))
                    predictions[f"{family}_{name}_{seed}_{tf}"]=pp
                print(f"{family:8s} {name:15s} seed={seed} selected={chosen} val={best:.5g} time={elapsed:.1f}s",flush=True)
                pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
                pd.DataFrame(histories).to_csv(out/"training.csv",index=False)
    # Non-neural baselines: nearest descriptor and train-only RBF interpolation.
    from scipy.interpolate import RBFInterpolator
    for family in ("constant","diverse"):
        train=np.array([i for i,c in enumerate(cases) if c["family"]==family and c["split"]=="train"])
        features=desc.numpy(); active=np.ptp(features[train],axis=0)>1e-7
        rbf=RBFInterpolator(features[train][:,active],data['psi'][train].reshape(len(train),-1),smoothing=1e-7)
        for tf in ("constant","diverse","ood_shape","ood_re"):
            ids=np.array([i for i,c in enumerate(cases) if c["family"]==tf and c["split"]=="test"])
            near=np.argmin(np.sum((desc[ids,None].numpy()-desc[train].numpy()[None])**2,-1),axis=1)
            for label,pp in [('nearest',data['psi'][train[near]]),('rbf',rbf(features[ids][:,active]).reshape(-1,config['n'],config['n']))]:
                met=evaluate(pp,data["psi"][ids],data["lid"][ids])
                predictions[f"{family}_{label}_0_{tf}"]=pp
                for j,i in enumerate(ids):rows.append(dict(train_family=family,test_family=tf,model=label,seed=0,case=cases[i]["id"],Re=cases[i]["Re"],**{k:float(v[j]) for k,v in met.items()}))
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    np.savez_compressed(out/"predictions.npz",**predictions)
    summary=pd.DataFrame(rows).groupby(["train_family","test_family","model"]).agg(velocity_mean=("velocity_rel_l2","mean"),velocity_std=("velocity_rel_l2","std"),psi_mean=("psi_rel_l2","mean")).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    return summary


def plot_results(out, cases, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    out=Path(out); metrics=pd.read_csv(out/"metrics.csv")
    combos=[("constant","constant"),("constant","diverse"),("diverse","constant"),("diverse","diverse")]
    fig,axs=plt.subplots(1,4,figsize=(15,4),sharey=True,layout="constrained")
    for ax,(a,b) in zip(axs,combos):
        sub=metrics[(metrics.train_family==a)&(metrics.test_family==b)]
        means=sub.groupby("model").velocity_rel_l2.mean()*100
        ax.bar(means.index,means.values); ax.tick_params(axis="x",rotation=75)
        ax.set_title(f"Train {a}\nTest {b}"); ax.set_yscale("log"); ax.grid(axis="y",alpha=.25)
    axs[0].set_ylabel("Mean interior velocity relative L2 (%)")
    fig.suptitle("Four transfer tests - educational finite-grid labels");fig.savefig(out/"four_combinations.png",dpi=150);plt.close(fig)
    pred=dict(np.load(out/"predictions.npz")); seed=int(metrics[metrics.model=="fno"].seed.iloc[0])
    ids=[i for i,c in enumerate(cases) if c["family"]=="diverse" and c["split"]=="test"]
    ref=data["psi"][ids[0]]
    ps=[ref,pred[f"constant_fno_{seed}_diverse"][0],pred[f"diverse_fno_{seed}_diverse"][0]]
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout="constrained"); x=np.linspace(0,1,ref.shape[0]); levels=np.linspace(min(z.min() for z in ps),max(z.max() for z in ps)+1e-10,22)
    for ax,p,title in zip(axs,ps,["FD reference","FNO trained on constant lid","FNO trained on diverse lids"]):
        im=ax.contourf(x,x,p,levels=levels,cmap="viridis");ax.contour(x,x,p,levels=levels[1:-1:2],colors="white",linewidths=.4); ax.set_aspect("equal");ax.set_title(title);ax.set_xlabel("x/L")
    axs[0].set_ylabel("y/L");fig.colorbar(im,ax=axs,label="psi / (U_ref L)");fig.savefig(out/"heldout_streamfunctions.png",dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--profile",choices=PROFILES,default="pilot");parser.add_argument("--out",default="results/cavity_diversity_pilot")
    args=parser.parse_args(); config=PROFILES[args.profile];out=Path(args.out)
    cases,data=generate_dataset(out,config)
    print(f"Accepted {len(cases)} CFD cases, max RHS={max(data['residual']):.3g}",flush=True)
    benchmark(out,config,cases,data);plot_results(out,cases,data)
    (out/"run_status.json").write_text(json.dumps(dict(status="executed",profile=args.profile,config=config,case_count=len(cases),dataset_sha256=hashlib.sha256((out/"dataset.npz").read_bytes()).hexdigest(),limitations=["Same-grid educational references; no claim of grid independence", "Pilot is one seed and small training budget", "Fixed square geometry; no mesh/geometry transfer", "Parameter counts differ; report architecture and budget together", "Test cases are regression cases after this first evaluation"]),indent=2))


if __name__=="__main__":main()
