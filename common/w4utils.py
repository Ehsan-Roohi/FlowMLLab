"""Week 4 cavity-project utilities (v6.3).

Each student generates the full Reynolds-number family. The primary
streamfunction-vorticity solver produces velocity, streamfunction, and
vorticity. A dimensionless zero-mean pressure field is then recovered from the
steady momentum equation by a least-squares pressure-gradient reconstruction.
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

W4_UTILS_VERSION = "6.3"
from scipy.fft import dstn, idstn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

GHIA = {
    100: {
        "y": np.array([1.0000, .9766, .9688, .9609, .9531, .8516, .7344, .6172,
                       .5000, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.0]),
        "u": np.array([1.0, .84123, .78871, .73722, .68717, .23151, .00332,
                       -.13641, -.20581, -.21090, -.15662, -.10150, -.06434,
                       -.04775, -.04192, -.03717, 0.0]),
        "x": np.array([1.0000, .9688, .9609, .9531, .9453, .9063, .8594, .8047,
                       .5000, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0.0]),
        "v": np.array([0.0, -.05906, -.07391, -.08864, -.10313, -.16914,
                       -.22445, -.24533, .05454, .17527, .17507, .16077,
                       .12317, .10890, .10091, .09233, 0.0]),
    },
    400: {
        "y": np.array([1.0000, .9766, .9688, .9609, .9531, .8516, .7344, .6172,
                       .5000, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.0]),
        "u": np.array([1.0, .75837, .68439, .61756, .55892, .29093, .16256,
                       .02135, -.11477, -.17119, -.32726, -.24299, -.14612,
                       -.10338, -.09266, -.08186, 0.0]),
        "x": np.array([1.0000, .9688, .9609, .9531, .9453, .9063, .8594, .8047,
                       .5000, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0.0]),
        "v": np.array([0.0, -.12146, -.15663, -.19254, -.22847, -.23827,
                       -.44993, -.38598, .05186, .30174, .30203, .28124,
                       .22965, .20920, .19713, .18360, 0.0]),
    },
}

CASES = (
    {"Re": 100, "split": "train"},
    {"Re": 150, "split": "train"},
    {"Re": 175, "split": "test"},
    {"Re": 200, "split": "train"},
    {"Re": 225, "split": "train"},
    {"Re": 250, "split": "train"},
    {"Re": 275, "split": "test"},
    {"Re": 300, "split": "train"},
    {"Re": 350, "split": "train"},
    {"Re": 375, "split": "test"},
    {"Re": 400, "split": "train"},
)
CASE_MAP = {i + 1: dict(case) for i, case in enumerate(CASES)}
PACKAGE_VERSION = "W4-v6.3"


def require_file(name: str | Path) -> Path:
    name = Path(name)
    candidates = [name, Path.cwd()/name.name, Path('/content')/name.name]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Required file '{name.name}' was not found. Put it in the same Colab "
        "Files pane / working folder as this notebook, then run the cell again."
    )


_DATASET_REQUIRED_KEYS = {"x", "y", "Re", "u", "v", "p", "psi", "split"}


def inspect_npz(path: str | Path, required_keys=None) -> dict:
    """Return a compact diagnostic without raising on a damaged upload.

    An ``.npz`` file is a ZIP archive. Colab can occasionally contain a zero-byte,
    interrupted, or HTML file that merely has an ``.npz`` suffix. This function
    distinguishes that situation from a valid NumPy archive.
    """
    p = Path(path)
    info = {"path": str(p), "exists": p.exists(), "size_bytes": 0,
            "is_zip": False, "valid": False, "keys": [], "error": ""}
    if not p.exists():
        info["error"] = "file not found"
        return info
    try:
        info["size_bytes"] = p.stat().st_size
        info["is_zip"] = zipfile.is_zipfile(p)
        if not info["is_zip"]:
            head = p.read_bytes()[:16]
            info["error"] = f"not a valid NPZ/ZIP archive; first bytes={head!r}"
            return info
        with zipfile.ZipFile(p, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                info["error"] = f"corrupt ZIP member: {bad_member}"
                return info
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            info["keys"] = sorted(keys)
            needed = set(required_keys or [])
            missing = sorted(needed - keys)
            if missing:
                info["error"] = "missing required arrays: " + ", ".join(missing)
                return info
        info["valid"] = True
        return info
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info


def _dataset_candidates(preferred: str | Path = "cavity_data.npz"):
    preferred = Path(preferred)
    roots = [preferred.parent if str(preferred.parent) not in ("", ".") else Path.cwd(),
             Path.cwd(), Path("/content")]
    candidates, seen = [], set()

    def add(p):
        p = Path(p)
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p.absolute())
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    for p in [preferred, Path.cwd()/preferred.name, Path('/content')/preferred.name]:
        add(p)
    for root in roots:
        if root.exists():
            for p in sorted(root.glob("cavity_data*.npz"),
                            key=lambda q: q.stat().st_mtime if q.exists() else 0,
                            reverse=True):
                add(p)
    return candidates


def find_valid_dataset(preferred: str | Path = "cavity_data.npz") -> Path:
    """Find a valid Week-4 dataset, tolerating Colab duplicate suffixes.

    The exact filename is preferred. If it is damaged, a valid duplicate such as
    ``cavity_data(2).npz`` is selected and the rejected files are reported.
    """
    diagnostics = [inspect_npz(p, _DATASET_REQUIRED_KEYS)
                   for p in _dataset_candidates(preferred)]
    valid = [d for d in diagnostics if d["valid"]]
    if valid:
        chosen = Path(valid[0]["path"])
        rejected = [d for d in diagnostics if d["exists"] and not d["valid"]]
        for d in rejected:
            warnings.warn(
                f"Ignoring invalid dataset candidate {d['path']} "
                f"({d['size_bytes']} bytes): {d['error']}"
            )
        if chosen.name != Path(preferred).name:
            print(f"Using valid dataset '{chosen.name}' instead of the damaged or absent "
                  f"'{Path(preferred).name}'.")
        return chosen
    present = [d for d in diagnostics if d["exists"]]
    if not present:
        raise FileNotFoundError(
            "No cavity_data*.npz file was found. Run Lab 1 through the dataset-assembly "
            "cell, then upload the generated cavity_data.npz beside this notebook."
        )
    lines = [f"- {d['path']} ({d['size_bytes']} bytes): {d['error']}" for d in present]
    raise ValueError(
        "No valid Week-4 cavity dataset was found. The files with an .npz suffix "
        "are damaged, incomplete, or missing required arrays:\n" + "\n".join(lines) +
        "\nDelete the invalid copies, rerun Lab 1, verify the success message, and "
        "upload the newly generated cavity_data.npz."
    )


def build_grid(N: int):
    x=np.linspace(0.,1.,N); y=np.linspace(0.,1.,N)
    X,Y=np.meshgrid(x,y)
    return x,y,X,Y,x[1]-x[0],y[1]-y[0]


def apply_vorticity_bc(psi,omega,U,dx,dy):
    omega[0,1:-1]=-2.*psi[1,1:-1]/dy**2
    omega[-1,1:-1]=-2.*psi[-2,1:-1]/dy**2-2.*U/dy
    omega[1:-1,0]=-2.*psi[1:-1,1]/dx**2
    omega[1:-1,-1]=-2.*psi[1:-1,-2]/dx**2
    omega[0,0]=.5*(omega[0,1]+omega[1,0])
    omega[0,-1]=.5*(omega[0,-2]+omega[1,-1])
    omega[-1,0]=.5*(omega[-1,1]+omega[-2,0])
    omega[-1,-1]=.5*(omega[-1,-2]+omega[-2,-1])
    return omega


def solve_poisson_dst(omega,dx,dy):
    rhs=-np.asarray(omega[1:-1,1:-1],dtype=float)
    ny,nx=rhs.shape
    ky=np.arange(1,ny+1); kx=np.arange(1,nx+1)
    lam_y=2.*(np.cos(np.pi*ky/(ny+1))-1.)/dy**2
    lam_x=2.*(np.cos(np.pi*kx/(nx+1))-1.)/dx**2
    rhs_hat=dstn(rhs,type=1,norm="ortho")
    psi_hat=rhs_hat/(lam_y[:,None]+lam_x[None,:])
    psi=np.zeros_like(omega,dtype=float)
    psi[1:-1,1:-1]=idstn(psi_hat,type=1,norm="ortho")
    return psi


def compute_velocity(psi,U,dx,dy):
    u=np.zeros_like(psi); v=np.zeros_like(psi)
    u[1:-1,1:-1]=(psi[2:,1:-1]-psi[:-2,1:-1])/(2.*dy)
    v[1:-1,1:-1]=-(psi[1:-1,2:]-psi[1:-1,:-2])/(2.*dx)
    u[-1,:]=U; u[:,[0,-1]]=0.
    v[[0,-1],:]=0.; v[:,[0,-1]]=0.
    return u,v


def advance_vorticity(omega,u,v,Re,dt,dx,dy):
    old=omega.copy()
    dwdx=(old[1:-1,2:]-old[1:-1,:-2])/(2.*dx)
    dwdy=(old[2:,1:-1]-old[:-2,1:-1])/(2.*dy)
    lap=((old[1:-1,2:]-2.*old[1:-1,1:-1]+old[1:-1,:-2])/dx**2+
         (old[2:,1:-1]-2.*old[1:-1,1:-1]+old[:-2,1:-1])/dy**2)
    conv=u[1:-1,1:-1]*dwdx+v[1:-1,1:-1]*dwdy
    omega[1:-1,1:-1]=old[1:-1,1:-1]+dt*(-conv+lap/Re)
    return omega,old


def _laplacian(field,dx,dy):
    d2x=np.gradient(np.gradient(field,dx,axis=1,edge_order=2),dx,axis=1,edge_order=2)
    d2y=np.gradient(np.gradient(field,dy,axis=0,edge_order=2),dy,axis=0,edge_order=2)
    return d2x+d2y


def momentum_pressure_gradient(u,v,Re,dx,dy):
    """Return dp/dx and dp/dy from the steady nondimensional momentum equation."""
    dudx=np.gradient(u,dx,axis=1,edge_order=2)
    dudy=np.gradient(u,dy,axis=0,edge_order=2)
    dvdx=np.gradient(v,dx,axis=1,edge_order=2)
    dvdy=np.gradient(v,dy,axis=0,edge_order=2)
    gx=-(u*dudx+v*dudy)+_laplacian(u,dx,dy)/Re
    gy=-(u*dvdx+v*dvdy)+_laplacian(v,dx,dy)/Re
    return gx,gy


@lru_cache(maxsize=8)
def _pressure_gradient_matrix(ny,nx,dx,dy):
    """Sparse edge-gradient matrix plus one gauge equation."""
    rows=[]; cols=[]; vals=[]; row=0
    for j in range(ny):
        for i in range(nx-1):
            a=j*nx+i; b=a+1
            rows.extend([row,row]); cols.extend([a,b]); vals.extend([-1./dx,1./dx]); row+=1
    for j in range(ny-1):
        for i in range(nx):
            a=j*nx+i; b=(j+1)*nx+i
            rows.extend([row,row]); cols.extend([a,b]); vals.extend([-1./dy,1./dy]); row+=1
    rows.append(row); cols.append(0); vals.append(1.0); row+=1
    return coo_matrix((vals,(rows,cols)),shape=(row,ny*nx)).tocsr()


def recover_pressure(u,v,Re,x,y):
    """Recover dimensionless pressure from velocity, then impose zero spatial mean.

    Pressure is absent from the primary streamfunction-vorticity solve. We first
    compute the pressure-gradient field implied by the steady momentum equation,
    then find the scalar pressure whose discrete edge gradients best match it.
    Because pressure is defined only up to an additive constant, the returned
    field is shifted to have zero spatial mean.
    """
    u=np.asarray(u,float); v=np.asarray(v,float); x=np.asarray(x,float); y=np.asarray(y,float)
    ny,nx=u.shape; dx=float(x[1]-x[0]); dy=float(y[1]-y[0])
    gx,gy=momentum_pressure_gradient(u,v,float(Re),dx,dy)
    rhs=[]
    for j in range(ny):
        for i in range(nx-1): rhs.append(.5*(gx[j,i]+gx[j,i+1]))
    for j in range(ny-1):
        for i in range(nx): rhs.append(.5*(gy[j,i]+gy[j+1,i]))
    rhs.append(0.0)
    A=_pressure_gradient_matrix(ny,nx,dx,dy)
    sol=lsqr(A,np.asarray(rhs),atol=1e-9,btol=1e-9,iter_lim=2500)
    p=sol[0].reshape(ny,nx)
    p-=p.mean()
    pgx=np.gradient(p,dx,axis=1,edge_order=2)
    pgy=np.gradient(p,dy,axis=0,edge_order=2)
    rel=float(np.sqrt(np.mean((pgx-gx)**2+(pgy-gy)**2))/(np.sqrt(np.mean(gx**2+gy**2))+1e-30))
    return p,{"pressure_grad_rel_residual":rel,"pressure_lsqr_iterations":int(sol[2])}


def run_cavity(Re=100,N=65,dt=1e-3,max_steps=24000,U=1.0,
               check_every=250,min_steps=5000,tol=5e-6,
               consecutive_required=3,verbose=True):
    if Re<=0 or N<17 or dt<=0: raise ValueError("Require Re>0, N>=17, and dt>0.")
    x,y,X,Y,dx,dy=build_grid(N)
    diffusion_limit=Re*min(dx,dy)**2/4.
    if dt>.8*diffusion_limit:
        warnings.warn(f"dt={dt:g} is close to/above explicit diffusion limit {diffusion_limit:g}.")
    psi=np.zeros((N,N)); omega=np.zeros_like(psi)
    omega=apply_vorticity_bc(psi,omega,U,dx,dy)
    history_steps=[]; history_residual=[]; passed=0; converged=False; final_step=max_steps
    for step in range(1,max_steps+1):
        psi=solve_poisson_dst(omega,dx,dy)
        omega=apply_vorticity_bc(psi,omega,U,dx,dy)
        u,v=compute_velocity(psi,U,dx,dy)
        omega,old=advance_vorticity(omega,u,v,Re,dt,dx,dy)
        omega=apply_vorticity_bc(psi,omega,U,dx,dy)
        if not np.isfinite(omega).all(): raise FloatingPointError("Non-finite vorticity: reduce dt.")
        if step%check_every==0:
            residual=np.linalg.norm(omega-old)/(np.linalg.norm(omega)+1e-30)
            history_steps.append(step); history_residual.append(residual)
            if verbose: print(f"Re={Re:>4g} | step {step:6d} | residual {residual:.3e}")
            passed=passed+1 if (step>=min_steps and residual<tol) else 0
            if passed>=consecutive_required:
                converged=True; final_step=step; break
    psi=solve_poisson_dst(omega,dx,dy)
    omega=apply_vorticity_bc(psi,omega,U,dx,dy)
    u,v=compute_velocity(psi,U,dx,dy)
    p,pinfo=recover_pressure(u,v,Re,x,y)
    return {"x":x,"y":y,"X":X,"Y":Y,"psi":psi,"omega":omega,"u":u,"v":v,"p":p,
            "Re":float(Re),"N":int(N),"dt":float(dt),"U":float(U),"steps":int(final_step),
            "converged":bool(converged),"residual_steps":np.asarray(history_steps),
            "residual_values":np.asarray(history_residual),
            "final_residual":float(history_residual[-1]) if history_residual else np.nan,
            "solver":"streamfunction-vorticity + DST Poisson; pressure recovered from momentum",
            **pinfo}


def centerline_profiles(result):
    mid=len(result["x"])//2
    return result["u"][:,mid]/result["U"], result["v"][mid,:]/result["U"]


def pressure_centerlines(p):
    mid=p.shape[0]//2
    return p[:,mid],p[mid,:]


def ghia_errors(result):
    key=int(round(float(result["Re"])))
    if key not in GHIA: return np.nan,np.nan
    ref=GHIA[key]; uc,vc=centerline_profiles(result)
    ui=np.interp(ref["y"],result["y"],uc); vi=np.interp(ref["x"],result["x"],vc)
    return float(np.linalg.norm(ui-ref["u"])/np.linalg.norm(ref["u"])),float(np.linalg.norm(vi-ref["v"])/np.linalg.norm(ref["v"]))


def physical_metrics(result):
    x,y,u,v,psi,p=result["x"],result["y"],result["u"],result["v"],result["psi"],result["p"]
    dx,dy=x[1]-x[0],y[1]-y[0]
    div=np.gradient(u,dx,axis=1)+np.gradient(v,dy,axis=0); interior=div[1:-1,1:-1]
    j,i=np.unravel_index(np.argmin(psi),psi.shape); eu,ev=ghia_errors(result)
    return {"div_l2":float(np.sqrt(np.mean(interior**2))),"div_linf":float(np.max(np.abs(interior))),
            "vortex_x":float(x[i]),"vortex_y":float(y[j]),"psi_min":float(psi[j,i]),
            "speed_max":float(np.max(np.hypot(u,v))),"ghia_Eu":eu,"ghia_Ev":ev,
            "pressure_min":float(p.min()),"pressure_max":float(p.max()),
            "pressure_range":float(p.max()-p.min()),"pressure_mean":float(p.mean()),
            "pressure_grad_rel_residual":float(result["pressure_grad_rel_residual"])}


def quality_gate(result,residual_limit=1e-5,ghia_limit=.20):
    m=physical_metrics(result)
    checks={"finite_fields":bool(all(np.isfinite(result[k]).all() for k in ("u","v","p","psi","omega"))),
            "residual_below_limit":bool(result["final_residual"]<residual_limit),
            "reasonable_speed":bool(m["speed_max"]<=1.25*result["U"]),
            "converged_flag":bool(result["converged"]),
            "pressure_zero_mean":bool(abs(m["pressure_mean"])<1e-10)}
    if int(round(result["Re"])) in GHIA:
        checks["Ghia_Eu"]=bool(m["ghia_Eu"]<ghia_limit); checks["Ghia_Ev"]=bool(m["ghia_Ev"]<ghia_limit)
    return {"accepted":bool(all(checks.values())),"checks":checks,"metrics":m}


def save_case(result,split,outdir="cases"):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    Re,N=int(round(result["Re"])),int(result["N"]); stem=f"case_Re{Re:04d}"; q=quality_gate(result)
    np.savez_compressed(out/f"{stem}.npz",x=result["x"],y=result["y"],u=result["u"],v=result["v"],p=result["p"],
        psi=result["psi"],omega=result["omega"],Re=result["Re"],N=result["N"],dt=result["dt"],U=result["U"],
        steps=result["steps"],final_residual=result["final_residual"],residual_steps=result["residual_steps"],
        residual_values=result["residual_values"],pressure_grad_rel_residual=result["pressure_grad_rel_residual"],
        split=str(split),package_version=PACKAGE_VERSION)
    meta={"split":str(split),"Re":Re,"N":N,"dt":result["dt"],"steps":result["steps"],
           "final_residual":result["final_residual"],"solver":result["solver"],"package_version":PACKAGE_VERSION,**q}
    qpath=out/f"{stem}_qc.json"; qpath.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return out/f"{stem}.npz",qpath,meta


def load_case(path):
    z=np.load(path,allow_pickle=False); return {k:z[k] for k in z.files}


def build_dataset(paths=None,output="cavity_data.npz",cases_dir="cases",require_accepted=True):
    if paths is None: paths=sorted(Path(cases_dir).glob("case_Re*.npz"))
    records=[]; quality=[]
    for p in map(Path,paths):
        qpath=p.with_name(p.stem+"_qc.json")
        if not qpath.exists(): raise FileNotFoundError(f"Missing quality card for {p.name}: {qpath.name}")
        q=json.loads(qpath.read_text(encoding="utf-8"))
        if require_accepted and not q.get("accepted",False): raise ValueError(f"Case {p.name} failed quality control.")
        records.append(load_case(p)); quality.append(q)
    if len(records)!=len(CASES): raise ValueError(f"Expected {len(CASES)} case files but found {len(records)}.")
    order=np.argsort([float(r["Re"]) for r in records]); records=[records[i] for i in order]; quality=[quality[i] for i in order]
    x0,y0=records[0]["x"],records[0]["y"]
    for r in records:
        if r["u"].shape!=records[0]["u"].shape or not np.allclose(r["x"],x0) or not np.allclose(r["y"],y0):
            raise ValueError("All production cases must use the same grid and coordinate ordering.")
    data={"x":x0,"y":y0,"Re":np.array([float(r["Re"]) for r in records]),
          "u":np.stack([r["u"] for r in records]),"v":np.stack([r["v"] for r in records]),
          "p":np.stack([r["p"] for r in records]),"psi":np.stack([r["psi"] for r in records]),
          "omega":np.stack([r["omega"] for r in records]),"split":np.array([str(r["split"]) for r in records]),
          "final_residual":np.array([float(q["final_residual"]) for q in quality]),
          "pressure_grad_rel_residual":np.array([float(q["metrics"]["pressure_grad_rel_residual"]) for q in quality]),
          "accepted":np.array([bool(q["accepted"]) for q in quality]),"package_version":np.array(PACKAGE_VERSION)}
    np.savez_compressed(output,**data); return data


def load_dataset(path="cavity_data.npz", search_variants=True):
    p = find_valid_dataset(path) if search_variants else require_file(path)
    info = inspect_npz(p, _DATASET_REQUIRED_KEYS)
    if not info["valid"]:
        raise ValueError(f"Invalid Week-4 dataset {p}: {info['error']}")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha256_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reproducibility_text():
    return f"package={PACKAGE_VERSION}\npython={platform.python_version()}\nnumpy={np.__version__}\n"


def pressure_errors(true_p,pred_p,crop=2):
    true_p=np.asarray(true_p,float)-np.mean(true_p); pred_p=np.asarray(pred_p,float)-np.mean(pred_p)
    dp=pred_p-true_p; den=np.linalg.norm(true_p)+1e-30
    s=np.s_[crop:-crop,crop:-crop] if crop else np.s_[:,:]
    den_i=np.linalg.norm(true_p[s])+1e-30
    return {"relative_L2_p":float(np.linalg.norm(dp)/den),
            "relative_L2_p_interior":float(np.linalg.norm(dp[s])/den_i),
            "MAE_p":float(np.mean(np.abs(dp))),"max_p_error":float(np.max(np.abs(dp)))}


def centerline_error_metrics(true_u,true_v,pred_u,pred_v,true_p=None,pred_p=None):
    mid=true_u.shape[0]//2
    def rel(a,b): return float(np.linalg.norm(b-a)/(np.linalg.norm(a)+1e-30))
    out={"centerline_rel_L2_u":rel(true_u[:,mid],pred_u[:,mid]),
         "centerline_rel_L2_v":rel(true_v[mid,:],pred_v[mid,:])}
    if true_p is not None and pred_p is not None:
        tp=np.asarray(true_p)-np.mean(true_p); pp=np.asarray(pred_p)-np.mean(pred_p)
        out["centerline_rel_L2_p_vertical"]=rel(tp[:,mid],pp[:,mid])
        out["centerline_rel_L2_p_horizontal"]=rel(tp[mid,:],pp[mid,:])
    return out


def field_errors(true_u,true_v,pred_u,pred_v):
    du,dv=pred_u-true_u,pred_v-true_v
    denom=np.sqrt(np.sum(true_u**2+true_v**2))+1e-30
    vec=np.hypot(du,dv)
    return {"relative_L2_uv":float(np.sqrt(np.sum(du**2+dv**2))/denom),
            "MAE_u":float(np.mean(np.abs(du))),"MAE_v":float(np.mean(np.abs(dv))),
            "p95_vector_error":float(np.percentile(vec,95)),"max_vector_error":float(np.max(vec))}


def field_physics_metrics(x,y,u,v,U=1.0):
    """Return divergence and wall-condition diagnostics.

    The two upper corners are excluded from the moving-lid metric because the
    lid condition ``u=U`` meets the side-wall condition ``u=0`` discontinuously
    there. Counting either corner against both walls creates an artificial
    nonzero error even for the reference CFD field.
    """
    x=np.asarray(x); y=np.asarray(y); u=np.asarray(u); v=np.asarray(v)
    if u.shape != v.shape or u.ndim != 2 or min(u.shape) < 3:
        raise ValueError("u and v must be matching 2-D fields with at least 3 points per direction")
    dx,dy=float(x[1]-x[0]),float(y[1]-y[0])
    div=np.gradient(u,dx,axis=1)+np.gradient(v,dy,axis=0); interior=div[1:-1,1:-1]

    # Exclude all four corners from wall metrics.  At the upper corners the
    # tangential velocity is mathematically discontinuous; the lower corners
    # are excluded as well so every wall contributes the same interior nodes.
    walls=[u[0,1:-1],u[-1,1:-1]-U,u[1:-1,0],u[1:-1,-1],
           v[0,1:-1],v[-1,1:-1],v[1:-1,0],v[1:-1,-1]]
    w=np.concatenate([a.ravel() for a in walls])
    lid=u[-1,1:-1]-U
    return {"div_l2_pred":float(np.sqrt(np.mean(interior**2))),"div_linf_pred":float(np.max(np.abs(interior))),
            "wall_rms_error":float(np.sqrt(np.mean(w**2))),"wall_max_error":float(np.max(np.abs(w))),
            "lid_u_mae":float(np.mean(np.abs(lid)))}


def field_validation_report(x,y,true_u,true_v,pred_u,pred_v,true_p=None,pred_p=None,U=1.0):
    out=field_errors(true_u,true_v,pred_u,pred_v)
    out.update(field_physics_metrics(x,y,pred_u,pred_v,U=U))
    out.update(centerline_error_metrics(true_u,true_v,pred_u,pred_v,true_p,pred_p))
    if true_p is not None and pred_p is not None: out.update(pressure_errors(true_p,pred_p))
    return out


def save_predictions(path,Re,x,y,u_true,v_true,u_pred,v_pred,model_name,p_true=None,p_pred=None):
    payload={"Re":float(Re),"x":x,"y":y,"u_true":u_true,"v_true":v_true,"u_pred":u_pred,"v_pred":v_pred,
             "model_name":str(model_name)}
    if p_true is not None and p_pred is not None: payload.update(p_true=p_true,p_pred=p_pred)
    np.savez_compressed(path,**payload)
