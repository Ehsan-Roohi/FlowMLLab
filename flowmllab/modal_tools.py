"""Small auditable modal algorithms; original code inspired by PyDMD/PySensors/PySINDy.

These are named textbook methods, not wrappers or full replacements for those
packages. POD fitting/reconstruction reuse the existing FlowMLLab implementation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr
from scipy.integrate import solve_ivp
from sklearn.preprocessing import PolynomialFeatures
from .cylinder_ml import fit_pod,project_pod,reconstruct_pod


def matrix(a):
    a=np.asarray(a,float)
    if a.ndim!=2 or min(a.shape)<1 or not np.isfinite(a).all():
        raise ValueError('Expected a finite nonempty matrix')
    return a


def fit_dmd(states):
    states=matrix(states)
    if len(states)<3:
        raise ValueError('Need at least three states')
    return np.linalg.lstsq(states[:-1],states[1:],rcond=None)[0]


def rollout_dmd(operator,initial,steps):
    a=matrix(operator);x=np.asarray(initial,float)
    if x.ndim!=1 or a.shape!=(len(x),len(x)) or not np.isfinite(x).all() or int(steps)!=steps or steps<1:
        raise ValueError('Invalid rollout dimensions')
    result=[]
    for _ in range(int(steps)):
        x=x@a
        if not np.isfinite(x).all() or np.linalg.norm(x)>1e12:
            raise ValueError('DMD rollout diverged; no clipping applied')
        result.append(x.copy())
    return np.array(result)


def dmd_modes(operator,dt):
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('dt must be positive')
    eigenvalues=np.linalg.eigvals(matrix(operator))
    return [{'real':float(v.real),'imag':float(v.imag),
             'frequency':float(np.angle(v)/(2*np.pi*dt)),
             'growth_rate':None if abs(v)==0 else float(np.log(abs(v))/dt)} for v in eigenvalues]


def select_sensors(modes,count):
    """QR seed of r locations, then D-optimal greedy oversampling (not QR tail)."""
    phi=matrix(modes);n,r=phi.shape
    if not r<=count<=n or int(count)!=count or np.linalg.matrix_rank(phi)<r:
        raise ValueError('Need full rank modes and rank <= sensor count <= points')
    selected=list(map(int,qr(phi.T,pivoting=True,mode='economic')[2][:r]))
    while len(selected)<count:
        gram=phi[selected].T@phi[selected]
        leverage=np.einsum('ij,jk,ik->i',phi,np.linalg.pinv(gram),phi)
        leverage[selected]=-np.inf
        selected.append(int(np.argmax(leverage)))
    return np.array(selected)


def reconstruct_sensors(pod,indices,observations):
    ids=np.asarray(indices);y=matrix(observations)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(pod.mean)) or y.shape[1]!=len(ids):
        raise ValueError('Invalid sensor indices/measurements')
    sampling=pod.modes[ids]
    if np.linalg.matrix_rank(sampling)<pod.rank:
        raise ValueError('Sensor matrix is rank deficient')
    coefficients=np.linalg.lstsq(sampling,(y-pod.mean[ids]).T,rcond=None)[0].T
    return reconstruct_pod(pod,coefficients)


@dataclass
class SparseODE:
    library: object
    coefficients: np.ndarray
    threshold: float

    def derivative(self,states):
        z=matrix(states)
        return self.library.transform(z)@self.coefficients

    def equations(self):
        terms=self.library.get_feature_names_out([f'z{i+1}' for i in range(self.coefficients.shape[1])])
        return [' + '.join(f'{c:.6g}*{name}' for name,c in zip(terms,column) if c!=0) or '0'
                for column in self.coefficients.T]


def fit_integral_sindy(states,dt,threshold=.05,window=4,degree=3):
    """Trapezoidal integral regression + normalized-column STLSQ, train only."""
    z=matrix(states)
    if not np.isfinite(dt) or dt<=0 or threshold<0 or not np.isfinite(threshold) or int(window)!=window or window<1 or len(z)<=window+2:
        raise ValueError('Invalid integral-regression settings')
    library=PolynomialFeatures(degree=degree,include_bias=True)
    theta=library.fit_transform(z)
    # Each row integrates the library over a training-only window.
    integrated=np.array([dt*(.5*theta[i]+theta[i+1:i+window].sum(axis=0)+.5*theta[i+window]) for i in range(len(z)-window)])
    target=z[window:]-z[:-window]
    scale=np.maximum(np.linalg.norm(integrated,axis=0),1e-12)
    design=integrated/scale
    coef=np.linalg.lstsq(design,target,rcond=1e-10)[0]
    for _ in range(20):
        # Threshold coefficients in z/time units after undoing column scaling.
        keep=abs(coef/scale[:,None])>=threshold
        new=np.zeros_like(coef)
        for j in range(z.shape[1]):
            if keep[:,j].any():
                new[keep[:,j],j]=np.linalg.lstsq(design[:,keep[:,j]],target[:,j],rcond=1e-10)[0]
        if np.array_equal(new,coef):
            break
        coef=new
    return SparseODE(library,coef/scale[:,None],float(threshold))


def rollout_sindy(model,initial,dt,steps):
    initial=np.asarray(initial,float)
    if initial.ndim!=1 or not np.isfinite(initial).all() or not np.isfinite(dt) or dt<=0 or int(steps)!=steps or steps<1:
        raise ValueError('Invalid ODE rollout')
    limit=100*max(1,float(np.linalg.norm(initial)))
    def stop(t,z):
        return limit-np.linalg.norm(z)
    stop.terminal=True
    times=np.arange(1,int(steps)+1)*dt
    solution=solve_ivp(lambda t,z:model.derivative(z[None])[0],(0,times[-1]),initial,
        t_eval=times,rtol=1e-8,atol=1e-10,max_step=dt,events=stop)
    if not solution.success or solution.y.shape[1]!=steps or not np.isfinite(solution.y).all():
        raise ValueError('Sparse ODE rollout failed/diverged; no clipping applied')
    return solution.y.T
