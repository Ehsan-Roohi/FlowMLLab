"""Bounded, data-free PINN for the constant-lid Ghia cavity.

The declared upper-corner collars bound a finite collocation experiment. They do
not smooth the lid and do not establish a corner-consistent full-domain solution.
Reference labels are deliberately absent from this training module.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn

REYNOLDS = (100, 400, 1000, 3200, 5000, 7500, 10000)
DTYPE = torch.float64
SCHEMA = 1

@dataclass(frozen=True)
class Config:
    reynolds: int = 100
    seed: int = 7
    width: int = 64
    depth: int = 4
    collar: float = 0.02
    collocation: int = 2048
    adam_steps: int = 2000
    lbfgs_steps: int = 500
    adam_lr: float = 0.001
    chunk_size: int = 256
    lbfgs_history: int = 50
    audit_points: int = 2048
    audit_seed: int = 983451
    export_grid: int = 129
    def __post_init__(self):
        if self.reynolds not in REYNOLDS:
            raise ValueError(f'reynolds must be one of {REYNOLDS}')
        if not 0 < self.collar <= 0.1:
            raise ValueError('explicit collar must be in (0, 0.1]')
        for key in ('width','depth','collocation','chunk_size','lbfgs_history','audit_points'):
            if getattr(self,key) < 1: raise ValueError(f'{key} must be positive')
        if self.adam_steps < 0 or self.lbfgs_steps < 0 or self.adam_steps+self.lbfgs_steps == 0:
            raise ValueError('nonnegative optimizer budgets with at least one step required')
        if self.export_grid < 3: raise ValueError('export_grid must be at least 3')
        if self.adam_lr <= 0 or self.seed < 0 or self.audit_seed < 0:
            raise ValueError('invalid rate or seed')

def config_hash(config):
    return hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()

def source_hash():
    return hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()

def tensor_hash(tensors):
    digest=hashlib.sha256()
    for key,value in sorted(tensors.items()):
        a=value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(a.shape).encode())
        digest.update(str(a.dtype).encode()); digest.update(a.tobytes())
    return digest.hexdigest()

def derivative(value,xy):
    if not value.requires_grad: return xy*0
    result=torch.autograd.grad(value,xy,torch.ones_like(value),create_graph=True,
                               retain_graph=True,allow_unused=True)[0]
    return xy*0 if result is None else result

def corner_stokes(xy,reynolds,side='left'):
    """Exact local Stokes streamfunction/pressure in an upper right-angle corner."""
    if side not in ('left','right'): raise ValueError('invalid side')
    x=xy[:,:1] if side=='left' else 1-xy[:,:1]
    z=1-xy[:,1:2]; r2=x*x+z*z
    if bool(torch.any(r2==0)): raise ValueError('exact lid corners are singular')
    theta=torch.atan2(z,x)
    denominator=math.pi**2-4
    a,b,d=-math.pi**2/denominator,2*math.pi/denominator,4/denominator
    psi=(a+b*theta)*z+d*theta*x
    p=-2/reynolds*(b*z+d*x)/r2
    return psi,p if side=='left' else -p

class CavityPINN(nn.Module):
    def __init__(self,width=64,depth=4,reynolds=100):
        super().__init__(); self.reynolds=float(reynolds)
        sizes=[2]+[width]*depth+[2]; layers=[]
        for i,(n,m) in enumerate(zip(sizes[:-1],sizes[1:])):
            layer=nn.Linear(n,m,dtype=DTYPE)
            nn.init.xavier_normal_(layer.weight); nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i<len(sizes)-2: layers.append(nn.Tanh())
        self.network=nn.Sequential(*layers)
    def forward(self,xy):
        if xy.dtype!=DTYPE: raise TypeError('float64 required')
        x,y=xy[:,:1],xy[:,1:2]
        pl,ql=corner_stokes(xy,self.reynolds,'left')
        pr,qr=corner_stokes(xy,self.reynolds,'right')
        wr=x*x/(x*x+(1-x)**2); wl=1-wr
        raw=self.network(2*xy-1)
        bubble=(16*x*(1-x)*y*(1-y))**2
        psi=y*y*(wl*pl+wr*pr)+bubble*raw[:,:1]
        # The corner pressure lifting vanishes at the centre by antisymmetry.
        center=xy.new_tensor([[0.5,0.5]])
        gauge=self.network(2*center-1)[:,1:2]
        p=y*y*(wl*ql+wr*qr)+raw[:,1:2]-gauge
        return psi,p

def physical_fields(psi,p,xy):
    d=derivative(psi,xy); u,v=d[:,1:2],-d[:,:1]
    du,dv=derivative(u,xy),derivative(v,xy)
    return dict(psi=psi,p=p,u=u,v=v,ux=du[:,:1],uy=du[:,1:2],vx=dv[:,:1],
                vy=dv[:,1:2],omega=dv[:,:1]-du[:,1:2],divergence=du[:,:1]+dv[:,1:2])

def momentum(fields,xy,reynolds,include_advection=True):
    dp=derivative(fields['p'],xy)
    lapu=derivative(fields['ux'],xy)[:,:1]+derivative(fields['uy'],xy)[:,1:2]
    lapv=derivative(fields['vx'],xy)[:,:1]+derivative(fields['vy'],xy)[:,1:2]
    rx,ry=dp[:,:1]-lapu/reynolds,dp[:,1:2]-lapv/reynolds
    if include_advection:
        rx=rx+fields['u']*fields['ux']+fields['v']*fields['uy']
        ry=ry+fields['u']*fields['vx']+fields['v']*fields['vy']
    return rx,ry

def residuals(model,points):
    xy=points.detach().clone().requires_grad_(True)
    psi,p=model(xy); f=physical_fields(psi,p,xy)
    rx,ry=momentum(f,xy,model.reynolds)
    return rx,ry,f['divergence']

def corner_radius(points):
    x,z=points[:,0],1-points[:,1]
    return torch.minimum(torch.sqrt(x*x+z*z),torch.sqrt((1-x)**2+z*z))

def uniform_points(n,generator,collar):
    rows=[]; count=0
    while count<n:
        xy=torch.rand((max(2*(n-count),32),2),generator=generator,dtype=DTYPE)
        keep=(xy>0).all(1)&(xy<1).all(1)&(corner_radius(xy)>=collar)
        accepted=xy[keep][:n-count]; rows.append(accepted); count+=len(accepted)
    return torch.cat(rows)

def audit_locations(n=2048,seed=983451):
    # Fixed across all Re/seeds/collars, and never optimized against.
    return uniform_points(n,torch.Generator().manual_seed(seed),0.02)

def shell_locations(low,high,n=128,seed=923):
    if not 0<low<high<=0.25: raise ValueError('invalid shell')
    z=torch.rand((n,2),generator=torch.Generator().manual_seed(seed),dtype=DTYPE)
    radius=torch.sqrt(low**2+(high**2-low**2)*z[:,0]); angle=(.01+.98*z[:,1])*math.pi/2
    x=radius*torch.cos(angle); y=1-radius*torch.sin(angle)
    return torch.stack((torch.stack((x,y),1),torch.stack((1-x,y),1)),1).reshape(-1,2)

def boundary_audit(model,device='cpu'):
    s=torch.unique(torch.cat((torch.linspace(.001,.999,129,dtype=DTYPE),
         torch.tensor([1e-6,1e-5,1e-4,1-1e-4,1-1e-5,1-1e-6],dtype=DTYPE)))).to(device)
    zero,one=torch.zeros_like(s),torch.ones_like(s)
    walls={'top':torch.stack((s,one),1),'bottom':torch.stack((s,zero),1),
           'left':torch.stack((zero,s),1),'right':torch.stack((one,s),1)}
    result={}
    for name,xy in walls.items():
        xy.requires_grad_(True); psi,p=model(xy); d=derivative(psi,xy)
        result[name]={'u_max_abs_error':float((d[:,1]-(1 if name=='top' else 0)).abs().max().detach().cpu()),
                      'v_max_abs_error':float(d[:,0].abs().max().detach().cpu())}
    return result

def residual_statistics(model,points,device='cpu',chunk_size=256):
    rows=[]
    for batch in points.split(chunk_size):
        rx,ry,div=residuals(model,batch.to(device))
        rows.append(torch.cat((rx,ry,div),1).detach().cpu())
    a=torch.cat(rows)
    return {'count':len(a),'minimum_corner_distance':float(corner_radius(points).min()),
            'rx_rms':float(a[:,0].square().mean().sqrt()),'ry_rms':float(a[:,1].square().mean().sqrt()),
            'momentum_rms':float(a[:,:2].square().sum(1).mean().sqrt()),
            'momentum_max':float(a[:,:2].square().sum(1).sqrt().max()),
            'divergence_rms':float(a[:,2].square().mean().sqrt()),'finite':bool(torch.isfinite(a).all())}

def final_audit(model,points,device='cpu',chunk_size=256):
    result={'boundary':boundary_audit(model,device),
            'outer_uniform':residual_statistics(model,points,device,chunk_size),'corner_shells':[]}
    edges=(1e-4,1e-3,.002,.005,.01,.02,.05)
    for i,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
        stats=residual_statistics(model,shell_locations(low,high,seed=923+i),device,chunk_size)
        result['corner_shells'].append({'r_min':low,'r_max':high,**stats})
    result['interpretation']='Raw unweighted samples; no full-domain integrability or corner-consistency claim.'
    return result

def predict(model,points,device='cpu',chunk_size=512):
    values={key:[] for key in ('psi','p','u','v','omega')}
    for batch in torch.as_tensor(points,dtype=DTYPE).split(chunk_size):
        xy=batch.to(device).detach().clone().requires_grad_(True)
        psi,p=model(xy); f=physical_fields(psi,p,xy)
        for key in values: values[key].append(f[key].detach().cpu().numpy().ravel())
    return {key:np.concatenate(value) for key,value in values.items()}

class Trainer:
    """Fixed collocation and finite Adam/L-BFGS budgets, with complete restart."""
    def __init__(self,config:Config,device='cpu'):
        self.config=config; self.device=torch.device(device)
        random.seed(config.seed); np.random.seed(config.seed%(2**32)); torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        self.model=CavityPINN(config.width,config.depth,config.reynolds).to(self.device)
        self.generator=torch.Generator().manual_seed(config.seed+104729)
        self.points=uniform_points(config.collocation,self.generator,config.collar)
        self.audit_points=audit_locations(config.audit_points,config.audit_seed)
        self.adam_completed=0; self.lbfgs_completed=0; self.closure_calls=0
        self.residual_point_evaluations=0; self.elapsed_seconds=0.; self.history=[]; self.frozen=None
        self.stage='adam' if config.adam_steps else 'lbfgs'
        self.objective_hash=tensor_hash({'points':self.points})
        self.optimizer=self.make_optimizer(self.stage)
        self.runtime_history=[self.runtime()]
    def runtime(self):
        return {'torch':torch.__version__,'device':str(self.device),'cuda':torch.version.cuda}
    def make_optimizer(self,stage):
        if stage=='adam': return torch.optim.Adam(self.model.parameters(),lr=self.config.adam_lr)
        return torch.optim.LBFGS(self.model.parameters(),lr=1.,max_iter=1,max_eval=20,
                  history_size=self.config.lbfgs_history,tolerance_grad=1e-12,tolerance_change=1e-14,
                  line_search_fn='strong_wolfe')
    @property
    def completed_steps(self): return self.adam_completed+self.lbfgs_completed
    @property
    def finished(self): return self.completed_steps>=self.config.adam_steps+self.config.lbfgs_steps
    def objective(self,backward=True):
        self.optimizer.zero_grad(set_to_none=True); total=0.
        for batch in self.points.split(self.config.chunk_size):
            rx,ry,_=residuals(self.model,batch.to(self.device))
            loss=(rx.square()+ry.square()).sum()/len(self.points)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite objective')
            if backward: loss.backward()
            total+=float(loss.detach().cpu()); self.residual_point_evaluations+=len(batch)
        if backward and not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
            raise FloatingPointError('nonfinite gradient')
        return total
    def step(self):
        if self.finished: return None
        if tensor_hash({'points':self.points})!=self.objective_hash: raise RuntimeError('fixed objective changed')
        started=time.perf_counter(); stage=self.stage; values=[]
        if stage=='adam':
            value=self.objective(); self.closure_calls+=1; self.optimizer.step()
            self.adam_completed+=1; values.append(value)
            if self.adam_completed==self.config.adam_steps and self.config.lbfgs_steps:
                self.stage='lbfgs'; self.optimizer=self.make_optimizer('lbfgs')
        else:
            def closure():
                value=self.objective(); self.closure_calls+=1; values.append(value)
                # Gradients were accumulated chunkwise, without retaining every graph.
                return torch.tensor(value,device=self.device,dtype=DTYPE)
            self.optimizer.step(closure); self.lbfgs_completed+=1
        self.elapsed_seconds+=time.perf_counter()-started
        row={'step':self.completed_steps,'stage':stage,'training_loss':values[-1],
             'loss_location':'last closure evaluation, not a recomputed accepted endpoint',
             'closure_calls':self.closure_calls,'residual_point_evaluations':self.residual_point_evaluations,
             'elapsed_seconds':self.elapsed_seconds}
        self.history.append(row)
        if self.finished:
            self.frozen={'model_sha256':tensor_hash(self.model.state_dict()),'config_sha256':config_hash(self.config),
                         'objective_sha256':self.objective_hash,'reference_labels_used':False,
                         'endpoint_rule':'last complete step of the preregistered finite budget',
                         'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        return row
    def state_dict(self):
        return {'schema':SCHEMA,'source_sha256':source_hash(),'config':asdict(self.config),
                'config_sha256':config_hash(self.config),'source_raw_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model':self.model.state_dict(),
                'model_sha256':tensor_hash(self.model.state_dict()),'optimizer':self.optimizer.state_dict(),
                'stage':self.stage,'adam_completed':self.adam_completed,'lbfgs_completed':self.lbfgs_completed,
                'closure_calls':self.closure_calls,'residual_point_evaluations':self.residual_point_evaluations,
                'elapsed_seconds':self.elapsed_seconds,'points':self.points,'audit_points':self.audit_points,
                'objective_hash':self.objective_hash,'sampler_rng':self.generator.get_state(),
                'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
                'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'adaptive_state':{'enabled':False},'frozen':self.frozen,'history':self.history,
                'runtime_history':self.runtime_history}
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as stream:
            torch.save(self.state_dict(),stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    @classmethod
    def load(cls,path,device='cpu',config=None):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['schema']!=SCHEMA or state['source_sha256']!=source_hash():
            raise ValueError('checkpoint schema/source differs; explicit migration required')
        restored_config=Config(**state['config'])
        if config_hash(restored_config)!=state['config_sha256']: raise ValueError('config hash failed')
        if config is not None and config!=restored_config: raise ValueError('configuration mismatch')
        if tensor_hash(state['model'])!=state['model_sha256']: raise ValueError('model hash failed')
        result=cls(restored_config,device); result.model.load_state_dict(state['model'])
        result.stage=state['stage']; result.optimizer=result.make_optimizer(result.stage)
        result.optimizer.load_state_dict(state['optimizer'])
        for key in ('adam_completed','lbfgs_completed','closure_calls','residual_point_evaluations','elapsed_seconds',
                    'points','audit_points','objective_hash','frozen','history','runtime_history'):
            setattr(result,key,state[key])
        if tensor_hash({'points':result.points})!=result.objective_hash: raise ValueError('collocation hash failed')
        if result.runtime()!=result.runtime_history[-1]: result.runtime_history.append(result.runtime())
        result.generator.set_state(state['sampler_rng']); random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy']); torch.set_rng_state(state['rng_torch'])
        if state['rng_cuda'] and torch.cuda.is_available():
            if len(state['rng_cuda'])!=torch.cuda.device_count(): raise ValueError('CUDA RNG device count mismatch')
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        if result.frozen and result.frozen['model_sha256']!=state['model_sha256']:
            raise ValueError('frozen model hash failed')
        return result

def numerical_admission(device='cpu'):
    """Independent BC, manufactured-equation and derivative gates; no CFD data."""
    torch.manual_seed(1907)
    model=CavityPINN(6,2,100).to(device)
    points=torch.tensor([[.13,.27],[.39,.62],[.71,.43],[.83,.19]],dtype=DTYPE,device=device,requires_grad=True)
    x,y=points[:,:1],points[:,1:2]
    f=physical_fields(-(x*x+y*y)/2,(x*x+y*y)/2,points)
    rx,ry=momentum(f,points,100)
    rotation_error=float(torch.cat((rx,ry,f['omega']-2,f['divergence']),1).abs().max().detach().cpu())
    psi=x**3*y**2; p=x*x*y
    f=physical_fields(psi,p,points); rx,ry=momentum(f,points,100)
    exact_rx=6*x**5*y*y+2*x*y-12*x*y/100
    exact_ry=6*x**4*y**3+x*x+6*(x*x+y*y)/100
    viscous_error=float(torch.cat((rx-exact_rx,ry-exact_ry),1).abs().max().detach().cpu())
    corner_errors={}
    for side in ('left','right'):
        rows=[]
        for radius in (.1,.01,.001):
            for angle in (.17,.41,.89,1.31):
                local_x=radius*math.cos(angle); yy=1-radius*math.sin(angle)
                rows.append([local_x if side=='left' else 1-local_x,yy])
        xy=torch.tensor(rows,dtype=DTYPE,device=device,requires_grad=True)
        psi,p=corner_stokes(xy,100,side); fields=physical_fields(psi,p,xy)
        rx,ry=momentum(fields,xy,100,include_advection=False)
        scale=derivative(p,xy).abs().max().detach().clamp_min(1.)
        corner_errors[side]=float(torch.cat((rx,ry),1).abs().max().detach().cpu()/scale.cpu())
    walls=boundary_audit(model,device)
    wall_error=max(v for row in walls.values() for v in row.values())
    rx,ry,div=residuals(model,points.detach())
    loss=(rx.square()+ry.square()).mean()
    parameters=list(model.parameters())
    gradients=torch.autograd.grad(loss,parameters)
    direction=[torch.randn_like(p) for p in parameters]
    norm=torch.sqrt(sum((d*d).sum() for d in direction))
    direction=[d/norm for d in direction]
    ad=float(sum((g*d).sum() for g,d in zip(gradients,direction)).detach().cpu())
    originals=[p.detach().clone() for p in parameters]
    def shifted_value(amount):
        with torch.no_grad():
            for p,base,d in zip(parameters,originals,direction): p.copy_(base+amount*d)
        a,b,_=residuals(model,points.detach())
        return float((a.square()+b.square()).mean().detach().cpu())
    h=1e-6; fd=(shifted_value(h)-shifted_value(-h))/(2*h)
    shifted_value(0)
    gradient_error=abs(ad-fd)/max(abs(ad),abs(fd),1.)
    report={'device':str(device),'dtype':'float64','rotation_residual_error':rotation_error,
            'manufactured_viscous_error':viscous_error,'corner_stokes_relative_errors':corner_errors,
            'boundary_max_abs_error':wall_error,'boundary':walls,
            'divergence_max_abs':float(div.abs().max().detach().cpu()),
            'directional_gradient_relative_error':gradient_error}
    report['passed']=(rotation_error<1e-10 and viscous_error<1e-10 and max(corner_errors.values())<1e-8
                      and wall_error<1e-10 and report['divergence_max_abs']<1e-10 and gradient_error<1e-5)
    return report

