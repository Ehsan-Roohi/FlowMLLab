"""CPU parameter-to-velocity POD-DeepONet with explicit full-case splits.

The POD trunk is computed only from training CFD fields. A tanh branch receives
log10(Re); Adam/L-BFGS train its modal coefficients. No Ghia targets enter fitting.
This restricted scalar-parameter operator is not a data-free PINN.
"""
from dataclasses import dataclass,asdict
from time import perf_counter
import numpy as np
from scipy.optimize import minimize

TRAIN=(100,250,1000,2500,4000,5000,6000,9000,10000)
VALIDATION=(400,6500)
TEST=(3200,7500)
SEEDS=(7,23,61)


@dataclass(frozen=True)
class Config:
    name:str
    depth:int=2
    width:int=32
    optimizer:str='hybrid'
    scaling:str='global'
    evaluations:int=4000
    adam_evaluations:int=1200
    learning_rate:float=.003


CONFIGS=(Config('adam_2',optimizer='adam'),Config('hybrid_2'),
         Config('hybrid_5',depth=5),
         Config('hybrid_5_balanced',depth=5,scaling='per_mode'),
         Config('hybrid_5_capacity_matched',depth=5,width=17,scaling='per_mode'))


def make_trunk(states):
    states=np.asarray(states,dtype=float)
    if states.ndim!=2 or len(states)!=len(TRAIN):raise ValueError('Expected complete training states only')
    mean=states.mean(axis=0);centered=states-mean
    left,singular,modes=np.linalg.svd(centered,full_matrices=False)
    rank=min(len(states)-1,int(np.sum(singular>singular[0]*1e-12)))
    modes=modes[:rank].copy();coeff=centered @ modes.T
    return dict(mean=mean,modes=modes,singular_values=singular,train_coefficients=coeff,
                mix_map=(left[:,:rank]/singular[:rank]),rank=rank,
                retained_training_energy=float(np.sum(singular[:rank]**2)/np.sum(singular**2)))


def features(reynolds):
    re=np.asarray(reynolds,dtype=float)
    if np.any(re<=0):raise ValueError('Re must be positive')
    # Fixed physical parameter map, using training endpoints only.
    return (np.log10(re).reshape(-1,1)-3.0)


def initialize(config,nout,seed):
    rng=np.random.default_rng(seed);sizes=[1]+[config.width]*config.depth+[nout]
    shapes=list(zip(sizes[:-1],sizes[1:]));pieces=[]
    for a,b in shapes:pieces.extend([rng.normal(size=(a,b)).ravel()*np.sqrt(2/(a+b)),np.zeros(b)])
    return np.concatenate(pieces),shapes,rng.bit_generator.state


def unpack(theta,shapes):
    offset=0;layers=[]
    for a,b in shapes:
        w=theta[offset:offset+a*b].reshape(a,b);offset+=a*b
        bias=theta[offset:offset+b];offset+=b;layers.append((w,bias))
    if offset!=len(theta):raise ValueError('Neural parameter shape mismatch')
    return layers


def forward(theta,shapes,x):
    layers=unpack(theta,shapes);acts=[np.asarray(x)]
    for i,(w,b) in enumerate(layers):
        z=acts[-1]@w+b;acts.append(np.tanh(z) if i<len(layers)-1 else z)
    return acts[-1],acts,layers


def objective(theta,shapes,x,target):
    pred,acts,layers=forward(theta,shapes,x);error=pred-target
    value=float(np.mean(error**2));delta=2*error/error.size;grads=[]
    for i in range(len(layers)-1,-1,-1):
        grads.append((acts[i].T@delta,delta.sum(axis=0)))
        if i:delta=(delta@layers[i][0].T)*(1-acts[i]**2)
    return value,np.concatenate([a.ravel() for pair in reversed(grads) for a in pair])


def fit(config,seed,train_coefficients):
    coefficients=np.asarray(train_coefficients)
    if coefficients.shape[0]!=len(TRAIN):raise ValueError('Wrong complete-case training split')
    if config.scaling=='global':scale=np.full(coefficients.shape[1],np.sqrt(np.mean(coefficients**2)))
    elif config.scaling=='per_mode':scale=np.sqrt(np.mean(coefficients**2,axis=0))
    else:raise ValueError('Unknown output scaling')
    scale=np.maximum(scale,1e-12)
    x=features(TRAIN);target=coefficients/scale
    theta,shapes,rng_state=initialize(config,target.shape[1],seed)
    m=np.zeros_like(theta);v=np.zeros_like(theta)
    best=theta.copy();best_value=float('inf');count=0;history=[];phase='Adam'
    started=perf_counter()
    class BudgetDone(Exception):pass
    def evaluate(parameters):
        nonlocal count,best,best_value
        if count>=config.evaluations:raise BudgetDone
        value,gradient=objective(parameters,shapes,x,target);count+=1
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite neural objective')
        if value<best_value:best_value=value;best=parameters.copy()
        if count==1 or count%25==0:
            history.append(dict(evaluations=count,phase=phase,best_training_loss=best_value,
                                elapsed_seconds=perf_counter()-started))
        return value,gradient
    adam_count=config.evaluations if config.optimizer=='adam' else config.adam_evaluations
    for step in range(1,adam_count+1):
        value,gradient=evaluate(theta)
        m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
        theta-=config.learning_rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    adam_last_theta=theta.copy()
    status='Adam fixed objective-evaluation budget'
    if config.optimizer=='hybrid':
        phase='L-BFGS';theta=best.copy()
        try:
            result=minimize(evaluate,theta,jac=True,method='L-BFGS-B',
                options=dict(maxiter=config.evaluations,maxfun=config.evaluations,maxls=30,
                             ftol=1e-15,gtol=1e-10,maxcor=30))
            status=str(result.message)
        except BudgetDone:status='fixed joint function/gradient evaluation budget reached'
    history.append(dict(evaluations=count,phase=phase,best_training_loss=best_value,
                        elapsed_seconds=perf_counter()-started))
    return dict(theta=best,shapes=shapes,scale=scale,config=asdict(config),seed=seed,
                evaluations=count,best_training_loss=best_value,history=history,status=status,
                training_seconds=perf_counter()-started,adam_last_theta=adam_last_theta,
                adam_m=m,adam_v=v,adam_steps=adam_count,rng_state=rng_state)


def predict_coefficients(model,reynolds):
    return forward(model['theta'],model['shapes'],features(reynolds))[0]*model['scale']


def baseline_coefficients(trunk,reynolds,coordinate='log'):
    train=np.asarray(TRAIN,dtype=float);target=np.asarray(reynolds,dtype=float)
    if np.any(target<train.min()) or np.any(target>train.max()):raise ValueError('Baseline cannot extrapolate')
    if coordinate=='log':train,target=np.log10(train),np.log10(target)
    elif coordinate!='linear':raise ValueError('Unknown interpolation coordinate')
    return np.column_stack([np.interp(target,train,c) for c in trunk['train_coefficients'].T])


def reconstruct(trunk,coefficients):
    return np.asarray(coefficients) @ trunk['modes']+trunk['mean']


def mixing_weights(trunk,coefficients):
    return 1/len(TRAIN)+np.asarray(coefficients) @ trunk['mix_map'].T


def velocity_error(prediction,truth):
    error=np.asarray(prediction)-truth
    return dict(relative_l2=float(np.linalg.norm(error)/np.linalg.norm(truth)),
                rmse_U=float(np.sqrt(np.mean(error**2))),linf_U=float(np.abs(error).max()))
