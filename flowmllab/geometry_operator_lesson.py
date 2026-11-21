"""A bounded manufactured geometry-to-field operator lesson for a CPU.

The reference is an analytic streamfunction, not CFD. A training-only POD trunk
is fixed, and a NumPy MLP learns geometry -> modal coefficients. Validation picks
the checkpoint; test geometries are generated only by the later evaluation call.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.optimize import minimize

TRAIN=(0.,.08,.16,.24,.40,.48,.56)
VALIDATION=(.12,.36)
TEST=(.32,.52)


@dataclass(frozen=True)
class Config:
    depth:int=2
    width:int=24
    rank:int=4
    optimizer:str="adam_lbfgs"
    adam_steps:int=500
    lbfgs_iterations:int=100
    learning_rate:float=.01
    v_weight:float=1.
    near_wall_weight:float=1.
    seed:int=17
    nx:int=49
    ny:int=25


def reference(a,nx=49,ny=25):
    """Exact kinematic flow in y>=a sin²(pi x); it is not a PDE solution claim."""
    if not 0<=a<.8:raise ValueError("Use a in [0,.8)")
    x=np.linspace(0.,1.,nx);eta=np.linspace(0.,1.,ny)
    X,E=np.meshgrid(x,eta)
    bottom=a*np.sin(np.pi*X)**2
    height=1-bottom
    slope=a*np.pi*np.sin(2*np.pi*X)
    Y=bottom+height*E
    U=6*E*(1-E)/height
    V=U*slope*(1-E)
    return dict(a=float(a),x=x,eta=eta,X=X,Y=Y,field=np.stack((U,V),axis=-1))


def unpack(theta,shapes):
    offset=0;layers=[]
    for ni,no in shapes:
        w=theta[offset:offset+ni*no].reshape(ni,no);offset+=ni*no
        b=theta[offset:offset+no];offset+=no
        layers.append((w,b))
    if offset!=len(theta):raise ValueError("Parameter-vector shape mismatch")
    return layers


def forward(theta,shapes,inputs):
    values=[np.asarray(inputs)]
    for i,(w,b) in enumerate(unpack(theta,shapes)):
        z=values[-1]@w+b
        values.append(np.tanh(z) if i<len(shapes)-1 else z)
    return values


def loss_gradient(theta,shapes,inputs,truth,mean,modes,scale,weights):
    """Analytic gradient of the declared weighted full-field objective."""
    layers=unpack(theta,shapes);values=forward(theta,shapes,inputs)
    prediction=mean+(values[-1]*scale)@modes
    error=prediction-truth
    denom=len(truth)*weights.sum()
    loss=float(np.sum(error*error*weights)/denom)
    delta=(2*error*weights)@modes.T*scale/denom
    gradients=[]
    for layer in range(len(layers)-1,-1,-1):
        w,_=layers[layer]
        gradients.append((values[layer].T@delta,delta.sum(0)))
        if layer:delta=(delta@w.T)*(1-values[layer]**2)
    gradient=np.concatenate([a.ravel() for pair in reversed(gradients) for a in pair])
    return loss,gradient


def fit(config=Config(),train=TRAIN,validation=VALIDATION):
    train=np.asarray(train,dtype=float);validation=np.asarray(validation,dtype=float)
    if train.ndim!=1 or validation.ndim!=1 or len(train)<2 or not len(validation):raise ValueError("Require one-dimensional nonempty geometry partitions and at least two training cases")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():raise ValueError("Geometry partitions must be finite")
    if min(config.nx,config.ny)<3 or not isinstance(config.nx,int) or not isinstance(config.ny,int):raise ValueError("nx and ny must be integers >=3")
    if min(config.adam_steps,config.lbfgs_iterations)<0:raise ValueError("Optimizer budgets must be nonnegative")
    if len(set(train))!=len(train) or np.any(np.diff(train)<=0):raise ValueError("Training geometries must be distinct and strictly increasing")
    if len(set(validation))!=len(validation) or not len(validation):raise ValueError("Validation geometries must be distinct and nonempty")
    if set(train)&set(validation):raise ValueError("Training and validation geometries overlap")
    if config.optimizer not in ("adam","lbfgs","adam_lbfgs"):raise ValueError("Unknown optimizer")
    if min(config.depth,config.width,config.rank)<1 or config.rank>=len(train):raise ValueError("Invalid model size")
    hyperparameters=(config.v_weight,config.near_wall_weight,config.learning_rate)
    if not np.isfinite(hyperparameters).all() or min(hyperparameters)<=0:raise ValueError("Weights/rate must be finite and positive")
    started=time.perf_counter()
    training=[reference(a,config.nx,config.ny) for a in train]
    validation_cases=[reference(a,config.nx,config.ny) for a in validation]
    truth=np.stack([c["field"].ravel() for c in training])
    validation_truth=np.stack([c["field"].ravel() for c in validation_cases])
    mean=truth.mean(0);_,singular,modes=np.linalg.svd(truth-mean,full_matrices=False)
    modes=modes[:config.rank]
    scale=np.maximum(((truth-mean)@modes.T).std(0),1e-10)
    lo,hi=min(train),max(train)
    normalized=lambda a:(2*np.asarray(a,dtype=float).reshape(-1,1)-lo-hi)/(hi-lo)
    if np.any(validation<lo) or np.any(validation>hi):raise ValueError("Validation must remain within the training geometry range")
    inputs=normalized(train);vinputs=normalized(validation)
    widths=[1]+[config.width]*config.depth+[config.rank]
    shapes=list(zip(widths[:-1],widths[1:]))
    rng=np.random.default_rng(config.seed)
    theta=np.concatenate([a.ravel() for ni,no in shapes for a in
                          (rng.normal(size=(ni,no))*np.sqrt(2/(ni+no)),np.zeros(no))])
    weights=np.ones((config.ny,config.nx,2));weights[:,:,1]=config.v_weight
    band=(training[0]["eta"]<.15)|(training[0]["eta"]>.85)
    weights[band]*=config.near_wall_weight;weights=weights.ravel()
    history=[];best=dict(score=np.inf,theta=theta.copy(),step=0,phase="initial")
    evaluations=0;step=0

    def objective(t):
        nonlocal evaluations
        evaluations+=1
        return loss_gradient(t,shapes,inputs,truth,mean,modes,scale,weights)

    def record(t,phase):
        vp=mean+(forward(t,shapes,vinputs)[-1]*scale)@modes
        score=float(np.linalg.norm(vp-validation_truth)/np.linalg.norm(validation_truth))
        training_prediction=mean+(forward(t,shapes,inputs)[-1]*scale)@modes
        train_loss=float(np.sum((training_prediction-truth)**2*weights)/(len(truth)*weights.sum()))
        history.append(dict(step=step,phase=phase,training_weighted_mse=train_loss,
                            validation_relative_l2=score,seconds=time.perf_counter()-started))
        if score<best["score"]:best.update(score=score,theta=t.copy(),step=step,phase=phase)

    record(theta,"initial")
    if config.optimizer in ("adam","adam_lbfgs"):
        m=np.zeros_like(theta);v=np.zeros_like(theta)
        for k in range(1,config.adam_steps+1):
            _,gradient=objective(theta)
            m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
            theta-=config.learning_rate*(m/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
            step+=1
            if k%25==0 or k==config.adam_steps:record(theta,"Adam")
    scipy_result=None
    if config.optimizer in ("lbfgs","adam_lbfgs") and config.lbfgs_iterations>0:
        def callback(t):
            nonlocal step
            step+=1;record(t,"L-BFGS")
        result=minimize(objective,theta,method="L-BFGS-B",jac=True,callback=callback,
                        options=dict(maxiter=config.lbfgs_iterations,maxls=30,ftol=1e-12,gtol=1e-8))
        record(result.x,"L-BFGS final")
        scipy_result=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
    elif config.optimizer in ("lbfgs","adam_lbfgs"):
        scipy_result=dict(success=False,message="Not run: zero L-BFGS iteration budget",iterations=0)
    return dict(config=asdict(config),theta=best["theta"],shapes=shapes,mean=mean,modes=modes,scale=scale,
                geometry_range=(lo,hi),train=list(train),validation=list(validation),
                training_fields=truth,history=history,selected_step=best["step"],selected_phase=best["phase"],
                validation_score=best["score"],objective_evaluations=evaluations,
                monitoring_evaluations=len(history),seconds=time.perf_counter()-started,lbfgs=scipy_result,
                retained_training_energy=float((singular[:config.rank]**2).sum()/(singular**2).sum()))


def predict(model,geometries):
    lo,hi=model["geometry_range"];a=np.asarray(geometries,dtype=float)
    if not np.isfinite(a).all():raise ValueError("Geometry values must be finite")
    if np.any(a<lo) or np.any(a>hi):raise ValueError("This lesson evaluates in-range geometry interpolation")
    inputs=(2*a.reshape(-1,1)-lo-hi)/(hi-lo)
    values=forward(model["theta"],model["shapes"],inputs)[-1]
    return model["mean"]+(values*model["scale"])@model["modes"]


def evaluate(model,geometries=TEST):
    if set(geometries)&(set(model["train"])|set(model["validation"])):
        raise ValueError("The reported test geometries must be separate")
    cfg=model["config"];records=[]
    predicted=predict(model,geometries)
    for a,prediction in zip(geometries,predicted):
        case=reference(a,cfg["nx"],cfg["ny"]);truth=case["field"].ravel()
        train=np.asarray(model["train"]);right=np.searchsorted(train,a);left=right-1
        fraction=(a-train[left])/(train[right]-train[left])
        linear=(1-fraction)*model["training_fields"][left]+fraction*model["training_fields"][right]
        oracle=model["mean"]+((truth-model["mean"])@model["modes"].T)@model["modes"]
        methods={"POD-DeepONet":prediction,"Linear geometry interpolation":linear,"POD oracle":oracle}
        scores={name:float(np.linalg.norm(field-truth)/np.linalg.norm(truth)) for name,field in methods.items()}
        records.append(dict(a=float(a),case=case,methods={name:f.reshape(case["field"].shape) for name,f in methods.items()},
                            scores=scores))
    return records


def plot_training(model):
    import matplotlib.pyplot as plt
    rows=model["history"]
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),constrained_layout=True)
    axes[0].semilogy([r["step"] for r in rows],[max(r["training_weighted_mse"],1e-30) for r in rows])
    axes[1].semilogy([r["step"] for r in rows],[max(r["validation_relative_l2"],1e-30) for r in rows])
    axes[0].set(xlabel="optimizer update",ylabel="training weighted MSE")
    axes[1].set(xlabel="optimizer update",ylabel="validation relative discrete L2")
    for ax in axes:
        ax.axvline(model["selected_step"],color="black",ls=":",label="validation-selected update")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,1.3),frameon=False);ax.grid(alpha=.2)
    return fig


def plot_case(record):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    case=record["case"];reference=case["field"];X,Y=case["X"],case["Y"]
    names=("POD-DeepONet","Linear geometry interpolation")
    predicted=[record["methods"][name] for name in names]
    ref_speed=np.linalg.norm(reference,axis=-1)
    levels=np.linspace(float(ref_speed.min()),float(ref_speed.max()),25)
    errors=[np.linalg.norm(value-reference,axis=-1) for value in predicted]
    maximum=max(float(e.max()) for e in errors)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    im=axes[0].contourf(X,Y,ref_speed,levels=levels,cmap="jet")
    for value,color,style in zip(predicted,("red","#6827a8"),("--",":")):
        axes[0].contour(X,Y,np.linalg.norm(value,axis=-1),levels=levels[2:-2:3],colors=color,linestyles=style,linewidths=.8)
    axes[0].legend(handles=[Patch(facecolor="#00bd96",label="Exact manufactured reference")]+
                   [Line2D([],[],color=c,ls=s,label=n) for n,c,s in
                   zip(names,("red","#6827a8"),("--",":"))],
                   loc="upper center",bbox_to_anchor=(.5,1.45),fontsize=8,frameon=False)
    axes[0].set_title("Exact manufactured reference")
    fig.colorbar(im,ax=axes[0],label="speed / U₀",shrink=.7)
    for ax,error,name in zip(axes[1:],errors,names):
        im=ax.contourf(X,Y,error,levels=np.linspace(0,max(maximum,1e-14),25),cmap="jet")
        ax.set_title(name+" error",fontsize=10)
        fig.colorbar(im,ax=ax,label="vector error / U₀",shrink=.7)
    for ax in axes:
        ax.plot(X[0],Y[0],color="black",lw=1)
        ax.set(xlabel="x/L",ylabel="y/L",aspect="equal",xlim=(0,1),ylim=(0,1))
    fig.suptitle(f"Unseen manufactured geometry a={record['a']:.2f}")
    j=int(np.argmin(abs(case["x"]-.375)))
    profiles,pa=plt.subplots(2,2,figsize=(11,6),sharey=True,constrained_layout=True)
    for component,label in enumerate(("u","v")):
        pa[component,0].plot(reference[:,j,component],Y[:,j],color="black",lw=2,label="Exact manufactured reference")
        for name,value,color,style in zip(names,predicted,("red","#6827a8"),("--",":")):
            pa[component,0].plot(value[:,j,component],Y[:,j],color=color,ls=style,label=name)
            pa[component,1].plot(value[:,j,component]-reference[:,j,component],Y[:,j],color=color,ls=style)
        pa[component,0].set(xlabel=label+" / U₀",ylabel="physical y/L")
        pa[component,1].set(xlabel=f"signed error in {label} / U₀")
        pa[component,1].axvline(0,color=".6",lw=.7)
    pa[0,0].legend(loc="upper center",bbox_to_anchor=(1.05,1.45),ncol=3,fontsize=8,frameon=False)
    profiles.suptitle(f"Manufactured a={record['a']:.2f}; native column x/L={case['x'][j]:.4g}")
    return fig,profiles
