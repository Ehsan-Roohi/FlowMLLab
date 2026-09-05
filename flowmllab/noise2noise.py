"""Original CPU Noise2Noise-style lab for paired real DSMC observations.

Method inspiration: Lehtinen et al., arXiv:1803.04189. No NVlabs code reused.
The learner never receives a clean/high-budget reference or archived predictions.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def check_fields(a, ndim):
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=ndim or not np.isfinite(a).all() or min(a.shape)<1:
        raise ValueError('Expected finite, nonempty field arrays')
    return a


def split_indices(seeds, split):
    seeds=list(map(int,seeds))
    declared=[int(s) for group in split.values() for s in group]
    if len(set(seeds))!=len(seeds) or len(set(declared))!=len(declared):
        raise ValueError('Duplicate seed or overlapping splits')
    if set(declared)!=set(seeds) or set(split)!={'train','validation','test'}:
        raise ValueError('Splits must partition all seeds exactly')
    if any(len(split[k])<2 for k in split):
        raise ValueError('Each split needs at least two independent units')
    return {k:np.array([seeds.index(int(s)) for s in group]) for k,group in split.items()}


def independent_targets(train):
    """For input seed i use only other training seeds as the noisy target."""
    train=check_fields(train,4)
    if len(train)<2:
        raise ValueError('At least two independent training observations required')
    return np.stack([train[np.arange(len(train))!=i].mean(axis=0) for i in range(len(train))])


def patch_features(field, width=5):
    field=check_fields(field,3)
    if width<1 or width%2==0 or min(field.shape[-2:])<=width//2:
        raise ValueError('Patch width must be odd and fit the field')
    p=width//2
    padded=np.pad(field,((0,0),(p,p),(p,p)),mode='reflect')
    windows=np.lib.stride_tricks.sliding_window_view(padded,(width,width),axis=(-2,-1))
    return windows.transpose(1,2,0,3,4).reshape(-1,field.shape[0]*width*width)


def restore_mean(estimate, observation):
    estimate=check_fields(estimate,3); observation=check_fields(observation,3)
    if estimate.shape!=observation.shape:
        raise ValueError('Mismatched fields')
    return estimate+observation.mean(axis=(-2,-1),keepdims=True)-estimate.mean(axis=(-2,-1),keepdims=True)


@dataclass
class PatchMLP:
    model: object
    scale: np.ndarray
    width: int=5

    def predict(self, observation):
        obs=check_fields(observation,3)
        x=patch_features(obs/self.scale[:,None,None],self.width)
        with threadpool_limits(limits=1):
            y=self.model.predict(x)
        return y.T.reshape(obs.shape)*self.scale[:,None,None]


def fit_patch_mlp(train, *, random_state=12, max_iter=160, samples_per_seed=3000):
    train=check_fields(train,4)
    targets=independent_targets(train)
    scale=np.maximum(train.std(axis=(0,2,3)),1e-12)
    rng=np.random.default_rng(random_state)
    xs=[];ys=[]
    for observation,target in zip(train,targets):
        x=patch_features(observation/scale[:,None,None])
        y=(target/scale[:,None,None]).reshape(len(scale),-1).T
        ids=rng.choice(len(x),min(samples_per_seed,len(x)),replace=False)
        xs.append(x[ids]);ys.append(y[ids])
    model=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32),
        activation='tanh',alpha=.01,batch_size=256,max_iter=max_iter,
        random_state=random_state,early_stopping=False,tol=1e-5,n_iter_no_change=20))
    with threadpool_limits(limits=1):
        model.fit(np.concatenate(xs),np.concatenate(ys))
    return PatchMLP(model,scale)


def spectral_gain(train):
    train=check_fields(train,4)
    if len(train)<2:
        raise ValueError('Need independent seeds for noise power')
    c=dctn(train,type=2,norm='ortho',axes=(-2,-1))
    noise=c.var(axis=0,ddof=1)
    signal=np.maximum(c.mean(axis=0)**2-noise/len(train),0)
    gain=signal/np.maximum(signal+noise,1e-30)
    gain[:,0,0]=1
    return gain


def spectral_predict(observation,gain):
    obs=check_fields(observation,3)
    if gain.shape!=obs.shape or not np.isfinite(gain).all() or np.any((gain<0)|(gain>1)):
        raise ValueError('Invalid spectral gain')
    return idctn(gain*dctn(obs,type=2,norm='ortho',axes=(-2,-1)),type=2,norm='ortho',axes=(-2,-1))


def noisy_validation_score(estimates, observations, scale):
    """Paired validation MSE; independent noisy target, never evaluation reference."""
    estimates=check_fields(estimates,4); observations=check_fields(observations,4)
    if estimates.shape!=observations.shape:
        raise ValueError('Mismatched validation fields')
    target=independent_targets(observations)
    return float(np.mean(((estimates-target)/np.asarray(scale)[None,:,None,None])**2))


def fit_experiment(train, validation, *, max_iter=160, samples_per_seed=3000):
    """Fit and select using only train/validation fields, with no reference argument."""
    train=check_fields(train,4); validation=check_fields(validation,4)
    if train.shape[1:]!=validation.shape[1:]:
        raise ValueError('Mismatched grids/components')
    mlp=fit_patch_mlp(train,max_iter=max_iter,samples_per_seed=samples_per_seed)
    candidates={}
    # Width is chosen using independent noisy validation targets only.
    for sigma in (.5,1.,1.5,2.):
        values=np.stack([gaussian_filter(o,(0,sigma,sigma),mode='reflect') for o in validation])
        candidates[sigma]=noisy_validation_score(values,validation,mlp.scale)
    width=min(candidates,key=candidates.get)
    return {'mlp':mlp,'gaussian_width':width,'gaussian_validation':candidates,
            'gain':spectral_gain(train),'training_mean':train.mean(axis=0)}


def predict_experiment(fitted, observation):
    obs=check_fields(observation,3); width=fitted['gaussian_width']
    neural=fitted['mlp'].predict(obs)
    return {'Raw(3)':obs.copy(),
      'Gaussian':gaussian_filter(obs,(0,width,width),mode='reflect'),
      'Spectral':spectral_predict(obs,fitted['gain']),
      'Training mean (prior only)':fitted['training_mean'].copy(),
      'Noise2Noise MLP':neural,
      'MLP + observed mean':restore_mean(neural,obs)}


def audit_errors(estimate,reference,observation):
    """Per-component direct finite-reference errors, not exact-truth scores."""
    a=check_fields(estimate,3);b=check_fields(reference,3);o=check_fields(observation,3)
    if a.shape!=b.shape or a.shape!=o.shape:
        raise ValueError('Mismatched grids')
    if np.any(np.linalg.norm(b,axis=(-2,-1))==0):
        raise ValueError('Zero reference norm')
    result=[]
    h,w=b.shape[-2:]; band=max(1,min(h,w)//10)
    edge=np.ones((h,w),bool);edge[band:-band,band:-band]=False
    for x,y,z in zip(a,b,o):
        gx=np.stack(np.gradient(x));gy=np.stack(np.gradient(y))
        result.append({'reference_nrmse':float(np.linalg.norm(x-y)/np.linalg.norm(y)),
            'gradient_nrmse':float(np.linalg.norm(gx-gy)/max(np.linalg.norm(gy),1e-30)),
            'edge_nrmse':float(np.linalg.norm((x-y)[edge])/max(np.linalg.norm(y[edge]),1e-30)),
            'observed_mean_change':float(x.mean()-z.mean()),
            'max_absolute_error':float(abs(x-y).max())})
    return result


def contour_figure(observation, prediction, reference, *, component='qy', seed=0):
    import matplotlib.pyplot as plt
    k={'qx':0,'qy':1}[component]
    arrays=[reference[k],observation[k],prediction[k]]
    limit=max(float(np.max(abs(a))) for a in arrays)
    fig,axes=plt.subplots(1,3,figsize=(14,5.5),layout='constrained')
    for ax,a,title in zip(axes,arrays,['Independent finite-budget reference','Raw DSMC: 3 blocks','Fresh Noise2Noise patch MLP']):
        im=ax.imshow(a,origin='lower',extent=[0,1,0,1],interpolation='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set(title=title,xlabel='normalized column position',ylabel='normalized row position')
        if title!='Independent finite-budget reference':
            error=np.linalg.norm(a-reference[k])/np.linalg.norm(reference[k])
            ax.text(.5,-.19,f'Reference NRMSE = {100*error:.2f}%',transform=ax.transAxes,ha='center',fontsize=12)
    fig.colorbar(im,ax=axes,shrink=.8,label=component+' (archive units)')
    fig.suptitle(f'Real DSMC Noise2Noise | {component} | held-out seed {seed}',fontsize=16)
    return fig


def diagnostic_figure(predictions,reference,rows):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    chosen=['Raw(3)','Gaussian','Spectral','Training mean (prior only)','Noise2Noise MLP']
    for k,component in enumerate(['qx','qy']):
        ax=axes[0,k]
        for method in chosen:
            ax.plot(np.linspace(.005,.995,100),predictions[method][k,:,49],label=method,lw=1.4)
        ax.plot(np.linspace(.005,.995,100),reference[k,:,49],'k--',label='Reference',lw=2)
        ax.set(title=component+': column 49, first held-out seed',xlabel='normalized row position',ylabel='archive units')
        ax=axes[1,k]
        for seed,marker in zip(sorted({r['seed'] for r in rows}),['o','x']):
            scores=[next(r['reference_nrmse']*100 for r in rows if r['field']==component and r['seed']==seed and r['method']==m) for m in chosen]
            ax.plot(range(len(chosen)),scores,marker,ms=7,label=str(seed))
        ax.set(xticks=range(len(chosen)),xticklabels=['Raw(3)','Gaussian','Spectral','Train mean','N2N MLP'],ylabel='Reference NRMSE (%)',title=component+': both held-out seeds')
        ax.legend(title='seed',fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    return fig
