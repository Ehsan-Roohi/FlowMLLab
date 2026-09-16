"""Create new diagnostic figures from source data and fresh solver executions."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tmp/w14'
OUT = ROOT / 'results/week14_validation'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.titleweight':'bold',
    'axes.labelcolor':'#24364b','text.color':'#24364b','axes.grid':True,
    'grid.alpha':.16, 'figure.facecolor':'white','savefig.facecolor':'white'})
C = ['#166c8d','#e56b35','#6f4c9b','#168779']
def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
def rel(q,r):
    return float(np.linalg.norm(q-r)/np.linalg.norm(r))

dns = np.loadtxt(SRC/'PINN-NN/LM_Channel_5200_mean_prof.dat',comments='%')
stress = np.loadtxt(SRC/'PINN-NN/LM_Channel_5200_vel_fluc_prof.dat',comments='%')
archive = np.loadtxt(SRC/'channel-5200-half-channel-PINN-vist-over-y-uv_tot-2nd-submission/y_u_k_om_uv_5200-RANS-half-channel.txt')
baseline = np.loadtxt(SRC/'channel-5200-half-channel-yfac1.1/y_u_k_om_uv_5200-RANS-half-channel.txt')
ck = np.loadtxt(SRC/'PINN-NN/c_k_pred_5200-plus-units-from-balance.txt')[:,0]
sigma = np.loadtxt(SRC/'PINN-NN/prand_k_5200-plus-units-from-balance-smooth.txt')
com = np.loadtxt(SRC/'PINN-NN/c_omega_2_pred_5200-plus-units-from-balance.txt')
kdns=stress[:,2:5].sum(axis=1)/2
np.testing.assert_allclose(archive[:,0],baseline[:,0],rtol=0,atol=1e-14)
np.savez_compressed(OUT/'teaching_data.npz',dns=dns,stress=stress,
                   corrected_archive=archive,baseline_archive=baseline,ck=ck,sigma=sigma,com=com)
metrics=[]
fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
for ax,truth,label in zip(axes,[dns[:,2],kdns],[r'$U^+$',r'$k^+$']):
    ax.semilogx(dns[:,1],truth,color='#24364b',lw=2.4,label='Lee-Moser DNS')
    ax.set(xlabel=r'$y^+$',ylabel=label,xlim=(.3,5200))
for i,(name,values) in enumerate([('Unmodified RANS (archive)',baseline),('PINN-corrected (archive)',archive)]):
    for j,ref in enumerate([dns[:,2],kdns]):
        axes[j].semilogx(values[:,0]*5200,values[:,j+1],color=C[i],lw=1.8,label=name)
    metrics.append({'case':name,'U_relL2':rel(values[:,1],np.interp(values[:,0],dns[:,0],dns[:,2])),
                    'k_relL2':rel(values[:,2],np.interp(values[:,0],stress[:,0],kdns))})
for name,color in [('baseline',C[0]),('pinn',C[1])]:
    f=OUT/(name+'.npz')
    if f.exists():
        a=np.load(f)
        for ax,key in zip(axes,['u','k']):
            ax.semilogx(a['y'][::5]*5200,a[key][::5],'o',mfc='white',ms=4,color=color,label=name+' fresh restart')
        metrics.append({'case':name+' fresh restart','U_relL2':rel(a['u'],np.interp(a['y'],dns[:,0],dns[:,2])),
                        'k_relL2':rel(a['k'],np.interp(a['y'],stress[:,0],kdns))})
axes[0].set_title('Mean flow: necessary, not sufficient')
axes[1].set_title('Turbulence energy reveals the difference')
axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False)
save(fig,'profiles')

fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
for ax,val,default,title in zip(axes,[sigma,ck,com],[2,1,.075],
                             [r'$\sigma_k$',r'$C_k$',r'$C_{\omega 2}$']):
    ax.semilogx(archive[:,0]*5200,val,color=C[0],lw=2,label='Released correction')
    ax.axhline(default,color=C[1],ls='--',label='Unmodified constant')
    ax.set(xlabel=r'$y^+$',title=title)
axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False)
save(fig,'coefficients')

y,u,k,om,_=archive.T
yp=y*5200
nut=k/om
# Historical notebook adaptation, intentionally distinct from upstream NN.
features=np.c_[np.log10(np.maximum(nut/y,1e-12)),
               np.minimum(abs((1/5200+nut)*np.gradient(u,y,edge_order=2)),.995)]
hold=(yp>=100)&(yp<=400)
model=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(10,10),
      activation='tanh',solver='lbfgs',alpha=1e-5,max_iter=5000,random_state=42))
model.fit(features[~hold],ck[~hold])
pred=np.clip(model.predict(features),0,1)
pc=np.clip(PchipInterpolator(np.log10(yp[~hold]),ck[~hold])(np.log10(yp)),0,1)
gap=[{'method':n,'relative_L2':rel(p[hold],ck[hold]),
       'RMSE':float(np.sqrt(np.mean((p[hold]-ck[hold])**2)))} for n,p in [('PCHIP',pc),('Adapted MLP',pred)]]
fig,ax=plt.subplots(figsize=(9,3.9),layout='constrained')
ax.axvspan(100,400,color='#e9edf2',label='Held-out gap (not unopened)')
ax.semilogx(yp,ck,color='#24364b',lw=2,label='Source target')
ax.semilogx(yp,pc,'--',color=C[0],label='PCHIP')
ax.semilogx(yp,pred,color=C[1],label='Adapted tanh MLP')
ax.set(xlabel=r'$y^+$',ylabel=r'$C_k$',title='A strong interpolation control')
ax.legend(fontsize=9,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False)
save(fig,'gap')

fig,ax=plt.subplots(figsize=(10,3),layout='constrained')
ax.axis('off')
boxes=[('REFERENCE\nDNS statistics',.02),('INVERSE PHYSICS\ndiffusion coefficient',.27),
       ('REGRESSION\nthree local closures',.52),('DEPLOYMENT\ncoupled RANS',.77)]
for i,(label,x) in enumerate(boxes):
    ax.text(x+.10,.56,label,ha='center',va='center',fontsize=10,color='white',
            bbox=dict(boxstyle='round,pad=1.0',fc=C[i],ec='none'),transform=ax.transAxes)
    if i<3: ax.annotate('',xy=(x+.235,.56),xytext=(x+.20,.56),xycoords='axes fraction',
                      arrowprops=dict(arrowstyle='->',color='#24364b',lw=2))
ax.text(.5,.08,'Target fidelity  /  regression error  /  solver convergence  /  physical accuracy',
        ha='center',transform=ax.transAxes,fontsize=11)
save(fig,'pipeline')

if (OUT/'train_ck.npz').exists():
    a=np.load(OUT/'train_ck.npz')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].semilogy(np.arange(1,len(a['loss'])+1),a['loss'],color=C[0])
    axes[0].set(xlabel='Epoch',ylabel='Random-point validation MSE',title='Original SGD / fixed seed 42')
    axes[1].plot(a['target'][a['test_indices']],a['prediction'],'o',color=C[1],ms=4)
    axes[1].plot([0,1],[0,1],'--',color='#8e9aaa')
    axes[1].set(xlabel='Source target',ylabel='NN prediction',title='Same-profile random test points')
    save(fig,'training')

fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
for i,name in enumerate(['baseline','pinn','nn10000','nn5200']):
    path=OUT/(name+'.log')
    if path.exists():
        matches=re.findall(r'--iter:(\d+), max residual:([\deE+.-]+)',path.read_text())
        if matches:
            a=np.array(matches,dtype=float)
            ax.semilogy(a[:,0],a[:,1],color=C[i],label=name)
ax.axhline(1e-6,color='#7b8795',ls='--',lw=1,label='1e-6')
ax.set(xlabel='Outer iteration (restart)',ylabel='Reported maximum scaled residual',title='Convergence is a separate claim')
ax.legend(fontsize=9,ncol=5,loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False)
save(fig,'convergence')
inverse_file=SRC/'PINN-NN-inverse/loss-vist-diffusion-pinn-5200-half-channel-load.txt'
if (OUT/'inverse.json').exists():
    loss=np.loadtxt(inverse_file)
    pred=np.loadtxt(SRC/'PINN-NN-inverse/vist_pred-PINN-from-vist-diffusion-pinn-5200-plus-units-load.txt')
    archived=np.loadtxt(SRC/'PINN-NN/vist_pred-PINN-from-vist-diffusion-pinn-5200-plus-units-load.txt')
    log=(OUT/'inverse.log').read_text()
    last=re.findall(r'Epoch (\d+), Learning Rate: ([^,]+), Loss: ([^,]+), Loss_min: ([^,]+)',log)[-1]
    record=json.loads((OUT/'inverse.json').read_text())
    record.update(epochs=int(last[0]),reported_final_total_loss=float(last[2]),
                  reported_minimum_total_loss=float(last[3]),
                  final_interior_loss=float(loss[-1,0]),
                  reference_diffusivity_relative_L2=rel(pred[:,1],archived[:,1]),
                  finite=bool(np.isfinite(loss).all() and np.isfinite(pred).all()),
                  status='Completed original checkpoint-resume script; downstream uses released coefficients, not these fresh outputs')
    (OUT/'inverse.json').write_text(json.dumps(record,indent=2))
    index=np.unique(np.r_[np.arange(0,len(loss),100),len(loss)-1])
    np.savez_compressed(OUT/'inverse.npz',epochs=index+1,interior_loss=loss[index,0],
                        y=pred[:,0],nu_t_pinn_plus=pred[:,1],released_nu_t_pinn_plus=archived[:,1])
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    axes[0].semilogy((index+1)/1000,loss[index,0],color=C[0])
    axes[0].set(xlabel='Epoch (thousands)',xticks=[0,50,100,150,200],ylabel='Interior residual sum of squares',title='Original inverse PINN: fresh continuation')
    axes[1].semilogx(pred[:,0]*5200,pred[:,1],color=C[0],label='New 200k-epoch execution')
    axes[1].semilogx(archived[:,0]*5200,archived[:,1],'--',color=C[1],label='Released output table')
    axes[1].set(xlabel=r'$y^+$',ylabel=r'$\nu_{t,PINN}/\nu$',title='Compare outputs; do not conflate runs')
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False)
    save(fig,'inverse')
if (OUT/'nn5200.npz').exists():
    a=np.load(OUT/'nn5200.npz')
    metrics.append({'case':'NN5200 assembled-case restart','U_relL2':rel(a['u'],np.interp(a['y'],dns[:,0],dns[:,2])),
                    'k_relL2':rel(a['k'],np.interp(a['y'],stress[:,0],kdns))})
records={p.stem:json.loads(p.read_text()) for p in OUT.glob('*.json') if p.stem in ['baseline','pinn','nn10000','nn5200','train_ck','balance','inverse']}
balance_differences={}
for f in ['c_k_pred_5200-plus-units-from-balance.txt',
          'c_omega_2_pred_5200-plus-units-from-balance.txt',
          'prand_k_5200-plus-units-from-balance-smooth.txt']:
    regenerated=SRC/'PINN-NN-balance'/f
    if regenerated.exists():
        original=np.loadtxt(SRC/'PINN-NN'/f)
        new=np.loadtxt(regenerated)
        balance_differences[f]={'max_absolute_difference':float(np.max(abs(new-original))),
                                'relative_L2':rel(new,original),
                                'matches_released_target':bool(np.allclose(new,original,rtol=1e-6,atol=1e-9))}
result={'profile_metrics':metrics,'held_out_gap':gap,'runs':records,
        'balance_regeneration':balance_differences,
        'claim_boundary':'Known-case reproduction; not prospective blind validation or mesh independence'}
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
