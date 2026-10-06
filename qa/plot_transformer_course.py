"""Explicitly generate figures from retained evidence; renderer never calls this."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from flowmllab import transformer_course as lab

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/transformer_course_v3'
cases,_=lab.load_data(ROOT);lab.seed_all()
plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def save(fig,week):
    fig.savefig(OUT/f'week{week}.png',dpi=180,bbox_inches='tight');plt.close(fig)

states=torch.load(OUT/'checkpoints.pt',map_location='cpu',weights_only=True)
bundle=states['sensors__Sensor-set-17'];model,rep=lab.checkpoint_model(bundle)
tokens=lab.sensor_tokens(cases[110]['v'][[80,120,160]],bundle['sensor_ids'].numpy(),cases[90],bundle['value_mean'],bundle['value_scale'])
with torch.no_grad():
    embedded=model.embed(torch.tensor(tokens,dtype=torch.float32));b,n,d=embedded.shape
    q,k,v=model.block.qkv(model.block.norm(embedded)).reshape(b,n,3,4,d//4).permute(2,0,3,1,4).unbind(0)
    _,weight=lab.attention(q,k,v)
fig,axes=plt.subplots(3,4,figsize=(11,7),sharex=True,sharey=True)
for i in range(3):
    for head in range(4):
        axes[i,head].imshow(weight[i,head],vmin=0,vmax=float(weight.max()),cmap='viridis')
        axes[i,head].set_title(f'frame {[80,120,160][i]}, head {head+1}',fontsize=12)
fig.suptitle('Trained sensor attention: content-dependent weights, not causal explanations');fig.tight_layout();save(fig,17)

decoder=lab.CausalDecoder();x=torch.randn(1,4,8);changed=x.clone();changed[:,2:]+=10
with torch.no_grad():
    a=(decoder(x)-decoder(changed)).abs().mean(-1)[0].numpy()
    for block in decoder.blocks:block.causal=False
    b=(decoder(x)-decoder(changed)).abs().mean(-1)[0].numpy()
fig,axes=plt.subplots(1,2,figsize=(10,3.6))
axes[0].imshow(np.tril(np.ones((4,4))),vmin=0,vmax=1,cmap='Blues');axes[0].set_title('Allowed dependencies');axes[0].set_xlabel('input position');axes[0].set_ylabel('output position')
axes[1].bar(np.arange(4)-.17,a,.34,label='causal');axes[1].bar(np.arange(4)+.17,b,.34,label='unmasked');axes[1].set_title('Response to changed future tokens');axes[1].set_xlabel('output position');axes[1].set_ylabel('mean absolute change');axes[1].legend();fig.tight_layout();save(fig,18)

fig,axes=plt.subplots(1,2,figsize=(10,3.8));field=cases[110]['v'][100]
bound=abs(field).max();axes[0].imshow(field,cmap='RdBu_r',vmin=-bound,vmax=bound)
iy,ix=np.unravel_index(bundle['sensor_ids'].numpy(),field.shape);axes[0].scatter(ix,iy,c='black',s=18);axes[0].set_title('Sixteen training-selected sensors')
axes[1].bar(['points','patches','one POD state'],[2496**2,104**2,1]);axes[1].set_yscale('log');axes[1].set_ylabel('pairwise scores per head');axes[1].set_title('Tokenization changes quadratic storage');fig.tight_layout();save(fig,19)

sensors=json.loads((OUT/'sensors.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
names=list(dict.fromkeys(r['method'] for r in sensors['rows']))
for ax,condition in zip(axes,['all','drop-half']):
    for i,name in enumerate(names):
        vals=[r['metrics']['field_relative_l2'] for r in sensors['rows'] if r['method']==name and r['condition']==condition]
        ax.scatter([i]*len(vals),vals,s=28,label=name)
    ax.axhline(sensors['rows'][0]['metrics']['representation_floor'],c='black',ls='--',label='POD floor')
    ax.set_xticks(range(len(names)),names,rotation=22,ha='right');ax.set_title(condition);ax.set_ylabel('retained field relative L2')
fig.tight_layout();save(fig,20)

forecast=json.loads((OUT/'forecast.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
for row in forecast['rows']:
    if row['seed'] in (None,17):
        axes[0].plot(np.arange(51,122),row['metrics']['per_frame_relative_l2'],label=row['key'])
        axes[1].scatter(row['metrics']['representation_floor'],row['metrics']['in_subspace_relative_l2'],label=row['key'])
axes[0].set_ylim(0,.16);axes[0].set_xlabel('steps since initialization');axes[0].set_ylabel('field relative L2');axes[0].set_title('Persistence exceeds this zoomed range')
axes[1].set_xlabel('representation floor');axes[1].set_ylabel('in-subspace error');axes[1].legend(fontsize=11);fig.tight_layout();save(fig,21)

transfer=json.loads((OUT/'transfer.json').read_text());fig,axes=plt.subplots(2,3,figsize=(12,7))
for j,reynolds in enumerate((100,105,110)):
    for arm in ('scratch','pretrained','matched-steps','target-POD-MLP','DMD'):
        rows=[r for r in transfer['rows'] if r['target_re']==reynolds and r['arm']==arm]
        for i,metric in enumerate(('field_relative_l2','in_subspace_relative_l2')):
            groups=[[r['metrics'][metric] for r in rows if r['labels']['total_target_labels']==n] for n in (44,84,154)]
            means=np.array([np.mean(g) for g in groups]);lo=np.array([min(g) for g in groups]);hi=np.array([max(g) for g in groups])
            axes[i,j].plot([44,84,154],means,marker='o',label=arm);axes[i,j].fill_between([44,84,154],lo,hi,alpha=.12)
    floor=next(r['metrics']['representation_floor'] for r in transfer['rows'] if r['target_re']==reynolds and r['arm']=='scratch')
    axes[0,j].axhline(floor,c='black',ls='--',label='source-POD floor');axes[0,j].set_title(f'Re = {reynolds}');axes[1,j].set_xlabel('total target labels')
axes[0,0].set_ylabel('field relative L2');axes[1,0].set_ylabel('in-subspace relative L2');axes[1,2].legend(fontsize=12);fig.suptitle('Fixed-start transfer: mean and seed range, not confidence intervals');fig.tight_layout();save(fig,22)

manifest=json.loads((OUT/'manifest.json').read_text())
manifest.setdefault('training_source_hashes',dict(manifest['source_hashes']))
manifest['source_hashes']['qa/plot_transformer_course.py']=lab.canonical_digest(Path(__file__))
manifest['figure_provenance']='Figures regenerated from unchanged numerical artifacts; training_source_hashes preserves the original training-run snapshot.'
manifest['files'].update({p.name:lab.digest(p) for p in OUT.glob('week*.png')})
lab.json_write(OUT/'manifest.json',manifest)
print('Six figures generated and hashed; scientific results unchanged.')
