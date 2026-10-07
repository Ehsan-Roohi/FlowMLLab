"""Render homepage CFD contours from frozen predictions, without fitting models.

The six instructional figures in results/ remain untouched. Display interpolation
is Matplotlib contour-level interpolation on the native 32x78 retained grid.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmllab import transformer_course as lab

SENSOR_FRAME = 140
FORECAST_FRAMES = (210, 245)
TRANSFER_RE = 105
TRANSFER_BLOCKS = 8
SEED = 17


def snapshot_error(pred, truth):
    return float(np.linalg.norm(pred-truth)/np.linalg.norm(truth))


def render(out, cases, week, title, subtitle, panels, shape, quantity, footer, sensor_ids=None):
    rows, cols = shape
    case = cases[panels[0]['re']]
    fields = [p['field'] for p in panels if not p.get('error', False)]
    errors = [p['field'] for p in panels if p.get('error', False)]
    field_limit = max(float(np.max(np.abs(a))) for a in fields)
    error_limit = max(float(np.max(a)) for a in errors)
    assert field_limit > 0 and error_limit > 0
    field_levels = np.linspace(-field_limit, field_limit, 81)
    error_levels = np.linspace(0, error_limit, 61)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.edgecolor':'#526579', 'axes.linewidth':.65,
                         'xtick.color':'#526579', 'ytick.color':'#526579',
                         'axes.labelcolor':'#334155', 'text.color':'#14283e',
                         'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(13.2, 3.7 if rows==1 else 6.4), facecolor='white')
    fig.text(.065, .965, f'FLOWMLLAB  /  WEEK {week}', fontsize=10.5,
             fontweight='bold', color='#247d93', va='top')
    fig.text(.065, .9, title, fontsize=21, fontweight='bold', va='top')
    fig.text(.065, .828, subtitle, fontsize=10.5, color='#526579', va='top')
    grid = fig.add_gridspec(rows, cols, left=.065, right=.965,
                           bottom=.215 if rows==1 else .155,
                           top=.71 if rows==1 else .735, wspace=.16,
                           hspace=.56 if rows==2 else .1)
    recorded = []
    for i, p in enumerate(panels):
        ax = fig.add_subplot(grid[i//cols, i%cols])
        a = np.asarray(p['field'], dtype=np.float64)
        assert a.shape==(len(case['y']),len(case['x'])) and np.isfinite(a).all()
        if p.get('error', False):
            assert np.min(a)>=0
            ax.contourf(case['x'],case['y'],a,levels=error_levels,cmap='inferno',antialiased=False)
        else:
            ax.contourf(case['x'],case['y'],a,levels=field_levels,cmap='RdBu_r',antialiased=False)
        if p.get('sensors') is not None:
            ids = sensor_ids[np.asarray(p['sensors'])]
            iy, ix = np.unravel_index(ids,a.shape)
            ax.scatter(case['x'][ix],case['y'][iy],s=13,facecolors='white',
                       edgecolors='#13283d',linewidths=.7,zorder=4)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlim(float(case['x'][0]),float(case['x'][-1]))
        ax.set_ylim(float(case['y'][0]),float(case['y'][-1]))
        ax.set_xticks([2,6,10,13]);ax.set_yticks([-2,0,2])
        ax.tick_params(labelsize=9,length=3,pad=2)
        ax.set_xlabel('x / D',fontsize=9.5,labelpad=2)
        if i%cols==0:ax.set_ylabel('y / D',fontsize=9.5,labelpad=2)
        ax.set_title(f'({chr(97+i)}) {p["title"]}',loc='left',fontsize=11.5,pad=7)
        recorded.append({k:v for k,v in p.items() if k not in ('field','sensors')})
        recorded[-1]['array_sha256']=hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()
        if p.get('sensors') is not None:
            recorded[-1]['sensor_indices']=sensor_ids[np.asarray(p['sensors'])].tolist()
    cb_y=.125 if rows==1 else .083
    for left,limit,cmap,label in [(.19,field_limit,'RdBu_r',quantity),(.64,error_limit,'inferno','Absolute error ('+quantity+')')]:
        axis=fig.add_axes([left,cb_y,.23,.018])
        cb=fig.colorbar(ScalarMappable(norm=Normalize(-limit,limit) if cmap=='RdBu_r' else Normalize(0,limit),cmap=cmap),cax=axis,orientation='horizontal')
        cb.ax.tick_params(labelsize=8,length=2,pad=1)
        cb.set_label(label,fontsize=9,labelpad=2)
    fig.text(.065,.016,footer,fontsize=8.4,color='#526579',va='bottom')
    path=out/f'week{week}.png'
    fig.savefig(path,dpi=200,metadata={'Description':title+'; native-grid CFD contours; frozen prediction evidence'})
    plt.close(fig)
    return {'week':week,'title':title,'file':path.name,'sha256':lab.digest(path),
            'shared_field_range':[-field_limit,field_limit],
            'shared_absolute_error_range':[0,error_limit],
            'grid_shape':[len(case['y']),len(case['x'])],
            'x_over_D':[float(case['x'][0]),float(case['x'][-1])],
            'y_over_D':[float(case['y'][0]),float(case['y'][-1])],
            'panels':recorded}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/transformer-contours')
    args=parser.parse_args()
    out=args.output.resolve()
    if not any(out.is_relative_to(base) and out!=base for base in (ROOT/'docs/assets',ROOT/'build')):
        parser.error('Output must be a subdirectory of docs/assets or build, never a numerical evidence directory.')
    out.mkdir(parents=True,exist_ok=True)
    evidence=ROOT/'results/transformer_course_v3'
    protected=[*list((ROOT/'data/modal_labs').glob('*.npz')),
               *list(evidence.glob('*'))]
    before={p.relative_to(ROOT).as_posix():lab.digest(p) for p in protected if p.is_file()}
    cases,inputs=lab.load_data(ROOT)
    stored=np.load(evidence/'predictions.npz',allow_pickle=False)
    states=torch.load(evidence/'checkpoints.pt',map_location='cpu',weights_only=True)
    sensor=states['sensors__Sensor-set-17']
    rep=lab.Representation.from_tensors(sensor['representation'])
    sensor_ids=sensor['sensor_ids'].numpy()
    assert len(sensor_ids)==16
    records=[]
    predkeys=set()
    def field(key,index,re,variable):
        predkeys.add(key)
        value=lab.prediction(stored,key)[index].reshape(cases[re][variable].shape[1:])
        return value
    def panel(title,a,re,frame,**kwargs):
        return dict(title=title,field=a,re=re,frame=frame,**kwargs)
    truth=cases[105]['v'][SENSOR_FRAME].astype(np.float64)
    all_pred=field('sensors__Sensor-set-17-all',SENSOR_FRAME,105,'v')
    half_pred=field('sensors__Sensor-set-17-drop-half',SENSOR_FRAME,105,'v')
    gappy_all=field('sensors__Gappy-POD-variable-all',SENSOR_FRAME,105,'v')
    gappy_half=field('sensors__Gappy-POD-variable-drop-half',SENSOR_FRAME,105,'v')
    all_ids=list(range(16));half_ids=list(range(0,16,2))
    footer='Native 32 x 78 LBM wake ROI; cylinder lies upstream outside the displayed region. L2 values describe this fixed snapshot, not aggregate performance.'
    label=lambda name,p,t: f'{name} | L2 {100*snapshot_error(p,t):.1f}%'
    records.append(render(out,cases,17,'Attention models reconstruct a wake from sparse sensors',
        f'Re = 105  |  fixed frame {SENSOR_FRAME}  |  16 known sensor locations  |  seed {SEED}',
        [panel('CFD reference + 16 sensors',truth,105,SENSOR_FRAME,sensors=all_ids,source='data/modal_labs/re105.npz:v'),
         panel(label('SensorSet',all_pred,truth),all_pred,105,SENSOR_FRAME,prediction_key='sensors__Sensor-set-17-all'),
         panel('Reconstruction error',abs(all_pred-truth),105,SENSOR_FRAME,error=True)],
        (1,3),'v / U',footer,sensor_ids))
    panels=[]
    for frame in FORECAST_FRAMES:
        truth_f=cases[110]['omega'][frame].astype(np.float64)
        pred=field('forecast__Causal-Transformer-17',frame-160,110,'omega')
        panels.extend([panel(f'CFD reference | frame {frame}',truth_f,110,frame,source='data/modal_labs/re110.npz:omega'),
                       panel(label('Causal decoder',pred,truth_f),pred,110,frame,prediction_key='forecast__Causal-Transformer-17'),
                       panel('Decoding error',abs(pred-truth_f),110,frame,error=True)])
    records.append(render(out,cases,18,'Causal wake decoding, seen as flow-field contours',
        'Re = 110  |  two fixed evaluation frames  |  uninterrupted forecast after frame 159  |  seed 17',
        panels,(2,3),r'$\omega D / U$',
        'This is the continuous-state CFD decoder branch. The separate required character-language model is evaluated in the notebook, not in these fields.'))
    projection=rep.decode(rep.encode(truth[None]))[0].reshape(truth.shape)
    records.append(render(out,cases,19,'Sensor placement and the representation limit',
        'Re = 105  |  fixed frame 140  |  training-selected sensor positions  |  retained noisy-observation predictions',
        [panel('CFD reference + 16 sensors',truth,105,140,sensors=all_ids,source='data/modal_labs/re105.npz:v'),
         panel(label('POD projection',projection,truth),projection,105,140,operation='Truth projected into the saved training basis; diagnostic, not a predictor'),
         panel(label('Gappy POD',gappy_all,truth),gappy_all,105,140,prediction_key='sensors__Gappy-POD-variable-all'),
         panel('Gappy reconstruction error',abs(gappy_all-truth),105,140,error=True)],
        (2,2),'v / U','Projection uses the saved representation without refitting. White markers are observation locations; the CFD region perimeter is not a wall.',sensor_ids))
    records.append(render(out,cases,20,'Missing observations: compare the fields and the errors',
        'Re = 105  |  fixed frame 140  |  fixed 8-of-16 subset  |  SensorSet seed 17  |  same stored noisy observations',
        [panel('CFD reference + 8 sensors',truth,105,140,sensors=half_ids,source='data/modal_labs/re105.npz:v'),
         panel(label('SensorSet, 8',half_pred,truth),half_pred,105,140,prediction_key='sensors__Sensor-set-17-drop-half'),
         panel(label('Gappy POD, 8',gappy_half,truth),gappy_half,105,140,prediction_key='sensors__Gappy-POD-variable-drop-half'),
         panel(label('SensorSet, 16 control',all_pred,truth),all_pred,105,140,prediction_key='sensors__Sensor-set-17-all'),
         panel('SensorSet error, 8',abs(half_pred-truth),105,140,error=True),
         panel('Gappy POD error, 8',abs(gappy_half-truth),105,140,error=True)],
        (2,3),'v / U','All field panels share one signed scale; both error panels share one absolute-error scale. This fixed snapshot is not an architecture ranking.',sensor_ids))
    frame=245;truth_f=cases[110]['omega'][frame].astype(np.float64)
    transformer=field('forecast__Causal-Transformer-17',frame-160,110,'omega')
    mlp=field('forecast__History-MLP-17',frame-160,110,'omega')
    dmd=field('forecast__DMD-r8',frame-160,110,'omega')
    records.append(render(out,cases,21,'Autonomous prediction: CFD, neural models and DMD',
        'Re = 110  |  fixed frame 245  |  86 forecast updates after the last observed frame 159  |  neural seed 17',
        [panel('CFD reference',truth_f,110,frame,source='data/modal_labs/re110.npz:omega'),
         panel(label('Transformer',transformer,truth_f),transformer,110,frame,prediction_key='forecast__Causal-Transformer-17'),
         panel(label('History-MLP',mlp,truth_f),mlp,110,frame,prediction_key='forecast__History-MLP-17'),
         panel(label('DMD',dmd,truth_f),dmd,110,frame,prediction_key='forecast__DMD-r8'),
         panel('Transformer error',abs(transformer-truth_f),110,frame,error=True),
         panel('History-MLP error',abs(mlp-truth_f),110,frame,error=True)],
        (2,3),r'$\omega D / U$','No observation resets occur at the evaluation boundary. Snapshot errors are separate from the full-trajectory and multi-seed scores in the retained evidence.'))
    truth_t=cases[TRANSFER_RE]['omega'][frame].astype(np.float64)
    pre=field('transfer__Re105-B8-pretrained-17',frame-160,105,'omega')
    matched=field('transfer__Re105-B8-matched-steps-17',frame-160,105,'omega')
    target=field('transfer__Re105-B8-target-POD-MLP-17',frame-160,105,'omega')
    records.append(render(out,cases,22,'Transfer auditing: pretraining, compute and representation',
        'Re = 105  |  fixed frame 245  |  84 target labels including validation and initialization  |  seed 17',
        [panel('CFD reference',truth_t,105,frame,source='data/modal_labs/re105.npz:omega'),
         panel(label('Pretrained',pre,truth_t),pre,105,frame,prediction_key='transfer__Re105-B8-pretrained-17'),
         panel(label('Matched steps',matched,truth_t),matched,105,frame,prediction_key='transfer__Re105-B8-matched-steps-17'),
         panel(label('Target-POD-MLP',target,truth_t),target,105,frame,prediction_key='transfer__Re105-B8-target-POD-MLP-17'),
         panel('Pretrained error',abs(pre-truth_t),105,frame,error=True),
         panel('Matched-step error',abs(matched-truth_t),105,frame,error=True)],
        (2,3),r'$\omega D / U$','Matched optimizer-update ceilings do not equal FLOPs or source access. Target-POD-MLP changes the representation; it is a diagnostic control.'))
    after={name:lab.digest(ROOT/name) for name in before}
    assert before==after,'A protected input or original instructional figure changed'
    manifest={'renderer':'qa/plot_transformer_homepage.py','renderer_sha256':lab.canonical_digest(Path(__file__)),
              'environment':dict(lab.runtime_environment(),matplotlib=matplotlib.__version__),'input_sha256':before,
              'data_source':inputs['source_release'],'fixed_selection':{'sensing_frame':140,'forecast_frames':[210,245],
              'transfer_re':105,'transfer_blocks':8,'transfer_frame':245,'neural_seed':17},
              'operations':['Reconstruct saved low-rank prediction storage','Read retained CFD arrays',
                            'Project one truth snapshot into its saved training basis for a labeled diagnostic',
                            'Compute instantaneous field-relative-L2 and absolute error','Native-grid contour display'],
              'fitting_performed':False,'checkpoint_inference_performed':False,
              'snapshot_metric':'float64 ||prediction - CFD||_2 / ||CFD||_2 over all native-grid cells in the displayed frame; labels rounded to one decimal percent',
              'original_inputs_and_instructional_figures_unchanged':True,
              'display':'81 shared signed field levels and 61 shared absolute-error levels per figure. No field regridding or smoothing; contour level crossings interpolate only for display. Equal x/D and y/D aspect.',
              'scope':'Previously inspected coarse LBM wake ROI; no new blind evaluation, grid-independence claim, full-domain CFD or language-model-generated CFD claim.',
              'reconstructed_prediction_keys':sorted(predkeys),'figures':records}
    lab.json_write(out/'manifest.json',manifest)
    print('PASS: six native-grid contour figures rendered; fixed snapshot choices; shared scales; original scientific inputs and instructional figures unchanged.',flush=True)


if __name__=='__main__':
    main()
