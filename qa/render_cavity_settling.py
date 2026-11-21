"""Render verified, time-matched native cavity continuations without resampling time."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from qa.analyze_cavity_settling import admit_sources, load_case
from qa.build_cavity_startup import (comparison_figure, draw_panels, frame_title, ffmpeg_path,
                                    write_json, sha256, FIELD_COLORS)


def load_display(campaign_path, data_root, seed_directory, reynolds, end_time, interval):
    campaign_path=Path(campaign_path).resolve()
    campaign=json.loads(campaign_path.read_text(encoding='utf-8'))
    admit_sources(campaign)
    data_root=Path(data_root or campaign_path.parent).resolve()
    policy_root=data_root/'policies'
    if not policy_root.is_dir(): policy_root=campaign_path.parent/'policies'
    selected=[]
    for re in reynolds:
        candidates=[c for c in campaign['cases'] if c['purpose']=='continue' and c['reynolds']==re]
        if len(candidates)!=1: raise ValueError(f'Expected exactly one native baseline for Re={re}')
        series=load_case(candidates[0],campaign,data_root,seed_directory,policy_root)
        if series.summary['admission']!='verified_checkpoint':
            raise ValueError(f'No admitted continuation for Re={re}')
        selected.append(series)
    base=selected[0].config
    for s in selected[1:]:
        for key in ('cells','dt','method','convection','outer_correctors','pressure_correctors'):
            if s.config[key]!=base[key]: raise ValueError(f'Matched movie control differs: {key}')
    spacing=Fraction(str(interval)); end=Fraction(str(end_time))
    if spacing<=0 or end<=0 or end%spacing: raise ValueError('Positive end time must be an integer multiple of frame interval')
    keys=[k*spacing for k in range(int(end/spacing)+1)]
    views=[]; proof=[]
    bounds={field:[float('inf'),-float('inf')] for field in ('speed','omega','p')}
    psi_bound=1e-6
    for s in selected:
        missing=[float(k) for k in keys if k not in s.frames]
        if missing: raise ValueError(f'Re={s.case["reynolds"]}: actual frames missing at {missing}; no hold-last or interpolation')
        # Bounds include every retained native frame in the interval, including
        # early .1-spaced seeds omitted from the .5-spaced movie clock.
        for key,frame in s.frames.items():
            if key>end: continue
            values=frame.fields()
            for field in bounds:
                bounds[field][0]=min(bounds[field][0],float(values[field].min()))
                bounds[field][1]=max(bounds[field][1],float(values[field].max()))
            psi_bound=max(psi_bound,float(np.abs(values['psi']).max()))
        native=[s.frames[k].native() for k in keys]
        view={'config':s.config,'times':np.asarray([float(k) for k in keys]),
              **{name:np.stack([frame[name] for frame in native]) for name in ('u_faces','v_faces','p')}}
        views.append(view)
        proof.append({'case':s.case,'admission':s.summary,'displayed_frames':[
            {'time':s.frames[k].time,'path':str(s.frames[k].path) if s.frames[k].path else None,
             'file_sha256':s.frames[k].digest,'seed_index':s.frames[k].seed_index,
             'contract_sha256':s.frames[k].contract,'step':s.frames[k].step,
             'checkpoint_payload':s.frames[k].arrays is not None} for k in keys]})
    limits={field:{'data_min':lo,'data_max':hi,'vmin':0. if field=='speed' else -max(abs(lo),abs(hi),1e-12),
                   'vmax':max(1.,hi) if field=='speed' else max(abs(lo),abs(hi),1e-12)} for field,(lo,hi) in bounds.items()}
    positive=np.geomspace(psi_bound*1e-5,psi_bound,17)
    return views,proof,limits,np.r_[-positive[::-1],positive]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--seed-directory',type=Path)
    p.add_argument('--reynolds',nargs='+',type=int,default=[100,400,1000,3200,5000,7500,10000])
    p.add_argument('--end-time',type=float,required=True)
    p.add_argument('--frame-interval',type=float,default=.5)
    p.add_argument('--fps',type=int,default=12)
    p.add_argument('--dpi',type=int,default=100)
    p.add_argument('--fields',nargs='+',choices=['speed','omega','p'],default=['speed','omega','p'])
    p.add_argument('--stills-only',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.fps<1 or args.dpi<1: raise ValueError('Positive movie fps and dpi required')
    cases,proof,bounds,levels=load_display(args.campaign,args.data_root,args.seed_directory,args.reynolds,args.end_time,args.frame_interval)
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':'#323232','axes.linewidth':.8})
    products={}
    for field in args.fields:
        records=[]
        for k in sorted({0,len(cases[0]['times'])-1,*[int(round(t/args.frame_interval)) for t in (10.,25.,50.,100.,200.) if t<=args.end_time and abs(t/args.frame_interval-round(t/args.frame_interval))<1e-10]}):
            fig,_,_=comparison_figure(cases,field,k,bounds,levels)
            stem=f'{field}_t{cases[0]["times"][k]:g}'
            for ext in ('png','pdf'): fig.savefig(args.output/(stem+'.'+ext),dpi=160)
            plt.close(fig)
            records.append({'time':float(cases[0]['times'][k]),'png':stem+'.png','png_sha256':sha256(args.output/(stem+'.png')),'pdf':stem+'.pdf','pdf_sha256':sha256(args.output/(stem+'.pdf'))})
        product={'stills':records,'limits':bounds[field]}
        if not args.stills_only:
            matplotlib.rcParams['animation.ffmpeg_path']=ffmpeg_path()
            fig,axes,title=comparison_figure(cases,field,0,bounds,levels)
            def update(k):
                draw_panels(cases,field,k,bounds,levels,axes=axes)
                title.set_text(frame_title(cases,field,k))
                return axes
            movie=args.output/f'cavity_continuation_{field}.mp4'
            animation=FuncAnimation(fig,update,frames=range(len(cases[0]['times'])),interval=1000/args.fps,blit=False)
            animation.save(movie,writer=FFMpegWriter(fps=args.fps,codec='libx264',extra_args=['-threads','1','-crf','18','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-movflags','+faststart']),dpi=args.dpi)
            plt.close(fig)
            product.update(movie=movie.name,movie_sha256=sha256(movie))
        products[field]=product
        print(f'Rendered {field} through actual t*={args.end_time:g}',flush=True)
    write_json(args.output/'movies.json',{'schema':'flowmllab.cavity.continuation.movies.v1','fields':products,
        'campaign_sha256':sha256(args.campaign),'render_source_sha256':sha256(Path(__file__)),
        'physical_times':cases[0]['times'].tolist(),'fps':args.fps,'frame_interval':args.frame_interval,
        'sources':proof,'palette_stops':FIELD_COLORS,'streamfunction_levels':levels.tolist(),
        'time_evidence':'Only actual time-matched native fields. No temporal interpolation or holding an earlier final state at a later physical time.',
        'limits_scope':'Full extrema of all retained fields in the displayed physical-time interval, including original finer-cadence startup frames.',
        'qualification':'2D fixed-grid physical-time continuation. Stationarity candidates and numerical sensitivity require separate diagnostic evidence; high-Re128-cell fields remain provisional.'})


if __name__=='__main__': main()
