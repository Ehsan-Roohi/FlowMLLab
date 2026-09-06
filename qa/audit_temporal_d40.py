import json,sys,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
report={"method":"Time-weighted window means and complete positive-going lift cycles; descriptive sensitivity, not independent-sample confidence intervals","cases":{}}
def avg(t,y,a,b):
 q=(t>a)&(t<b); x=np.r_[a,t[q],b]; z=np.r_[np.interp(a,t,y),y[q],np.interp(b,t,y)]
 return float(np.trapezoid(z,x)/(b-a))
for d in (18,27,40):
 p=(root/'results/cylinder_grid_convergence'/f're100_D{d:03d}.npz') if d!=40 else root/'tmp/unity-d40-64026823/production/re100_D040.npz'
 with np.load(p,allow_pickle=False) as z:
  m=json.loads(str(z['metadata'])); c=m['config']; t=z['time'].astype(float)*c['inflow_velocity']/c['diameter']; cd=z['drag_coefficient'].astype(float);cl=z['lift_coefficient'].astype(float)
 assert np.all(np.diff(t)>0) and np.isfinite(cd).all() and np.isfinite(cl).all()
 start=float(m['statistics_start_step'])*c['inflow_velocity']/c['diameter']; end=float(t[-1]); mid=(start+end)/2
 means={str(a):avg(t,cd,a,end) for a in (45,55,65,75)}
 first=avg(t,cd,start,mid);last=avg(t,cd,mid,end)
 sel=t>=start; tt=t[sel];ll=cl[sel]-avg(t,cl,start,end)
 ids=np.flatnonzero((ll[:-1]<=0)&(ll[1:]>0)); crossings=tt[ids]-ll[ids]*(tt[ids+1]-tt[ids])/(ll[ids+1]-ll[ids])
 cyc=[avg(t,cd,a,b) for a,b in zip(crossings[:-1],crossings[1:])]; sts=1/np.diff(crossings)
 assert len(cyc)>=3
 r={'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'window':[start,end],'Cd_window_means':means,'Cd_first_half':first,'Cd_second_half':last,'half_change_percent':100*abs(last-first)/abs(last),'complete_cycles':len(cyc),'cycle_Cd_mean':float(np.mean(cyc)),'cycle_Cd_min':min(cyc),'cycle_Cd_max':max(cyc),'cycle_Cd_first':cyc[0],'cycle_Cd_last':cyc[-1],'cycle_Cd_slope_per_cycle':float(np.polyfit(np.arange(len(cyc)),cyc,1)[0]),'cycle_St_mean':float(np.mean(sts)),'cycle_St_min':float(min(sts)),'cycle_St_max':float(max(sts)),'Lr_temporal_uncertainty':'UNAVAILABLE: only one time-mean velocity field retained; no time-resolved velocity snapshots'}
 np.savetxt(out/f'D{d}_cycles.csv',np.c_[crossings[:-1],crossings[1:],cyc,sts],delimiter=',',header='start_tU_D,end_tU_D,Cd_cycle,St_cycle',comments='')
 report['cases'][str(d)]=r
x=report['cases'];report['D40_minus_D27_by_start']={k:x['40']['Cd_window_means'][k]-x['27']['Cd_window_means'][k] for k in ('45','55','65','75')}
(out/'temporal_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
