from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
root=Path(__file__).resolve().parents[1]
out=root/'results/cylinder_d40'
z=np.load(out/'re100_D040.npz',allow_pickle=False)
m=json.loads(str(z['metadata'])); c=m['config']; D=c['diameter']; U=c['inflow_velocity']
x=(z['x']-c['center_x'])/D; y=(z['y']-c['center_y'])/D
w=np.ma.masked_where(z['solid'].astype(bool),z['vorticity']*D/U)
t=z['time']*U/D
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','axes.titleweight':'bold'})
fig=plt.figure(figsize=(13,11),facecolor='white')
gs=fig.add_gridspec(3,2,height_ratios=[4,1,1],left=.075,right=.94,bottom=.11,top=.87,hspace=.55,wspace=.30)
ax=fig.add_subplot(gs[0,:]); im=ax.pcolormesh(x,y,w,cmap='RdBu_r',vmin=-2,vmax=2,shading='auto',rasterized=True)
ax.add_patch(Circle((0,0),.5,color='#202936')); ax.set(xlim=(-2,14),ylim=(-3,3),aspect='equal',xlabel=r'$(x-x_c)/D$',ylabel=r'$(y-y_c)/D$')
ax.set_title('(a) Alternating vortex wake | D40, final field at tU/D = 100',loc='left',pad=10)
fig.colorbar(im,ax=ax,pad=.015,fraction=.025,extend='both',label=r'$\omega_z D/U_\infty$')
a=fig.add_subplot(gs[1,0]); b=fig.add_subplot(gs[2,0],sharex=a)
for q,key,col,title,lab in [(a,'lift_coefficient','#2563eb','(b) Lift history | D40',r'$C_L$'),(b,'drag_coefficient','#e57624','(c) Drag history | D40',r'$C_D$')]:
    q.plot(t,z[key],color=col,lw=1); q.set_xlim(20,100); q.set_ylabel(lab); q.set_title(title,loc='left'); q.grid(alpha=.2); q.axvline(45,color='gray',ls='--',lw=1)
b.set_ylim(1.25,1.60); b.set_xlabel(r'Time, $tU_\infty/D$'); a.tick_params(labelbottom=False)
a.text(.02,.92,'Dashed line: statistics start',transform=a.transAxes,fontsize=9,va='top')
q=fig.add_subplot(gs[1:,1])
for d,col in [(18,'#78909c'),(27,'#159581'),(40,'#2563eb')]:
    v=np.genfromtxt(out/f'temporal_audit/D{d}_cycles.csv',delimiter=',',names=True)
    mid=(v['start_tU_D']+v['end_tU_D'])/2
    q.plot(mid,v['Cd_cycle'],'o-',ms=4,lw=1.8,color=col,label=f'D{d}')
q.set_title('(d) Cycle-averaged drag | three grids',loc='left'); q.set_xlabel(r'Cycle midpoint, $tU_\infty/D$'); q.set_ylabel(r'Cycle mean $C_D$'); q.grid(alpha=.2); q.legend(frameon=False)
fig.text(.075,.965,'FlowMLLab | Cylinder wake at Re = 100',fontsize=23,weight='bold',color='#172238')
fig.text(.075,.924,'D2Q9 TRT lattice Boltzmann | 40 nodes per diameter | fixed domain, blockage = 12.5%',fontsize=12,color='#526075')
fig.text(.075,.043,'Teaching dataset: grid and domain independence are not established. Cycle means show temporal drift, not confidence intervals.',fontsize=10,color='#526075')
for ext in ['png','pdf']: fig.savefig(out/f'cylinder_d40_overview.{ext}',dpi=220)
print('FIGURE_READY',out/'cylinder_d40_overview.png')