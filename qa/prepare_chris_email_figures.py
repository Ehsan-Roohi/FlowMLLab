"""Build an email figure gallery and a loss plot from an explicit retained log."""
import argparse
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--log',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=[]
for line in a.log.read_text().splitlines():
    m=re.match(r'^\s*(\d+)\s+\[([^\]]+)\]',line)
    if m:
        vals=[float(v) for v in m[2].split(',')]
        if len(vals)==2: rows.append((int(m[1]),*vals))
if not rows: raise ValueError('No training loss records')
# Never connect across a restart/reset of the displayed counter.
segments=[[]]
for row in rows:
    if segments[-1] and row[0]<=segments[-1][-1][0]: segments.append([])
    segments[-1].append(row)
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for index,segment in enumerate(segments):
    x,u,v=zip(*segment)
    for values,label,color in [(u,'Momentum x','#1764ab'),(v,'Momentum y','#d95f02'),([i+j for i,j in zip(u,v)],'Sum','#252525')]:
        ax.semilogy(x,values,'o-',ms=3,label=label if index==0 else None,color=color)
ax.set(xlabel='Logged step since restart (not cumulative training iterations)',ylabel='Logged mean-square residual loss',title='Chris deep-cavity PINN | resumed SSBroyden2 training')
ax.grid(alpha=.2,which='both'); ax.legend()
fig.supxlabel('Job 64412797; parameter warm restart, inverse Hessian reset.\nTraining log values; not independent CFD error or an independent test-set metric.',fontsize=10)
a.output.mkdir(parents=True,exist_ok=True)
for ext in ('png','pdf'): fig.savefig(a.output/f'pinn_loss.{ext}',dpi=180)
plt.close(fig)
names=['fields.png','profiles.png','streamfunction_comparison.png','vorticity_comparison.png','pressure_comparison.png','speed_comparison.png','lower_vortices_comparison.png','pinn_loss.png']
for name in names:
    if not (a.output/name).is_file(): raise FileNotFoundError(name)
(a.output/'email_gallery.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Cavity comparison figures</title></head><body>'+''.join(f'<h2>{n}</h2><img src="{n}" style="max-width:100%">' for n in names)+'</body></html>')
print('Loss points:',len(rows),'last:',rows[-1])
