"""Original flowcharts and a measured one-step coupling experiment at Re=100.

The startup experiment measures an algebraic momentum defect. It does not
measure temporal truncation error or establish a best physical time step.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab.pressure_velocity import CavityConfig,coupling_step,_residual,_momentum,pressure_gradient,divergence

OUT=ROOT/'results/week01_2_pressure_velocity'


def typography():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ['times.ttf','timesbd.ttf','timesi.ttf']:
        path=Path('C:/Windows/Fonts')/name
        if path.exists():font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':11,'mathtext.fontset':'stix','svg.fonttype':'path'})


def flowchart(method):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,Polygon,FancyArrowPatch
    typography()
    if method=='simple':
        nodes=[('Initialize q, p; choose relaxation','box'),
            ('Assemble steady momentum A, b\nApply equation relaxation to A and b','box'),
            ('Predict q*: solve A q* = b - Gp','box'),
            ("Solve pressure correction\nL p' = -D q*; L = -D d G",'box'),
            ("Correct face velocity fully\nq = q* - d G p'",'box'),
            ("Relax pressure: p = p + alpha_p p'\nSet pressure mean to zero",'box'),
            ('Momentum AND continuity\nbelow tolerances?','decision'),
            ('Accept steady solution','end')]
        loops=[(6,1,.95,'No')]
        title='SIMPLE: a steady coupling iteration'
        caption='The iteration index is not physical time. d = 1 / diag(A).'
    elif method=='piso':
        nodes=[('Initialize q, p, physical time','box'),
            ('Store immutable old-time velocities\nChoose dt; keep them fixed within this step','box'),
            ('Assemble BE A, b; predict A q* = b - Gp\nReset the inner counter; keep A fixed','box'),
            ('Recompute H(q) = b - offdiag(A) q\nKeep the predictor matrix A fixed','box'),
            ('Solve absolute pressure\nL p = -D(d H); L = -D d G','box'),
            ('Correct q = d(H - Gp)\nIncrement inner correction counter','box'),
            ('All pressure corrections\ncompleted?','decision'),
            ('Accept the physical step; t = t + dt','box'),
            ('Requested end time or\nsteady target reached?','decision'),
            ('Finish and report residuals','end')]
        loops=[(6,3,.95,'No'),(8,1,.05,'No')]
        title='PISO: inner corrections inside one physical step'
        caption='An inner correction updates H; it is not a linear-solver sweep.'
    else:
        nodes=[('Initialize q, p, physical time','box'),
            ('Store immutable old-time velocities\nReset the outer counter; choose dt','box'),
            ('Rebuild backward-Euler A, b\nPredict q; reset the inner counter','box'),
            ('H update; L p = -D(dH); q = d(H - Gp)\nIncrement inner correction counter','box'),
            ('All inner pressure\ncorrections completed?','decision'),
            ('Increment the outer counter','box'),
            ('All outer momentum\nloops completed?','decision'),
            ('Accept the physical step; t = t + dt','box'),
            ('Requested end time or\nsteady target reached?','decision'),
            ('Finish and report residuals','end')]
        loops=[(4,3,.86,'No'),(6,2,.96,'No'),(8,1,.04,'No')]
        title='PIMPLE: outer momentum loops around PISO'
        caption='Old-time fields stay fixed in BOTH loops. One outer loop = PISO.'
    fig,ax=plt.subplots(figsize=(7.2,9.0));fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.5,.98,title,ha='center',va='top',fontsize=13,fontweight='bold')
    ys=np.linspace(.9,.095,len(nodes));height=.059
    for (label,kind),y in zip(nodes,ys):
        if kind=='decision':
            patch=Polygon([[.5,y+height*.7],[.82,y],[.5,y-height*.7],[.18,y]],
                closed=True,facecolor='white',edgecolor='black',lw=1.15)
        else:
            width=.65 if kind=='box' else .46
            patch=FancyBboxPatch((.5-width/2,y-height/2),width,height,
                boxstyle='round,pad=0.003,rounding_size=0.009',facecolor='white',edgecolor='black',lw=1.1)
        ax.add_patch(patch);ax.text(.5,y,label,ha='center',va='center',fontsize=10.7,linespacing=1.2)
    def arrow(x0,y0,x1,y1):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=10,color='black',lw=1))
    for i in range(len(nodes)-1):
        bottom=ys[i]-(height*.7 if nodes[i][1]=='decision' else height/2+.003)
        top=ys[i+1]+(height*.7 if nodes[i+1][1]=='decision' else height/2+.003)
        arrow(.5,bottom,.5,top)
        if nodes[i][1]=='decision':ax.text(.52,(bottom+top)/2,'Yes',fontsize=9,va='center')
    for src,dst,lane,label in loops:
        right=lane>.5;edge=.82 if right else .18;target=.83 if right else .17
        ax.plot([edge,lane,lane],[ys[src],ys[src],ys[dst]],color='black',lw=1)
        arrow(lane,ys[dst],target,ys[dst])
        ax.text((edge+lane)/2,ys[src]+.012,label,ha='center',fontsize=9)
    ax.text(.5,.025,caption,ha='center',va='bottom',fontsize=10)
    folder=OUT/'flowcharts';folder.mkdir(exist_ok=True)
    for ext in ['png','svg']:
        path=folder/(method+'.'+ext)
        fig.savefig(path,dpi=220,facecolor='white')
        if ext=='svg':
            path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)


def experiment():
    n=32;u=np.zeros((n,n+1));v=np.zeros((n+1,n));p=np.zeros((n,n));rows=[]
    # PIMPLE with one outer loop is the same implementation as two-correction PISO.
    for dt in [.01,.05,.2]:
        for method,outer,inner in [('piso',1,1),('piso',1,2),('piso',1,4),('pimple',3,2),('pimple',6,2)]:
            cfg=CavityConfig(method=method,cells=n,dt=dt,outer_correctors=outer,pressure_correctors=inner)
            start=time.perf_counter();un,vn,pn,count=coupling_step(u.copy(),v.copy(),p.copy(),cfg)
            elapsed=time.perf_counter()-start
            residual=_residual(un,vn,pn,cfg,u,v,dt)
            frozen=None
            if method=='piso':
                blocks=_momentum(u,v,cfg,u,v,dt)
                gradients=pressure_gradient(pn,1/n)
                fields=[un[:,1:-1],vn[1:-1,:]]
                frozen=float(max(np.abs(a@q.ravel()-(b-g).ravel()).max()
                    for (a,b,_),q,g in zip(blocks,fields,gradients)))
            mass=float(np.abs(divergence(un,vn,1/n)).max())
            assert np.isfinite(residual) and mass<1e-9
            rows.append({'method':method,'outer':outer,'inner':inner,'dt':dt,
                'full_nonlinear_step_momentum_linf':residual,'continuity_linf':mass,
                'frozen_predictor_momentum_linf':frozen,
                'pressure_solves':count,'momentum_system_solves':2*outer,'wall_seconds':elapsed})
    record={'case':'One backward-Euler startup step from rest; constant unit lid, Re=100, 32 MAC cells/side',
        'source_sha256':hashlib.sha256((ROOT/'flowmllab/pressure_velocity.py').read_bytes()).hexdigest(),
        'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
        'scope':'Algebraic coupling defect at fixed dt; not a time-truncation error, continuum error or long-time convergence result.',
        'timing_scope':'One local sample per configuration; solve counts are deterministic, wall times are illustrative.',
        'rows':rows}
    (OUT/'coupling_step_study.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    return record


def experiment_figure(record):
    import matplotlib.pyplot as plt
    typography();fig,axes=plt.subplots(1,3,figsize=(12,4.4),layout='constrained')
    labels=['PISO 1','PISO 2','PISO 4','PIMPLE 3x2','PIMPLE 6x2']
    for ax,dt in zip(axes,[.01,.05,.2]):
        chosen=[r for r in record['rows'] if r['dt']==dt]
        values=[r['full_nonlinear_step_momentum_linf'] for r in chosen]
        ax.bar(np.arange(5),values,color=['#485e75']*3+['#91734e']*2,width=.65)
        ax.set_yscale('log');ax.set_xticks(np.arange(5),labels,rotation=45,ha='right')
        ax.set_title('dt = '+str(dt));ax.set_ylabel('Full nonlinear step momentum defect, Linf')
        for i,r in enumerate(chosen):ax.annotate(str(r['pressure_solves'])+' p solves',(i,values[i]),
            xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Re = 100 startup: coupling consistency versus extra solves\nOne physical step; not temporal-accuracy validation')
    fig.savefig(OUT/'figures/coupling_step_study.png',dpi=180,facecolor='white');plt.close(fig)


def main():
    for method in ['simple','piso','pimple']:flowchart(method)
    record=experiment();experiment_figure(record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
