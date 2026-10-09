"""Evidence loading and matched scientific figures for Week 1.2."""
from pathlib import Path
import hashlib,json
import numpy as np
from .pressure_velocity import centerlines,GHIA_X,GHIA_Y,GHIA_U,GHIA_V

LABELS={'simple':'SIMPLE','piso':'PISO (2 corrections)','pimple':'PIMPLE (3 outer, 2 inner)',
        'streamfunction-vorticity':'Streamfunction-vorticity'}


def load_results(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    rows=[]
    for record in manifest['runs']:
        path=directory/(record['name']+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['field_sha256']:
            raise ValueError('Recorded field checksum mismatch: '+record['name'])
        with np.load(path,allow_pickle=False) as f:
            rows.append({**record,**{k:f[k].copy() for k in f.files}})
    return manifest,rows


def figures(rows,cells=64):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts=Path('C:/Windows/Fonts')
    if (fonts/'times.ttf').exists():
        for name in ['times.ttf','timesbd.ttf','timesi.ttf']:
            font_manager.fontManager.addfont(str(fonts/name))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':11,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,
        'figure.facecolor':'white','axes.facecolor':'white','mathtext.fontset':'stix'})
    selected=[r for r in rows if r['config']['cells']==cells]
    names=[r['config']['method'] for r in selected]
    plots={}
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    styles=['-','--','-.',':']
    for r,ls in zip(selected,styles):
        y,u,x,v=centerlines(r)
        axes[0].plot(u,y,ls=ls,label=LABELS[r['config']['method']],lw=1.6)
        axes[1].plot(x,v,ls=ls,label=LABELS[r['config']['method']],lw=1.6)
    axes[0].plot(GHIA_U,GHIA_Y,'ko',ms=4,label='Ghia et al. (1982)')
    axes[1].plot(GHIA_X,GHIA_V,'ko',ms=4)
    axes[0].set(xlabel='u / U at x = 0.5',ylabel='y / L')
    axes[1].set(xlabel='x / L',ylabel='v / U at y = 0.5')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle(f'Re = 100: centerlines on matched spacing, h = 1/{cells}')
    plots['centerlines']=fig
    for key,title in [('speed','Speed / U'),('p','Zero-mean pressure / (rho U^2)'),('omega','Vorticity L / U')]:
        fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
        values=[np.hypot(r['u'],r['v']) if key=='speed' else r[key] for r in selected]
        if key=='speed':lo=0.;hi=max(float(q.max()) for q in values);cmap='viridis'
        else:
            bound=max(float(np.max(np.abs(q))) for q in values)
            lo,hi=-bound,bound;cmap='RdBu_r'
        for ax,r,q in zip(axes.flat,selected,values):
            im=ax.pcolormesh(r['x'],r['y'],q,cmap=cmap,vmin=lo,vmax=hi,shading='auto',rasterized=True)
            if key=='speed':ax.streamplot(r['x'],r['y'],r['u'],r['v'],color='black',density=.8,linewidth=.5,arrowsize=.6)
            ax.set(title=LABELS[r['config']['method']],xlabel='x / L',ylabel='y / L',xlim=(0,1),ylim=(0,1),aspect='equal')
        fig.colorbar(im,ax=list(axes.flat),label=title,shrink=.82)
        fig.suptitle(f'Re = 100: {title}; shared color scale')
        plots[key]=fig
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for r in selected:
        if 'history' not in r:continue
        history=r['history'];label=LABELS[r['config']['method']]
        axes[0].semilogy(history[:,2],history[:,3],label=label)
        axes[1].semilogy(history[:,2],np.maximum(history[:,4],1e-16),label=label)
    axes[0].set(xlabel='Elapsed CPU wall time (s)',ylabel='Steady FV momentum defect, Linf')
    axes[1].set(xlabel='Elapsed CPU wall time (s)',ylabel='Cell divergence, Linf')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Coupling convergence: one measured local CPU solve per method')
    plots['convergence']=fig
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for method,ls in zip(names,styles):
        chosen=sorted([r for r in rows if r['config']['method']==method],key=lambda r:r['config']['cells'])
        for ax,key in zip(axes,['ghia_u_relative_l2','ghia_v_relative_l2']):
            ax.loglog([r['config']['cells'] for r in chosen],[r[key] for r in chosen],ls=ls,marker='o',label=LABELS[method])
            ax.set(xlabel='Uniform cell count per side',ylabel=key.replace('_',' '),xticks=[32,64],xticklabels=['32','64'])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Two-grid benchmark check; no asymptotic order claim')
    plots['grid_errors']=fig
    return plots


def write_report(directory,manifest,rows):
    directory=Path(directory)
    lines=['# Week 1.2: executed Re=100 cavity comparison','',
        'All three FV coupling algorithms solve the same steady discrete equations. PISO and PIMPLE approach that state using backward-Euler physical steps. The existing Week-1-family streamfunction-vorticity source is used unchanged as an independent reference.','',
        '| Method | Cells/side | Iterations or steps | CPU s | Ghia u relative L2 | Ghia v relative L2 | FV momentum Linf | FV divergence Linf |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append('| '+ ' | '.join([LABELS[r['config']['method']],str(r['config']['cells']),
            str(r.get('iterations',r.get('steps'))),f'{r["runtime_seconds"]:.3f}',
            f'{r["ghia_u_relative_l2"]:.6g}',f'{r["ghia_v_relative_l2"]:.6g}',
            f'{r["momentum_residual_linf"]:.3e}' if 'momentum_residual_linf' in r else 'not the same discrete norm',
            f'{r["continuity_linf"]:.3e}' if 'continuity_linf' in r else 'streamfunction construction'])+' |')
    lines+=['','## What this establishes','',
        '- All six new FV runs reach momentum Linf < 1e-7 and continuity Linf < 1e-9.',
        '- The three FV algorithms agree on each grid within 2e-6 in u, v and zero-mean p.',
        '- The FD reference velocity errors decrease on refinement. The FV Ghia errors increase slightly from 32 to 64 cells: this nonmonotone trend is retained explicitly, not labelled an accuracy/refinement pass.',
        '- Ghia values are benchmark data, not an exact solution. Pressure is not validated by Ghia velocity tables.',
        '- The independent streamfunction discretization and pressure recovery need not match the FV pressure pointwise, especially at the singular lid corners.',
        '- CPU values are single-run, local measurements. They do not establish a universal fastest algorithm.',
        '- Two grids do not establish an asymptotic convergence order. Only Re=100 steady coupling, conservation and the stated benchmark-error bounds are qualified. Further refinement and another velocity reference are required before an asymptotic accuracy claim.',
        '- Backward Euler is first order in time. Startup accuracy and larger-time-step PIMPLE benefits need a separate temporal-refinement study.','',
        '## Settings','',
        'Unit square, lid speed 1, density 1, nu=0.01, no-slip impermeable walls; pressure mean zero. Uniform MAC finite-volume grids, central convection and central diffusion. SIMPLE alpha_u=0.7, alpha_p=0.3. PISO: dt=0.05, 2 pressure corrections. PIMPLE: same dt, 3 outer loops and 2 pressure corrections per outer loop. Linear systems use sparse LU (SciPy); no multigrid or packaged CFD solver.','',
        'Normal wall velocities are exactly zero. Tangential wall velocities enter the momentum equations using half-cell distances; the lid corner discontinuity is not silently smoothed. All plotted curves are computed fields.','',
        '## Figures','']
    for name in ['centerlines','speed','p','omega','convergence','grid_errors']:
        lines += [f'![{name.replace("_"," ")}](figures/{name}.png)','']
    lines+=['## Reproduce','',
        '```bash','python qa/run_week01_2.py','python qa/build_week01_2.py','python -m unittest discover -s tests -p test_pressure_velocity.py -v','```','',
        '[Executed field/configuration manifest](manifest.json) records hashes, Python/NumPy/SciPy versions, stopping conditions and all measured quantities.','',
        '[Notebook](../../notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb) | [Lecture](../../lectures/week01_2_pressure_velocity.pdf) | [Algorithm and source notes](../../notebooks/week01_2/README.md)']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
