"""Compare retained author PINN and Nektar exports, including boundary mismatch."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from audit_week13_nektar import structured, integrate_paths
from check_week13_nektar_vtu import read_vtu
from evaluate_chris_deep_original import vortex_candidates
from audit_week13_cfd import audit as foam_audit, foam_list, mesh


def main():
    p = argparse.ArgumentParser()
    for k in ('pinn', 'vtu', 'previous', 'output'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--foam', type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.pinn.parent/'audit.json').read_text())
    assert audit['case'] == {'Re':1000.0, 'depth_over_width':2.2, 'tri':0.0}
    pn = np.load(a.pinn)
    xn, yn, fn, continuity = structured(read_vtu(a.vtu), depth=2.2)
    xo, yo, fo, _ = structured(read_vtu(a.previous), depth=2.2)
    x, y = pn['x'], pn['y']
    xx, yy = np.meshgrid(x,y)
    q = np.c_[yy.ravel(),xx.ravel()]
    def interp(xs, ys, f):
        return RegularGridInterpolator((ys,xs), f)(q).reshape(xx.shape)
    nf = {k:interp(xn,yn,fn[k]) for k in ('u','v')}
    old = {k:interp(xo,yo,fo[k]) for k in ('u','v')}
    # Equal-area midpoint quadrature avoids boundary/corner sample overweighting.
    mx=(np.arange(160)+.5)/160; my=(np.arange(352)+.5)*2.2/352
    mxx,myy=np.meshgrid(mx,my); mq=np.c_[myy.ravel(),mxx.ravel()]
    def vel(xs,ys,f):
        return np.column_stack([RegularGridInterpolator((ys,xs),f[k])(mq) for k in ('u','v')])
    ref=vel(xn,yn,fn)
    rel=lambda v: float(np.linalg.norm(v-ref)/np.linalg.norm(ref))
    nh,nv=integrate_paths(x,y,nf['u'],nf['v'])
    ph,pv=integrate_paths(x,y,pn['u'],pn['v'])
    report=dict(case=audit['case'],checkpoint=audit['checkpoint'],
        nektar_file=str(a.vtu),previous_file=str(a.previous),
        velocity_relative_L2=rel(vel(x,y,pn)),
        nektar_last_interval_relative_L2=rel(vel(xo,yo,fo)),
        lid_max_difference=float(np.max(abs(nf['u'][-1]-pn['u'][-1]))),
        nektar_vortices=vortex_candidates(nh,x,y),
        pinn_vortices=vortex_candidates(ph,x,y),
        path_disagreement=dict(nektar=float(np.max(abs(nh-nv))),pinn=float(np.max(abs(ph-pv)))),
        continuity=continuity,
        limitations=['Boundary profiles must match before interpreting differences as PINN error.',
            'Last saved interval does not establish steady or mesh convergence.',
            'Streamfunction extrema use export-grid integration, not spectral roots.'])
    report['openfoam'] = []
    foam_fields = []
    for case in sorted(a.foam.glob('production-*')):
        record,(xf,yf,vf,psif,area)=foam_audit(case)
        if record['re'] != 1000 or not np.isclose(record['depth_over_width'],2.2):
            raise ValueError('OpenFOAM case mismatch')
        fx,fy=np.meshgrid(xf,yf); fq=np.c_[fy.ravel(),fx.ravel()]
        def at_cells(xs,ys,f):
            return np.stack([RegularGridInterpolator((ys,xs),f[k])(fq).reshape(fx.shape) for k in ('u','v')],axis=-1)
        ref_f=at_cells(xn,yn,fn); pin_f=at_cells(x,y,pn)
        def weighted(v):
            return float(np.sqrt(np.sum(area[:,:,None]*(v-vf[:,:,:2])**2)/np.sum(area[:,:,None]*vf[:,:,:2]**2)))
        record['nektar_relative_L2_vs_foam']=weighted(ref_f)
        record['pinn_relative_L2_vs_foam']=weighted(pin_f)
        report['openfoam'].append(record)
        foam_fields.append((xf,yf,vf,psif))
    if len(foam_fields)!=3:
        raise ValueError('Expected all three OpenFOAM grids')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plt.rcParams.update({'font.size':13,'axes.titlesize':14})
    fig,ax=plt.subplots(1,3,figsize=(13,9),layout='constrained')
    levels=np.unique(np.r_[-.12,-.10,-.08,-.06,-.04,-.02,-.005,-1e-3,-1e-4,-1e-5,1e-5,1e-4,1e-3,.003,.006,.009,.012,.015])
    for axis,f,title in zip(ax[:2],[nh,ph],['Nektar++ at t=120','Chris PINN checkpoint 55118']):
        axis.contour(x,y,f,levels=levels,colors='black',linewidths=.8)
        axis.set(xlabel='x/W',ylabel='y/W',title=title,aspect='equal')
    im=ax[2].pcolormesh(x,y,np.hypot(nf['u']-pn['u'],nf['v']-pn['v']),shading='auto',cmap='magma')
    ax[2].set(xlabel='x/W',ylabel='y/W',title='Velocity difference / lid speed',aspect='equal')
    fig.colorbar(im,ax=ax[2],shrink=.6)
    fig.savefig(a.output/'fields.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for f,label in [(nf,'Nektar++'),(pn,'PINN')]:
        ax[0].plot(f['u'][:,len(x)//2],y,label=label)
        ax[1].plot(x,f['v'][len(y)//2],label=label)
        ax[2].plot(x,f['u'][-1],label=label)
    for rec,(xf,yf,vf,_) in zip(report['openfoam'],foam_fields):
        vi=RegularGridInterpolator((yf,xf),vf)
        ax[0].plot(vi(np.c_[yf,np.full(len(yf),.5)])[:,0],yf,'--',label=f"Foam {rec['nx']}")
        ax[1].plot(xf,vi(np.c_[np.full(len(xf),1.1),xf])[:,1],'--',label=f"Foam {rec['nx']}")
    for axis in ax: axis.legend(); axis.grid(alpha=.2)
    ax[0].set(xlabel='u/U',ylabel='y/W',title='Vertical centreline')
    ax[1].set(xlabel='x/W',ylabel='v/U',title='Horizontal centreline')
    ax[2].set(xlabel='x/W',ylabel='u/U',title='Lid boundary comparison')
    fig.savefig(a.output/'profiles.png',dpi=180); plt.close(fig)
    # Identical physical sample grid and derivative operator for all three plots.
    finest=max(range(len(foam_fields)),key=lambda i:len(foam_fields[i][0]))
    cx,cy,cv,_=foam_fields[finest]
    gx,gy=np.meshgrid(cx,cy); cq=np.c_[gy.ravel(),gx.ravel()]
    def common(xs,ys,f):
        return [RegularGridInterpolator((ys,xs),f[k])(cq).reshape(gx.shape) for k in ('u','v')]
    velocities=[common(xn,yn,fn),[cv[:,:,0],cv[:,:,1]],common(x,y,pn)]
    titles=['Nektar++ | t = 120','OpenFOAM | 180 x 396','Chris PINN | checkpoint 55118']
    stream=[]; vort=[]
    for u,v in velocities:
        stream.append(-cumulative_trapezoid(np.pad(v,((0,0),(1,0))),np.r_[0,cx],axis=1,initial=0)[:,1:])
        vort.append(np.gradient(v,cx,axis=1,edge_order=2)-np.gradient(u,cy,axis=0,edge_order=2))
    contour_levels=np.unique(np.r_[-.12,-.10,-.08,-.06,-.04,-.02,-.005,
        -1e-3,-1e-4,-3e-5,-1e-5,-3e-6,-1e-6,-3e-7,-1e-7,
        1e-7,1e-6,1e-5,1e-4,1e-3,.003,.006,.009,.012,.015])
    plt.rcParams.update({'font.size':15,'axes.titlesize':15,'axes.labelsize':16})
    for name,values in [('streamfunction',stream),('vorticity',vort)]:
        fig,axes=plt.subplots(1,3,figsize=(13,9),layout='constrained')
        if name=='vorticity':
            limit=max(float(np.max(abs(w))) for w in values)
            norm=SymLogNorm(linthresh=.01,vmin=-limit,vmax=limit)
        for axis,value,title in zip(axes,values,titles):
            if name=='streamfunction':
                axis.contour(cx,cy,value,levels=contour_levels,colors='#172b4d',linewidths=.9)
            else:
                im=axis.pcolormesh(cx,cy,value,cmap='RdBu_r',norm=norm,shading='auto')
            axis.set(title=title,xlabel='x/W',ylabel='y/W',xlim=(0,1),ylim=(0,2.2),aspect='equal')
        if name=='vorticity':
            fig.colorbar(im,ax=axes,shrink=.7,label=r'$\omega_z W/U$ (symmetric log; linear within $\pm0.01$)')
        fig.suptitle('Re = 1000 | depth / width = 2.2\n'+('Streamfunction contours (identical levels)' if name=='streamfunction' else 'Vorticity: dv/dx - du/dy (identical colour scale)'),fontsize=18)
        fig.supxlabel('Retained checkpoint comparison; lid profiles differ near corners.\n'+('Streamfunction reconstructed from velocity; weak corner eddies require further verification.' if name=='streamfunction' else 'Same 180 x 396 cell-centre grid and finite-difference derivatives; not spectral residuals.'),fontsize=11)
        for ext in ('png','pdf'):
            fig.savefig(a.output/f'{name}_comparison.{ext}',dpi=220)
        plt.close(fig)
    fine_case=sorted(a.foam.glob('production-*'))[finest]
    _,_,weights=mesh(fine_case,json.loads((fine_case/'case-spec.json').read_text()))
    fp=foam_list(fine_case/str(report['openfoam'][finest]['time'])/'p',field=True,components=1).reshape(gx.shape)
    pressures=[RegularGridInterpolator((yn,xn),fn['p'])(cq).reshape(gx.shape),fp,
               RegularGridInterpolator((y,x),pn['p'])(cq).reshape(gx.shape)]
    means=[float(np.sum(weights*f)/np.sum(weights)) for f in pressures]
    pressures=[f-m for f,m in zip(pressures,means)]
    pressure_limit=max(float(np.max(abs(f))) for f in pressures)
    report['pressure']={'gauge':'area-weighted mean removed on common cell-centre grid',
        'removed_means':means,'scale':'p/(rho U_lid^2); OpenFOAM kinematic pressure with U_lid=1',
        'pinn_vs_nektar_relative_L2':float(np.sqrt(np.sum(weights*(pressures[2]-pressures[0])**2)/np.sum(weights*pressures[0]**2)))}
    for name in ('pressure','speed','lower_vortices'):
        fig,axes=plt.subplots(1,3,figsize=(13,9 if name!='lower_vortices' else 5),layout='constrained')
        for i,(axis,title) in enumerate(zip(axes,titles)):
            if name=='pressure':
                im=axis.contourf(cx,cy,pressures[i],levels=np.linspace(-pressure_limit,pressure_limit,41),cmap='RdBu_r')
                axis.contour(cx,cy,pressures[i],levels=np.linspace(-pressure_limit,pressure_limit,15),colors='k',alpha=.25,linewidths=.4)
            elif name=='speed':
                u,v=velocities[i]
                vmax=max(float(np.max(np.hypot(*uv))) for uv in velocities)
                im=axis.pcolormesh(cx,cy,np.hypot(u,v),shading='auto',cmap='viridis',vmin=0,vmax=vmax)
                axis.contour(cx,cy,stream[i],levels=contour_levels,colors='white',alpha=.65,linewidths=.6)
            else:
                axis.contour(cx,cy,stream[i],levels=contour_levels,colors='#17456b',linewidths=1)
            axis.set(title=title,xlabel='x/W',ylabel='y/W',xlim=(0,1),ylim=(0,.5 if name=='lower_vortices' else 2.2),aspect='equal')
        if name!='lower_vortices':
            fig.colorbar(im,ax=axes,shrink=.7,label=r'$(p-\overline{p})/(\rho U^2)$' if name=='pressure' else r'$|\mathbf{u}|/U$')
        fig.suptitle('Re = 1000 | depth / width = 2.2\n'+{'pressure':'Pressure with a common mean-zero gauge','speed':'Speed and streamfunction contours','lower_vortices':'Lower cavity: identical streamfunction levels'}[name],fontsize=18)
        fig.supxlabel('Lid profiles differ near corners; retained PINN checkpoint 55118.\n'+('Area-weighted pressure means removed separately; shared linear colour scale.' if name=='pressure' else 'Derived from retained velocity fields; weak eddies require resolution checks.'),fontsize=11)
        for ext in ('png','pdf'): fig.savefig(a.output/f'{name}_comparison.{ext}',dpi=220)
        plt.close(fig)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
