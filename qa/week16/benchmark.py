"""Independent Taylor-Maccoll cone pressure benchmark for axisymmetric Euler."""
import json, numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from pathlib import Path
import cfd
from analyze import load_csv

def cone_exact(mach=1.8,theta_deg=7.):
    gamma=1.4; theta=np.deg2rad(theta_deg)
    v1=1/np.sqrt(1+2/((gamma-1)*mach**2))
    def shoot(beta,dense=False):
        mn=mach*np.sin(beta); ratio=(gamma+1)*mn**2/(2+(gamma-1)*mn**2)
        y0=[v1*np.cos(beta),-v1*np.sin(beta)/ratio]
        def fun(t,y):
            vr,vt=y; h=.5*(gamma-1)*(1-vr*vr-vt*vt)
            return [vt,(vt*vt*vr-h*(2*vr+vt/np.tan(t)))/(h-vt*vt)]
        s=solve_ivp(fun,[beta,theta],y0,rtol=1e-10,atol=1e-12)
        return s.y[:,-1]
    bs=np.linspace(np.arcsin(1/mach)+.001,np.deg2rad(55),100)
    vals=[shoot(z)[1] for z in bs]
    pairs=[(x,y) for x,y,u,v in zip(bs[:-1],bs[1:],vals[:-1],vals[1:]) if u*v<0]
    beta=brentq(lambda z:shoot(z)[1],*pairs[0]); vr,vt=shoot(beta)
    mn=mach*np.sin(beta); pr=1+2*gamma/(gamma+1)*(mn**2-1)
    ratio=(gamma+1)*mn**2/(2+(gamma-1)*mn**2)
    post_v2=(v1*np.cos(beta))**2+(v1*np.sin(beta)/ratio)**2
    pc=pr*((1-vr**2-vt**2)/(1-post_v2))**(gamma/(gamma-1))
    return {'shock_angle_deg':float(np.rad2deg(beta)),'pressure_ratio':float(pc),'cp':float((pc-1)/(.5*gamma*mach**2))}

def run():
    folder=cfd.ROOT/'results/week16_lowboom/runs/cone_benchmark'
    original=cfd.radius
    cfd.radius=lambda x,a=0,b=0:np.asarray(x)*np.tan(np.deg2rad(7))
    # Cone ends at domain outflow, so only one continuous body surface is used.
    import gmsh
    folder.mkdir(parents=True,exist_ok=True)
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0);g=gmsh.model.geo
    points=[g.addPoint(*p) for p in [(0,0,0),(1,float(np.tan(np.deg2rad(7))),0),(1,1,0),(0,1,0)]]
    lines=[g.addLine(points[i],points[(i+1)%4]) for i in range(4)]
    sf=g.addPlaneSurface([g.addCurveLoop(lines)]);g.synchronize()
    for i,l in enumerate(lines):gmsh.model.mesh.setTransfiniteCurve(l,201 if i%2==0 else 121)
    gmsh.model.mesh.setTransfiniteSurface(sf);gmsh.model.mesh.setRecombine(2,sf)
    for name,ls in [('body',[lines[0]]),('farfield',lines[1:])]:
        t=gmsh.model.addPhysicalGroup(1,ls);gmsh.model.setPhysicalName(1,t,name)
    gmsh.model.addPhysicalGroup(2,[sf],1);gmsh.model.mesh.generate(2);gmsh.write(str(folder/'mesh.su2'));gmsh.finalize()
    cfd.radius=original;cfd.config(folder,1.8,2000)
    p=folder/'flow.cfg';p.write_text(p.read_text().replace('MARKER_SYM= (axis)','MARKER_SYM= (NONE)'))
    import subprocess,os,time
    t=time.time()
    with (folder/'solver.log').open('w') as f:r=subprocess.run([os.environ.get('SU2_CFD','SU2_CFD'),'flow.cfg','-t','1'],cwd=folder,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError((folder/'solver.log').read_text()[-3000:])
    s=load_csv(folder/'surface_flow.csv');mask=(s['x']>.3)&(s['x']<.8)
    p=.4*(s['Energy']-.5*(s['Momentum_x']**2+s['Momentum_y']**2)/s['Density'])
    cp=(p-101325)/(.5*1.4*101325*1.8**2)
    exact=cone_exact();measured=float(np.mean(cp[mask]))
    result=dict(exact=exact,cfd_cp_mean=measured,relative_error=abs(measured/exact['cp']-1),wall_seconds=time.time()-t)
    result['passed']=result['relative_error']<.03
    (folder/'benchmark.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':run()
