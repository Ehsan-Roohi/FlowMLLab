"""Gmsh/SU2 axisymmetric Euler experiment. Length=1 m; dimensional pressure.
Run from repository root. SU2_CFD must be on PATH or supplied through SU2_CFD.
"""
from pathlib import Path
import os, json, time, subprocess, hashlib, argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
VOLUME=np.pi*.06**2/2

def radius(x,a=0.,b=0.):
    x=np.asarray(x); z=np.linspace(0,1,4001)
    raw=lambda t: np.sin(np.pi*t)*np.exp(a*(2*t-1)+b*np.cos(2*np.pi*t))
    scale=np.sqrt(VOLUME/(np.pi*np.trapezoid(raw(z)**2,z)))
    return scale*raw(x)

def mesh(folder,a=0.,b=0.,level=1.,height=1.5,end=3.):
    import gmsh
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0)
    gmsh.model.add('axisymmetric_body'); g=gmsh.model.geo
    xs=[-.5,0,1,end]
    lo=[g.addPoint(x,0,0) for x in xs]; hi=[g.addPoint(x,height,0) for x in xs]
    verts=[g.addLine(lo[i],hi[i]) for i in range(4)]
    pts=[lo[1]]+[g.addPoint(float(x),float(radius(x,a,b)),0) for x in np.linspace(0,1,101)[1:-1]]+[lo[2]]
    bottom=[g.addLine(lo[0],lo[1]),g.addSpline(pts),g.addLine(lo[2],lo[3])]
    top=[g.addLine(hi[i],hi[i+1]) for i in range(3)]
    surfs=[]
    for i in range(3):
        loop=g.addCurveLoop([bottom[i],verts[i+1],-top[i],-verts[i]])
        surfs.append(g.addPlaneSurface([loop]))
    g.synchronize()
    q=1.04**(1/level)
    nr=round(np.log(1+(height/1.5)*(q**round(64*level)-1))/np.log(q))+1
    for v in verts: gmsh.model.mesh.setTransfiniteCurve(v,nr,'Progression',1.04**(1/level))
    for i,s in enumerate(surfs):
        nx=round((xs[i+1]-xs[i])*90*level)+1
        for c in (bottom[i],top[i]): gmsh.model.mesh.setTransfiniteCurve(c,nx)
        gmsh.model.mesh.setTransfiniteSurface(s,'Left',[lo[i],lo[i+1],hi[i+1],hi[i]])
        gmsh.model.mesh.setRecombine(2,s)
    for name,curves in [('body',[bottom[1]]),('axis',[bottom[0],bottom[2]]),('farfield',top+[verts[0],verts[-1]])]:
        tag=gmsh.model.addPhysicalGroup(1,curves); gmsh.model.setPhysicalName(1,tag,name)
    gmsh.model.addPhysicalGroup(2,surfs,1)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(folder/'mesh.su2')); gmsh.write(str(folder/'mesh.msh'))
    tags,coords,_=gmsh.model.mesh.getNodes()
    ets,etags,_=gmsh.model.mesh.getElements(2)
    all_tags=np.concatenate(etags)
    quality=gmsh.model.mesh.getElementQualities(all_tags,'minSJ')
    info={'nodes':len(tags),'cells':len(all_tags),'min_scaled_jacobian':float(min(quality))}
    gmsh.finalize()
    if info['min_scaled_jacobian']<=0: raise ValueError('Invalid mesh')
    return info

def config(folder,mach=1.8,iterations=1800,axisym=True):
    text=f'''SOLVER= EULER
MATH_PROBLEM= DIRECT
AXISYMMETRIC= {'YES' if axisym else 'NO'}
RESTART_SOL= NO
MACH_NUMBER= {mach}
AOA= 0.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
REF_DIMENSIONALIZATION= DIMENSIONAL
REF_AREA= 1.0
REF_LENGTH= 1.0
MARKER_EULER= (body)
MARKER_SYM= (axis)
MARKER_FAR= (farfield)
MARKER_MONITORING= (body)
MARKER_PLOTTING= (body)
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= (0.5, 1.5, 1.0, 100.0)
ITER= {iterations}
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.03
LIMITER_ITER= 300
TIME_DISCRE_FLOW= EULER_IMPLICIT
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 0.05
LINEAR_SOLVER_ITER= 10
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= -9
CONV_STARTITER= 500
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
OUTPUT_FILES= (RESTART_ASCII, PARAVIEW, SURFACE_CSV)
VOLUME_OUTPUT= (COORDINATES, SOLUTION, PRIMITIVE)
OUTPUT_WRT_FREQ= {iterations}
CONV_FILENAME= history
RESTART_FILENAME= restart_flow
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, DRAG)
SCREEN_WRT_FREQ_INNER= 100
'''
    (Path(folder)/'flow.cfg').write_text(text)

def run(name,a=0.,b=0.,level=1.,mach=1.8,iterations=1800,height=1.5,end=3.):
    folder=ROOT/'results/week16_lowboom/runs'/name; folder.mkdir(parents=True,exist_ok=True)
    meta={'name':name,'a':a,'b':b,'level':level,'mach':mach,'height':height,'end':end,'volume':VOLUME,'axisymmetric':True}
    meta.update(mesh(folder,a,b,level,height,end)); config(folder,mach,iterations)
    start=time.time()
    exe=os.environ.get('SU2_CFD','SU2_CFD')
    with (folder/'solver.log').open('w') as f:
        proc=subprocess.run([exe,'flow.cfg','-t','1'],cwd=folder,stdout=f,stderr=subprocess.STDOUT)
    meta.update(returncode=proc.returncode,wall_seconds=time.time()-start)
    for fn in ['mesh.su2','flow.cfg']:
        meta[fn+'_sha256']=hashlib.sha256((folder/fn).read_bytes()).hexdigest()
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    if proc.returncode: raise RuntimeError((folder/'solver.log').read_text()[-4000:])
    return meta
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--name',default='baseline_medium'); p.add_argument('--a',type=float,default=0); p.add_argument('--b',type=float,default=0); p.add_argument('--level',type=float,default=1); p.add_argument('--mach',type=float,default=1.8); p.add_argument('--iterations',type=int,default=1800); p.add_argument('--height',type=float,default=1.5); p.add_argument('--end',type=float,default=3)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
