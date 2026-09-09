"""Independent rectangular cavity setup; qualification is NOT retained CFD evidence.

Session syntax follows https://www.nektar.info/notebooks/tutorials/basics-incns-solver/.
Width=1, depth=5, lid speed=1, Re based on width. No lid smoothing or SVV.
The discontinuous lid/side-wall corner must be audited before benchmark use.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def make_case(output, re=100, order=4, nx=8, ny=40, restart=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = ET.Element('NEKTAR')
    geo = ET.SubElement(root, 'GEOMETRY', DIM='2', SPACE='2')
    verts = ET.SubElement(geo, 'VERTEX')
    for j in range(ny+1):
        for i in range(nx+1):
            ET.SubElement(verts, 'V', ID=str(j*(nx+1)+i)).text = f'{i/nx:.17g} {5*j/ny:.17g} 0'
    edges = ET.SubElement(geo, 'EDGE')
    lookup = {}
    def edge(a, b):
        key = tuple(sorted((a,b)))
        if key not in lookup:
            k = len(lookup)
            lookup[key] = k
            ET.SubElement(edges, 'E', ID=str(k)).text = f'{a} {b}'
        return lookup[key]
    elems = ET.SubElement(geo, 'ELEMENT')
    walls, lid = [], []
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i; b=a+1; d=a+nx+1; c=d+1
            es=[edge(a,b),edge(b,c),edge(c,d),edge(d,a)]
            ET.SubElement(elems, 'Q', ID=str(j*nx+i)).text=' '.join(map(str,es))
            if j==0: walls.append(es[0])
            if i==nx-1: walls.append(es[1])
            if j==ny-1: lid.append(es[2])
            if i==0: walls.append(es[3])
    comp=ET.SubElement(geo,'COMPOSITE')
    ET.SubElement(comp,'C',ID='0').text=f'Q[0-{nx*ny-1}]'
    ET.SubElement(comp,'C',ID='1').text='E['+','.join(map(str,walls))+']'
    ET.SubElement(comp,'C',ID='2').text='E['+','.join(map(str,lid))+']'
    ET.SubElement(geo,'DOMAIN').text='C[0]'
    exp=ET.SubElement(root,'EXPANSIONS')
    ET.SubElement(exp,'E',COMPOSITE='C[0]',NUMMODES=str(order+1),FIELDS='u,v,p',TYPE='MODIFIED')
    cond=ET.SubElement(root,'CONDITIONS')
    solver=ET.SubElement(cond,'SOLVERINFO')
    for key,value in {'SolverType':'VelocityCorrectionScheme','EQTYPE':'UnsteadyNavierStokes',
                      'EvolutionOperator':'Nonlinear','Projection':'Continuous',
                      'SPECTRALHPDEALIASING':'True'}.items():
        ET.SubElement(solver,'I',PROPERTY=key,VALUE=value)
    time=ET.SubElement(cond,'TIMEINTEGRATIONSCHEME')
    ET.SubElement(time,'METHOD').text='IMEX'
    ET.SubElement(time,'ORDER').text='2'
    params=ET.SubElement(cond,'PARAMETERS')
    for key,value in {'TimeStep':0.0005,'NumSteps':100,'IO_CheckSteps':50,
                      'IO_InfoSteps':10,'IO_CFLSteps':10,'Kinvis':1/re}.items():
        ET.SubElement(params,'P').text=f'{key} = {value}'
    variables=ET.SubElement(cond,'VARIABLES')
    for i,v in enumerate(('u','v','p')): ET.SubElement(variables,'V',ID=str(i)).text=v
    regions=ET.SubElement(cond,'BOUNDARYREGIONS')
    ET.SubElement(regions,'B',ID='0').text='C[1]'
    ET.SubElement(regions,'B',ID='1').text='C[2]'
    bcs=ET.SubElement(cond,'BOUNDARYCONDITIONS')
    for i in range(2):
        region=ET.SubElement(bcs,'REGION',REF=str(i))
        ET.SubElement(region,'D',VAR='u',VALUE=str(i))
        ET.SubElement(region,'D',VAR='v',VALUE='0')
        ET.SubElement(region,'N',VAR='p',VALUE='0',USERDEFINEDTYPE='H')
    initial=ET.SubElement(cond,'FUNCTION',NAME='InitialConditions')
    if restart:
        ET.SubElement(initial,'F',VAR='u,v,p',FILE=str(Path(restart).resolve()))
    else:
        for v in ('u','v','p'): ET.SubElement(initial,'E',VAR=v,VALUE='0')
    ET.indent(root)
    ET.ElementTree(root).write(output/'cavity.xml',encoding='utf-8',xml_declaration=True)
    (output/'spec.json').write_text(json.dumps(dict(Re=re,depth=5,width=1,order=order,
        nx=nx,ny=ny,dt=0.0005,steps=100,restart=restart,status='qualification_only'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--re',type=int,choices=[100,500],default=100)
    p.add_argument('--order',type=int,choices=[4,6,8],default=4)
    p.add_argument('--restart')
    a=p.parse_args()
    make_case(a.output,a.re,a.order,restart=a.restart)
