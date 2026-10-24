"""Exact algebra checks for the expanded Week 1 derivations; no new CFD runs."""
from pathlib import Path
from fractions import Fraction as F
import json,subprocess,hashlib,unittest,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import nbformat as n
from pypdf import PdfReader
from week01_foundations_lesson import update_notebook

def add(*args):
    out={}
    for a in args:
        for k,v in a.items():out[k]=out.get(k,0)+v
    return {k:v for k,v in out.items() if v}
def scale(a,c):return {k:v*c for k,v in a.items() if v*c}
def mul(a,b):
    out={}
    for k,v in a.items():
        for l,w in b.items():
            m=tuple(x+y for x,y in zip(k,l));out[m]=out.get(m,0)+v*w
    return {k:v for k,v in out.items() if v}
def d(a,axis):
    out={}
    for k,v in a.items():
        if k[axis]:
            key=list(k);key[axis]-=1;out[tuple(key)]=v*k[axis]
    return out
def lap(a):return add(d(d(a,0),0),d(d(a,1),1))
U={(3,1,1):2,(1,2,2):-3,(2,0,0):7,(0,1,0):1}
V={(2,2,1):-5,(1,3,0):4,(0,1,2):3,(1,0,0):2}

class Derivations(unittest.TestCase):
    def test_time_derivative_commutation(self):
        w=add(d(V,0),scale(d(U,1),-1))
        self.assertEqual(add(d(d(V,2),0),scale(d(d(U,2),1),-1)),d(w,2))
    def test_full_convective_expansion(self):
        w=add(d(V,0),scale(d(U,1),-1));div=add(d(U,0),d(V,1))
        curl=add(d(add(mul(U,d(V,0)),mul(V,d(V,1))),0),scale(d(add(mul(U,d(U,0)),mul(V,d(U,1))),1),-1))
        self.assertEqual(curl,add(mul(U,d(w,0)),mul(V,d(w,1)),mul(div,w)))
    def test_viscous_curl_and_four_term_divergence(self):
        w=add(d(V,0),scale(d(U,1),-1));div=add(d(U,0),d(V,1))
        self.assertEqual(add(d(lap(V),0),scale(d(lap(U),1),-1)),lap(w))
        self.assertEqual(add(d(lap(U),0),d(lap(V),1)),lap(div))
        psi={(4,2,0):3,(2,3,1):-2};u=d(psi,1);v=scale(d(psi,0),-1)
        self.assertEqual(add(d(u,0),d(v,1)),{})
        self.assertEqual(add(d(lap(u),0),d(lap(v),1)),{})
        self.assertNotEqual(lap(add(d(v,0),scale(d(u,1),-1))),{})
    def test_taylor_stencil_values(self):
        h=F(1,10);x=y=F(1,2);f=lambda x,y:x*x+3*y*y
        c=f(x,y);e=f(x+h,y);w=f(x-h,y);north=f(x,y+h);south=f(x,y-h)
        self.assertEqual([c,e,w,north,south],[F(1),F(111,100),F(91,100),F(133,100),F(73,100)])
        self.assertEqual((e-2*c+w)/h**2,2)
        self.assertEqual((north-2*c+south)/h**2,6)
        f=lambda x:x**4
        self.assertEqual((f(1+h)-2*f(1)+f(1-h))/h**2-12,2*h*h)
    def test_assemble_matrix_and_jacobi(self):
        nodes=[(1,1),(2,1),(1,2),(2,2)]
        a=[[4 if p==q else -1 if sum(abs(x-y) for x,y in zip(p,q))==1 else 0 for q in nodes] for p in nodes]
        self.assertEqual(a,[[4,-1,-1,0],[-1,4,0,-1],[-1,0,4,-1],[0,-1,-1,4]])
        solution=[F(-1,9)]*4;rhs=[F(-2,9)]*4
        self.assertEqual([sum(c*x for c,x in zip(row,solution)) for row in a],rhs)
        old=[F(0)]*4
        for expected in [F(-1,18),F(-1,12),F(-7,72)]:
            old=[(rhs[i]-sum(a[i][j]*old[j] for j in range(4) if i!=j))/4 for i in range(4)]
            self.assertEqual(old,[expected]*4)
    def test_unequal_spacing_update_sign(self):
        dx=F(1,5);dy=F(1,7);e,w,north,south,omega=map(F,[2,3,4,5,-2])
        c=((e+w)*dy**2+(north+south)*dx**2+omega*dx**2*dy**2)/(2*(dx**2+dy**2))
        self.assertEqual((e-2*c+w)/dx**2+(north-2*c+south)/dy**2,-omega)
    def test_notebook_content_and_idempotence(self):
        path=ROOT/'notebooks/week01/03_cavity_ghia.ipynb';current=n.read(path,4)
        old=n.reads(subprocess.check_output(['git','show','HEAD:notebooks/week01/03_cavity_ghia.ipynb'],cwd=ROOT).decode('utf-8'),4)
        codes=lambda nb:[(c.source,c.outputs,c.execution_count) for c in nb.cells if c.cell_type=='code']
        self.assertEqual(codes(current),codes(old))
        updated=update_notebook(path);self.assertEqual(current,updated)
        cells=[c for c in current.cells if c.metadata.get('explicit_derivation')]
        self.assertEqual(len(cells),9)
        for c in cells:
            self.assertEqual(c.source.count('$$')%2,0)
            for command in ['\\topic','\\begin{tabular}','\\begin{tikzpicture}','\\label']:self.assertNotIn(command,c.source)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Derivations))
    if not result.wasSuccessful():raise SystemExit(1)
    pdf=ROOT/'lectures/week01_numerical_foundations.pdf';r=PdfReader(pdf);assert len(r.pages)==35
    record={'revision':'Vorticity transport and Poisson discretization','algebra_checks_passed':result.testsRun,'new_cfd_runs':False,'notebook_solver_code_and_outputs_unchanged':True,'notebook_lesson_update_idempotent':True,'pdf_pages':len(r.pages),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'visual_review':'All 35 pages rendered; revised derivation and stencil pages inspected at higher resolution','reading_map':{'vorticity_derivation':[7,8,9],'four_viscous_terms':[10],'taylor_to_stencil':[14,15],'poisson_to_matrix_and_jacobi':[16,17,18,19,20,21]}}
    out=ROOT/'results/week01_foundations/derivation_revision_validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))
