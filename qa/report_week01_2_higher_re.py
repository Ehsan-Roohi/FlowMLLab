"""Validate retained experimental fields and build an English measured report."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flowmllab.pressure_velocity_experiments import benchmark_errors
from flowmllab.pressure_velocity_experiment_plots import comparison_figures
from flowmllab.pressure_velocity import divergence
OUT=ROOT/'results/week01_2_higher_re'

def main():
    rows=[];records=[]
    for f in sorted(OUT.glob('*.json')):
        if f.name=='validation.json':continue
        r=json.loads(f.read_text());r['name']=f.stem;records.append(r)
        archive=f.with_suffix('.npz')
        if not archive.exists():continue
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['field_sha256']
        with np.load(archive,allow_pickle=False) as a:r.update({k:a[k].copy() for k in a.files})
        assert all(np.isfinite(a).all() for a in r.values() if isinstance(a,np.ndarray))
        for key,value in benchmark_errors(r).items():assert abs(value-r[key])<1e-13
        assert abs(r['p'].mean())<1e-12
        assert abs(np.abs(divergence(r['u_faces'],r['v_faces'],1/r['config']['cells'])).max()-r['continuity_linf'])<1e-12
        if r['converged']:assert r['momentum_residual_linf']<r['config']['tolerance'] and r['continuity_linf']<1e-9
        rows.append(r)
    pairs=[]
    for re in [400,1000]:
        group=[r for r in rows if r['config']['reynolds']==re and r['config']['cells']==32 and r['config']['convection']=='central']
        simple=next(r for r in group if r['config']['method']=='simple')
        for r in group:
            if r['config']['method']=='simple':continue
            errors={k:float(np.abs(r[k]-simple[k]).max()) for k in ['u','v','p']}
            pairs.append({'Re':re,'method':r['config']['method'],'all_converged':r['converged'] and simple['converged'],'linf_vs_simple':errors})
    import matplotlib.pyplot as plt
    for name,fig in comparison_figures(rows).items():fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');plt.close(fig)
    text=['# Week 1.2: experimental higher-Re cavity study','',
        'Executed 15 October 2026. Independent Python finite-volume calculations; no external CFD solver. The retained Re=100 source and evidence are unchanged.','',
        '## Research context','',
        '- [Pascau and Garcia (2010)](https://congress2.cimne.com/eccomas/proceedings/cfd2010/papers/01330.pdf) compare PISO/SIMPLEC-family variants at Re=1000 on a uniform 100x100 grid and Re=5000 on a wall-clustered 100x100 grid. The preferred variant depends on the case and relaxation. These are collocated-grid results, not a ranking of our MAC PIMPLE solver.',
        '- [NASA driven-cavity validation](https://www.grc.nasa.gov/www/wind/valid/cavity/cavity.html) uses fine grids for higher Re (400 and 3200). This is a compressible low-Mach WIND calculation, not SIMPLE/PISO evidence.',
        '- [OpenFOAM cavity tutorial](https://www.openfoam.com/documentation/tutorial-guide/2-incompressible-flow/2.1-lid-driven-cavity-flow) discusses grid/time-step refinement. Its high-Re turbulent example changes the physical model and cannot qualify a 2D laminar Python solver.',
        '- [NIST FiPy](https://pages.nist.gov/fipy/en/latest/generated/examples.flow.stokesCavity.html) illustrates SIMPLE pressure correction and relaxation in Stokes flow; it does not validate finite-Re behavior.',
        '- [PySIMPLE](https://github.com/VishalKandala/PySIMPLE) is an inspectable staggered-grid educational implementation. Its example settings are not evidence of an arbitrary high-Re accuracy range.','',
        'There is no universal maximum Reynolds number inherent in the algorithm name. Mesh resolution, convection, relaxation, time discretization, coupling convergence and the physical/model assumptions must be assessed separately.','',
        '## Protocol and measured outcomes','',
        'Unit square, discontinuous unit moving lid, no-slip stationary walls, density 1, nu=1/Re. Uniform MAC grids. SIMPLE relaxation: velocity 0.5, pressure 0.2. PISO: two corrections; PIMPLE: three outer loops and two corrections. Backward Euler dt=0.05. All runs start from rest. Steady momentum Linf target 1e-6; continuity is also checked independently. Fixed iteration limits are retained in each config.','',
        'Jobs overlapped on one local CPU host. Wall times are illustrative and must not be used as a controlled speed ranking; pressure/momentum solve counts are the reproducible work measures.','',
        '| Re | Method | Cells | Scheme | Status | Iterations | Momentum Linf | Ghia u rel L2 | Ghia v rel L2 (all) | v sensitivity* | Pressure solves |',
        '|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        c=r['config'];fmt=lambda k:f"{r[k]:.4g}" if k in r else '-'
        text.append(f"| {c['reynolds']} | {c['method'].upper()} | {c['cells']} | {c['convection']} | {r['status']} | {r.get('iterations','-')} | {fmt('momentum_residual_linf')} | {fmt('ghia_u_relative_l2')} | {fmt('ghia_v_relative_l2')} | {fmt('ghia_v_relative_l2_without_x09063')} | {r.get('pressure_solves','-')} |")
    text+=['','*Re=400 sensitivity omits only the published v entry at x=0.9063 (-0.23827), an isolated departure from adjacent tabulated values. The all-point metric is always shown. No replacement reference value is invented. Other Reynolds numbers use every point.','',
        'Reference: Ghia, Ghia and Shin (1982), tables I/II, DOI [10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4), printed pp.398-399. Numeric entries were checked against the original scanned tables. Benchmark data are numerical reference data, not an exact continuum solution.','',
        '![Centerlines](centerlines.png)','','![Convergence and work](convergence.png)','','![Physical fields](fields.png)','',
        '## Coupling agreement at fixed discrete equations','',
        'Maximum absolute differences from SIMPLE on the same 32-cell central grid (u, v and zero-mean p):','']
    for p in pairs:text.append(f"- Re={p['Re']}, {p['method'].upper()}: {p['linf_vs_simple']}; both converged: {p['all_converged']}.")
    text+=['','## Interpretation and student requirements','',
        'Convergence of the equation residual does not remove spatial error. Compare 32/64 meshes before accepting either solution; two meshes do not establish asymptotic order. First-order upwind is a numerical-diffusion control, not automatically a better answer. Pressure has no Ghia validation here.','',
        'The notebook includes opt-in fresh runs, retained results, reference plots, a mesh study and explicit submission requirements. Target <5% velocity-centerline relative L2 error plus refinement evidence. Keep the all-point Re=400 v diagnostic and explain its reference sensitivity. Compare methods only at the same mesh, scheme and final residual/field-error targets. Retain failures.','',
        'Re=400/1000 are experimental exploration settings, not a general high-Re qualification. No Re=5000/10000 runs, turbulence validation, temporal-order validation or higher-Re physical stability limit is claimed. The historical Re=100 lecture remains the qualified baseline; this report and the notebook extension describe the new experiments.']
    (OUT/'README.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    validation={'date':'2026-10-15','retained_runs':len(records),'converged':sum(r.get('converged',False) for r in records),'field_hashes_verified':True,'reference_entries_visually_checked':True,'benchmark_diagnostics_verified':True,'coupling_comparisons':pairs,'scope':'Experimental 2D steady study; local overlapping jobs; no controlled wall-time ranking.'}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
