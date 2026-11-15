"""Rebuild high-Re CFD plots and audit tables from hash-verified native fields."""
from pathlib import Path
import csv, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flowmllab import cavity_highre as high


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    out=ROOT/'results/cavity_highre_ghia'
    from flowmllab.cavity_field_operator import TRAIN,VALIDATION,TEST
    cases=high.load_cases(ROOT,reynolds=sorted(set(TRAIN+VALIDATION+TEST)))
    rows=high.score_rows(cases);mesh=high.mesh_rows(cases)
    time_rows,pairs=high.load_temporal(ROOT)
    write_csv(out/'ghia_scores.csv',rows)
    write_csv(out/'mesh_sensitivity.csv',mesh)
    write_csv(out/'time_sensitivity.csv',time_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False})
    plots={'highre_cfd_ghia.png':high.plot_overview(cases),
           'highre_streamlines.png':high.plot_streamlines(cases),
           'temporal_audit.png':high.plot_temporal(pairs)}
    for re in high.REYNOLDS:plots[f're{re}_cfd_ghia.png']=high.plot_case(cases,re)
    for name,fig in plots.items():
        fig.savefig(out/name,dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.08)
        plt.close(fig)
    provenance=dict(cases=[dict(Re=c['Re'],cells=c['cells'],source=c['source'],sha256=c['sha256'],
                         independent_audit=c['audit']) for c in cases],
                    steady_gate='independent Poisson, vorticity, continuity, wall and velocity defects < 1e-6',
                    all_fields_pass=all(max(c['audit'].values())<1e-6 for c in cases),
                    temporal_pairs_verified=len(pairs),
                    temporal_checks_pass=all(r['finite_horizon_time_check_pass'] for r in time_rows),
                    no_reference_fitting=True,neural_fits_unchanged=True,
                    temporal_scope='256 cells, tU/L=0..10, common deterministic perturbation; not long-time stability',
                    mesh_scope='three native grids; report differences, do not assume Ghia is exact truth',
                    files_sha256={p.relative_to(out).as_posix():high.file_hash(p)
                        for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.npz','.csv','.png')})
    (out/'verification.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(cases),finest_scores=[r for r in rows if r['cells']==512],
                         fine_mesh_sensitivity=[r for r in mesh if r['cells']==256],time_audit=time_rows),indent=2),flush=True)


if __name__=='__main__':main()
