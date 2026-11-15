"""Replot retained CFD against Ghia; no simulation or neural retraining."""
from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flowmllab import cavity_validation as cv


def main():
    out=ROOT/'results/cavity_cfd_ghia'
    out.mkdir(parents=True,exist_ok=True)
    cases=cv.load_evidence(ROOT)
    pd.DataFrame(cv.score_rows(cases)).to_csv(out/'cfd_ghia_metrics.csv',index=False,lineterminator='\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'mathtext.fontset':'stix'})
    plots={'cfd_ghia_validation':cv.plot_overview(cases),
           'cfd_streamlines':cv.plot_streamlines(cases)}
    for re in cv.CFD_RUNS:
        plots[f're{re}_cfd_profiles']=cv.plot_profiles(cases,re)
    for name,fig in plots.items():
        fig.savefig(out/(name+'.png'),dpi=170,facecolor='white')
        plt.close(fig)
    receipt={'purpose':'Recomputed native CFD/Ghia comparisons, no fitted CFD curves',
        'sources':[{'path':c['source'],'sha256':c['sha256'],
                    'metadata_sha256_lf':c['metadata_sha256_lf']} for c in cases],
        'plots':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.png'))},
        'raw_and_screened_scores_retained':True,
        'high_Re_CFD_missing':[3200,5000,7500,10000],
        'limits':'Native iteration gates differ by method; no new mesh/time qualification. No changes to source archives or frozen neural models.'}
    (out/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(pd.DataFrame(cv.score_rows(cv.finest(cases)))[['Re','method','cells','u_published_relative_l2','v_published_relative_l2','v_screened_relative_l2']].to_string(index=False))


if __name__=='__main__':main()
