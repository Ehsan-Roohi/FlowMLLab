"""Re-evaluate author-supplied JCP2 cavity results, without fitting on references.

The archive is historical prospective evidence, not a newly blind experiment.
Outputs are derived teaching evidence; the original archive is never modified.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

METHODS = {'raw_b3':'Raw (3 blocks)', 'raw_b10':'Raw (10 blocks)',
           'pnet_alone':'Archived neural prior',
           'promoted_full_hierarchy':'Archived conditioned estimator',
           'gaussian_fixed':'Gaussian (sigma = 1 cell)'}


def nrmse(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.linalg.norm(b)==0:
        raise ValueError('Invalid field or zero reference norm')
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('tmp/week12_research_v1'))
    args=parser.parse_args(); out=args.output
    if out.exists():
        raise FileExistsError('Choose a new output directory')
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None
        lock=json.loads(z.read('prediction_lock.json'))
        raw=z.read('predictions.npz')
        assert hashlib.sha256(raw).hexdigest()==lock['prediction_sha256']
        pred=dict(np.load(io.BytesIO(raw),allow_pickle=False))
        ref=dict(np.load(io.BytesIO(z.read('reference_stats.npz')),allow_pickle=False))
        archived=list(csv.DictReader(io.StringIO(z.read('metrics.csv').decode())))
    out.mkdir(parents=True)
    fields=[str(f) for f in pred['field_names']]; seeds=pred['seeds']
    # Fixed before reading errors. This comparator is not a paper method or tuned winner.
    pred['method_gaussian_fixed']=np.stack([np.stack([gaussian_filter(f,1,mode='reflect')
                                                    for f in unit]) for unit in pred['method_raw_b3']])
    rows=[]; subset={}; differences=[]
    for field in ('qx','qy'):
        k=fields.index(field); target=ref['reference'][k]
        subset[field+'_reference']=target
        for method in METHODS:
            values=pred['method_'+method][:,k]
            subset[field+'_'+method]=values[0]
            for i,seed in enumerate(seeds):
                error=nrmse(values[i],target)
                if method!='gaussian_fixed':
                    old=next(r for r in archived if int(r['seed'])==int(seed) and
                             r['field']==field and r['method']==method)
                    differences.append(abs(error-float(old['nrmse_observed'])))
                rows.append({'seed':int(seed),'field':field,'method':method,
                             'reference_nrmse':error})
        # A declared first-seed illustration, with all-seed scores below.
        names=['reference','raw_b3','raw_b10','pnet_alone','promoted_full_hierarchy']
        labels=['Independent reference']+[METHODS[n] for n in names[1:]]
        arrays=[subset[field+'_'+n] for n in names]
        limit=max(float(abs(a).max()) for a in arrays)
        fig,axes=plt.subplots(1,5,figsize=(19,4.7),layout='constrained')
        for ax,a,label in zip(axes,arrays,labels):
            im=ax.imshow(a,origin='lower',extent=[0,1,0,1],cmap='RdBu_r',
                         vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set(title=label,xlabel='normalized column position',ylabel='normalized row position')
            if label!='Independent reference':
                ax.text(.5,-.27,f'NRMSE = {100*nrmse(a,target):.2f}%',
                        ha='center',transform=ax.transAxes,fontsize=12)
        fig.colorbar(im,ax=axes,shrink=.75,label=f'{field} (stored archive units)')
        fig.suptitle(f'Real DSMC cavity | Kn = 0.085, lid speed = 350 m/s | {field} | seed {seeds[0]}',fontsize=17)
        fig.savefig(out/f'cavity_{field}.png',dpi=300);fig.savefig(out/f'cavity_{field}.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,5.8),layout='constrained')
        for ax,method,title in zip(axes,['reference','raw_b3','promoted_full_hierarchy'],
                                  ['Independent DSMC reference','Raw DSMC: 3 blocks','Neural prior + observation']):
            a=subset[field+'_'+method]
            im=ax.imshow(a,origin='lower',extent=[0,1,0,1],cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set(title=title,xlabel='normalized column position',ylabel='normalized row position')
            if method!='reference':
                ax.text(.5,-.19,f'Reference NRMSE: {100*nrmse(a,target):.2f}%',transform=ax.transAxes,ha='center',fontsize=13)
        fig.colorbar(im,ax=axes,shrink=.8,label=f'{field} (stored archive units)')
        fig.suptitle(f'DSMC heat-flux reconstruction | {field} | Kn = 0.085 | seed {seeds[0]}',fontsize=17)
        fig.savefig(out/f'cavity_{field}_hero.png',dpi=300);fig.savefig(out/f'cavity_{field}_hero.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
        for method in METHODS:
            a=subset[field+'_'+method]
            axes[0].plot(np.linspace(.005,.995,100),a[:,49],label=METHODS[method],lw=1.5)
        axes[0].plot(np.linspace(.005,.995,100),target[:,49],'k--',lw=2,label='Independent reference')
        axes[0].set(title='Column 49 profile (no interpolation)',xlabel='normalized row position',ylabel=f'{field} (archive units)')
        for j,method in enumerate(METHODS):
            errors=[r['reference_nrmse']*100 for r in rows if r['field']==field and r['method']==method]
            axes[1].plot(np.arange(1,9),errors,'o-',label=METHODS[method],ms=4)
        axes[1].set(title='All eight observation seeds',xlabel='seed index (declared archive order)',ylabel='Reference NRMSE (%)',xticks=range(1,9))
        err=abs(subset[field+'_promoted_full_hierarchy']-target)
        im=axes[2].imshow(err,origin='lower',extent=[0,1,0,1],cmap='magma',vmin=0,vmax=float(err.max()),interpolation='nearest')
        axes[2].set(title='Conditioned estimator: absolute error',xlabel='normalized column position',ylabel='normalized row position')
        fig.colorbar(im,ax=axes[2],shrink=.8,label='absolute error (archive units)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle(f'{field}: profiles, seed variation and remaining error',fontsize=16)
        fig.savefig(out/f'cavity_{field}_audit.png',dpi=300);fig.savefig(out/f'cavity_{field}_audit.pdf');plt.close(fig)
    # Floating-point accumulation differs from the original archive runtime.
    assert max(differences)<2e-6, max(differences)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'first_seed_fields.npz',**subset)
    report={'source_archive':args.archive.name,'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            'prediction_sha256':lock['prediction_sha256'],'condition':'S2_kn0p085_u350',
            'seeds':[int(s) for s in seeds], 'illustrated_seed':int(seeds[0]),
            'scope':'Historical JCP2 archive re-evaluation; not a new prospective trial or neural training. Archive method names retained; no claim of reproducing the final paper estimator.',
            'citation':'Ehsan Roohi, Geometry-native machine learning reconstruction of DSMC moment fields with support monitoring, arXiv:2609.01637 (2026).',
            'reference':'Independent finite-budget DSMC reference; raw-3 and raw-10 observations are disjoint according to archive summary.',
            'metrics':'Equal-cell L2 relative error against stored reference, not reference-noise-deconvolved error.',
            'recomputed_scores':len(rows),'archived_scores_checked':len(differences),
            'max_archived_score_difference':max(differences),
            'fresh_baseline':'16 qx/qy Gaussian-filter evaluations, fixed sigma=1 cell; not selected on this reference.',
            'display':'No interpolation or clipping; symmetric shared color limits cover all five displayed fields. Normalized array positions, not verified physical coordinates.',
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out/'research_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('recomputed_scores','archived_scores_checked','max_archived_score_difference')},indent=2))


if __name__=='__main__':
    main()
