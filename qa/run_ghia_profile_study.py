"""Reproduce the CPU seven-Re centreline-surrogate teaching experiment."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flowmllab import ghia, ghia_surrogate as model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/cavity_ghia_allre'
SEEDS = (7, 23, 61)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = {}; histories = []; validation = []; timings = []
    for config in model.CONFIGS:
        for seed in SEEDS:
            fit = model.fit(config, seed)
            models[config.name, seed] = fit
            filename = f'{config.name}_seed{seed}.npz'
            np.savez_compressed(OUT/filename, theta=fit['theta'], shapes=fit['shapes'])
            # Selection sees only Re=400. No held-out target is evaluated here.
            errors = []
            for component in ('u', 'v'):
                ref = ghia.GHIA[400]; t = ref['y' if component == 'u' else 'x']
                keep = ghia.valid_mask(400, component)
                error = model.predict(fit, 400, t, component)-ref[component]
                errors.append(float(np.sqrt(np.mean(error[keep]**2))))
            validation.append({'model': config.name, 'seed': seed, 'validation_mean_rmse_U': np.mean(errors)})
            timings.append({'model': config.name, 'seed': seed, 'seconds': fit['seconds'],
                            'evaluations': fit['evaluations'], 'parameters': len(fit['theta']), 'status': fit['status']})
            histories.extend(dict(model=config.name, seed=seed, **h) for h in fit['history'])
            print(config.name, seed, 'validation RMSE', np.mean(errors), 'seconds', fit['seconds'], flush=True)
    v = pd.DataFrame(validation); rank = v.groupby('model').validation_mean_rmse_U.mean().sort_values()
    selected = str(rank.index[0])
    freeze = {'selected': selected, 'selection': 'mean u/v RMSE on Re=400, averaged over predeclared seeds',
              'seeds': list(SEEDS), 'roles': ghia.case_roles(), 'configs': [vars(c) for c in model.CONFIGS],
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT/'flowmllab/ghia.py', ROOT/'flowmllab/ghia_surrogate.py', Path(__file__))},
              'models_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*seed*.npz'))}}
    (OUT/'frozen_selection.json').write_text(json.dumps(freeze, indent=2)+'\n')
    # All roles, including the two held-out cases, are first scored after freeze.
    rows = [dict(model='linear_logRe', seed=-1, **r) for r in model.scores(model.linear_baseline)]
    for (name, seed), fit in models.items():
        rows.extend(dict(model=name, seed=seed, **r) for r in model.scores(lambda re,t,c: model.predict(fit,re,t,c)))
    pd.DataFrame(rows).to_csv(OUT/'scores.csv', index=False)
    pd.DataFrame(histories).to_csv(OUT/'history.csv', index=False)
    pd.DataFrame(timings).to_csv(OUT/'timing.csv', index=False)
    v.to_csv(OUT/'selection.csv', index=False)
    for re in ghia.REYNOLDS:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for col, component in enumerate(('u', 'v')):
            ref = ghia.GHIA[re]; t = ref['y' if component == 'u' else 'x']; order = np.argsort(t)
            keep = ghia.valid_mask(re, component)
            axes[0,col].plot(t[keep], ref[component][keep], 'ko', ms=4, label='Ghia samples')
            if (~keep).any():
                axes[0,col].plot(t[~keep],ref[component][~keep],'kx',ms=8,label='Flagged printed entry')
            dense = np.linspace(0,1,301)
            for name, color, style in [('linear_logRe','#777777','--'), ('adam_2','#c33d3d',':'),(selected,'#006aa0','-')]:
                if name == 'linear_logRe':
                    dp = model.linear_baseline(re,dense,component); at = model.linear_baseline(re,t,component)
                else:
                    dp = np.mean([model.predict(models[name,s],re,dense,component) for s in SEEDS],axis=0)
                    at = np.mean([model.predict(models[name,s],re,t,component) for s in SEEDS],axis=0)
                axes[0,col].plot(dense,dp,style,color=color,label=name.replace('_',' '))
                # Errors at actual benchmark samples, never a fabricated dense truth.
                axes[1,col].plot(t[order][keep[order]],(at-ref[component])[order][keep[order]],
                                 style+'o',color=color,ms=3)
            axes[0,col].set_ylabel(component+'/U')
            axes[1,col].set_ylabel('Signed discrepancy / U'); axes[1,col].axhline(0,color='k',lw=.6)
            for ax in axes[:,col]:
                ax.set_xlabel('y/L at x/L=0.5' if component=='u' else 'x/L at y/L=0.5')
                ax.set_xlim(0,1); ax.grid(alpha=.2)
        handles, labels = [], []
        for ax in axes[0]:
            hs,ls = ax.get_legend_handles_labels()
            for h,l in zip(hs,ls):
                if l not in labels: handles.append(h);labels.append(l)
        fig.suptitle(f'Re = {re}  |  {ghia.case_roles()[re]} case  |  centreline surrogate',y=.995)
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
        fig.subplots_adjust(top=.80,bottom=.09,hspace=.38,wspace=.30)
        fig.savefig(OUT/f're{re}_profiles.png',dpi=160);plt.close(fig)
    print('Frozen selection',selected,flush=True)


if __name__ == '__main__':
    main()
