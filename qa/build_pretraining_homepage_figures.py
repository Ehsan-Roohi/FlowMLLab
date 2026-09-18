"""Readable homepage figures from retained numerical records, without retraining."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 17, 'axes.labelsize': 15,
                         'legend.fontsize': 12, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'font.family': 'DejaVu Sans'})
    ink, red, blue, purple = '#17324D', '#C74440', '#2374AB', '#8258A5'
    phase = json.loads((ROOT/'results/cylinder_phase/phase_stable_metrics.json').read_text())
    groups = ['validation', 'fresh_test', 'retained_test']
    labels = ['Re100\nValidation', 'Re95\nHeld-out at model selection', 'Re105\nHistorical test']
    global_error = [100*phase[g]['vorticity_global_relative_l2'] for g in groups]
    worst_error = [100*phase[g]['vorticity_max_frame_relative_l2'] for g in groups]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=.11, right=.97, top=.76, bottom=.22)
    fig.text(.07,.95,'WEEK 7  |  AUTONOMOUS WAKE PREDICTION',color=ink,weight='bold',fontsize=18)
    fig.text(.07,.89,'4 initial fields  •  277 future frames  •  No future CFD inputs',fontsize=14)
    x=np.arange(3);width=.32
    a=ax.bar(x-width/2,global_error,width,label='Global error',color=blue)
    b=ax.bar(x+width/2,worst_error,width,label='Worst-frame error',color='#7BB8CE')
    ax.bar_label(a,fmt='%.2f%%',padding=5,fontsize=13)
    ax.bar_label(b,fmt='%.2f%%',padding=5,fontsize=13)
    ax.set_xticks(x,labels);ax.set_ylabel('Vorticity relative L2 error (%)');ax.set_ylim(0,10.5)
    ax.legend(loc='upper right',frameon=False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.07,.07,'Phase-stable Fourier decoder; educational LBM reference.',fontsize=12,color=ink)
    fig.text(.07,.03,'Prediction accuracy does not establish grid-independent CFD accuracy.',fontsize=12,color=ink)
    fig.savefig(ROOT/'results/cylinder_phase/homepage_week07.png',dpi=180);plt.close(fig)

    folder=ROOT/'results/week07_4_diverse_pretraining'
    m=json.loads((folder/'metrics.json').read_text());ks=m['protocol']['ks']
    names=['pretrained_encoder','random_encoder','pod32']
    titles=['Pretrained encoder + ridge','Random encoder + ridge','POD-32 + ridge']
    colors=[red,purple,blue]
    fig,axs=plt.subplots(2,1,figsize=(10,10.2),gridspec_kw={'height_ratios':[1.35,1]})
    fig.subplots_adjust(left=.12,right=.96,top=.85,bottom=.14,hspace=.48)
    fig.text(.07,.96,'WEEK 7.4  |  LEARNING FROM UNLABELLED WAKES',fontsize=18,weight='bold',color=ink)
    fig.text(.07,.92,'Lift decoding on 4 Reynolds cases excluded from pretraining',fontsize=14)
    for name,label,color in zip(names,titles,colors):
        c=[m['curves'][name][str(k)] for k in ks];y=np.array([v['mean'] for v in c])
        err=np.array([y-[v['min'] for v in c],[v['max'] for v in c]-y])
        axs[0].errorbar(ks,y,yerr=err,label=label,color=color,marker='o',lw=2,capsize=3)
    axs[0].set(xscale='log',yscale='log',ylabel='Lift NRMSE (%)',xlabel='Labelled frames per target trajectory (k)')
    axs[0].set_xticks(ks,[str(k) for k in ks]);axs[0].set_yticks([5,10,20,50,100,200],['5','10','20','50','100','200'])
    axs[0].legend(frameon=False,fontsize=11,loc='upper right');axs[0].grid(alpha=.2,which='both')
    e=m['example'];t=np.asarray(e['time'])
    axs[1].plot(t,e['truth'],color=ink,lw=2.5,label='LBM lift')
    for name,label,color in zip(names,titles,colors):
        axs[1].plot(t,e[name],color=color,lw=1.8,ls='--' if name=='pod32' else '-')
    axs[1].set(title='Re115: instantaneous lift, 32 target labels',xlabel='Lattice timestep',ylabel='Lift coefficient')
    axs[1].legend(frameon=False,loc='upper right');axs[1].grid(alpha=.2)
    fig.text(.07,.075,'Top: mean and range across 4 cases, after averaging 3 label draws.',fontsize=12,color=ink)
    fig.text(.07,.048,'Source labels for probe selection are additional to k. One encoder seed.',fontsize=12,color=ink)
    fig.text(.07,.021,'Fixed-rank POD baseline; coarse LBM data; instantaneous decoding, not forecasting.',fontsize=11,color=ink)
    fig.savefig(folder/'homepage_week07_4.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    main()
