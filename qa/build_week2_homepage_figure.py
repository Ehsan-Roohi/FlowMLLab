"""Render the Week-2 notebook's synthetic response, not a DSMC result."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    log_kn = np.linspace(-4, 1.5, 600)
    kn = 10**log_kn
    fig, ax = plt.subplots(figsize=(10, 4.3), layout='constrained')
    bounds = [(-4, -3, 'Continuum'), (-3, -1, 'Slip'), (-1, 1, 'Transition'), (1, 1.5, 'Free molecular')]
    for i, (a, b, label) in enumerate(bounds):
        ax.axvspan(a, b, color=('#edf2f7', '#dceaf5')[i % 2], zorder=0)
        ax.text((a+b)/2, 1.02, label, ha='center', transform=ax.get_xaxis_transform(), fontsize=9)
    for alpha in (.6, .8, 1.):
        # Same noise-free synthetic_Q equation as the Week-2 laboratory.
        q = (1-1/3) * (1+1.25*((2-alpha)/alpha)*(kn/(1+kn))) * (1+.10*np.tanh(log_kn+1))
        ax.plot(log_kn, q, lw=2.5, label=fr'$\alpha={alpha}$')
    ax.set(xlabel=r'$\log_{10}(Kn)$', ylabel=r'Synthetic $Q^*$', xlim=(-4, 1.5))
    ax.legend(title='Accommodation', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Week 2 | Features, rarefaction and model validity', fontsize=16, fontweight='bold')
    fig.supxlabel('Notebook teaching equation at pressure ratio 3 and temperature ratio 1; synthetic data, not CFD/DSMC.', fontsize=9)
    output = root/'assets/week02_synthetic_response.png'
    fig.savefig(output, dpi=250)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
