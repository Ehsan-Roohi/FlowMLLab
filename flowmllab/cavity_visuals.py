"""Shared RGB field colors and velocity-derived streamlines for cavity plots."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
RGB = LinearSegmentedColormap.from_list("cavity_rgb", ["#1238db", "#00bf65", "#ed2024"], N=256)

def flow_structure(ax, x, y, u, v):
    """Show recirculation from the computed velocities, without inventing vortices."""
    ax.streamplot(x, y, u, v, density=1.8, color="#151515",
                  linewidth=.55, arrowsize=.7, minlength=.035, zorder=4)
    # Separate corner seeds help expose small resolved recirculation cells.
    xx, yy = np.meshgrid([.025,.05,.09,.14,.86,.91,.95,.975],
                         [.025,.05,.09,.14,.86,.91,.95,.975])
    seeds=np.column_stack([xx.ravel(),yy.ravel()])
    inside=(seeds[:,0]>=min(x))&(seeds[:,0]<=max(x))&(seeds[:,1]>=min(y))&(seeds[:,1]<=max(y))
    ax.streamplot(x,y,u,v,start_points=seeds[inside],density=2.2,
                  color="#151515",linewidth=.5,arrowsize=.65,minlength=.015,zorder=4)
