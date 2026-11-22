"""Original alpha-input and pressure-input cavitation methods, frozen replay.

Pressure interfaces deliberately do not accept vapor fraction or reference masks.
Pressure deficit means stored gauge pressure + operating pressure - configured
vapor pressure, in Pa. These spatial models do not implement phase transport.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .cavitation_detection import ContextUNet, digest, predict


def pressure_patch_model():
    return nn.Sequential(nn.Linear(45,48), nn.Tanh(), nn.Linear(48,32), nn.Tanh(), nn.Linear(32,1))


class PressureBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci,co,3,padding=1), nn.GroupNorm(1,co), nn.SiLU(),
                                 nn.Conv2d(co,co,3,padding=1), nn.GroupNorm(1,co), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class PressureUNet(nn.Module):
    def __init__(self, topology=False):
        super().__init__()
        self.topology = topology
        self.e1 = PressureBlock(5,8); self.e2 = PressureBlock(8,16); self.mid = PressureBlock(16,32)
        self.u2 = nn.ConvTranspose2d(32,16,2,2); self.d2 = PressureBlock(32,16)
        self.u1 = nn.ConvTranspose2d(16,8,2,2); self.d1 = PressureBlock(16,8)
        if topology:
            self.alpha_head = nn.Conv2d(8,1,1)
            self.topology_head = nn.Conv2d(8,3,1)
        else:
            self.head = nn.Conv2d(8,1,1)

    def forward(self, x):
        a = self.e1(x); b = self.e2(F.max_pool2d(a,2)); z = self.mid(F.max_pool2d(b,2))
        z = self.d2(torch.cat([self.u2(z),b],1))
        z = self.d1(torch.cat([self.u1(z),a],1))
        return (self.alpha_head(z), self.topology_head(z)) if self.topology else self.head(z)


def load_comparison(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(root/name) != expected:
            raise ValueError('Checksum mismatch: '+name)
    frames = []
    for item in manifest['frames']:
        with np.load(root/item['file'], allow_pickle=False) as z:
            frame = {k:z[k] for k in z.files}
        frame.update(key=item['key'], case=item['case'])
        frames.append(frame)
    return manifest, frames


def load_method(root, method, seed=11):
    if method == 'alpha_topology':
        model = ContextUNet()
        name = 'alpha_topology.npz'
    elif method == 'pressure_patch':
        model = pressure_patch_model(); name = f'{method}_{seed}.npz'
    elif method in ('pressure_unet','pressure_topology'):
        model = PressureUNet(topology=method == 'pressure_topology'); name = f'{method}_{seed}.npz'
    else:
        raise ValueError(method)
    with np.load(Path(root)/name, allow_pickle=False) as z:
        model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files}, strict=True)
    return model.eval()


def patch(a):
    return np.lib.stride_tricks.sliding_window_view(np.pad(a,1,mode='edge'),(3,3)).reshape(-1,9).copy()


def pressure_predict(model, method, pressure_deficit, wall):
    """Predict from pressure and geometry only; never accepts alpha or labels."""
    p = np.asarray(pressure_deficit, dtype='float32')
    wall = np.asarray(wall, dtype=bool)
    if p.shape != wall.shape or p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('Pressure/geometry arrays must match and be finite')
    p = p.copy(); p[wall] = 0
    with torch.inference_mode():
        if method == 'pressure_patch':
            q = torch.from_numpy(patch(p))
            w = torch.from_numpy(patch(wall.astype('float32')))
            x = torch.cat([torch.asinh(q/s) for s in [10,100,1000,10000]]+[w],1)
            value = torch.cat([model(x[i:i+8192]).sigmoid() for i in range(0,len(x),8192)])
            return {'alpha':value[:,0].numpy().reshape(p.shape)}
        channels = np.concatenate([np.arcsinh(p[None]/s) for s in [10,100,1000,10000]]+
                                  [wall[None].astype('float32')],0).astype('float32')
        out = model(torch.from_numpy(channels[None]))
        if method == 'pressure_topology':
            return {'alpha':out[0].sigmoid()[0,0].numpy(),
                    'topology':out[1].argmax(1)[0].numpy().astype('uint8')}
        if method != 'pressure_unet':
            raise ValueError(method)
        return {'alpha':out.sigmoid()[0,0].numpy()}


def binary_score(prediction, reference, valid):
    p, q = np.asarray(prediction,dtype=bool)&valid, np.asarray(reference,dtype=bool)&valid
    tp, fp, fn = int((p&q).sum()), int((p&~q).sum()), int((~p&q).sum())
    den = 2*tp+fp+fn
    return dict(tp=tp, fp=fp, fn=fn, dice=2*tp/den if den else None)


def infer_comparison(root, frames, seed=11):
    """Return all five methods for the same frames; labels enter only evaluation."""
    alpha_model = load_method(root,'alpha_topology')
    models = {name:load_method(root,name,seed) for name in
              ['pressure_patch','pressure_unet','pressure_topology']}
    outputs = []
    for frame in frames:
        out = {'alpha_topology':predict(alpha_model,frame['alpha'],frame['wall'])[0],
               'pressure_threshold':frame['pressure_deficit'] < 0}
        for name, model in models.items():
            value = pressure_predict(model,name,frame['pressure_deficit'],frame['wall'])
            out[name] = value['topology'] if name == 'pressure_topology' else value['alpha']
            if name == 'pressure_topology':
                out['pressure_topology_alpha'] = value['alpha']
        outputs.append(out)
    return outputs


def method_mask(name, value):
    if name in ('alpha_topology','pressure_topology'):
        return (value == 1)|(value == 2)
    if name == 'pressure_threshold':
        return value.astype(bool)
    return value >= .20


def methods_figure(frame, output):
    """Six-panel same-frame comparison; color separates topology from total vapor."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    import matplotlib.patheffects as pe
    outline_effects = [pe.Stroke(linewidth=4.2, foreground='white'), pe.Normal()]
    total_color = '#00843d'
    x,y,wall = frame['x'],frame['y'],frame['wall']
    extent = [x[0]-(x[1]-x[0])/2,x[-1]+(x[1]-x[0])/2,
              y[0]-(y[1]-y[0])/2,y[-1]+(y[1]-y[0])/2]
    fig,axes = plt.subplots(2,3,figsize=(15,6.3),layout='constrained')
    panels = [('CFD vapor fraction + native reference',None),
              ('Alpha-input context U-Net','alpha_topology'),
              ('Pressure threshold: p_abs < p_v','pressure_threshold'),
              ('Pressure-only 3x3 model','pressure_patch'),
              ('Pressure-only U-Net','pressure_unet'),
              ('Pressure-only topology U-Net','pressure_topology')]
    valid = (frame['reference'] != 255)&~wall
    for ax,(title,name) in zip(axes.flat,panels):
        im = ax.imshow(frame['alpha'],origin='lower',extent=extent,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
        ax.imshow(np.ma.masked_where(~wall,wall),origin='lower',extent=extent,
                  cmap=ListedColormap(['#d4dbe2']),vmin=0,vmax=1,interpolation='nearest')
        ax.contour(x,y,wall,levels=[.5],colors='#243746',linewidths=.8)
        if name is None or name in ('alpha_topology','pressure_topology'):
            labels = frame['reference'] if name is None else output[name]
            for cls,color in [(1,'#dc8d00'),(2,'#c32983')]:
                if (labels==cls).any():
                    contour = ax.contour(x,y,labels==cls,levels=[.5],colors=color,linewidths=2.2)
                    contour.set_path_effects(outline_effects)
        else:
            mask = method_mask(name,output[name])
            if mask.any():
                contour = ax.contour(x,y,mask,levels=[.5],colors=total_color,linewidths=2.2)
                contour.set_path_effects(outline_effects)
        if name:
            score = binary_score(method_mask(name,output[name]),frame['alpha']>=.2,valid)['dice']
            title += '\nFluid-only cavity Dice = '+('N/A (both empty)' if score is None else f'{score:.3f}')
            solid_fp = method_mask(name,output[name])&wall
            if solid_fp.any():
                yy,xx = np.nonzero(solid_fp)
                ax.annotate(f'False vapor in solid\n{int(solid_fp.sum())} pixels',
                            xy=(float(x[xx].mean()),float(y[yy].mean())),
                            xytext=(.73,.50),textcoords='axes fraction',
                            fontsize=9,color='#9b3016',ha='center',va='center',
                            bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor='#9b3016',alpha=.96),
                            arrowprops=dict(arrowstyle='->',color='#9b3016',lw=1.3),
                            zorder=10)
        else:
            title += '\nAlgorithmic alpha >= 0.20 reference'
        ax.set(title=title,xlabel='x [m]',ylabel='y [m]')
    fig.colorbar(im,ax=axes,shrink=.7,label='CFD vapor fraction (same background in every panel)')
    fig.suptitle(f'Hydrofoil vapor-cloud detection | {frame["case"]}, t = {float(frame["time"]):.4f} s\n'
                 'Same field and time | frozen original models | pressure-model seed 11',fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color='#dc8d00',lw=2.2,label='Attached'),
                        Line2D([0],[0],color='#c32983',lw=2.2,label='Disconnected cloud (2-D)'),
                        Line2D([0],[0],color=total_color,lw=2.2,label='Total cavity only (no topology output)')],
               loc='outside lower center',ncol=3,frameon=False)
    return fig
