"""Replay the author's hydrofoil vapor-cloud vision model on retained CFD rasters.

The architecture and preprocessing match native_alpha20_v6 (2026-09-12).
Classes: 0 background, 1 attached cavity, 2 disconnected vapor in 2-D.
Native weak references use 255 for solid/uncertain support. They are never inputs
to neural inference. This is spatial segmentation, not a temporal forecast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.ReLU())


class ContextUNet(nn.Module):
    """Original 126,275-parameter encoder-decoder, including global context."""
    def __init__(self):
        super().__init__()
        width = [8, 16, 24, 32, 48]
        self.enc = nn.ModuleList([Block(a, b) for a, b in zip([3]+width[:-1], width)])
        self.dec = nn.ModuleList([Block(a+b, b) for a, b in
                                 [(48, 32), (32, 24), (24, 16), (16, 8)]])
        self.global_proj = nn.Conv2d(48, 48, 1)
        self.head = nn.Conv2d(8, 3, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.enc):
            x = encoder(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = x + self.global_proj(F.adaptive_avg_pool2d(x, 1))
        for decoder, skip in zip(self.dec, reversed(skips[:-1])):
            x = decoder(torch.cat([F.interpolate(x, size=skip.shape[-2:],
                mode='bilinear', align_corners=False), skip], 1))
        return self.head(x)


def inputs(alpha, wall):
    """Preserve the original raster/pixel-distance convention exactly."""
    alpha = np.asarray(alpha, dtype=np.float32).copy()
    wall = np.asarray(wall, dtype=bool)
    if alpha.shape != wall.shape or alpha.ndim != 2 or min(alpha.shape) < 16:
        raise ValueError('alpha and wall must be matching 2-D arrays, at least 16x16')
    if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError('Vapor fraction must be finite and within [0, 1]')
    alpha[wall] = 0
    distance = np.minimum(ndi.distance_transform_edt(~wall)/48., 1).astype('float32')
    return np.stack([alpha, wall.astype('float32'), distance]).astype('float32')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(root):
    """Verify public file hashes before loading numeric arrays (no pickle)."""
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    cases = {}
    for entry in manifest['cases']:
        with np.load(root/entry['file'], allow_pickle=False) as z:
            case = {name: z[name] for name in z.files}
        case.update(name=entry['name'], role=entry['role'])
        if case['alpha'].shape != case['reference'].shape:
            raise ValueError('Mismatched field/reference dimensions')
        if not np.isin(case['reference'], [0, 1, 2, 255]).all():
            raise ValueError('Unknown label')
        if not np.all(np.diff(case['times']) > 0):
            raise ValueError('Times must be increasing')
        cases[entry['name']] = case
    return manifest, cases


def load_model(path):
    model = ContextUNet()
    with np.load(path, allow_pickle=False) as z:
        model.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files}, strict=True)
    return model.eval()


def predict(model, alpha, wall):
    """Untouched neural argmax: no reference, connectivity repair or wall veto."""
    alpha = np.asarray(alpha)
    if alpha.ndim == 2:
        alpha = alpha[None]
    with torch.inference_mode():
        return np.stack([model(torch.from_numpy(inputs(a, wall)[None])).argmax(1)
                         [0].cpu().numpy().astype('uint8') for a in alpha])


def scores(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError('Prediction/reference shapes differ')
    valid = reference != 255
    result = {}
    for cls, name in [(1, 'attached_2d'), (2, 'disconnected_2d')]:
        p, q = (prediction == cls) & valid, (reference == cls) & valid
        tp, fp, fn = int((p&q).sum()), int((p&~q).sum()), int((~p&q).sum())
        denom = 2*tp+fp+fn
        result[name] = dict(dice=2*tp/denom if denom else None, tp=tp, fp=fp, fn=fn)
    return result


def raster_baseline(alpha, wall, threshold=.20):
    """Explicit classroom comparator; NOT the native-face research teacher.

    Four-connected raster support, one-cell wall contact, no size filtering.
    Raster adjacency can disagree with contact on the original Fluent mesh.
    """
    support = (alpha >= threshold) & ~wall
    structure = ndi.generate_binary_structure(2, 1)
    components, _ = ndi.label(support, structure=structure)
    contact = ndi.binary_dilation(wall, structure=structure) & ~wall
    attached_ids = np.unique(components[contact & support])
    labels = np.zeros(wall.shape, dtype='uint8')
    labels[support] = 2
    labels[support & np.isin(components, attached_ids)] = 1
    return labels


def comparison_figure(case, prediction, indices, heading=None):
    """Physical coordinates and common vapor colors for raw/reference/ML panels."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    x, y, wall = case['x'], case['y'], case['wall']
    extent = [x[0]-(x[1]-x[0])/2, x[-1]+(x[1]-x[0])/2,
              y[0]-(y[1]-y[0])/2, y[-1]+(y[1]-y[0])/2]
    fig, axes = plt.subplots(len(indices), 3, figsize=(14, 1.9*len(indices)+.7),
                             squeeze=False, layout='constrained')
    for row, i in enumerate(indices):
        for col, title in enumerate(['CFD vapor fraction', 'Native weak reference', 'Learned detection']):
            ax = axes[row, col]
            im = ax.imshow(case['alpha'][i], origin='lower', extent=extent,
                           cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
            ax.imshow(np.ma.masked_where(~wall, wall), origin='lower', extent=extent,
                      cmap=ListedColormap(['#d4dbe2']), vmin=0, vmax=1, interpolation='nearest')
            ax.contour(x, y, wall, levels=[.5], colors='#243746', linewidths=.8)
            if col:
                lab = case['reference'][i] if col == 1 else prediction[i]
                for cls, color in [(1, '#dc8d00'), (2, '#c32983')]:
                    if np.any(lab == cls):
                        ax.contour(x, y, lab == cls, levels=[.5], colors=color, linewidths=1.1)
                # Preserve the native uncertainty region, distinct from solid.
                if col == 1:
                    unknown = (lab == 255) & ~wall
                    ax.imshow(np.ma.masked_where(~unknown, unknown), origin='lower',
                              extent=extent, cmap=ListedColormap(['#999999']), alpha=.55,
                              vmin=0, vmax=1, interpolation='nearest')
            ax.set(title=f'{title} | t = {case["times"][i]:.2f} s', xlabel='x [m]', ylabel='y [m]')
        # Keep aspect ratio physical in every panel.
    fig.colorbar(im, ax=axes, shrink=.7, label='Vapor volume fraction')
    fig.suptitle(heading or f'{case["name"]} | {case["role"]}', fontsize=13)
    fig.legend(handles=[Line2D([0], [0], color='#dc8d00', label='Attached cavity'),
                        Line2D([0], [0], color='#c32983', label='Disconnected cloud (2-D)')],
               loc='outside lower center', ncol=2, frameon=False)
    return fig


def adapt_from_parent(parent, cases, protocol, steps=None):
    """Original final-stage training loop, using only the four declared TRAIN cases.

    Default: 1000 fixed updates from native_v4. A shorter budget is a classroom
    experiment, not reproduction of v6. This function writes no files and never
    mutates the supplied parent. Checkpoint selection is the fixed final step.
    """
    import copy
    torch.manual_seed(protocol['seed'])
    rng = np.random.default_rng(protocol['seed'])
    names = protocol['training_cases']
    training = [cases[n] for n in names]
    targets = []
    features = []
    for c in training:
        y = c['reference'].copy()
        y[:, c['wall']] = 0  # original explicit solid negative, ignored in evaluation
        targets.append(y)
        features.append(np.stack([inputs(a, c['wall']) for a in c['alpha']]))
    counts = sum(np.bincount(y[y != 255], minlength=3) for y in targets)
    weights = np.sqrt(counts.sum()/np.maximum(counts, 1))
    weights = np.clip(weights/weights[1], .1, 10)
    np.testing.assert_allclose(weights, protocol['class_weights'], rtol=1e-12)
    model = copy.deepcopy(parent).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    history, started = [], time.perf_counter()
    budget = protocol['steps'] if steps is None else int(steps)
    if budget <= 0:
        raise ValueError('steps must be positive')
    for step in range(1, budget+1):
        k = int(rng.integers(len(training)))
        positive = np.flatnonzero((targets[k] == 2).any((1, 2)))
        i = int(rng.choice(positive)) if len(positive) and rng.random() < .5 else int(rng.integers(len(targets[k])))
        x, y = features[k][i].copy(), targets[k][i].copy()
        if rng.random() < .5:
            x, y = x[:, ::-1].copy(), y[::-1].copy()
        tx, ty = torch.from_numpy(x[None]), torch.from_numpy(y[None].astype('int64'))
        logits = model(tx)
        loss = F.cross_entropy(logits, ty, weight=torch.tensor(weights, dtype=torch.float32), ignore_index=255)
        prob, valid = logits.softmax(1), ty != 255
        for cls in [1, 2]:
            p, target = prob[:, cls]*valid, (ty == cls).float()
            loss += .5*(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % 100 == 0 or step == budget:
            record = dict(step=step, loss=float(loss.detach()), seconds=time.perf_counter()-started)
            history.append(record)
            print(record)
    return model.eval(), history
