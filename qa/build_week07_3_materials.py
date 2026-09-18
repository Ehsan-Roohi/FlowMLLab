#!/usr/bin/env python3
"""Build the Week 7.3 notebook (masked pretraining and label efficiency) and its lecture PDF.

Usage:
    python qa/build_week07_3_materials.py                 # write the notebook text and the PDF
    python qa/build_week07_3_materials.py --execute       # also execute the notebook in place and
                                                          # refresh results/week07_3_pretraining/
    python qa/build_week07_3_materials.py --notebook-only # notebook text only
    python qa/build_week07_3_materials.py --pdf-only      # lecture PDF only (reads retained metrics)

The notebook is the single source of the retained evidence: executed with
FLOWMLLAB_WRITE_RESULTS=1 (which --execute sets) it writes metrics.json and the
figures into results/week07_3_pretraining/.  Without that variable a Run All
writes into a fresh temporary folder and compares its numbers with the retained
ones, so students can never modify the tracked evidence by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))
from add_colab_entrypoints import badge, bootstrap  # noqa: E402

NOTEBOOK = ROOT / "notebooks/week07_3/W7_3_Masked_Pretraining_Label_Efficiency.ipynb"
RESULTS = ROOT / "results/week07_3_pretraining"
SOURCE = ROOT / "lectures/source/week07_3_masked_pretraining.md"
PDF = ROOT / "lectures/week07_3_masked_pretraining.pdf"


def cell(kind: str, source: str) -> dict:
    source = source.strip("\n") + "\n"
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


# --------------------------------------------------------------------------- notebook text
TITLE = r'''# Week 7.3 - Self-supervised pretraining for label-efficient wake reconstruction
<!-- MIE690A article-aligned validation v4 -->

Scientific question: if we own many *unlabelled* wake fields and only a handful of
*labelled* ones, does pretraining a network on the unlabelled fields reduce the
number of labels needed for a new trajectory, and does it beat a classical method
that receives exactly the same information?

The protocol follows the recipe of MAPA (Tang, Spalding and Cogan, 2026: masked
autoencoding on intracranial recordings, then a label-efficiency curve for a new
subject) on the FlowMLLab lattice-Boltzmann cylinder wake: pretrain a masked
autoencoder (He et al., 2022) on two unlabelled trajectories, then measure error
against the number of labelled frames of a third trajectory. Gappy POD
(Everson and Sirovich, 1995) is the matched classical baseline.

Prerequisites: Weeks 5 and 7 (POD, the wake data) and Week 7.2 (validation-only
selection). Allow 90 minutes. CPU only; the long cells are the pretraining
(about 1 minute), the label-efficiency loop (about 4 minutes) and the budget
control (about 3 minutes) on a two-core machine; `QUICK = True` in the setup cell
cuts the total to about 3 minutes. Lecture: `lectures/week07_3_masked_pretraining.pdf`.

What this is not: the four trajectories are educational LBM runs at nearly the
same Reynolds number (90 to 110) on a 32 x 78 window, not grid-independent CFD,
and the target trajectory (Re = 110) was inspected in Weeks 5, 7 and 7.2. It is a
protocol demonstration with retained numbers, not a blind benchmark.'''

SETUP = r'''from pathlib import Path
import copy, hashlib, json, os, platform, sys, tempfile, time
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'flowmllab/masked_pretraining.py').is_file())
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from flowmllab import masked_pretraining as mp
from flowmllab.cylinder_ml import fit_pod

torch.set_num_threads(min(2, os.cpu_count() or 1))
QUICK = False            # True: one mask seed and four k values (about a quarter of the runtime)
WRITE_RESULTS = bool(os.environ.get('FLOWMLLAB_WRITE_RESULTS'))
RETAINED = ROOT / 'results/week07_3_pretraining'
OUT = RETAINED if WRITE_RESULTS else Path(tempfile.mkdtemp(prefix='flowmllab-week73-'))
OUT.mkdir(parents=True, exist_ok=True)
tracked = sorted(p for p in (ROOT / 'data/modal_labs').rglob('*.npz'))
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}

fields = {re: mp.load_wake_field(ROOT, re) for re in mp.TRAJECTORIES}
layout = mp.PatchLayout()
cfg = mp.ProtocolConfig()
if QUICK:
    cfg = mp.ProtocolConfig(ks=(1, 4, 16, 64), seeds=(0,))
print({re: f.shape for re, f in fields.items()}, 'field =', mp.FIELD)
print('patch grid', layout.grid, '->', layout.n_patches, 'patches of', layout.py, 'x', layout.px, 'pixels')
print('outputs go to', OUT.name if not WRITE_RESULTS else RETAINED.relative_to(ROOT))'''

DEFINITIONS = r'''## 1. Definitions before use

- **Label.** Here a label is a *complete* frame of the target trajectory: the full
  32 x 78 transverse-velocity field. A method that is "given k labelled frames"
  may read k complete frames of Re = 110 and nothing else from that trajectory.
- **Unlabelled data.** Complete frames of *other* trajectories (Re = 90 and 100).
  They are unlabelled with respect to the target: nothing about Re = 110 is in them.
  Self-supervised methods manufacture their own training signal from such data.
- **Patch.** A 4 x 6 block of pixels. The 32 x 78 window is an 8 x 13 grid of 104 patches.
- **Masking.** Hiding a random 75% of the patches of a frame (78 of 104). The
  *visible* patches are the input; the hidden ones are the prediction target.
- **Masked autoencoder (MAE).** A transformer *encoder* that reads only the
  visible patches and a small *decoder* that fills in the hidden ones. The
  training loss is the mean squared error on the hidden patches only.
- **Self-supervised pretraining.** Training the MAE on unlabelled frames with
  random masks. No label is used; the frames supervise themselves.
- **Zero-shot.** Applying the pretrained model to the target trajectory without
  any labelled frame of it.
- **Linear probe.** Freezing the pretrained encoder and fitting only a linear
  (ridge) map from its features to the full frame, using the k labelled frames.
- **Fine-tuning.** Continuing to train *all* weights of the pretrained model on
  the k labelled frames.
- **From scratch.** The same network, randomly initialised, trained on the k
  labelled frames only, with the same optimiser budget as fine-tuning.
- **Gappy POD.** The classical answer to the same completion task: fit a POD
  basis to complete frames, then for a masked frame solve a least-squares
  problem for the modal coefficients using the visible pixels only, and
  reconstruct the hidden pixels from the basis (Everson and Sirovich, 1995).
- **Matched baseline.** A baseline that sees exactly the same frames and the same
  visible pixels as the neural method. Without it, an improvement cannot be
  attributed to the method rather than to the information it received.
- **Label-efficiency curve.** Test error against k, the number of labelled frames.
  A pretrained method is more label-efficient if it reaches a given error with
  fewer labels. The *label saving* at a given error is the ratio of the labels
  the reference method needs to the labels the pretrained method needs.
- **Relative L2 error.** 100 x ||prediction - truth|| / ||truth||. We report it over
  the hidden pixels only (the quantity actually predicted); every method copies the
  visible pixels into its output first, so the full-frame error is always smaller.'''

CONTRACT = r'''## 2. Data and information contract

Four retained trajectories of the transverse velocity `v/U` on the 32 x 78 wake
window (`data/modal_labs/re*.npz`, 281 frames each, about 52 frames per shedding
period, hash-checked at the end of the notebook). Their roles are frozen before
any number is looked at:

| Trajectory | Role | Who may read it |
| --- | --- | --- |
| Re = 90, Re = 100 | unlabelled pretraining data | MAE pretraining; the transferred and pooled POD bases |
| Re = 105 | validation | every selection: POD rank, ridge strength, early stopping |
| Re = 110, frames `[0, 160)` | labelled pool | the k frames drawn per seed |
| Re = 110, frames `[210, 281)` | test | scoring only, with fixed random masks |

Frames `[160, 210)` of Re = 110 separate the pool from the test window, as in Week
7.2. The test masks hide 75% of the patches of every test frame; three mask
seeds also change which k frames are drawn from the pool. Selecting anything on
the test window is forbidden, so every hyperparameter is chosen on Re = 105.

Below: one Re = 110 test frame, the patch grid and a 75% mask. The 26 visible
patches are all that any method receives about that frame.'''

MASK_FIGURE = r'''test_frames = fields[110][slice(*cfg.test_window)]
demo_visible = mp.random_patch_masks(1, layout.n_patches, cfg.mask_ratio, seed=2000)
demo_pixels = layout.pixel_mask(demo_visible)[0]
frame = test_frames[0]
vmax = float(np.abs(frame).max())
fig, axes = plt.subplots(1, 3, figsize=(15, 3.1), constrained_layout=True)
axes[0].imshow(frame, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
axes[0].set_title('test frame 210 of Re = 110: v/U')
axes[1].imshow(demo_visible.reshape(layout.grid), origin='lower', cmap='Greys_r', vmin=0, vmax=1, aspect='equal')
axes[1].set_title(f'{int(demo_visible.sum())} visible (white) of {layout.n_patches} patches')
axes[2].imshow(np.where(demo_pixels, frame, np.nan), origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
axes[2].set_title('what every method receives')
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(OUT / 'masking_example.png', dpi=150)
plt.show()
print('visible pixel fraction:', demo_pixels.mean())'''

GAPPY_TEXT = r'''## 3. The classical baseline: gappy POD

Write a complete frame as a vector $q \in \mathbb{R}^{2496}$ and a POD basis fitted
on complete frames as the mean $\bar q$ and modes $\Phi \in \mathbb{R}^{2496 \times r}$.
For a masked frame let $V$ be the index set of visible pixels. Gappy POD solves

$$\min_a \|\Phi_V a - (q_V - \bar q_V)\|^2 + \epsilon\|a\|^2, \qquad \hat q = \bar q + \Phi a,$$

with $\Phi_V$ the visible rows of the modes and a tiny ridge $\epsilon$ for
numerical safety. With 624 visible pixels and $r \le 64$ the system is
overdetermined ten times over, so the fit is a least-squares projection, not an
interpolation. The rank $r$ is the only choice and it is selected on Re = 105:
larger ranks represent more of the field but make the visible-pixel system
worse conditioned.

Two bases are used: the **transferred** basis from the unlabelled Re = 90 and
100 frames (zero labels of the target), and later a **target-only** basis from
the k labelled frames (rank at most k - 1) and a **pooled** basis from the
unlabelled frames plus the k labelled ones, the classical analogue of fine-tuning.'''

GAPPY_CODE = r'''unlabelled = np.concatenate([fields[90], fields[100]])
validation = fields[105]
val_visible = mp.random_patch_masks(len(validation), layout.n_patches, cfg.mask_ratio, seed=1000)
val_pixels = layout.pixel_mask(val_visible)
val_patches = layout.patchify(validation)

t0 = time.perf_counter()
rank_transfer, rank_scores = mp.select_pod_rank(unlabelled, validation, val_pixels, cfg.rank_candidates)
basis_transfer = mp.fit_gappy_basis(unlabelled, rank_transfer)
print(f'transferred basis: rank {rank_transfer} selected on Re = 105 in {time.perf_counter() - t0:.1f} s')
display(pd.Series(rank_scores, name='Re105 masked error (%)').rename_axis('rank').to_frame().T.round(3))

test_masks = {seed: mp.random_patch_masks(len(test_frames), layout.n_patches, cfg.mask_ratio, seed=2000 + seed)
              for seed in cfg.seeds}
zero_shot = []
for seed, visible in test_masks.items():
    pixels = layout.pixel_mask(visible)
    prediction = mp.gappy_reconstruct(basis_transfer, test_frames, pixels)
    zero_shot.append({'method': 'pod_transfer', 'k': 0, 'seed': seed, **mp.score(prediction, test_frames, pixels)})
print('transferred gappy POD, zero-shot on the Re = 110 test window (masked-pixel relative L2, %):',
      [round(r['masked'], 2) for r in zero_shot])'''

MAE_TEXT = r'''## 4. The masked autoencoder

Each visible patch (24 pixels) is embedded into a 64-dimensional token and given
a learned position embedding. A two-layer transformer encoder (4 heads, GELU,
pre-norm) attends over the 26 visible tokens only; it never sees a hidden patch,
so it cannot learn to copy. A one-layer decoder of width 32 receives the encoded
visible tokens plus a shared learned *mask token* at every hidden position, each
with its own position embedding, and a linear head predicts the 24 pixels of
every patch. The loss is the mean squared error over hidden patches only. The
model has about 90 thousand parameters.

Training uses AdamW (weight decay 0.05), a cosine learning-rate schedule from
2e-3, batches of 8 frames and a *fresh random mask for every frame at every
step*, so the 562 unlabelled frames yield an unlimited supply of training
problems. The masked loss on Re = 105 (with fixed masks) is evaluated once per
pass and the weights with the lowest validation loss are kept (early stopping).

Watch the loss curve: for the first few hundred steps the network predicts the
mean field (loss equal to the variance of `v`, the dashed line), then it breaks
through. A run stopped inside the plateau would look like a failed method.'''

MAE_CODE = r'''PRETRAIN_STEPS = 5000 if not QUICK else 2500
mae = mp.build_mae(layout, dim=64, depth=2, heads=4, decoder_dim=32, decoder_depth=1, seed=0)
n_parameters = sum(p.numel() for p in mae.parameters())
with torch.no_grad():
    probe_out = mae(torch.as_tensor(layout.patchify(test_frames[:2]), dtype=torch.float32),
                    torch.as_tensor(test_masks[cfg.seeds[0]][:2]))
assert probe_out.shape == (2, layout.n_patches, layout.patch_dim)
print(f'MAE parameters: {n_parameters:,}')

t0 = time.perf_counter()
history, best_step = mp.train_masked_model(mae, layout.patchify(unlabelled), layout, mask_ratio=cfg.mask_ratio,
                                           steps=PRETRAIN_STEPS, batch_size=8, lr=2e-3, seed=0,
                                           val_patches=val_patches, val_visible=val_visible, log_every=10)
pretrain_seconds = time.perf_counter() - t0
pretrained_state = copy.deepcopy(mae.state_dict())
print(f'pretraining: {PRETRAIN_STEPS} steps in {pretrain_seconds:.0f} s; weights restored from step {best_step}')

variance = float(np.var(unlabelled))
steps = [h['step'] for h in history]
fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
ax.semilogy(steps, [h['train_loss'] for h in history], label='train (Re = 90, 100; fresh masks)')   # NaN at step 0
ax.semilogy(steps, [h['val_loss'] for h in history], label='validation (Re = 105; fixed masks)')
ax.axhline(variance, color='k', ls='--', lw=1, label='predict the mean field')
ax.axvline(best_step, color='grey', lw=1, label=f'restored step {best_step}')
ax.set(xlabel='optimiser step', ylabel='masked-patch MSE', title='MAE pretraining on unlabelled frames')
ax.legend(fontsize=8)
fig.savefig(OUT / 'pretraining_loss.png', dpi=150)
plt.show()'''

ZERO_SHOT_TEXT = r'''## 5. Zero-shot reconstruction of the new trajectory

Neither the pretrained MAE nor the transferred POD basis has seen a single frame
of Re = 110. Both receive the same 26 visible patches of every test frame. The
first test frame (seed 0) is shown with both reconstructions and their absolute
error maps on a common scale; the table gives the masked-pixel error over all
71 test frames and the three mask seeds.'''

ZERO_SHOT_CODE = r'''for seed, visible in test_masks.items():
    prediction = mp.predict_masked(mae, layout.patchify(test_frames), visible, layout)
    zero_shot.append({'method': 'mae_zero_shot', 'k': 0, 'seed': seed, **mp.score(prediction, test_frames, layout.pixel_mask(visible))})
zero_table = pd.DataFrame(zero_shot).groupby('method')['masked'].agg(['mean', 'std']).rename(
    columns={'mean': 'masked error mean (%)', 'std': 'sample SD (pp)'})
display(zero_table)

visible0 = test_masks[cfg.seeds[0]]
pixels0 = layout.pixel_mask(visible0)
pod0 = mp.fill_visible(mp.gappy_reconstruct(basis_transfer, test_frames, pixels0), test_frames, pixels0)
mae0 = mp.predict_masked(mae, layout.patchify(test_frames), visible0, layout)
i = 0
vmax = float(np.abs(test_frames[i]).max())
emax = float(max(np.abs(pod0[i] - test_frames[i]).max(), np.abs(mae0[i] - test_frames[i]).max()))
fig, axes = plt.subplots(2, 3, figsize=(15, 5), constrained_layout=True)
panels = [(test_frames[i], 'truth', vmax), (np.where(pixels0[i], test_frames[i], np.nan), 'visible input (25%)', vmax),
          (None, None, None),
          (pod0[i], f'gappy POD, transferred basis (rank {rank_transfer})', vmax), (mae0[i], 'MAE zero-shot', vmax)]
for ax, (array, title, limit) in zip(axes.ravel()[:5], panels):
    if array is None:
        ax.axis('off'); continue
    ax.imshow(array, origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='equal'); ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 2].axis('on')
axes[0, 2].imshow(np.abs(pod0[i] - test_frames[i]), origin='lower', cmap='magma', vmin=0, vmax=emax, aspect='equal')
axes[0, 2].set_title(f'|POD - truth|, masked error {mp.score(pod0, test_frames, pixels0)["masked"]:.2f}%')
axes[1, 2].imshow(np.abs(mae0[i] - test_frames[i]), origin='lower', cmap='magma', vmin=0, vmax=emax, aspect='equal')
axes[1, 2].set_title(f'|MAE - truth|, masked error {mp.score(mae0, test_frames, pixels0)["masked"]:.2f}%')
for ax in (axes[0, 2], axes[1, 2]):
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Zero-shot completion of test frame 210 (Re = 110), mask seed 0; error panels share one scale')
fig.savefig(OUT / 'zero_shot_fields.png', dpi=150)
plt.show()'''

PROTOCOL_TEXT = r'''## 6. Using k labelled frames: five methods, one budget

For each mask seed and each k in {1, 2, 4, 8, 16, 32, 64, 128}, k complete frames
are drawn from the labelled pool `[0, 160)` of Re = 110 (evenly spread over the
pool for k <= 8, random otherwise) and every method is refitted:

| Method | Uses unlabelled frames | Uses the k labels | What is fitted |
| --- | --- | --- | --- |
| `pod_target` | no | yes | POD basis of the k frames (rank <= k - 1, chosen on Re = 105; k = 1 gives the mean field) |
| `pod_pooled` | yes | yes | POD basis of the 562 unlabelled frames plus the k frames (rank chosen on Re = 105) |
| `mae_probe` | yes (pretrained) | yes | ridge readout from the visible-token-pooled mean of visible encoder tokens to the full frame; 8 random masks per labelled frame; ridge strength chosen on Re = 105 |
| `mae_scratch` | no | yes | the MAE architecture from random weights, 300 AdamW steps on the k frames with fresh masks, early stopping on Re = 105 |
| `mae_finetune` | yes (pretrained) | yes | the pretrained MAE, the same 300 steps at a lower learning rate (5e-4), early stopping on Re = 105 |

`mae_scratch` and `mae_finetune` receive identical labelled data and identical
downstream compute; the only difference is the starting point. A separate
control at the end gives the from-scratch model the full pretraining budget on
the 128 labelled frames, to answer the objection that it merely needed longer.

Errors are averaged over the 71 test frames of each seed; the curve shows the
mean and sample SD over the three seeds. The seeds change the masks and the
drawn frames of one CFD trajectory; they are not independent flows.'''

PROTOCOL_CODE = r'''pool = np.arange(*cfg.label_pool)
records = list(zero_shot)
timings = {}
T0 = time.perf_counter()
for seed in cfg.seeds:
    visible = test_masks[seed]
    pixels = layout.pixel_mask(visible)
    test_patches = layout.patchify(test_frames)
    test_features = mp.encoder_features(mae, test_patches, visible, pooling='visible_mean')
    for k in cfg.ks:
        labelled = fields[110][mp.sample_labelled_frames(pool, k, seed=100 * seed + k)]
        t = time.perf_counter()
        rank, _ = mp.select_pod_rank(labelled, validation, val_pixels, cfg.rank_candidates)
        prediction = mp.gappy_reconstruct(mp.fit_gappy_basis(labelled, rank), test_frames, pixels)
        records.append({'method': 'pod_target', 'k': k, 'seed': seed, 'rank': rank, **mp.score(prediction, test_frames, pixels)})
        timings.setdefault('pod_target', []).append(time.perf_counter() - t)

        t = time.perf_counter()
        pooled = np.concatenate([unlabelled, labelled])
        rank, _ = mp.select_pod_rank(pooled, validation, val_pixels, cfg.rank_candidates)
        prediction = mp.gappy_reconstruct(mp.fit_gappy_basis(pooled, rank), test_frames, pixels)
        records.append({'method': 'pod_pooled', 'k': k, 'seed': seed, 'rank': rank, **mp.score(prediction, test_frames, pixels)})
        timings.setdefault('pod_pooled', []).append(time.perf_counter() - t)

        t = time.perf_counter()
        readout, alpha, _ = mp.fit_ridge_probe(mae, labelled, layout, mask_ratio=cfg.mask_ratio,
                                               copies=cfg.masks_per_labelled_frame, seed=300 + seed,
                                               val_frames=validation, val_visible=val_visible, alphas=cfg.alpha_candidates)
        prediction = readout.predict(test_features).reshape(test_frames.shape)
        records.append({'method': 'mae_probe', 'k': k, 'seed': seed, 'alpha': alpha, **mp.score(prediction, test_frames, pixels)})
        timings.setdefault('mae_probe', []).append(time.perf_counter() - t)

        t = time.perf_counter()
        scratch = mp.build_mae(layout, seed=seed)
        _, step = mp.train_downstream(scratch, labelled, layout, cfg, lr=cfg.scratch_lr, seed=400 + seed,
                                      val_patches=val_patches, val_visible=val_visible)
        prediction = mp.predict_masked(scratch, test_patches, visible, layout)
        records.append({'method': 'mae_scratch', 'k': k, 'seed': seed, 'best_step': step, **mp.score(prediction, test_frames, pixels)})
        timings.setdefault('mae_scratch', []).append(time.perf_counter() - t)

        t = time.perf_counter()
        tuned = mp.build_mae(layout, seed=seed)
        tuned.load_state_dict(pretrained_state)
        _, step = mp.train_downstream(tuned, labelled, layout, cfg, lr=cfg.finetune_lr, seed=500 + seed,
                                      val_patches=val_patches, val_visible=val_visible)
        prediction = mp.predict_masked(tuned, test_patches, visible, layout)
        records.append({'method': 'mae_finetune', 'k': k, 'seed': seed, 'best_step': step, **mp.score(prediction, test_frames, pixels)})
        timings.setdefault('mae_finetune', []).append(time.perf_counter() - t)
        print(f'seed {seed}  k {k:3d}  ' + '  '.join(f"{r['method']} {r['masked']:6.2f}%" for r in records[-5:])
              + f'   {time.perf_counter() - T0:5.0f} s', flush=True)
protocol_seconds = time.perf_counter() - T0
print({m: f'{sum(v):.0f} s' for m, v in timings.items()})'''

CURVE_TEXT = r'''## 7. The label-efficiency curve

The figure is the analogue of MAPA's main result: masked-pixel error against the
number of labelled frames, one curve per method, with the two zero-shot methods
as horizontal lines. Read it in three steps: (1) compare `mae_finetune` with
`mae_scratch` (does pretraining help the *network*?); (2) compare `mae_finetune`
with `pod_pooled` (does the network beat the classical method with the same
information?); (3) find the label saving: how many labels does the from-scratch
model need to match what the pretrained one does with its smallest k.'''

CURVE_CODE = r'''curves = mp.summarize_curves([r for r in records if r['k'] > 0])
zero = {m: float(np.mean([r['masked'] for r in records if r['method'] == m])) for m in ('pod_transfer', 'mae_zero_shot')}
table = pd.DataFrame({m: {k: f"{v['mean']:.2f} +/- {v['std']:.2f}" for k, v in c.items()} for m, c in curves.items()})
table.index.name = 'k labelled frames'
display(table)

labels = {'pod_target': 'gappy POD, target-only basis', 'pod_pooled': 'gappy POD, pooled basis',
          'mae_probe': 'MAE frozen encoder + ridge probe', 'mae_scratch': 'MAE from scratch (300 steps)',
          'mae_finetune': 'MAE pretrained + fine-tuned (300 steps)'}
styles = {'pod_target': ('tab:blue', 's', '--'), 'pod_pooled': ('tab:blue', 'o', '-'), 'mae_probe': ('tab:orange', '^', ':'),
          'mae_scratch': ('tab:red', 'x', '--'), 'mae_finetune': ('tab:red', 'o', '-')}
fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
for method, curve in curves.items():
    ks = sorted(curve)
    color, marker, ls = styles[method]
    means = np.array([curve[k]['mean'] for k in ks])
    lower = means - np.array([curve[k]['min'] for k in ks])
    upper = np.array([curve[k]['max'] for k in ks]) - means
    ax.errorbar(ks, means, yerr=[lower, upper], color=color, marker=marker, ls=ls, capsize=3, label=labels[method])
ax.axhline(zero['pod_transfer'], color='tab:blue', lw=1, ls='-.', label=f"gappy POD, transferred basis, zero-shot ({zero['pod_transfer']:.2f}%)")
ax.axhline(zero['mae_zero_shot'], color='tab:red', lw=1, ls='-.', label=f"MAE zero-shot ({zero['mae_zero_shot']:.2f}%)")
ax.set(xscale='log', yscale='log', xlabel='k labelled frames of Re = 110', ylabel='masked-pixel relative L2 error (%)',
       title='Label efficiency on the Re = 110 test window (mean of 3 mask seeds; bars span the seeds)')
ax.set_xticks(list(cfg.ks)); ax.set_xticklabels([str(k) for k in cfg.ks])
ax.grid(True, which='both', alpha=.3); ax.legend(fontsize=8, loc='lower left')
fig.savefig(OUT / 'label_efficiency.png', dpi=150)
plt.show()

mean_curve = lambda m: {k: v['mean'] for k, v in curves[m].items()}
comparisons = [('mae_finetune', 'mae_scratch'), ('mae_finetune', 'pod_target'), ('pod_pooled', 'pod_target')]
savings = {f'{pretrained}_vs_{reference}': mp.label_saving(mean_curve(reference), mean_curve(pretrained))
           for pretrained, reference in comparisons}
saving_rows = []
for pretrained, reference in comparisons:
    s = savings[f'{pretrained}_vs_{reference}']
    for k, match in s['matches'].items():
        saving_rows.append({'comparison': f'{pretrained} vs {reference}', 'k (pretrained)': k,
                            'pretrained error (%)': round(curves[pretrained][k]['mean'], 2),
                            'reference k needed': match if match is not None else f'> {max(cfg.ks)}',
                            'label saving': f'{s["saving"][k]:.0f}x' if s['saving'][k] is not None else 'not reached'})
print('Label saving: the smallest k at which the reference method reaches the pretrained method\'s error')
display(pd.DataFrame(saving_rows).set_index(['comparison', 'k (pretrained)']))'''

CONTROL_TEXT = r'''## 8. Budget control: is pretraining just "more steps"?

`mae_scratch` above received 300 steps, and at that budget it never leaves the
mean-field plateau. The fair objection is that it needed as many steps as
pretraining. The control below trains the architecture from scratch on the
labelled frames of seed 0 for the *full pretraining budget* at k = 2, 8 and 128
and places the results next to the 300-step fine-tuned model on the same frames.
It uses far more labelled compute than any other entry; if it still cannot match
the fine-tuned model at small k, the advantage is the pretraining, not the steps.'''

CONTROL_CODE = r'''seed0 = cfg.seeds[0]
control_ks = (2, 8, 128) if not QUICK else (4,)     # every control k must also be in cfg.ks
controls = []
for k in control_ks:
    labelled = fields[110][mp.sample_labelled_frames(pool, k, seed=100 * seed0 + k)]   # the same frames as the loop
    t0 = time.perf_counter()
    control = mp.build_mae(layout, seed=seed0)
    _, control_step = mp.train_masked_model(control, layout.patchify(labelled), layout, mask_ratio=cfg.mask_ratio,
                                            steps=PRETRAIN_STEPS, batch_size=8, lr=2e-3, seed=600 + k,
                                            val_patches=val_patches, val_visible=val_visible, val_every=50)
    result = mp.score(mp.predict_masked(control, layout.patchify(test_frames), test_masks[seed0], layout),
                      test_frames, layout.pixel_mask(test_masks[seed0]))
    finetuned = [r['masked'] for r in records if r['method'] == 'mae_finetune' and r['k'] == k and r['seed'] == seed0]
    controls.append({'k': k, 'seed': seed0, 'steps': PRETRAIN_STEPS, 'best_step': control_step,
                     'seconds': time.perf_counter() - t0, **result,
                     'finetune_300_steps_masked': finetuned[0] if finetuned else None})
    print(f"k = {k:3d}: from scratch, {PRETRAIN_STEPS} steps -> {result['masked']:6.2f}%  (restored step {control_step}, "
          f"{controls[-1]['seconds']:.0f} s);  pretrained + {cfg.downstream_steps} steps -> "
          f"{controls[-1]['finetune_300_steps_masked']:.2f}%" if finetuned else '', flush=True)
print(f"pretrained, zero labels (seed {seed0}): "
      f"{[r['masked'] for r in records if r['method'] == 'mae_zero_shot' and r['seed'] == seed0][0]:.2f}%")'''

READING_TEXT = r'''## 9. Reading the result honestly

Three findings, each bounded by the data it was measured on:

1. **Pretraining helps the network, most where labels are fewest.** With the
   same labelled frames and the same downstream budget, the fine-tuned model is
   never worse than its zero-shot starting point (early stopping may keep the
   pretrained weights), improves on it from k = 4 (about 10%) and settles near
   8% from k = 8 upwards, while the from-scratch model is still predicting the
   mean field (about 100%) at every k; the
   label-saving table above is the honest summary. The budget control sharpens
   the claim rather than overturning it: with the full pretraining budget on
   the target frames, the from-scratch model still fails at k = 2, comes within
   about a percentage point at k = 8, and at k = 128 is at least as good as the
   300-step fine-tuned model. The benefit of pretraining is concentrated in the
   low-label regime, which is the regime MAPA is about; with enough labels and
   enough compute, the unlabelled trajectories add nothing on this data.
2. **The classical method wins.** Gappy POD with a transferred basis, zero labels,
   is already an order of magnitude more accurate than the fine-tuned network,
   and the pooled basis improves further with every labelled frame. The wake at
   Re = 90 to 110 is periodic and low-rank: 64 modes fitted on other Reynolds
   numbers span the target field almost exactly, and 624 visible pixels
   overdetermine the coefficients. A transformer with 90 thousand weights and
   562 training frames cannot beat a subspace that is nearly the truth.
3. **The frozen probe needs enough labels.** The visible-token-pooled 64-component
   probe improves rapidly and reaches about 8% at k = 64 to 128. At k = 1 to 4,
   however, a linear map from very few augmented examples to 2496 output pixels
   is underdetermined, whereas the pretrained decoder already contains the
   nonlinear completion map.

When does the pretraining recipe pay off? When the data are not low-rank, when
no matched linear baseline exists, or when the downstream label is not the
field itself (MAPA decodes behaviour and speech from the recordings). The
exercises ask you to move this experiment towards those conditions rather than
to declare a winner on this one.

Limits of the evidence: educational LBM at 12 lattice nodes per diameter;
four trajectories at nearly the same Reynolds number; the target trajectory
was inspected in earlier weeks; three mask seeds of one trajectory, not
independent flows; CPU-sized models and budgets.'''

WRITE_CODE = r'''after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
assert after == before, 'the notebook must not modify the retained wake data'
metrics = {
    'protocol': {'field': mp.FIELD, 'mask_ratio': cfg.mask_ratio, 'patch': [layout.py, layout.px], 'patch_grid': list(layout.grid),
                 'label_pool': list(cfg.label_pool), 'test_window': list(cfg.test_window), 'ks': list(cfg.ks), 'seeds': list(cfg.seeds),
                 'pretraining_trajectories': [90, 100], 'validation_trajectory': 105, 'target_trajectory': 110,
                 'rank_candidates': list(cfg.rank_candidates), 'alpha_candidates': list(cfg.alpha_candidates),
                 'masks_per_labelled_frame': cfg.masks_per_labelled_frame, 'downstream_steps': cfg.downstream_steps,
                 'downstream_batch': cfg.downstream_batch, 'scratch_lr': cfg.scratch_lr, 'finetune_lr': cfg.finetune_lr,
                 'pretrain_steps': PRETRAIN_STEPS, 'pretrain_batch': 8, 'pretrain_lr': 2e-3, 'quick': QUICK},
    'model': {'parameters': n_parameters, 'dim': 64, 'depth': 2, 'heads': 4, 'decoder_dim': 32, 'decoder_depth': 1},
    'pretraining': {'best_step': best_step, 'seconds': pretrain_seconds, 'history': history, 'variance_of_v': variance},
    'transferred_pod': {'rank': rank_transfer, 'validation_scores': {str(k): v for k, v in rank_scores.items()}},
    'zero_shot': zero, 'curves': {m: {str(k): v for k, v in c.items()} for m, c in curves.items()},
    'records': records, 'label_saving': savings,
    'scratch_controls': controls,
    'timings_seconds': {m: float(sum(v)) for m, v in timings.items()} | {'protocol_total': protocol_seconds},
    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                    'threads': torch.get_num_threads(), 'platform': platform.platform()},
    'source_hashes': {f'data/modal_labs/{name}': digest for name, digest in after.items()},
}
mp.write_json(OUT / 'metrics.json', metrics)
print('written:', sorted(p.name for p in OUT.iterdir()))

retained_path = RETAINED / 'metrics.json'
if retained_path.is_file() and not WRITE_RESULTS:
    retained = json.loads(retained_path.read_text())
    rows = []
    for method, curve in retained['curves'].items():
        for k, v in curve.items():
            fresh = curves.get(method, {}).get(int(k))
            if fresh:
                rows.append({'method': method, 'k': int(k), 'retained (%)': round(v['mean'], 2), 'this run (%)': round(fresh['mean'], 2)})
    print('Your run against the retained evidence (differences come from CPU arithmetic, thread count and QUICK):')
    display(pd.DataFrame(rows).pivot(index='k', columns='method', values=['retained (%)', 'this run (%)']))
print('PASS: retained data unchanged; outputs in', OUT.relative_to(ROOT) if WRITE_RESULTS else OUT.name)'''

EXERCISES = r'''## 10. Exercises

1. **Structured masks.** Replace the random patch mask by a contiguous hidden
   block (for example the downstream half of the window). Which method degrades
   more, and why does the least-squares argument of Section 3 change?
2. **A harder downstream label.** Use the masked `v` field as input but the
   full `omega` field (also in the archives) as the label. Build the matched
   classical baseline (a linear map from gappy-POD coefficients of `v` to
   `omega`, fitted on the k frames) and redraw the curve.
3. **Noise.** Add Gaussian noise of 10% of the field RMS to the visible pixels,
   as in Week 7.2. Reselect the POD rank and the ridge strength on Re = 105
   and report which method is more robust.
4. **Budget fairness.** Give `mae_scratch` 1000 and 3000 steps at every k and
   plot the family of curves. At what budget does it catch the zero-shot model,
   and does it ever catch `mae_finetune`?
5. **Falsification.** Write down, before running anything, the observation that
   would make you conclude that pretraining does *not* help on this data. Then
   check whether the retained evidence contains it.

## References

- K. He, X. Chen, S. Xie, Y. Li, P. Dollar and R. Girshick, "Masked autoencoders
  are scalable vision learners," CVPR 2022. https://doi.org/10.1109/CVPR52688.2022.01553
- B. Tang, T. Spalding and E. Cogan (Duke University), "Pretraining for
  sample-efficient neural interfaces" (MAPA), arXiv:2609.13507, 2026.
  https://bentang18.github.io/mapa-page/
- R. Everson and L. Sirovich, "Karhunen-Loeve procedure for gappy data,"
  J. Opt. Soc. Am. A 12, 1657-1664, 1995. https://doi.org/10.1364/JOSAA.12.001657
- M. McCabe et al., "Multiple physics pretraining for physical surrogate
  models," arXiv:2310.02994, 2023.
- M. Herde et al., "Poseidon: efficient foundation models for PDEs," NeurIPS
  2024. arXiv:2405.19101.

Data: FlowMLLab cylinder-cfd-v1 (`data/modal_labs`). Code in
`flowmllab/masked_pretraining.py` is independently authored for this repository.'''


def build_cells() -> list[dict]:
    cells = [
        cell("markdown", TITLE + badge(NOTEBOOK.relative_to(ROOT).as_posix())),
        cell("code", bootstrap(NOTEBOOK.parent.relative_to(ROOT).as_posix()) + SETUP),
        cell("markdown", DEFINITIONS),
        cell("markdown", CONTRACT),
        cell("code", MASK_FIGURE),
        cell("markdown", GAPPY_TEXT),
        cell("code", GAPPY_CODE),
        cell("markdown", MAE_TEXT),
        cell("code", MAE_CODE),
        cell("markdown", ZERO_SHOT_TEXT),
        cell("code", ZERO_SHOT_CODE),
        cell("markdown", PROTOCOL_TEXT),
        cell("code", PROTOCOL_CODE),
        cell("markdown", CURVE_TEXT),
        cell("code", CURVE_CODE),
        cell("markdown", CONTROL_TEXT),
        cell("code", CONTROL_CODE),
        cell("markdown", READING_TEXT),
        cell("code", WRITE_CODE),
        cell("markdown", EXERCISES),
    ]
    for index, c in enumerate(cells):
        c["id"] = hashlib.sha256(f"week73-{index}-{''.join(c['source'])}".encode()).hexdigest()[:12]
    return cells


def make_notebook(execute: bool = False) -> Path:
    notebook = {"cells": build_cells(),
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if execute:
        import nbformat
        from nbclient import NotebookClient
        RESULTS.mkdir(parents=True, exist_ok=True)
        os.environ["FLOWMLLAB_WRITE_RESULTS"] = "1"
        nb = nbformat.read(NOTEBOOK, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
        nbformat.write(nb, NOTEBOOK)
    print(NOTEBOOK.relative_to(ROOT))
    return NOTEBOOK


# --------------------------------------------------------------------------- results README
def make_results_readme() -> Path:
    """Write results/week07_3_pretraining/README.md from the retained metrics so that no number is typed by hand."""
    evidence = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    protocol, curves, zero = evidence["protocol"], evidence["curves"], evidence["zero_shot"]
    ks = [str(k) for k in protocol["ks"]]
    methods = [("pod_target", "gappy POD, target-only basis"), ("pod_pooled", "gappy POD, pooled basis"),
               ("mae_probe", "MAE frozen encoder + ridge probe"), ("mae_scratch", "MAE from scratch, 300 steps"),
               ("mae_finetune", "MAE pretrained + fine-tuned, 300 steps")]
    lines = ["# Week 7.3 retained pretraining and label-efficiency evidence", "",
             "Generated by executing `notebooks/week07_3/W7_3_Masked_Pretraining_Label_Efficiency.ipynb` through",
             "`python qa/build_week07_3_materials.py --execute` (this README is written by the same builder from",
             "`metrics.json`). Regenerate only through the builder; a plain Run All of the notebook writes to a",
             "temporary folder.", "",
             "## Frozen protocol", "",
             f"- Field: transverse velocity `v/U` on the 32 x 78 wake window; patches of {protocol['patch'][0]} x "
             f"{protocol['patch'][1]} pixels ({protocol['patch_grid'][0]} x {protocol['patch_grid'][1]} = "
             f"{protocol['patch_grid'][0] * protocol['patch_grid'][1]} patches); mask ratio {protocol['mask_ratio']}.",
             f"- Unlabelled pretraining trajectories: Re = {', '.join(str(r) for r in protocol['pretraining_trajectories'])}; "
             f"validation trajectory for every selection (POD rank, ridge strength, early stopping): Re = "
             f"{protocol['validation_trajectory']}; target: Re = {protocol['target_trajectory']}, labelled pool frames "
             f"`[{protocol['label_pool'][0]}, {protocol['label_pool'][1]})`, test window "
             f"`[{protocol['test_window'][0]}, {protocol['test_window'][1]})` with fixed random masks.",
             f"- Labelled frames k in {protocol['ks']}; mask seeds {protocol['seeds']} (each seed changes the test masks "
             f"and the drawn frames).",
             f"- MAE: {evidence['model']['parameters']:,} parameters (dim {evidence['model']['dim']}, depth "
             f"{evidence['model']['depth']}, {evidence['model']['heads']} heads, decoder dim {evidence['model']['decoder_dim']}, "
             f"decoder depth {evidence['model']['decoder_depth']}); pretraining {protocol['pretrain_steps']} AdamW steps, "
             f"batch {protocol['pretrain_batch']}, learning rate {protocol['pretrain_lr']}, fresh masks every step, weights "
             f"restored from step {evidence['pretraining']['best_step']} ({evidence['pretraining']['seconds']:.0f} s).",
             f"- Downstream: {protocol['downstream_steps']} steps, batch {protocol['downstream_batch']}, learning rate "
             f"{protocol['scratch_lr']} from scratch and {protocol['finetune_lr']} for fine-tuning; "
             f"{protocol['masks_per_labelled_frame']} random masks per labelled frame for the probe; ridge strengths "
             f"{protocol['alpha_candidates']}; POD ranks {protocol['rank_candidates']} (at most k - 1 for the target-only basis).",
             f"- Transferred POD basis: rank {evidence['transferred_pod']['rank']} selected on Re = 105.", "",
             "## Hidden-pixel relative L2 error (%) on the Re = 110 test window", "",
             f"Mean +/- sample SD over {len(protocol['seeds'])} mask seeds of one trajectory (not independent flows).", "",
             "| k | " + " | ".join(label for _, label in methods) + " |",
             "| ---: | " + " | ".join("---:" for _ in methods) + " |"]
    for k in ks:
        lines.append(f"| {k} | " + " | ".join(f"{curves[m][k]['mean']:.2f} +/- {curves[m][k]['std']:.2f}" for m, _ in methods) + " |")
    lines += ["", f"Zero-shot (no labelled frame): gappy POD with the transferred basis {zero['pod_transfer']:.2f}%, "
              f"pretrained MAE {zero['mae_zero_shot']:.2f}%.", "",
              "Budget control (seed {seed}; the architecture from scratch for the full pretraining budget of {steps} steps on "
              "the same labelled frames as the loop):".format(seed=evidence['scratch_controls'][0]['seed'],
                                                              steps=evidence['scratch_controls'][0]['steps']), ""]
    lines += ["| k | from scratch, full budget | pretrained + 300 steps (same seed) |", "| ---: | ---: | ---: |"]
    for c in evidence["scratch_controls"]:
        lines.append(f"| {c['k']} | {c['masked']:.2f}% (restored step {c['best_step']}) | {c['finetune_300_steps_masked']:.2f}% |")
    lines += ["", "## Label saving", "",
              "Smallest k at which the reference method reaches the error of the pretrained method at each k of the "
              "pretrained curve (mean curves).", "",
              "| comparison | " + " | ".join(f"k = {k}" for k in ks) + " |", "| --- | " + " | ".join("---:" for _ in ks) + " |"]
    for name, saving in evidence["label_saving"].items():
        cells = []
        for k in ks:
            match = saving["matches"].get(k)
            cells.append("not reached" if match is None else f"{match}")
        lines.append(f"| {name.replace('_vs_', ' vs ')} | " + " | ".join(cells) + " |")
    lines += ["", "## Files", "",
              "- `metrics.json`: protocol, model, pretraining history, transferred-basis validation scores, every record "
              "(method, k, seed, rank or ridge strength or restored step, hidden-pixel and full-frame error), curves, label "
              "savings, budget controls, timings, environment and the SHA-256 of the four wake archives.",
              "- `label_efficiency.png`: the curve (Figure 1 of the lecture); `zero_shot_fields.png`: truth, visible input, "
              "both zero-shot reconstructions and error maps for test frame 210; `pretraining_loss.png`: masked MSE during "
              "pretraining with the mean-field plateau; `masking_example.png`: one frame, its patch mask and what every "
              "method receives.", "",
              "## Reading", "",
              "Fine-tuning the pretrained model beats the same architecture from scratch at every k under the same "
              "downstream budget, and by the largest margin at small k; with the full pretraining budget the from-scratch "
              "model still fails at k = 2. Gappy POD with the transferred basis is an order of magnitude more accurate "
              "than the network with zero labels, and the pooled basis improves with every label: on this periodic, "
              "low-rank educational wake the classical method with matched information wins. The target trajectory was "
              "inspected in earlier weeks; this is a protocol demonstration, not a blind benchmark. Environment: Python "
              f"{evidence['environment']['python']}, NumPy {evidence['environment']['numpy']}, PyTorch "
              f"{evidence['environment']['torch']}, {evidence['environment']['threads']} CPU threads; total loop time "
              f"{evidence['timings_seconds']['protocol_total']:.0f} s."]
    path = RESULTS / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))
    return path


# --------------------------------------------------------------------------- lecture PDF
def make_pdf() -> Path:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    evidence = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    out = ROOT / "output/pdf" / PDF.name
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    ink, muted = colors.HexColor("#173b56"), colors.HexColor("#506070")
    styles.add(ParagraphStyle(name="Title73", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=ink, spaceAfter=12))
    styles.add(ParagraphStyle(name="Heading73", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=ink,
                              spaceBefore=10, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name="Body73", fontName="Helvetica", fontSize=10.2, leading=14.4, spaceAfter=7))
    styles.add(ParagraphStyle(name="Small73", fontName="Helvetica", fontSize=8.8, leading=12, textColor=muted, spaceAfter=7))
    styles.add(ParagraphStyle(name="Bullet73", fontName="Helvetica", fontSize=10.2, leading=14.4, leftIndent=14,
                              bulletIndent=2, spaceAfter=3))
    styles.add(ParagraphStyle(name="Equation73", fontName="Courier", fontSize=9.6, leading=13, leftIndent=14, spaceAfter=6))

    sections = [page.strip().split("\n\n") for page in SOURCE.read_text(encoding="utf-8").split("\n---\n")]
    story: list = []

    def paragraph(text: str, style: str = "Body73") -> None:
        story.append(Paragraph(text, styles[style]))

    def block(text: str) -> None:
        """Render one Markdown block of the source: heading, bullet list, equation or paragraph."""
        if text.startswith("# "):
            paragraph(html.escape(text[2:]), "Heading73")
        elif text.startswith("- "):
            for item in text.split("\n- "):
                item = item.removeprefix("- ").replace("\n", " ")
                story.append(Paragraph(inline(item), styles["Bullet73"], bulletText="•"))
        elif re.match(r"\d+\. ", text):
            for item in re.split(r"\n(?=\d+\. )", text):
                number, body = item.split(". ", 1)
                story.append(Paragraph(inline(body.replace("\n", " ")), styles["Bullet73"], bulletText=f"{number}."))
        elif text.startswith("    "):
            for line in text.splitlines():
                paragraph(html.escape(line.strip()), "Equation73")
        elif text.startswith("> "):
            paragraph(inline(text[2:].replace("\n> ", " ")), "Small73")
        else:
            paragraph(inline(text.replace("\n", " ")))

    def inline(text: str) -> str:
        text = html.escape(text)
        while "**" in text:
            text = text.replace("**", "<b>", 1).replace("**", "</b>", 1)
        while "`" in text:
            text = text.replace("`", "<font face='Courier'>", 1).replace("`", "</font>", 1)
        return text

    def table(rows, widths) -> None:
        obj = Table(rows, colWidths=widths, hAlign="LEFT")
        obj.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8.8),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), .7, ink), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white]),
            ("BACKGROUND", (0, 0), (-1, 0), ink), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
        story.extend([obj, Spacer(1, 8)])

    def figure(name: str, width: float, caption: str) -> None:
        path = RESULTS / name
        w, h = ImageReader(str(path)).getSize()
        story.append(Image(str(path), width=width, height=width * h / w))
        paragraph(caption, "Small73")

    # Page 1: question, definitions, contract.
    paragraph("07.3  /  CYLINDER WAKE", "Small73")
    paragraph("Self-supervised pretraining<br/>and label efficiency", "Title73")
    for text in sections[0][1:]:
        block(text)
    for text in sections[1]:
        block(text)

    # Page 2: methods.
    for index in (2, 3):
        for text in sections[index]:
            block(text)

    # Page 3: protocol table and retained numbers.
    for text in sections[4]:
        block(text)
    curves = evidence["curves"]
    ks = [str(k) for k in evidence["protocol"]["ks"]]
    labels = [("pod_target", "POD target-only"), ("pod_pooled", "POD pooled"),
              ("mae_probe", "MAE probe"), ("mae_scratch", "MAE scratch"), ("mae_finetune", "MAE fine-tuned")]
    rows = [["k"] + [label for _, label in labels]]
    for k in ks:
        rows.append([k] + [f"{curves[m][k]['mean']:.2f} +/- {curves[m][k]['std']:.2f}" for m, _ in labels])
    paragraph("Retained masked-pixel relative L2 error (%) on the Re = 110 test window, mean +/- sample SD over "
              f"{len(evidence['protocol']['seeds'])} mask seeds", "Heading73")
    table(rows, [34, 92, 92, 92, 92, 100])
    paragraph("POD target-only: gappy POD with a basis from the k labelled frames; POD pooled: basis from the unlabelled "
              "frames plus the k labelled ones; MAE probe: frozen encoder with a ridge readout; MAE scratch: the "
              "architecture from random weights, 300 steps; MAE fine-tuned: the pretrained model, 300 steps.", "Small73")
    zero = evidence["zero_shot"]
    controls = evidence["scratch_controls"]
    control_text = "; ".join(f"k = {c['k']}: {c['masked']:.2f}% from scratch against {c['finetune_300_steps_masked']:.2f}% "
                             f"fine-tuned" for c in controls)
    paragraph(f"Zero-shot rows (no labelled frame): transferred gappy POD {zero['pod_transfer']:.2f}%, pretrained MAE "
              f"{zero['mae_zero_shot']:.2f}%. Budget control (seed {controls[0]['seed']}, the architecture trained from "
              f"scratch for the full pretraining budget of {controls[0]['steps']} steps on the same labelled frames): "
              f"{control_text}. Pretraining took {evidence['pretraining']['seconds']:.0f} s and the label-efficiency "
              f"loop {evidence['timings_seconds']['protocol_total']:.0f} s on {evidence['environment']['threads']} CPU threads.",
              "Small73")
    for text in sections[5]:
        block(text)

    # Figures follow the reading; no forced page break so that the pages fill.
    figure("label_efficiency.png", 470, "Figure 1. Masked-pixel error against the number of labelled Re = 110 frames, mean "
           "over three mask seeds of one trajectory; the bars span the three seeds. Horizontal lines are the zero-shot methods.")
    figure("zero_shot_fields.png", 511, "Figure 2. Zero-shot completion of test frame 210 (mask seed 0): truth, the 25% of pixels "
           "every method receives, and the two reconstructions with absolute-error maps on one scale.")
    figure("pretraining_loss.png", 380, "Figure 3. Masked-patch MSE during pretraining. The dashed line is the variance of v, "
           "the loss of predicting the mean field; the network sits on that plateau for a few hundred steps before it learns.")

    # Reproducibility, exercises, references.
    for index in range(6, len(sections)):
        for text in sections[index]:
            block(text)

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d4dfe6")); canvas.line(42, 39, 553, 39)
        canvas.setFont("Helvetica", 8); canvas.setFillColor(muted)
        canvas.drawString(42, 25, "FlowMLLab  /  Ehsan Roohi  /  Week 7.3")
        canvas.drawRightString(553, 25, str(doc.page))

    SimpleDocTemplate(str(out), pagesize=(595, 842), rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=52,
                      title="Week 7.3 - Self-supervised pretraining and label efficiency",
                      author="Ehsan Roohi / FlowMLLab").build(story, onFirstPage=footer, onLaterPages=footer)
    shutil.copy2(out, PDF)
    print(PDF.relative_to(ROOT))
    return PDF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="execute the notebook in place and refresh the retained results")
    parser.add_argument("--notebook-only", action="store_true")
    parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()
    if not args.pdf_only:
        make_notebook(execute=args.execute)
    if not args.notebook_only:
        make_results_readme()
        make_pdf()
