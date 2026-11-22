"""Masked-patch pretraining and label-efficiency protocol for cylinder-wake fields (Week 7.3).

The module implements, for the retained 32 x 78 LBM wake windows of
``data/modal_labs``:

* a fixed patch layout with random patch masking (the self-supervised task);
* a small masked autoencoder (MAE) with a transformer encoder that sees only
  visible patches and a lightweight decoder that reconstructs the missing ones;
* gappy-POD reconstruction (the classical baseline for the same completion task);
* a frozen-encoder linear readout and a same-architecture supervised model
  trained from scratch, both fitted on ``k`` labelled frames; and
* the label-efficiency protocol that evaluates every method on the same masked
  test frames for increasing ``k``.

All functions are deterministic for a given seed and run on CPU.  PyTorch is
imported lazily so that the POD parts work without it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cylinder_ml import PODBasis, fit_pod

FIELD = "v"
TRAJECTORIES = (90, 100, 105, 110)


# --------------------------------------------------------------------------- data
def load_wake_field(root: str | Path, reynolds: int, field: str = FIELD) -> np.ndarray:
    """Return the (frames, 32, 78) array of one retained wake trajectory."""
    with np.load(Path(root) / "data" / "modal_labs" / f"re{reynolds:03d}.npz", allow_pickle=False) as archive:
        values = np.asarray(archive[field], dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid wake field for Re={reynolds}")
    return values


def relative_l2_error(prediction: np.ndarray, reference: np.ndarray, where: np.ndarray | None = None) -> float:
    """100 * ||prediction - reference|| / ||reference||, optionally restricted to a boolean mask."""
    p = np.asarray(prediction, float).ravel()
    r = np.asarray(reference, float).ravel()
    if where is not None:
        w = np.asarray(where, bool).ravel()
        p, r = p[w], r[w]
    denominator = np.linalg.norm(r)
    return float(100.0 * np.linalg.norm(p - r) / denominator) if denominator > 0 else float("nan")


# --------------------------------------------------------------------------- patches and masks
@dataclass(frozen=True)
class PatchLayout:
    """Non-overlapping rectangular patches over an (ny, nx) grid."""

    ny: int = 32
    nx: int = 78
    py: int = 4
    px: int = 6

    def __post_init__(self) -> None:
        if self.ny % self.py or self.nx % self.px:
            raise ValueError("patch size must divide the grid")

    @property
    def grid(self) -> tuple[int, int]:
        return self.ny // self.py, self.nx // self.px

    @property
    def n_patches(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def patch_dim(self) -> int:
        return self.py * self.px

    def patchify(self, frames: np.ndarray) -> np.ndarray:
        """(N, ny, nx) -> (N, P, py*px), row-major over the patch grid."""
        frames = np.asarray(frames, float)
        n = frames.shape[0]
        gy, gx = self.grid
        blocks = frames.reshape(n, gy, self.py, gx, self.px).transpose(0, 1, 3, 2, 4)
        return blocks.reshape(n, self.n_patches, self.patch_dim)

    def unpatchify(self, patches: np.ndarray) -> np.ndarray:
        patches = np.asarray(patches, float)
        n = patches.shape[0]
        gy, gx = self.grid
        blocks = patches.reshape(n, gy, gx, self.py, self.px).transpose(0, 1, 3, 2, 4)
        return blocks.reshape(n, self.ny, self.nx)

    def pixel_mask(self, visible: np.ndarray) -> np.ndarray:
        """Boolean (N, ny, nx) array that is True where the patch is visible."""
        visible = np.asarray(visible, bool)
        return self.unpatchify(np.repeat(visible[:, :, None], self.patch_dim, axis=2)).astype(bool)


def random_patch_masks(n_frames: int, n_patches: int, mask_ratio: float, seed: int) -> np.ndarray:
    """Boolean (N, P) visibility array with exactly round((1 - ratio) * P) visible patches per frame."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    n_visible = int(round((1.0 - mask_ratio) * n_patches))
    if n_visible < 1:
        raise ValueError("mask_ratio leaves no visible patch")
    visible = np.zeros((n_frames, n_patches), dtype=bool)
    for i in range(n_frames):
        visible[i, rng.choice(n_patches, size=n_visible, replace=False)] = True
    return visible


# --------------------------------------------------------------------------- gappy POD
def gappy_pod_reconstruct(pod, frames: np.ndarray, pixel_visible: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Least-squares POD coefficients from the visible pixels of each frame, then full reconstruction.

    For frame f with visible index set V, solve
    (Phi_V^T Phi_V + ridge I) a = Phi_V^T (f_V - mean_V) and return mean + Phi a.
    """
    frames = np.asarray(frames, float).reshape(len(frames), -1)
    visible = np.asarray(pixel_visible, bool).reshape(len(frames), -1)
    out = np.empty_like(frames)
    eye = np.eye(pod.rank)
    for i in range(len(frames)):
        v = visible[i]
        phi = pod.modes[v]
        rhs = phi.T @ (frames[i, v] - pod.mean[v])
        coefficients = np.linalg.solve(phi.T @ phi + ridge * eye, rhs)
        out[i] = pod.mean + pod.modes @ coefficients
    return out


@dataclass(frozen=True)
class MeanField:
    """Rank-0 'basis': the prediction is the mean of the fitting frames (used when k = 1)."""

    mean: np.ndarray

    @property
    def rank(self) -> int:
        return 0


def fit_gappy_basis(train_frames: np.ndarray, rank: int):
    """POD basis of the given rank from full training frames (rank 0 gives the mean field)."""
    train = np.asarray(train_frames, float).reshape(len(train_frames), -1)
    if rank == 0:
        return MeanField(train.mean(axis=0))
    return fit_pod(train, rank=rank)


def gappy_reconstruct(basis, frames: np.ndarray, pixel_visible: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Gappy reconstruction with a POD basis or the rank-0 mean field; output shaped like ``frames``."""
    frames = np.asarray(frames, float)
    if isinstance(basis, MeanField):
        return np.broadcast_to(basis.mean.reshape(frames.shape[1:]), frames.shape).copy()
    return gappy_pod_reconstruct(basis, frames, pixel_visible, ridge).reshape(frames.shape)


def select_pod_rank(train_frames: np.ndarray, val_frames: np.ndarray, val_visible: np.ndarray,
                    candidates) -> tuple[int, dict[int, float]]:
    """Choose the POD rank (at most k - 1 for k training frames) by gappy-reconstruction error on validation frames."""
    train = np.asarray(train_frames, float).reshape(len(train_frames), -1)
    val = np.asarray(val_frames, float)
    val_pixels = np.asarray(val_visible, bool)
    max_rank = min(train.shape[0] - 1, train.shape[1])
    ranks = sorted({0, *(int(r) for r in candidates if r <= max_rank)})
    full = fit_gappy_basis(train, ranks[-1]) if ranks[-1] > 0 else None   # one SVD, then truncations
    scores: dict[int, float] = {}
    for rank in ranks:
        if rank == 0:
            basis = fit_gappy_basis(train, 0)
        else:
            basis = PODBasis(mean=full.mean, modes=full.modes[:, :rank], singular_values=full.singular_values,
                             cumulative_energy=full.cumulative_energy)
        scores[rank] = relative_l2_error(gappy_reconstruct(basis, val, val_pixels), val, ~val_pixels)
    best = min(scores, key=scores.get)
    return best, scores


def fill_visible(prediction: np.ndarray, frames: np.ndarray, pixel_visible: np.ndarray) -> np.ndarray:
    """Copy the observed pixels into the prediction; every method is scored after this step."""
    return np.where(np.asarray(pixel_visible, bool), np.asarray(frames, float), np.asarray(prediction, float))


# --------------------------------------------------------------------------- masked autoencoder
def _torch():
    import torch  # noqa: PLC0415

    return torch


def build_mae(layout: PatchLayout, dim: int = 64, depth: int = 2, heads: int = 4,
              decoder_dim: int = 32, decoder_depth: int = 1, seed: int = 0, norm_first: bool = True):
    """A small masked autoencoder (He et al., 2022) for one-channel patch grids."""
    torch = _torch()
    from torch import nn  # noqa: PLC0415

    torch.manual_seed(seed)

    class MaskedAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            P, D = layout.n_patches, layout.patch_dim
            self.embed = nn.Linear(D, dim)
            self.pos = nn.Parameter(torch.zeros(1, P, dim))
            layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.0,
                                               batch_first=True, activation="gelu", norm_first=norm_first)
            self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.encoder_norm = nn.LayerNorm(dim)
            self.to_decoder = nn.Linear(dim, decoder_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.decoder_pos = nn.Parameter(torch.zeros(1, P, decoder_dim))
            dlayer = nn.TransformerEncoderLayer(decoder_dim, heads, dim_feedforward=2 * decoder_dim,
                                                dropout=0.0, batch_first=True, activation="gelu",
                                                norm_first=norm_first)
            self.decoder = nn.TransformerEncoder(dlayer, decoder_depth, enable_nested_tensor=False)
            self.decoder_norm = nn.LayerNorm(decoder_dim)
            self.head = nn.Linear(decoder_dim, D)
            nn.init.normal_(self.pos, std=0.02)
            nn.init.normal_(self.decoder_pos, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)

        def encode(self, patches, visible):
            """Encoder sees visible patches only; returns (N, P, dim) with zeros at masked positions."""
            tokens = self.embed(patches) + self.pos
            n, p, d = tokens.shape
            # every frame has the same number of visible patches (random_patch_masks guarantees it)
            index = torch.stack([torch.nonzero(v, as_tuple=False).squeeze(1) for v in visible])
            gathered = torch.gather(tokens, 1, index[:, :, None].expand(-1, -1, d))
            encoded = self.encoder_norm(self.encoder(gathered))
            full = torch.zeros(n, p, d, dtype=encoded.dtype, device=encoded.device)
            full.scatter_(1, index[:, :, None].expand(-1, -1, d), encoded)
            return full, index

        def forward(self, patches, visible):
            full, index = self.encode(patches, visible)
            n, p, _ = full.shape
            dec = self.to_decoder(full)
            keep = visible[:, :, None].to(dec.dtype)
            dec = dec * keep + self.mask_token.expand(n, p, -1) * (1 - keep) + self.decoder_pos
            return self.head(self.decoder_norm(self.decoder(dec)))

    return MaskedAutoencoder()


def _to_tensor(array, dtype=None):
    torch = _torch()
    return torch.as_tensor(np.asarray(array), dtype=dtype or torch.float32)


def masked_mse(prediction, target, visible):
    """Mean squared error over masked patches only (the MAE objective)."""
    torch = _torch()
    weight = (~visible).to(prediction.dtype)[:, :, None]
    return torch.sum(weight * (prediction - target) ** 2) / torch.sum(weight) / prediction.shape[-1]


def train_masked_model(model, patches: np.ndarray, layout: PatchLayout, *, mask_ratio: float, steps: int,
                       batch_size: int, lr: float, seed: int, val_patches: np.ndarray | None = None,
                       val_visible: np.ndarray | None = None, val_every: int | None = None,
                       log_every: int = 0, weight_decay: float = 0.05):
    """Train the masked-reconstruction objective with a fresh random mask for every frame at every step.

    The optimizer is AdamW with a cosine learning-rate schedule over ``steps``.  When validation
    frames and masks are given, the masked loss on them is evaluated every ``val_every`` steps
    (default: once per pass over the training frames) and the weights with the lowest
    validation loss are restored at the end (early stopping).  Returns the history (one record
    per validation point) and the step of the restored weights.
    """
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = _to_tensor(patches)
    n = len(x)
    batch_size = min(batch_size, n)
    if val_every is None:
        val_every = max(1, int(np.ceil(n / batch_size)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    history: list[dict[str, float]] = []
    best = (float("inf"), None, -1)
    started = time.perf_counter()
    if val_patches is not None:
        # the starting weights are a candidate too: fine-tuning that only hurts on validation is rejected
        initial = evaluate_masked_loss(model, val_patches, val_visible)
        history.append({"step": 0, "train_loss": float("nan"), "seconds": 0.0, "val_loss": initial})
        best = (initial, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0)
    order, cursor, window_loss, window_count = rng.permutation(n), 0, 0.0, 0
    for step in range(1, steps + 1):
        model.train()
        if cursor + batch_size > n:
            order, cursor = rng.permutation(n), 0
        idx = order[cursor:cursor + batch_size]
        cursor += batch_size
        vis = _to_tensor(random_patch_masks(len(idx), layout.n_patches, mask_ratio, int(rng.integers(1 << 30))), torch.bool)
        loss = masked_mse(model(x[idx], vis), x[idx], vis)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        window_loss += float(loss.detach())
        window_count += 1
        if step % val_every == 0 or step == steps:
            record = {"step": step, "train_loss": window_loss / max(window_count, 1),
                      "seconds": time.perf_counter() - started}
            window_loss, window_count = 0.0, 0
            if val_patches is not None:
                record["val_loss"] = evaluate_masked_loss(model, val_patches, val_visible)
                if record["val_loss"] < best[0]:
                    best = (record["val_loss"], {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
            history.append(record)
            if log_every and (len(history) % log_every == 0 or step == steps):
                print(f"  step {step:6d}  train {record['train_loss']:.4e}"
                      + (f"  val {record['val_loss']:.4e}" if "val_loss" in record else "")
                      + f"  {record['seconds']:6.1f} s")
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return history, best[2]


def evaluate_masked_loss(model, patches: np.ndarray, visible: np.ndarray) -> float:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        x, v = _to_tensor(patches), _to_tensor(visible, torch.bool)
        return float(masked_mse(model(x, v), x, v))


def predict_masked(model, patches: np.ndarray, visible: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """Reconstruct full frames; visible patches are copied from the input, masked ones come from the model."""
    torch = _torch()
    model.eval()
    with torch.no_grad():
        x, v = _to_tensor(patches), _to_tensor(visible, torch.bool)
        out = model(x, v).numpy()
    out = np.where(np.asarray(visible)[:, :, None], np.asarray(patches), out)
    return layout.unpatchify(out)


def encoder_features(model, patches: np.ndarray, visible: np.ndarray, *, pooling: str = "flatten") -> np.ndarray:
    """Return frozen encoder features with an explicit mask-handling rule.

    ``flatten`` retains every spatial token (zeros at masked positions).
    ``visible_mean`` averages only encoded visible tokens to a fixed-size
    vector; its values still depend on which patches were visible. This is the compact
    representation used by the Week 7.3 linear probe.
    """
    torch = _torch()
    model.eval()
    with torch.no_grad():
        full, _ = model.encode(_to_tensor(patches), _to_tensor(visible, torch.bool))
    encoded = full.numpy()
    if pooling == "flatten":
        return encoded.reshape(len(patches), -1)
    if pooling == "visible_mean":
        weights = np.asarray(visible, bool)[..., None]
        return (encoded * weights).sum(axis=1) / weights.sum(axis=1)
    raise ValueError("pooling must be 'flatten' or 'visible_mean'")


# --------------------------------------------------------------------------- linear readout
@dataclass
class RidgeReadout:
    """Ridge regression from frozen features to full frames (fitted in dual form when k < features)."""

    alpha: float
    x_mean: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeReadout":
        x = np.asarray(features, float)
        y = np.asarray(targets, float).reshape(len(x), -1)
        self.x_mean, self.y_mean = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - self.x_mean, y - self.y_mean
        if xc.shape[0] <= xc.shape[1]:
            gram = xc @ xc.T
            self.weights = xc.T @ np.linalg.solve(gram + self.alpha * np.eye(len(xc)), yc)
        else:
            self.weights = np.linalg.solve(xc.T @ xc + self.alpha * np.eye(xc.shape[1]), xc.T @ yc)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.y_mean + (np.asarray(features, float) - self.x_mean) @ self.weights


# --------------------------------------------------------------------------- protocol helpers
def sample_labelled_frames(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Choose k frame indices from the labelled pool without replacement (evenly spread for small k)."""
    pool = np.asarray(pool)
    rng = np.random.default_rng(seed)
    if k >= len(pool):
        return pool.copy()
    if k <= 8:
        offset = int(rng.integers(len(pool)))
        positions = (offset + np.linspace(0, len(pool), k, endpoint=False).astype(int)) % len(pool)
        return np.sort(pool[positions]).astype(int)
    return np.sort(rng.choice(pool, size=k, replace=False)).astype(int)


def label_saving(curve_reference: dict[int, float], curve_pretrained: dict[int, float]) -> dict:
    """How many labelled frames the reference method needs to match the pretrained method at each k.

    Both curves map k -> error (percent).  ``matches`` maps every pretrained k to the smallest
    reference k whose error is at or below the pretrained error at that k (None when the reference
    never gets there); ``saving`` is the reference k divided by the pretrained k where defined.
    The summary fields describe the smallest pretrained k.
    """
    matches: dict[int, int | None] = {}
    for k in sorted(curve_pretrained):
        matched = [kr for kr in sorted(curve_reference) if curve_reference[kr] <= curve_pretrained[k]]
        matches[int(k)] = int(matched[0]) if matched else None
    saving = {k: (m / k if m is not None else None) for k, m in matches.items()}
    k0 = min(matches)
    return {"pretrained_k": int(k0), "pretrained_error": float(curve_pretrained[k0]),
            "reference_k_to_match": matches[k0], "matches": matches, "saving": saving}


def write_json(path: str | Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def score(prediction: np.ndarray, frames: np.ndarray, pixel_visible: np.ndarray) -> dict[str, float]:
    """Relative L2 errors (percent) of a filled-in prediction over masked pixels and over the full frame."""
    filled = fill_visible(prediction, frames, pixel_visible)
    return {"masked": relative_l2_error(filled, frames, ~np.asarray(pixel_visible, bool)),
            "full": relative_l2_error(filled, frames)}


# --------------------------------------------------------------------------- label-efficiency protocol
@dataclass(frozen=True)
class ProtocolConfig:
    """Frozen choices of the Week 7.3 label-efficiency protocol."""

    mask_ratio: float = 0.75
    label_pool: tuple[int, int] = (0, 160)        # Re110 frames that may be labelled
    test_window: tuple[int, int] = (210, 281)     # Re110 frames that are scored (never fitted)
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    seeds: tuple[int, ...] = (0, 1, 2)
    rank_candidates: tuple[int, ...] = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    alpha_candidates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    masks_per_labelled_frame: int = 8
    downstream_steps: int = 300
    downstream_batch: int = 8
    scratch_lr: float = 2e-3
    finetune_lr: float = 5e-4
    val_every: int = 50


def augmented_masked_set(frames: np.ndarray, layout: PatchLayout, mask_ratio: float, copies: int, seed: int):
    """Repeat each labelled frame ``copies`` times with independent random masks (masks are free)."""
    frames = np.asarray(frames, float)
    repeated = np.repeat(frames, copies, axis=0)
    visible = random_patch_masks(len(repeated), layout.n_patches, mask_ratio, seed)
    return layout.patchify(repeated), visible, repeated


def fit_ridge_probe(model, labelled_frames: np.ndarray, layout: PatchLayout, *, mask_ratio: float, copies: int,
                    seed: int, val_frames: np.ndarray, val_visible: np.ndarray, alphas) -> tuple[RidgeReadout, float, dict]:
    """Frozen-encoder features of masked labelled frames -> full frames; ridge strength chosen on validation."""
    patches, visible, targets = augmented_masked_set(labelled_frames, layout, mask_ratio, copies, seed)
    features = encoder_features(model, patches, visible, pooling="visible_mean")
    val_features = encoder_features(model, layout.patchify(val_frames), val_visible, pooling="visible_mean")
    val_pixels = layout.pixel_mask(val_visible)
    scores: dict[float, float] = {}
    best: tuple[float, RidgeReadout | None, float] = (float("inf"), None, float("nan"))
    for alpha in alphas:
        readout = RidgeReadout(alpha).fit(features, targets)
        prediction = readout.predict(val_features).reshape(val_frames.shape)
        scores[alpha] = score(prediction, val_frames, val_pixels)["masked"]
        if scores[alpha] < best[0]:
            best = (scores[alpha], readout, alpha)
    return best[1], best[2], scores


def train_downstream(model, labelled_frames: np.ndarray, layout: PatchLayout, cfg: ProtocolConfig, *, lr: float,
                     seed: int, val_patches: np.ndarray, val_visible: np.ndarray):
    """Train (from scratch or from pretrained weights) on the labelled frames with fresh masks each step."""
    return train_masked_model(model, layout.patchify(labelled_frames), layout, mask_ratio=cfg.mask_ratio,
                              steps=cfg.downstream_steps, batch_size=cfg.downstream_batch, lr=lr, seed=seed,
                              val_patches=val_patches, val_visible=val_visible, val_every=cfg.val_every)


def summarize_curves(records: list[dict]) -> dict[str, dict[int, dict[str, float]]]:
    """Summarize masked-error records by method and label count.

    ``std`` is the sample standard deviation (``ddof=1``), matching the
    uncertainty reported in the notebook, retained-results README and lecture.
    It is zero when only one repeat is available.
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    for method in sorted({r["method"] for r in records}):
        out[method] = {}
        for k in sorted({r["k"] for r in records if r["method"] == method}):
            values = np.array([r["masked"] for r in records if r["method"] == method and r["k"] == k], float)
            sample_std = values.std(ddof=1) if len(values) > 1 else 0.0
            out[method][int(k)] = {"mean": float(values.mean()), "std": float(sample_std),
                                   "min": float(values.min()), "max": float(values.max()), "n": int(len(values))}
    return out
