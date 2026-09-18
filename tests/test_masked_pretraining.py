"""Unit tests for the Week 7.3 masked-pretraining module (NumPy parts run without PyTorch)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flowmllab import masked_pretraining as mp

ROOT = Path(__file__).resolve().parents[1]


def test_patchify_round_trip_and_ordering():
    layout = mp.PatchLayout(ny=8, nx=12, py=4, px=6)
    assert layout.grid == (2, 2) and layout.n_patches == 4 and layout.patch_dim == 24
    frames = np.arange(2 * 8 * 12, dtype=float).reshape(2, 8, 12)
    patches = layout.patchify(frames)
    assert patches.shape == (2, 4, 24)
    # patch 1 of frame 0 is rows 0-3, columns 6-11, row-major
    np.testing.assert_array_equal(patches[0, 1], frames[0, 0:4, 6:12].ravel())
    np.testing.assert_array_equal(layout.unpatchify(patches), frames)
    with pytest.raises(ValueError):
        mp.PatchLayout(ny=8, nx=12, py=3, px=6)


def test_random_masks_have_fixed_visible_count_and_are_reproducible():
    visible = mp.random_patch_masks(5, 104, 0.75, seed=3)
    assert visible.shape == (5, 104) and visible.dtype == bool
    assert (visible.sum(axis=1) == 26).all()
    np.testing.assert_array_equal(visible, mp.random_patch_masks(5, 104, 0.75, seed=3))
    assert not np.array_equal(visible, mp.random_patch_masks(5, 104, 0.75, seed=4))
    pixels = mp.PatchLayout().pixel_mask(visible)
    assert pixels.shape == (5, 32, 78) and np.isclose(pixels.mean(), 0.25)
    with pytest.raises(ValueError):
        mp.random_patch_masks(1, 104, 1.0, seed=0)


def test_gappy_pod_recovers_exactly_low_rank_frames():
    rng = np.random.default_rng(0)
    layout = mp.PatchLayout(ny=8, nx=12, py=4, px=6)
    modes = rng.normal(size=(3, 96))
    train = rng.normal(size=(40, 3)) @ modes + 0.5
    test = (rng.normal(size=(6, 3)) @ modes + 0.5).reshape(6, 8, 12)
    visible = mp.random_patch_masks(6, layout.n_patches, 0.5, seed=1)
    pixels = layout.pixel_mask(visible)
    basis = mp.fit_gappy_basis(train, rank=3)
    reconstruction = mp.gappy_reconstruct(basis, test, pixels)
    assert reconstruction.shape == test.shape
    assert mp.relative_l2_error(reconstruction, test, ~pixels) < 1e-4  # percent
    rank, scores = mp.select_pod_rank(train, test, pixels, candidates=[1, 2, 3, 5])
    assert set(scores) == {0, 1, 2, 3, 5} and scores[3] < 1e-4 and scores[rank] <= scores[3]
    assert scores[0] > scores[1] > scores[2] > scores[3]
    # rank candidates above (frames - 1) are skipped: three frames allow at most rank 2
    rank_small, scores_small = mp.select_pod_rank(train[:3], test, pixels, candidates=[1, 2, 3, 5])
    assert set(scores_small) == {0, 1, 2} and rank_small in scores_small


def test_mean_field_used_for_a_single_labelled_frame():
    layout = mp.PatchLayout(ny=8, nx=12, py=4, px=6)
    frames = np.random.default_rng(2).normal(size=(1, 96))
    basis = mp.fit_gappy_basis(frames, rank=0)
    assert isinstance(basis, mp.MeanField) and basis.rank == 0
    pixels = layout.pixel_mask(mp.random_patch_masks(2, layout.n_patches, 0.5, seed=0))
    out = mp.gappy_reconstruct(basis, np.zeros((2, 8, 12)), pixels)
    np.testing.assert_allclose(out[0].ravel(), frames[0])


def test_ridge_readout_dual_and_primal_agree_and_fit_linear_map():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(12, 40))
    weights = rng.normal(size=(40, 7))
    targets = features @ weights + 1.0
    dual = mp.RidgeReadout(1e-8).fit(features, targets)            # 12 samples < 40 features
    np.testing.assert_allclose(dual.predict(features), targets, atol=1e-5)
    wide = rng.normal(size=(200, 40))
    primal = mp.RidgeReadout(1e-3).fit(wide, wide @ weights + 1.0)  # 200 samples > 40 features
    dual_wide = mp.RidgeReadout(1e-3)
    dual_wide.x_mean, dual_wide.y_mean = primal.x_mean, primal.y_mean
    xc = wide - primal.x_mean
    yc = wide @ weights + 1.0 - primal.y_mean
    dual_wide.weights = xc.T @ np.linalg.solve(xc @ xc.T + 1e-3 * np.eye(200), yc)
    np.testing.assert_allclose(primal.weights, dual_wide.weights, atol=1e-6)


def test_labelled_frame_sampling_is_a_subset_of_the_pool():
    pool = np.arange(20, 180)
    for k in (1, 2, 4, 8, 16, 128, 500):
        idx = mp.sample_labelled_frames(pool, k, seed=1)
        assert len(idx) == min(k, len(pool)) and len(set(idx)) == len(idx)
        assert set(idx) <= set(pool.tolist())
    np.testing.assert_array_equal(mp.sample_labelled_frames(pool, 8, 1), mp.sample_labelled_frames(pool, 8, 1))


def test_score_and_label_saving():
    frames = np.ones((2, 4, 6))
    pixels = np.zeros((2, 4, 6), bool)
    pixels[:, :, :3] = True
    prediction = np.full_like(frames, 1.5)
    result = mp.score(prediction, frames, pixels)
    assert np.isclose(result["masked"], 50.0) and np.isclose(result["full"], 50.0 / np.sqrt(2))
    saving = mp.label_saving({1: 30.0, 4: 12.0, 16: 4.0}, {1: 10.0, 4: 6.0})
    assert (saving["pretrained_k"], saving["pretrained_error"], saving["reference_k_to_match"]) == (1, 10.0, 16)
    assert saving["matches"] == {1: 16, 4: 16} and saving["saving"] == {1: 16.0, 4: 4.0}
    assert mp.label_saving({1: 30.0}, {1: 10.0})["reference_k_to_match"] is None


def test_curve_summary_reports_sample_standard_deviation():
    records = [
        {"method": "demo", "k": 2, "seed": 0, "masked": 1.0, "full": 0.5},
        {"method": "demo", "k": 2, "seed": 1, "masked": 2.0, "full": 1.0},
        {"method": "demo", "k": 2, "seed": 2, "masked": 3.0, "full": 1.5},
        {"method": "single", "k": 1, "seed": 0, "masked": 4.0, "full": 2.0},
    ]
    summary = mp.summarize_curves(records)
    assert summary["demo"][2] == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0, "n": 3}
    assert summary["single"][1]["std"] == 0.0


def test_retained_week07_3_evidence_is_consistent():
    path = ROOT / "results/week07_3_pretraining/metrics.json"
    if not path.exists():
        pytest.skip("retained Week 7.3 evidence not present")
    retained = json.loads(path.read_text())
    curves = retained["curves"]
    assert set(curves) == {"pod_target", "pod_pooled", "mae_probe", "mae_scratch", "mae_finetune"}
    protocol = retained["protocol"]
    assert protocol["test_window"] == [210, 281] and protocol["label_pool"] == [0, 160]
    assert protocol["mask_ratio"] == 0.75 and protocol["validation_trajectory"] == 105
    assert not protocol["quick"], "retained evidence must come from the full protocol"
    ks = [str(k) for k in protocol["ks"]]
    for method in curves:
        assert set(curves[method]) == set(ks) and all(curves[method][k]["n"] == len(protocol["seeds"]) for k in ks)
    # retained findings: the pretrained model beats the same architecture from scratch at every k,
    # and the classical baseline with matched information wins on this low-rank wake
    assert all(curves["mae_finetune"][k]["mean"] < curves["mae_scratch"][k]["mean"] for k in ks)
    assert retained["zero_shot"]["pod_transfer"] < retained["zero_shot"]["mae_zero_shot"]
    assert all(curves["pod_pooled"][k]["mean"] < curves["mae_finetune"][k]["mean"] for k in ks)
    for relative, digest in retained["source_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_masked_autoencoder_learns_a_fixed_completion_task():
    torch = pytest.importorskip("torch", reason="the MAE test needs the optional PyTorch dependency")
    torch.set_num_threads(1)
    layout = mp.PatchLayout(ny=8, nx=12, py=4, px=6)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2 * np.pi, 12)[None, None, :]
    phase = rng.uniform(0, 2 * np.pi, size=(64, 1, 1))
    frames = np.sin(x + phase) * np.ones((1, 8, 1))
    patches = layout.patchify(frames)
    model = mp.build_mae(layout, dim=16, depth=1, heads=2, decoder_dim=16, decoder_depth=1, seed=0)
    assert sum(p.numel() for p in model.parameters()) < 20_000
    before = mp.evaluate_masked_loss(model, patches, mp.random_patch_masks(64, 4, 0.5, seed=9))
    history, best_step = mp.train_masked_model(model, patches, layout, mask_ratio=0.5, steps=150, batch_size=16,
                                               lr=3e-3, seed=0, val_patches=patches,
                                               val_visible=mp.random_patch_masks(64, 4, 0.5, seed=9), val_every=25)
    after = history[-1]["val_loss"]
    assert history[0]["step"] == 0 and np.isclose(history[0]["val_loss"], before)
    assert after < before and best_step >= 25
    visible = mp.random_patch_masks(64, 4, 0.5, seed=11)
    prediction = mp.predict_masked(model, patches, visible, layout)
    pixels = layout.pixel_mask(visible)
    np.testing.assert_allclose(prediction[pixels], frames[pixels])    # visible pixels are copied
    features = mp.encoder_features(model, patches, visible)
    assert features.shape == (64, 4 * 16)
    assert np.all(features.reshape(64, 4, 16)[~visible] == 0)        # masked tokens never enter the encoder
    pooled = mp.encoder_features(model, patches, visible, pooling="visible_mean")
    assert pooled.shape == (64, 16) and np.isfinite(pooled).all()
    with pytest.raises(ValueError):
        mp.encoder_features(model, patches, visible, pooling="unknown")
