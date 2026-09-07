from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rectangular_runner_has_physical_scaling_and_two_optimizers():
    text = (ROOT / "qa" / "run_week13_rectangular_pinn.py").read_text(encoding="utf-8")
    assert "grad[:, 1:2] / aspect" in text
    assert "uee / aspect**2" in text
    assert "gp[:, 1:2] / aspect" in text
    assert "torch.optim.Adam" in text
    assert "module.SSBroyden2" in text
    assert "top_corner_momentum_x_rms" in text
    assert "heldout_momentum_rms" in text
    assert "wall_audit" in text
    assert "UPSTREAM_COMMIT" in text
    assert "residual-audited-no-field-reference" in text
    assert "field-qualified-square-case" in text
    assert "field-comparison-failed-square-case" in text
    assert 'Path("COMPLETE").write_text' in text
    assert 'immutable = ("reynolds_number"' in text


def test_matrix_job_is_restartable_gpu_preempt_array():
    text = (ROOT / "qa" / "unity_week13_pinn_matrix.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu-preempt" in text
    assert "#SBATCH --gres=gpu:a100:1" in text
    assert "#SBATCH --array=0-3" in text
    assert "#SBATCH --requeue" in text
    assert "--resume" in text
    for case in ('"100 1"', '"400 1"', '"100 2"', '"400 2"'):
        assert case in text


def test_protocol_does_not_overclaim_deep_cavity_validation():
    text = (ROOT / "qa" / "WEEK13_PINN_MATRIX_PROTOCOL.md").read_text(encoding="utf-8")
    assert "not** labelled field-validated" in text
    assert "top-corner residual" in text
    assert "SIGUSR1" in text


def test_harvester_excludes_checkpoints_and_normalizes_legacy_labels():
    text = (ROOT / "qa" / "harvest_week13_results.py").read_text(encoding="utf-8")
    assert "checkpoint.pt" not in text
    assert "residual-audited-no-field-reference" in text
    assert "field-qualified-square-case" in text
