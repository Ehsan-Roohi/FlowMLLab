from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runner_pins_upstream_source_and_requires_cuda():
    text = (ROOT / "qa" / "run_week42_deepplasma.py").read_text(encoding="utf-8")
    assert "fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b" in text
    assert "0a917522a757a442647e9c255b98b8b6d932692a855dc3d6ec5596c2a8507b18" in text
    assert "CUDA GPU is required" in text
    assert "raw_residuals" in text


def test_slurm_requests_a100_and_runs_dependency_gate_first():
    text = (ROOT / "qa" / "unity_week42_deepplasma.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:a100:1" in text
    assert "torch.cuda.is_available" in text
    assert text.index("pip check") < text.index('"$FLOWML_PINN_PYTHON" qa/run_week42_deepplasma.py')
    assert "dtype=torch.float64" in text
    assert "FLOWML_PINN_DEPS" in text


def test_optional_dependency_versions_are_pinned():
    lines = (ROOT / "qa" / "requirements-week42-unity.txt").read_text(encoding="utf-8")
    assert "scikit-optimize==0.10.2" in lines
    assert "pytorch-optimizer==3.10.1" in lines
