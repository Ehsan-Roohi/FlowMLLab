from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runner_pins_upstream_source_and_requires_cuda():
    text = (ROOT / "qa" / "run_week42_deepplasma.py").read_text(encoding="utf-8")
    assert "fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b" in text
    assert "391a2174cb9f6e8863c14d7b350077e54209134de9f85a17719c398146f91458" in text
    assert "CUDA GPU is required" in text
    assert "raw_residuals" in text
    assert '"qualification": (100.0, 16384, 300)' in text
    assert '"u_centerline_relative_l2": 0.10' in text


def test_slurm_requests_a100_and_runs_dependency_gate_first():
    text = (ROOT / "qa" / "unity_week42_deepplasma.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu-preempt" in text
    assert "#SBATCH --gres=gpu:a100:1" in text
    assert "check_week42_environment.py" in text
    assert text.index("check_week42_environment.py") < text.index('"$FLOWML_PINN_PYTHON" qa/run_week42_deepplasma.py')
    gate = (ROOT / "qa" / "check_week42_environment.py").read_text(encoding="utf-8")
    assert 'dtype=torch.float64' in gate
    assert '"status": "READY"' in gate
    assert "FLOWML_PINN_DEPS" in text
    assert "#SBATCH --requeue" in text
    assert "FLOWML_PINN_RUN_ID" in text
    assert "FLOWML_PINN_REFERENCE" in text
    assert "FLOWML_PINN_STEPS" in text
    runner = (ROOT / "qa" / "run_week42_deepplasma.py").read_text(encoding="utf-8")
    for required in ("optimizer.state", "collocation_points", "cuda_rng", "os.replace"):
        assert required in runner
    assert 'saved_config.pop("optimizer_steps", None)' in runner
    assert 'map_location="cpu"' in runner
    assert 'value.to(module.device)' in runner


def test_optional_dependency_versions_are_pinned():
    lines = (ROOT / "qa" / "requirements-week42-unity.txt").read_text(encoding="utf-8")
    assert "scikit-optimize==0.10.2" in lines
    assert "pytorch-optimizer==3.10.1" in lines
    assert "sympy==1.13.1" in lines
