#!/usr/bin/env python3
"""Actual-solver checks plus regression checks for costly job failure modes."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from unittest import mock

spec = importlib.util.spec_from_file_location("step_pilot", Path(__file__).with_name("pilot.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    with tempfile.TemporaryDirectory(prefix="sparta-step-verification-") as work:
        root = Path(work)
        for name, ratio, pr in [("h25", .25, 2), ("h50", .5, 2), ("h75", .75, 2), ("equilibrium", .5, 1)]:
            out = root / name
            extra = {"ppc": 40, "warmup": 1000, "sample": 1000} if name == "equilibrium" else {}
            pilot.generate(out, smoke=True, ratio=ratio, pressure_ratio=pr, **extra)
            with open(out / "in.step") as inp, open(out / "solver.stdout", "w") as stdout:
                subprocess.run([binary], cwd=out, stdin=inp, stdout=stdout, stderr=subprocess.STDOUT, check=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = pilot.report(out)
            assert result["stuck_particles"] == 0
            if name == "equilibrium":
                # A coarse finite-particle equilibrium control, not a flow-convergence test.
                p_target = json.loads((out / "case.json").read_text())["p_out_Pa"]
                assert all(abs(p / p_target - 1) < .12 for p in result["boundary_adjacent_mean_pressure_Pa"])
                assert abs(sum(b["bulk_u_m_s"] for b in result["blocks"]) / len(result["blocks"])) < 15
            print(f"SOLVER_CHECK_PASS {name} cells={result['fluid_cells']} final={result['final_step']}")
        bad = root / "h50"
        original = (bad / "grid.final.gz").read_bytes()
        (bad / "grid.final.gz").write_bytes(original[:80])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pilot.report(bad)
        except (EOFError, OSError, ValueError):
            print("TRUNCATED_OUTPUT_REJECTED")
        else:
            raise AssertionError("Truncated final dump was accepted")
        with mock.patch.object(pilot.subprocess, "check_output", side_effect=["101\n", "102;unity\n", "103\n"]) as sbatch:
            pilot.submit(Path(__file__).parent, root / "submission", "a" * 40, "test_account", "openmpi/5.0.3", 16)
            calls = sbatch.call_args_list
            assert "--dependency=afterok:101" in calls[1].args[0]
            assert "--dependency=afterok:102" in calls[2].args[0]
            assert all("--nodes=1" in c.args[0] and "--export=ALL" in c.args[0] for c in calls)
            assert "--kill-on-invalid-dep=yes" in calls[1].args[0]
        print("SUBMISSION_DEPENDENCIES_PASS")
        with mock.patch.object(pilot.subprocess, "check_output", side_effect=["201\n", subprocess.CalledProcessError(1, "sbatch")]):
            try:
                pilot.submit(Path(__file__).parent, root / "partial", "b" * 40, "test_account", "openmpi/5.0.3", 16)
            except subprocess.CalledProcessError:
                out = Path((root / "partial" / "LATEST_SPARTA_STEP_PILOT").read_text().strip())
                assert json.loads((out / "submission.json").read_text())["jobs"] == {"build": "201"}
            else:
                raise AssertionError("Failed submission was hidden")
        print("PARTIAL_SUBMISSION_RECORDED")
    print("LOCAL_VERIFICATION_COMPLETE")


if __name__ == "__main__":
    main()
