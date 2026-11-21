"""Freeze a policy, then continue verified native cavity trajectories on CPU.

Examples (paths are supplied by the caller, never guessed HPC workspaces)::

    python qa/continue_cavity_startup.py policy --reynolds 1000 --output policy.json
    python qa/continue_cavity_startup.py run --seed fine128/re1000.npz \
        --seed-receipt fine128/re1000.json --policy policy.json \
        --output continuation/re1000 --target-time 25 --progress

Repeat the same run command with absolute targets 50/100 to resume. A signal
requests a checkpoint after the current complete physical step. Existing locks
or checksum/source/policy mismatch are refusals; do not launch a second writer.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab.cavity_startup_continuation import (
    continue_case, production_policy, atomic_json, fingerprint,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    policy = commands.add_parser("policy", help="Write the agreed pre-run engineering policy for explicit review/freezing")
    policy.add_argument("--reynolds", type=int, choices=[100, 400, 1000, 3200, 5000, 7500, 10000], required=True)
    policy.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--seed", type=Path, required=True)
    run.add_argument("--seed-receipt", type=Path, required=True)
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--target-time", type=float, required=True)
    run.add_argument("--stop-on-candidate", action="store_true",
                     help="Stop only after all frozen complete-window gates pass; confirmation remains separate")
    run.add_argument("--progress", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "policy":
        if args.output.exists():
            raise FileExistsError("Refusing to replace a frozen policy; choose/review its existing bytes")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        document = production_policy(args.reynolds)
        atomic_json(args.output, document)
        print(json.dumps({"policy": str(args.output), "canonical_sha256": fingerprint(document)}, indent=2))
        return 0
    document = json.loads(args.policy.read_text(encoding="utf-8"))
    stopping = {"signal": None}
    def request_stop(signum, _frame):
        stopping["signal"] = signum
    previous = {}
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            previous[sig] = signal.signal(sig, request_stop)
    try:
        report = continue_case(args.seed, args.seed_receipt, document, args.output, args.target_time,
                               stop_requested=lambda: stopping["signal"] is not None,
                               stop_on_candidate=args.stop_on_candidate, progress=args.progress)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if stopping["signal"] is not None:
        return 75  # temporary/scheduler stop, complete checkpoint retained
    return 3 if report["reason"].startswith("frozen_storage_cap") else 0


if __name__ == "__main__":
    raise SystemExit(main())
