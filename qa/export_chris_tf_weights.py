"""Export ordered dense-layer arrays from a retained TensorFlow checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--width", type=int, default=64)
    args = ap.parse_args()
    widths = [5] + [args.width] * 6 + [3]
    reader = tf.train.load_checkpoint(args.checkpoint)
    shapes = reader.get_variable_to_shape_map()
    kernels = [(name, tuple(shape)) for name, shape in shapes.items()
               if len(shape) == 2 and "Adam" not in name and "optimizer" not in name.lower()]
    arrays, names, used = {}, [], set()
    for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        candidates = sorted((name for name, shape in kernels
                             if shape == (fan_in, fan_out) and name not in used),
                            key=lambda name: (name.count("/"), name))
        if not candidates:
            raise RuntimeError(f"Cannot identify kernel {(fan_in, fan_out)}")
        kernel_name = candidates[0]
        used.add(kernel_name)
        stem = kernel_name.rsplit("/", 1)[0]
        bias_names = [name for name, shape in shapes.items()
                      if tuple(shape) == (fan_out,) and name.startswith(stem + "/")
                      and "Adam" not in name]
        if len(bias_names) != 1:
            raise RuntimeError(f"Cannot identify bias for {kernel_name}: {bias_names}")
        arrays[f"kernel_{index}"] = reader.get_tensor(kernel_name)
        arrays[f"bias_{index}"] = reader.get_tensor(bias_names[0])
        names.append({"kernel": kernel_name, "bias": bias_names[0]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "architecture": widths, "variables": names}, indent=2
    ) + "\n")


if __name__ == "__main__":
    main()
