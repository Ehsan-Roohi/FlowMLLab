"""Convert a retained SSBroyden parameter vector to an evaluation checkpoint.

This reconstructs the *same* TensorFlow-v1/DeepXDE graph used by
``evaluate_chris_deep_original.py``. DeepXDE's SciPy adapter packs
``tf.trainable_variables()`` in list order, flattening each tensor in C order;
we reverse that exact mapping and verify it before saving. The training run is
never modified. The author-supplied source remains private on Unity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from evaluate_chris_deep_original import load_source, normalized, vortex_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--hidden-width", type=int, required=True)
    parser.add_argument("--re", type=float, required=True)
    parser.add_argument("--depth", type=float, required=True)
    parser.add_argument("--tri", type=float, default=0.0)
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--ny", type=int, default=1001)
    args = parser.parse_args()

    architecture = [5] + [args.hidden_width] * 6 + [3]
    provenance = json.loads(args.provenance.read_text())
    if provenance.get("architecture") != architecture:
        raise ValueError("Network architecture differs from training provenance")
    parameters = json.loads(args.parameters.read_text())
    source = load_source(args.source.resolve())
    for name in ("ReMin", "ReMax", "DMin", "DMax"):
        value = float(parameters[name])
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite {name}")
        setattr(source, name, value)
        source.pde.__globals__[name] = value
        source.output_transform_cavity_flow.__globals__[name] = value
    normalized(args.re, source.ReMin, source.ReMax, "Re")
    normalized(args.depth, source.DMin, source.DMax, "depth")
    normalized(args.tri, source.triMin, source.triMax, "tri")

    dde = source.dde
    import tensorflow.compat.v1 as tf

    geometry = dde.geometry.Rectangle([0] * 5, [1] * 5)
    net = dde.maps.FNN(architecture, "tanh", "Glorot normal")
    net.apply_output_transform(source.output_transform_cavity_flow)
    data = dde.data.PDE(geometry, source.pde, [], num_domain=2,
                        num_boundary=0, num_test=2,
                        train_distribution="Hammersley")
    model = dde.Model(data, net)
    model.compile("adam", lr=source.lr, loss=["MSE"] * 2,
                  loss_weights=[1, 1])

    variables = tf.trainable_variables()
    shapes = [tuple(int(d) for d in v.shape.as_list()) for v in variables]
    sizes = [int(np.prod(shape)) for shape in shapes]
    vector = np.load(args.vector, allow_pickle=False)
    if vector.ndim != 1 or len(vector) != sum(sizes):
        raise ValueError(f"Vector length {vector.shape} != graph size {sum(sizes)}")
    if not np.isfinite(vector).all():
        raise ValueError("Parameter vector has nonfinite values")

    offset = 0
    assignments = []
    for variable, shape, size in zip(variables, shapes, sizes):
        value = vector[offset:offset + size].reshape(shape)
        assignments.append(tf.assign(variable, value.astype(
            variable.dtype.as_numpy_dtype, copy=False)))
        offset += size
    model.sess.run(assignments)
    packed = np.concatenate([value.ravel() for value in model.sess.run(variables)])
    if not np.array_equal(packed, vector):
        raise ValueError("TensorFlow assignment failed exact vector round-trip")

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = model.save(str(args.output / "interim"), verbose=0)
    audit = {
        "status": "interim SSB parameter vector, not an optimizer convergence result",
        "source": str(args.source),
        "vector": str(args.vector),
        "vector_sha256": hashlib.sha256(args.vector.read_bytes()).hexdigest(),
        "checkpoint": checkpoint,
        "architecture": architecture,
        "parameter_count": len(vector),
        "case": {"Re": args.re, "depth_over_width": args.depth, "tri": args.tri},
        "mapping": "DeepXDE TF1 SciPy tf.trainable_variables(), C-order flattened",
    }
    (args.output / "conversion.json").write_text(json.dumps(audit, indent=2) + "\n")

    x = np.linspace(0, 1, args.nx)
    y = np.linspace(0, args.depth, args.ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), (yy / args.depth).ravel(),
                              np.full(xx.size, normalized(args.re, source.ReMin,
                                                            source.ReMax, "Re")),
                              np.full(xx.size, normalized(args.depth, source.DMin,
                                                            source.DMax, "depth")),
                              np.full(xx.size, normalized(args.tri, source.triMin,
                                                            source.triMax, "tri"))))
    prediction = np.concatenate([
        model.predict(points[start:start + 8192])
        for start in range(0, len(points), 8192)
    ]).reshape(args.ny, args.nx, 3)
    if not np.isfinite(prediction).all():
        raise ValueError("Interim vector predicts nonfinite fields")
    u, v, p = np.moveaxis(prediction, -1, 0)
    psi = cumulative_trapezoid(u, y, axis=0, initial=0)
    omega = np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)
    divergence = np.gradient(u, x, axis=1) + np.gradient(v, y, axis=0)
    np.savez_compressed(args.output / "field.npz", x=x, y=y, u=u, v=v,
                        p=p, psi=psi, omega=omega)
    field_audit = {
        "status": audit["status"],
        "case": audit["case"],
        "grid": {"nx": args.nx, "ny": args.ny},
        "vector_sha256": audit["vector_sha256"],
        "max_abs_divergence_fd": float(np.max(np.abs(divergence))),
        "wall_max_abs_velocity": {
            "bottom": float(np.max(np.hypot(u[0], v[0]))),
            "left": float(np.max(np.hypot(u[:, 0], v[:, 0]))),
            "right": float(np.max(np.hypot(u[:, -1], v[:, -1]))),
        },
        "vortex_candidates": vortex_candidates(psi, x, y),
        "limitations": [
            "SSB optimizer has not reported convergence.",
            "Streamfunction is reconstructed by integrating u from the bottom wall.",
            "Finite-difference divergence includes wall differentiation error.",
        ],
    }
    (args.output / "audit.json").write_text(json.dumps(field_audit, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(10, 11), constrained_layout=True)
    axes[0].contour(x, y, psi, levels=35, colors="black", linewidths=.65)
    axes[0].set_title("Streamfunction")
    speed = np.hypot(u, v)
    image = axes[1].pcolormesh(x, y, speed, cmap="viridis", shading="auto")
    fig.colorbar(image, ax=axes[1], shrink=.5)
    axes[1].set_title("Speed")
    limit = float(np.quantile(np.abs(omega), .995))
    image = axes[2].pcolormesh(x, y, omega, cmap="RdBu_r", shading="auto",
                               vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=axes[2], shrink=.5)
    axes[2].set_title("Vorticity")
    for axis in axes:
        axis.set(xlabel="x/W", ylabel="y/W", aspect="equal")
    fig.suptitle("Interim PINN vector, Re=100, D/W=5; not converged")
    fig.savefig(args.output / "interim_fields.png", dpi=200)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
