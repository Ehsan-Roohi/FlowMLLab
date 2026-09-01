"""Interactive, read-only viewer for retained FlowMLLab blind evidence."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab.demo import available_blind_reynolds, load_blind_demo_case


st.set_page_config(
    page_title="FlowMLLab blind-case explorer",
    page_icon="🌊",
    layout="wide",
)

st.title("FlowMLLab blind-case explorer")
st.markdown(
    "Compare a retained three-seed POD–DeepONet prediction with its untouched "
    "CFD reference. Every value shown here comes from the versioned release evidence."
)

left, right = st.columns([1, 2])
with left:
    reynolds_values = available_blind_reynolds(ROOT)
    reynolds = st.select_slider(
        "Blind Reynolds number",
        options=reynolds_values,
        value=reynolds_values[1],
        format_func=lambda value: f"Re = {value:g}",
    )
with right:
    field_name = st.radio(
        "Displayed field",
        (
            "Velocity magnitude",
            "Horizontal velocity",
            "Vertical velocity",
            "Pressure",
        ),
        horizontal=True,
    )

case = load_blind_demo_case(reynolds, ROOT)

metric_columns = st.columns(5)
metric_columns[0].metric("Blind relative $L_2$", f"{100 * case.relative_l2_uv:.4f}%")
metric_columns[1].metric("Pressure relative $L_2$", f"{100 * case.relative_l2_p:.4f}%")
metric_columns[2].metric("Maximum vector error", f"{case.maximum_vector_error:.3e}")
metric_columns[3].metric("Wall RMS error", f"{case.wall_rms_error:.1e}")
metric_columns[4].metric("Recorded ensemble inference", f"{case.inference_ms:.3f} ms")

if field_name == "Pressure":
    reference = case.reference_p
    prediction = case.prediction_p
    symbol = "$p^*$ (zero-mean gauge)"
    cmap = "RdBu_r"
elif field_name == "Horizontal velocity":
    reference = case.reference_u
    prediction = case.prediction_u
    symbol = "$u/U_{lid}$"
    cmap = "coolwarm"
elif field_name == "Vertical velocity":
    reference = case.reference_v
    prediction = case.prediction_v
    symbol = "$v/U_{lid}$"
    cmap = "coolwarm"
else:
    reference = case.reference_speed
    prediction = case.prediction_speed
    symbol = "$|\\mathbf{u}|/U_{lid}$"
    cmap = "YlGnBu_r"

absolute_error = np.abs(prediction - reference)
field_limit = max(float(np.max(np.abs(reference))), float(np.max(np.abs(prediction))))
if field_name == "Velocity magnitude":
    vmin, vmax = 0.0, field_limit
else:
    vmin, vmax = -field_limit, field_limit
field_norm = (
    SymLogNorm(linthresh=0.02, vmin=-field_limit, vmax=field_limit, base=10)
    if field_name == "Pressure"
    else None
)

figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
extent = [float(case.x[0]), float(case.x[-1]), float(case.y[0]), float(case.y[-1])]
panels = (
    (reference, f"CFD reference: {symbol}", cmap, vmin, vmax, field_norm),
    (
        prediction,
        f"POD–DeepONet: {symbol}",
        cmap,
        vmin,
        vmax,
        field_norm,
    ),
    (
        absolute_error, "Absolute error", "magma",
        0.0, float(np.max(absolute_error)), None,
    ),
)
for axis, (values, title, palette, panel_min, panel_max, panel_norm) in zip(axes, panels):
    color_scale = (
        {"norm": panel_norm}
        if panel_norm is not None
        else {"vmin": panel_min, "vmax": panel_max}
    )
    image = axis.imshow(
        values, origin="lower", extent=extent, cmap=palette,
        interpolation="nearest", **color_scale,
    )
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("$x/L$")
    axis.set_ylabel("$y/L$")
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

st.pyplot(figure, clear_figure=True, width="stretch")

st.info(
    "This viewer replays the frozen blind cases Re = 175, 275, and 375. "
    "It intentionally does not generate predictions at unvalidated intermediate values. "
    "Pressure is a direct output of a separately scaled POD–DeepONet pressure head. "
    "Its diverging colors use a symmetric-log normalization so the interior structure "
    "remains visible without clipping the corner extrema."
)

with st.expander("Protocol and interpretation"):
    st.markdown(
        """
- **Development cases:** Re = 100, 150, 200, 225, 250, 300, 350, and 400.
- **Blind cases:** Re = 175, 275, and 375; the selector above never exposes a training case.
- **Prediction:** retained mean of seeds 690, 691, and 692. The velocity head uses a rank-3 POD trunk and `(32, 32)` branch; the direct-pressure head uses a separate rank-3 pressure trunk and an 8-neuron branch.
- **Physics check:** the output transform preserves cavity-wall velocities exactly; aggregate and local errors remain visible.
- **Pressure:** the network learns the stored zero-mean pressure labels directly. Separate trunks prevent pressure scaling from changing the divergence-free velocity basis; agreement with these labels does not constitute an independent validation of the pressure-recovery method used to create them.
- **Timing:** the displayed 0.804 ms is the recorded three-seed ensemble timing. The 8.45 s CFD timing applies to Re = 275 on the recorded CPU and is not silently generalized to other hardware.
        """
    )

st.markdown(
    "[Open the 20-minute Colab](https://colab.research.google.com/github/"
    "Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P0_Project_Setup.ipynb) · "
    "[Inspect the evidence](https://github.com/Ehsan-Roohi/FlowMLLab/tree/main/"
    "results/pod_deeponet) · "
    "[Read the full notebook guide](https://github.com/Ehsan-Roohi/FlowMLLab/"
    "blob/main/notebooks/README.md)"
)
