# Cavity circulation: physical time 0–100

[Watch the three matched-time movies](https://ehsan-roohi.github.io/FlowMLLab/cavity-continuation/) · [Original startup lesson, 0–10](https://ehsan-roohi.github.io/FlowMLLab/cavity-startup/) · [Stationarity guide](../../docs/CAVITY_STARTUP_STATIONARITY.md)

Compare speed and instantaneous streamlines, signed vorticity and zero-mean pressure at **Re = 100, 400, 1000, 3200, 5000, 7500 and 10000**. These previews extend the original startup trajectories using their native velocity and pressure states at the actual physical time.

| Shared condition | Setting |
|---|---|
| Domain and initial state | Unit square; fluid starts from rest |
| Walls | Impulsive top lid moving right at U = 1; other walls stationary; all impermeable |
| Grid and timestep | 128 × 128 MAC cells; Δt⋆ = 0.01 |
| Coupling and convection | PIMPLE, 3 outer / 2 pressure corrections; central convection |
| Physical interval | t⋆ = tU/L from 0 to 100 |
| Displayed frames | 201 actual common times at intervals of 0.5 |
| Playback | 8 frames/s; physical timestamps remain unchanged |
| Pressure gauge | Zero cell-area mean; pressure scaled by ρU² |

A common endpoint does not establish that every case is steady. The 128-cell high-Re fields remain provisional and mesh-sensitive. Use the frozen physical-window criteria and separate coupling, timestep and grid tests described in the guide. These two-dimensional movies do not establish three-dimensional stability.

## View and compare

- [Speed and instantaneous streamlines](media/cavity_continuation_speed.mp4) · [t⋆ = 100 poster](media/speed_t100.png).
- [Signed vorticity](media/cavity_continuation_omega.mp4) · [t⋆ = 100 poster](media/omega_t100.png).
- [Zero-mean pressure](media/cavity_continuation_p.mp4) · [t⋆ = 100 poster](media/p_t100.png).

Speed uses a linear scale. Signed vorticity uses a symmetric-log scale, linear for |ωL/U| ≤ 1; pressure is linear within |p/(ρU²)| ≤ 0.01 and symmetric-log outside. Each field has fixed cross-case and cross-time limits covering all retained native extrema, including the original finer-cadence startup samples. No temporal interpolation, repeated frozen tail, smoothing or clipping is used. Streamlines describe the velocity at an instant, not particle paths.

Pause at times 10, 25, 50 and 100, then compare physical profiles and signed probe histories. A fixed upper-left probe is not a tracked vortex center. Follow the [five stationarity investigations](../../docs/CAVITY_STARTUP_STATIONARITY.md#five-student-investigations) to distinguish a changing field, a fixed-grid candidate and numerical accuracy. The [original notebook](../../notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb) and all its prior exercises remain unchanged.

## Evidence and reproduction

[Public asset manifest](media/preview_manifest.json) records every selected file's SHA-256 and size, the common time axis, encoded-video checks and the retained archive identity. [Full render receipt](media/movies.json) records exact native-frame identities for all seven cases. That receipt also names additional stills and PDFs kept in the external archive; they are not implied to be part of this public subset.

The separately retained research archive has SHA-256 `2b21477f4a044854f0886f654bfc28a95bfb7e09bf211b98afacf485f2ab1a01` (66,296,857 bytes; 88 verified members). It contains the full diagnostic reports and exact remote rendering/analysis scripts. Its renderer hash is recorded explicitly; the currently tracked renderer is not claimed to be byte-identical after source text normalization. No live job status or future steady-state outcome is inferred from this immutable movie receipt.

From the repository root, install an already verified source bundle with:

```bash
python qa/build_cavity_continuation_viewer.py --source /path/to/verified/media_next
```

Rebuild the viewer from its retained public files, or check it without rewriting:

```bash
python qa/build_cavity_continuation_viewer.py
python qa/build_cavity_continuation_viewer.py --check
python qa/build_site.py --output tmp/pages-site
python qa/check_site.py tmp/pages-site
```

These commands package and verify retained evidence; they do not run CFD or train a network. The [original 0–10 data and lesson](../cavity_startup/README.md) remain independent of this longer preview.
