# Archived phase/POD cylinder-wake failure baseline

> **Status:** retained failure evidence. After this `Re=100` field was inspected
> to design the corrected CNN, it became a validation case and must no longer
> be described as an untouched blind result.

`blind_re100_lbm_vs_neural.mp4` is an executed 1920x1080 comparison of
dimensionless vorticity from a withheld `Re=100` D2Q9--TRT LBM trajectory with
Bouzidi interpolated bounce-back on the analytical circular wall, and the
corresponding phase-conditioned neural POD prediction.  The full
`Re=100` case is absent from the POD basis, input/target scalers, and neural
training.  Training uses complete trajectories at
`Re=60,80,90,110,120,140`.

The bottom trace locates every displayed snapshot on the blind LBM lift
history.  The right panel is the signed neural-minus-LBM vorticity error.  The
poster, exact split, frozen architecture, physical diagnostics, field errors,
non-neural harmonic-POD baseline, numerical card, runtime, and claim scope are
retained beside the video.

Regenerate the evidence with:

```bash
python qa/run_cylinder_blind_video.py --workers 7
```

The 16.70% vorticity error and visibly diffuse far wake are intentionally
preserved as the motivation for the four-frame multi-scale CNN in
`results/cylinder_cnn/`. This is an educational, phase-conditioned Reynolds
interpolation. Phase is measured from the Re=100 LBM lift signal, so the model
is not an autonomous time predictor. The LBM target uses a stable low-Mach
teaching grid and is not claimed to be a grid-converged DNS benchmark.
