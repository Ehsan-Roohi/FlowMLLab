# Week 9 — Roohi--Mahdavi DeepONet case studies

This optional research-to-classroom extension contains two CPU-friendly labs:

1. [`W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb`](W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb)
   turns the May 2026 micro-step paper into a lesson on complete-geometry
   splits, recirculation-zone metrics, and validation-only loss selection. The
   executable experiment uses compact derivatives of all nine real DSMC
   height fields, with the seven learning/validation cases physically separate
   from the two held-out tests.
2. [`W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb`](W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb)
   uses compact full-field and centerline derivatives of all 15 public
   micro-nozzle DSMC snapshots. Students reproduce the physical/shock-centered
   density POD, freeze pressures 16, 25, and 30 kPa, and generate new 2-D
   density, velocity, Mach, and pressure predictions with a POD trunk and
   neural branch.

Both notebooks have direct Colab launchers and run on CPU. Rebuild them from
the reviewable source with:

```bash
python notebooks/week09/make_week9_notebooks.py
```

The full evidence and licensing contract is in
[`results/mahdavi_deeponet/`](../../results/mahdavi_deeponet/).
