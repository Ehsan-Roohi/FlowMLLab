# Week 9 — Roohi--Mahdavi DeepONet case studies

This optional research-to-classroom extension contains two CPU-friendly labs:

1. [`W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb`](W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb)
   turns the May 2026 micro-step paper into a lesson on complete-geometry
   splits, recirculation-zone metrics, and validation-only loss selection. The
   paper's DSMC fields are not public, so the executable field experiment is
   explicitly manufactured and kept separate from retained article metrics.
2. [`W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb`](W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb)
   uses a compact attributed derivative of all 15 public micro-nozzle DSMC
   snapshots. Students reproduce the physical/shock-centered density POD,
   freeze pressures 16, 25, and 30 kPa, and test a small POD-trunk/neural-branch
   model plus a training-only shock locator.

Both notebooks have direct Colab launchers and run on CPU. Rebuild them from
the reviewable source with:

```bash
python notebooks/week09/make_week9_notebooks.py
```

The full evidence and licensing contract is in
[`results/mahdavi_deeponet/`](../../results/mahdavi_deeponet/).
