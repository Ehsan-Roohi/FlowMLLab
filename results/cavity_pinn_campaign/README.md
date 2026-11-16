# Label-free PINN audit across all seven Ghia Reynolds numbers

**The 42-run finite-budget baseline failed its scientific acceptance gates.**
All saved outputs passed integrity checks. Exact wall values and continuity did
not imply a correct interior circulation: none passed both direct Ghia-profile
criteria or the independent momentum-residual criterion.

- [Physical report, all seeds and signed errors](figures/README.md)
- [Seven-Re centerline comparison PDF](figures/all_reynolds_centerlines.pdf)
- [CFD/PINN fields and streamline comparison PDF](figures/representative_physical_fields.pdf)
- [Frozen results and complete metrics](evaluation_manifest.json)
- [Actual sampling coverage and the 14 repeated collar objectives](sampling_coverage.json)
- [Measured compute-cost experiment](cost_study/README.md)
- [New student notebook](../../notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb)
- [Protocol and information budget](../../docs/CAVITY_PINN_CAMPAIGN_PROTOCOL.md)

The three seeds are 7, 23, 61, with collars 0.02W and 0.01W at each of
Re = 100, 400, 1000, 3200, 5000, 7500, 10000. Inputs are coordinates and the declared case;
no CFD or Ghia labels enter training or endpoint selection. All projected
predictions were frozen before reference scoring. The square-cavity pressure
field has no retained CFD pressure reference and is not assigned a pressure
accuracy score.

These case-specific PINNs are distinct from the previously completed
[supervised POD-DeepONet](../cavity_field_deeponet/README.md).
Native AD vorticity and transported CFD vorticity remain in the numerical score
record; figures that use a common centered curl of velocity are explicitly
labelled as a separate derived comparison. CFD mesh sensitivity remains part
of the interpretation, especially at high Reynolds number.
