# Week 4.2 Unity qualification protocol

This protocol executes the lid-driven-cavity PINN from Christopher J. McDevitt's
DeepPlasma repository, used with his permission. The external source is not
redistributed: Unity checks out commit `fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b`
and verifies SHA-256 `0a917522a757a442647e9c255b98b8b6d932692a855dc3d6ec5596c2a8507b18`.

The first submitted job is a three-step GPU smoke test at Re=100. Its purpose is
to verify imports, CUDA float64 automatic differentiation, the dense
SSBroyden2 optimizer, plotting, and evidence capture. It is explicitly not a
scientific accuracy result. A long Re=5,000 job may be submitted only after the
smoke evidence has been inspected and a matching CFD reference and acceptance
thresholds have been frozen.

The independent audit reports unmasked momentum residuals, continuity, hard-wall
errors, environment versions, hardware, source identity, and job identity. A
contour is watermarked as unvalidated until comparison with the matched CFD
reference is complete.
