# Week 16 publication status

This development branch contains the v1.9.0 reference extension. Main and the final release have not been updated. **The complete release remains blocked.**

## Accepted retained-checkpoint comparison

Eight fresh computations with the pinned official SU2 8.0.1 binary passed numerical convergence, positive pressure/density, maximum total-enthalpy deviation of 10%, and a stagnation-density bound. The existing portable neural checkpoint was unchanged. Aggregate errors are 7.1499% for the waveform, 2.6343% for peak pressure and 2.0615% for drag; the worst individual waveform error is 15.9701%. Aggregate acceptance does not mean every case is below 10%. See `results/week16_lowboom/reference/weakwall_checkpoint_audit.json` and workflow 35942843724.

This is a retrospective comparison on eight known teaching geometries, not blind testing, experimental validation of the network, or reproduction of the Beihang aircraft. Historical training labels still come from SU2 8.5.0; unchanged predictions are assessed against new physically checked 8.0.1 reference fields.

## Clean student dataset

All 44 original geometries were regenerated using the pinned official SU2 8.0.1 binary in workflow 35944299803. Every case passed numerical convergence and full-field physical checks. The maximum total-enthalpy deviation over the complete set was 7.4271%. The original 24/6/8/6 geometry split is unchanged. A new fixed-architecture model passed workflow 35944888983 and an independent no-fit audit: finer CFD waveform/peak/drag errors are 7.2036%/2.2373%/2.2740%, with worst-case waveform error 15.3716%. Extrapolation waveform error is 34.2862%; extrapolation reliability is not claimed. The one fixed fit converged in 800 iterations without warnings. It has a separate checkpoint and training log and does not overwrite historical weights.

## Remaining gates

- **NASA SEEB-ALR:** the original underresolved nose family is rejected. A locally resolved replacement improves nose pressure, but its SU2 8.5.0 coarse run also failed convergence and full-field physical checks (workflow 35941629706). A controlled comparison on the exact same mesh/configuration with official SU2 8.0.1 is the next diagnostic. Neither mesh family is accepted experimental validation.
- **Taylor-Maccoll cone: accepted.** Three-mesh pressure errors are 4.5632%, 2.1563% and 1.1616%, with 1.0064% last-two change. The coarse failure is preserved; the unchanged refined-family criteria passed in workflow 35943668572.
- **Retained optimized design: accepted.** Ten new baseline, optimized, alternative and off-design computations passed physical and numerical checks in workflow 35943668623. Peak reductions on two meshes are 21.1876% and 20.6646%; drag changes are -3.7971% and -3.5218%. These are actual new CFD results for the retained geometry, not predictions attributed to the new clean model.

## Preserved failures and scope

All eight earlier SU2 8.5.0 checkpoint cases passed numerical and prediction-error gates but failed the additional maximum-field enthalpy criterion (17.15–18.70% deviation at a sharp tip). Their raw evidence and diagnostics remain available; see `notebooks/week16/POINTED_BODY_PHYSICS_AUDIT.md`. Residual convergence alone is insufficient.

Original NASA records, source hashes, paper references, portable model and instructional sources are retained. Full Beihang aircraft reproduction and atmospheric propagation/ground-level PLdB are not claimed. No failed or incomplete result is a validated final release.
