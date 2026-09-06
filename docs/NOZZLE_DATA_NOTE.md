# Micro-nozzle data note — status 1 (source correction pending)

This is a versioned record of an unresolved defect, **not a corrected dataset**.
It concerns the fifteen research snapshots associated with
[10.1063/5.0343101](https://doi.org/10.1063/5.0343101), imported from revision
`e1b234ba499408d3b6224633972f939f3b2301d6` of the
[source repository](https://github.com/Ehsan-Roohi/roohi-nozzle-pod-reproducibility).
The original values and hashes in the [provenance manifest](../results/mahdavi_deeponet/provenance.json)
remain unchanged.

## Observations and limits

The stated symmetry row contains nonzero exported transverse velocity.
Streamwise mass-flow diagnostics also show variation; its magnitude depends
on case and integration convention. See the retained
[registered-POD diagnostics](../results/nozzle_transport/) for case-level values.
The approximately 6% variation flagged in review is not itself proof of a
solver conservation failure: export, boundary, cell/node and quadrature
conventions must first be resolved.

The full-domain post-processing defect is now demonstrated. Relative to the
recovered half-domain exports, the first two zones translate `Y` by -92
micrometres exactly at float32 precision. The two added mirror zones share every
non-`Y` variable unchanged, so the odd components `QY`, `V` and `Txy` receive
the wrong mirrored sign. A recovered 2009 source candidate also fills its
symmetry-plane nodal row from adjacent cell-centered samples rather than
explicitly imposing `V=0`. This supports an export/post-processing explanation,
not a particle-solver boundary-condition failure. Prescribing V=0 only in a
plot or prediction wrapper would still not correct the retained source fields.

## Required correction and publication sequence

1. Start from the fifteen full-domain back-pressure derivatives recovered at
   `/work/pi_roohie_umass_edu/Nozzle`, then recover the exact solver revision,
   input decks, raw accumulated moments, exporter revision, grid convention and
   sampling metadata for each snapshot. Available derived fields alone cannot
   establish those details; see the [Unity inventory](../qa/UNITY_BIRD_SOURCE_INVENTORY.md).
2. Reproduce the export, identify the faulty indexing/normalization if present,
   and fix it at source. Re-export from valid raw moments; rerun the solver if
   the source moments are affected. Do not zero a row and call it new data.
3. Check symmetry, wall conditions, integrated flux with documented quadrature,
   and sampling/grid sensitivity. Retain before/after values and checksums.
4. Recompute every affected paper figure and metric using the same protocol.
   The authors have agreed to proceed with a corrigendum if this source audit
   establishes that the published result is affected. The verified correction
   and its numerical consequences must be prepared before journal submission;
   see the [author-decision record](../qa/AUTHOR_DECISIONS_2026-09-06.md).
5. Publish a distinct corrected dataset version, with its own identifier,
   machine-readable change log, source lineage and license. Preserve the old
   version and link it to the correction. Then update the course derivatives
   and compare old/new results without silently replacing historical evidence.

**Current gate:** the author-decision gate is closed, the Unity snapshot
directory is located, the pressure labels and full-domain transformation are
resolved, and the mirror-sign defect is identified. Exact Bird-2D
producing-run lineage and corrected re-export/rerun verification are still
required. No corrected-data DOI, corrected field, submitted corrigendum or
journal action is claimed.
