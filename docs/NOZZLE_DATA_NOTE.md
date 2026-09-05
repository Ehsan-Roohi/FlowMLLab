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

An exporter defect is a hypothesis to verify against the exact producing run.
Prescribing V=0 in a boundary plot or prediction wrapper is not correction of
the underlying measured fields. Existing historical test cases have already
been inspected; they are regression tests, not fresh blind validation.

## Required correction and publication sequence

1. Recover the exact solver revision, input decks, raw accumulated moments,
   exporter revision, grid convention and sampling metadata for each snapshot.
   Available derived fields alone cannot establish those details.
2. Reproduce the export, identify the faulty indexing/normalization if present,
   and fix it at source. Re-export from valid raw moments; rerun the solver if
   the source moments are affected. Do not zero a row and call it new data.
3. Check symmetry, wall conditions, integrated flux with documented quadrature,
   and sampling/grid sensitivity. Retain before/after values and checksums.
4. Recompute every affected paper figure and metric using the same protocol.
   Ehsan Roohi and Amirmehran Mahdavi must decide whether the differences
   require a corrigendum and coordinate with the journal. No decision or
   message on their behalf is implied by this note.
5. Publish a distinct corrected dataset version, with its own identifier,
   machine-readable change log, source lineage and license. Preserve the old
   version and link it to the correction. Then update the course derivatives
   and compare old/new results without silently replacing historical evidence.

**Current gate:** exact producing-run lineage and author review are still
required. No corrected-data DOI, corrected field, or corrigendum is claimed.
