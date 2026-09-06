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

## Archive recovery checkpoint — 2026-09-05

The [machine-readable archive audit](nozzle_source_audit.json) identifies all
fifteen published `P=*.dat` snapshots in both author-supplied archives
`Nozzle (1).zip` and `Nozzle(1).zip`. Their contents match the published SHA-256
values **after CRLF-to-LF conversion only**. They are not byte-identical before
that conversion. No physical value was changed, and this is recovery of the
existing exports, not discovery of corrected data.

The audit also inspected `Nozzle.zip`, `Nozzle Results.zip`, and the ZIP-format
archive `AllresultsNozzleHeat`. None of those three yielded the same named
published snapshots. Across these five archives, a scoped filename search for
Fortran source, `.inp`, and named DS2V/DS2VD decks found no candidates. This
does not rule out raw moments in unknown formats, nested archives, or other
folders. It does not establish which exporter caused the observed inconsistency.

Reproduce comparisons locally using the original archives; no archived code
is executed, extracted, or uploaded:

```bash
python qa/audit_nozzle_archives.py --archive "/path/to/Nozzle (1).zip" \
  --archive "/path/to/Nozzle(1).zip" --output tmp/nozzle-source-audit.json
```

Still needed: the producing solver/exporter revision, matching input deck,
raw accumulated moments, grid convention and sampling record. A different
nozzle project or a file with a similar name is not a substitute without a
documented lineage match.

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
