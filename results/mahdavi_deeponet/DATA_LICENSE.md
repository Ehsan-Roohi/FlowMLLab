# Data attribution and license

## Micro-step source fields

The nine step-height fields used by Lab 1 originate in the separate
[`roohi-step-dnn-mahdavi`](https://github.com/Ehsan-Roohi/roohi-step-dnn-mahdavi)
repository, pinned here at commit
`c3f211376b42b8dc30daad380eaef5e0ab800b5c`. At that revision the source
repository states that no general license has been assigned. The corresponding
author explicitly authorized publication of the compact teaching derivatives
`step_height_learning_7cases.npz` and `step_height_test_2cases.npz` in
FlowMLLab on 2026-09-04. That permission does not relicense the upstream
Tecplot files or grant broader reuse rights. The two archives preserve the
complete source point rows (including repeated zone-boundary coordinates),
store `U,V` as float32, and separate the seven learning/validation cases from
the two sealed tests.

## Micro-nozzle derivative

`nozzle_centerline_15cases.npz` is an adapted, reduced centerline derivative of
the 15 DSMC snapshots published in **Shock-Centered Micro-Nozzle POD
Reproducibility** by Ehsan Roohi and Amirmehran Mahdavi:

<https://github.com/Ehsan-Roohi/roohi-nozzle-pod-reproducibility>

Source revision: `e1b234ba499408d3b6224633972f939f3b2301d6`.

The source dataset and reference numerical outputs are licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
FlowMLLab changes the representation by retaining the max-y symmetry
centerline, selecting seven fields, compressing them into NumPy format, and
recomputing density-shock diagnostics. The source values are otherwise not
smoothed or synthesized.

The CSV files of article results are factual metric transcriptions provided
for criticism, comparison, and teaching. They are not new model runs.
