# Data attribution and license

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
