# Optional self-study: Geometry–time operator evaluation

[Student notebook](Geometry_Time_Operator_Bridge.ipynb) · [Lecture PDF](../../lectures/optional_geometry_time_bridge.pdf) · [Editable lecture](../../lectures/source/optional_geometry_time_bridge.md) · [Reference answers](../../instructor/optional_geometry_time_bridge/Geometry_Time_Bridge_Solutions.ipynb)

This extension is **outside the mandatory Weeks 1–22**. Complete Week 15's
geometry/SDF material and Week 21's rollout/phase material first; Week 18's
causality discussion is useful preparation. Allow **150–180 minutes of study**
(an estimate, not measured student workload).

The notebook uses manufactured circles, rectangles and oscillating arrays.
It does not run CFD, train a network, download data or reproduce the paper.
The four tasks cover group splits, boundary-strip metrics, autonomous feedback
and circular phase with undefined values. Each is worth 25 points: 15 for
numerical behavior and 10 for interpretation. Tasks remain NOT SUBMITTED until
you implement them and enable `RUN_EXERCISES`.

## Run locally

Use Python 3.10–3.13 in a full repository checkout or extracted ZIP. In the
repository root, install the minimal notebook dependencies:

```bash
python -m pip install numpy matplotlib jupyterlab
python -m jupyter lab notebooks/optional_geometry_time_bridge/Geometry_Time_Operator_Bridge.ipynb
```

The setup cell searches upward for the repository's `flowmllab` package and
gives an actionable error if the notebook was copied without the source.
It modifies no retained data or result files. There are no credentials or GPU
requirements. Worked results appear inline. The measured notebook elapsed time
is separate from the study-time estimate above.

To run the focused contract checks without a notebook server:

```bash
python -m unittest discover -s tests -p test_geometry_time_bridge.py -v
```

Use the reference-answer notebook only after attempting the tasks. Its passing
checks demonstrate the software contract; they are not proof of CFD fidelity
or educational effectiveness. No LinkedIn content is used.

**Review execution:** the student and reference code cells were checked in
separate fresh Python processes. Kernel-server execution was unavailable in the
review container because socket binding was denied; browser/Jupyter interaction
acceptance remains unperformed. This limitation is not a claim that the local
Jupyter setup above fails on a normal computer.
