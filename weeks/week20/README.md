# Week 20 — Learning with missing sensors

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 19](../week19/README.md) · [Week 21 →](../week21/README.md)

Sensor-to-field reconstruction, baselines and validation-only selection.

![Eight-sensor velocity reconstruction: CFD, SensorSet, gappy POD, sixteen-sensor control and two absolute-error contours](../../docs/assets/transformer-contours/week20.png)

## Lecture and notebooks

**Lecture:** [Lecture 20](../../lectures/week20_cfd_transformer.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Learning with missing sensors | [Open notebook](../../notebooks/week20/W20_CFD_Transformer.ipynb) | [Setup and reproduction guide](../../notebooks/week20/README.md) |

## What you will work on

- Sensor-to-field learning with missing observations

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Reconstruct a velocity field when some observations are unavailable.<br>
**CFD / data:** Re90/Re110 fitting, Re100 selection and previously inspected Re105 retained evaluation.<br>
**Learning method:** Compare SensorSet, POD-DeepONet, ridge imputation and variable-sensor gappy POD using the same observations.

The fixed Re105 snapshot compares SensorSet and gappy POD using the same stored noisy observations from 8 of 16 sensors; the 16-sensor SensorSet prediction is a control. Shared field and error scales expose spatial differences. This snapshot does not rank architectures; the [paired three-seed augmentation ablation](../../results/transformer_sensor_ablation/README.md) separates training-recipe effects from architecture claims.



[Student notebook](../../notebooks/week20/W20_CFD_Transformer.ipynb) · [Lecture](../../lectures/week20_cfd_transformer.pdf) · [Setup and assignment](../../notebooks/week20/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 19](../week19/README.md) · [Week 21 →](../week21/README.md)
