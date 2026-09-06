# Recovered full-domain micro-nozzle DSMC derivatives

This directory contains fifteen full-domain Tecplot files recovered from the author's Unity working directory `/work/pi_roohie_umass_edu/Nozzle` on 2026-09-06. Their back pressures are 15, 16, 18, 19, 20, 22--30 and 33 kPa. An author-supplied legacy input with `PIN=100 kPa` and `POUT=25 kPa` confirms that `P=25` denotes back pressure, not pressure ratio.

Only files named exactly `P=<integer>full.dat` are included. The Unity directory also contains transformed files such as `P=*_full.dat`, `P=*_full_full.dat` and model predictions; those are excluded so that the recovered inputs are not mixed with later processing outputs.

`manifest.json` records the Unity source path, selection rule, byte count, SHA-256 and Tecplot zone dimensions for every case. `nozzle_canonical_sha256.txt` is the checksum list produced on Unity before transfer. The transferred files pass that checksum list byte for byte.

These files are not solver-native raw moments. Comparison with the recovered
half-domain exports in the article repository shows that their first two zones
are exact float32 copies after translating `Y` by -92 micrometres. Two additional
zones mirror `Y` while sharing all other variables. The shared-variable rule does
not reverse the physically odd components `QY`, `V` and `Txy`.

The maintainer identifies the fields as derivatives of outputs from a modified
two-dimensional Bird DSMC code. A 2009 source/input candidate now establishes
the solver family, P=25 input and grid construction, but its output-moment schema
does not match the article exports. The exact executable revision, fifteen
matching inputs and accumulated moments remain open. See [the correction note](../../docs/NOZZLE_DATA_NOTE.md) and [Unity inventory](../../qa/UNITY_BIRD_SOURCE_INVENTORY.md).

The dataset is associated with Roohi and Mahdavi, *Shock-Centered Micro-Nozzle POD Reproducibility*, and the article DOI [10.1063/5.0343101](https://doi.org/10.1063/5.0343101). Reuse is governed by [CC BY 4.0](DATA_LICENSE.md).
