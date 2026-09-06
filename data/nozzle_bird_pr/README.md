# Canonical micro-nozzle DSMC fields

This directory contains the fifteen full Tecplot snapshots recovered from the author's Unity working directory `/work/pi_roohie_umass_edu/Nozzle` on 2026-09-06. The pressure ratios are 15, 16, 18, 19, 20, 22--30 and 33.

Only files named exactly `P=<integer>full.dat` are included. The Unity directory also contains transformed files such as `P=*_full.dat`, `P=*_full_full.dat` and model predictions; those are excluded so that the recovered inputs are not mixed with later processing outputs.

`manifest.json` records the Unity source path, selection rule, byte count, SHA-256 and Tecplot zone dimensions for every case. `nozzle_canonical_sha256.txt` is the checksum list produced on Unity before transfer. The transferred files pass that checksum list byte for byte.

The maintainer identifies these fields as outputs of the two-dimensional Bird DSMC code. The exact executable revision, input decks, accumulated moments and exporter mapping have not yet been recovered, so these files establish the field archive but not the complete producing-run lineage. See [the correction note](../../docs/NOZZLE_DATA_NOTE.md) and [Unity inventory](../../qa/UNITY_BIRD_SOURCE_INVENTORY.md).

The dataset is associated with Roohi and Mahdavi, *Shock-Centered Micro-Nozzle POD Reproducibility*, and the article DOI [10.1063/5.0343101](https://doi.org/10.1063/5.0343101). Reuse is governed by [CC BY 4.0](DATA_LICENSE.md).
