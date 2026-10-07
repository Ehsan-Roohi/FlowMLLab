# SensorSet augmentation ablation

Post-review paired comparison within the SensorSet architecture. Identical initialization, noisy training/validation/test arrays, POD basis, sensor locations, Adam optimizer and 1200-step ceiling. The same early-stopping rule can yield different executed step counts. Checkpoint selection uses all-sensor validation only.

| Seed | Evaluation sensors | Augmented relative L2 (%) | No augmentation relative L2 (%) | Augmentation relative gain |
|---|---|---:|---:|---:|
| 17 | all | 13.5542 | 6.0781 | -1.2300 |
| 17 | drop-half | 27.1587 | 60.4606 | 0.5508 |
| 29 | all | 14.4320 | 4.6498 | -2.1038 |
| 29 | drop-half | 24.9896 | 39.4073 | 0.3659 |
| 43 | all | 11.6094 | 5.7340 | -1.0247 |
| 43 | drop-half | 22.6201 | 36.8877 | 0.3868 |

Gain is 1 minus augmented error divided by unaugmented error. Positive favors augmentation. These three retained-case pairs support a descriptive comparison; they do not establish significance, convergence, or superiority of one architecture. Missing-sensor performance was not the validation objective.

Largest difference between augmented relative-L2 results and the primary retained augmented baseline: 2.78e-16. No primary evidence files were modified.

Run `python -m qa.ablate_transformer_sensors --output NEW_EMPTY_DIRECTORY` to retrain. Run `python -m qa.ablate_transformer_sensors --verify --output results/transformer_sensor_ablation` to reload checkpoints and score the saved predictions.
