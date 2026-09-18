# Week 7.4 - Diverse-wake pretraining and label-efficient force decoding

Scientific question: after masked self-supervised pretraining on many unlabelled
wake trajectories, how many labelled frames from a new trajectory are needed to
decode lift, and does the representation outperform both an identical random
encoder and a classical POD representation?

## Why this differs from Week 7.3

Week 7.3 asks the network to complete a low-rank field; gappy POD nearly solves
that task analytically.  MAPA succeeds because it pretrains on heterogeneous
unlabelled recordings and transfers the representation to an external label.
Week 7.4 follows that logic: 4016 LBM frames from 16 Reynolds trajectories are
generated, the MAE sees only unlabelled development fields, and the downstream
label is instantaneous cylinder lift.

## Frozen information contract

- Development trajectories: Re = 60, 65, 70, 75, 80, 85, 90, 100, 110, 120, 130.
- Validation trajectory: Re = 105, used to select ridge strength and stopping.
- Test trajectories: Re = 95, 115, 125, 135, opened only after choices freeze.
- Within a test trajectory, labelled pool `[0,150)`, gap `[150,180)`, scored
  window `[180,251)`.
- Metric: lift NRMSE = RMSE divided by the test-window lift standard deviation.

## Model and baselines

The 90k-parameter MAE masks 75% of 4 x 6 patches.  Its transformer encoder sees
visible tokens only.  Pretraining uses 6000 AdamW steps on 2761 development
frames and validation-only early stopping.  At downstream time all patches are
visible and the encoded tokens are mean pooled to one 64-component vector.

Three frozen representations receive exactly the same k target labels and ridge
readout: pretrained encoder, identical random encoder, and 32 POD coefficients.
Ridge strength is selected on Re = 105.  Seeds change labelled frames within a
trajectory; uncertainty is summarized across the four independent test
trajectories, not by pretending the within-trajectory draws are new flows.

## Retained result

The pretrained encoder beats the random encoder at every k.  At k = 32 it gives
12.31% NRMSE, already below the random encoder's 16.02% at k = 128: measured
accuracy is better with one-quarter as many target labels.  It reaches 6.05% at k = 64 and 4.22% at k =
128.  POD is stronger through k = 32, but the pretrained representation passes
POD at k = 64 (6.05% versus 7.00%) and k = 128 (4.22% versus 4.73%).

The correct conclusion is not that transformers always beat POD.  Diverse
pretraining makes the neural representation label-efficient for an external
force label; POD remains the strongest very-low-label representation.  The
result is restricted to quick educational D2Q9-TRT trajectories and is not a
grid-independent force-validation claim.

## Reproducibility

Generate data with `qa/generate_week07_4_wakes.py`, run the protocol with
`qa/run_week07_4_protocol.py`, and inspect the exact records and hashes in
`results/week07_4_diverse_pretraining/metrics.json`.

References: Tang, Spalding and Cogan, "Pretraining for Sample-Efficient Neural
Interfaces" (MAPA), arXiv:2609.13507 (2026); He et al., CVPR 2022; Everson and
Sirovich, JOSA A 1995.

## Audit qualifications

Ridge selection additionally uses 2761 development and 251 validation lift labels.
The reported k counts target labels only. One encoder initialization is retained;
the three seeds resample target labels, not pretraining. POD rank 32 is fixed,
not selected from a rank sweep. One trajectory per Reynolds number varies its
initial perturbation together with Reynolds number, so initial-condition
generalization is not isolated. Mean improvements do not establish statistical
significance or superiority on every trajectory. These test cases are now
inspected retained evidence and cannot be reused for future model selection.
