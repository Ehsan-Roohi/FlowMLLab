# Week 7.2 - Sparse-sensor state estimation of a cylinder wake

[Lecture PDF](../../lectures/week07_2_cylinder_state_estimation.pdf) ·
[Executable notebook](W7_2_Cylinder_Wake_State_Estimation.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb)

Allow 75-90 minutes after Weeks 5 and 7. CPU only. The lab fits a POD-space
linear-Gaussian model on frames 0:160 of the existing Re110 LBM wake, chooses
rank, sensor count and process-covariance inflation on frames 160:210, then
compares causal filtering with open-loop DMD, persistence and instantaneous
sensor-only reconstruction on frames 210:281.

The retained result uses 32 sensors, rank 8 and 10% RMS synthetic measurement
noise. The Kalman filter improves the mean test relative L2 error over both
open-loop and sensor-only baselines, but its nominal 95% marginal intervals
cover only about 55% of sampled field values. The under-coverage is a retained
failure of the linear-Gaussian uncertainty model, not silently recalibrated.

This is a previously inspected, finite-resolution trajectory with artificial
sensor noise. It is not new CFD, a new-Re generalization result, simultaneous
field uncertainty, or offline smoothing with future observations.
