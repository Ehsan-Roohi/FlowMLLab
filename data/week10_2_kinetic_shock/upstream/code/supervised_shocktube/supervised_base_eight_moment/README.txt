Eight-moment data-assisted BGK shock tube (Kn = 1, t_f = 0.02)
================================================================

Material behind section 5.2, table 3, figures 4, 5 and 33 and table 15 of the manuscript
"Physics-informed neural networks (PINNs) for kinetic shock waves: simultaneous accuracy of lower- and
higher-order moments, with and without moment supervision".

accepted_state/
  STATE.pt                 final model state (common-tail envelope, after 35 output-layer updates), with the
                           velocity grid, point sets, observation scales and normalisations it was trained with
  code/                    model classes: all_moment_model.py, q_amplitude_adapter.py (eight-moment representation),
                           common_tail_model.py and common_tail_envelope.py (common-tail envelope), legacy.py
  inputs/                  REFERENCE6400_FIELDS.npz (6400-cell BGK DVM fields at t_f) and the frozen observation locations
  evidence/EXECUTED_CODE/  the output-layer refinement and its problem definition exactly as executed
  evidence/                run settings, accepted-update history, gradient check, assessment (JSON and arrays)
                           and the runs declared before execution
  verify_accepted_source.py  reloads STATE.pt in a fresh process and recomputes the fifteen training components
                           and the full-domain discrepancies (needs PyTorch and NumPy); RELOAD_VERIFICATION.json is
                           its recorded result. It rewrites RELOAD_VERIFICATION.json and a MANIFEST.json when run.

manuscript_checks/
  FACTS_FOR_METHODS.json            per-state discrepancies, upstream error shares, decay rates, extrema, point sets
  FACTS_REGIONAL.json               regional squared errors of the three eight-moment states
  FACTS_HELDOUT_AND_SCALES.json     discrepancies excluding the reference nodes that bracket observation locations
  stage0/STAGE0_FIVE_FIELD.json     the five-field state on the 48x20x20 grid against the 6400-cell reference
  REFERENCE_REFINEMENT_3200_6400.json  difference between the 3200- and 6400-cell DVM solutions
  PHYSICS_MAGNITUDES.json           balance residuals relative to the time-derivative terms; residual versus collision term
  KERNEL_CHECK.json                 Gaussian inner products of the velocity kernels and first-order moment responses
  FACTS_AFTER_REVIEW.json           comparison with the equation-trained BGK state on the 6400-cell reference and on its
                                    first-order extrapolate; sign changes and decay rates of the tails; location of the errors
  FACTS_STAGE1_LINEAGE.md           audit of the 170 accepted updates with the original envelopes
  *.py                              the scripts that produced these files and the three figures
  eight_field_grid.py               layout shared by the eight-field figures 4, 10 and 11 (three columns at print width)
  make_equation_trained_figures.py  draws figures 10 and 11 in that layout from the saved equation-trained fields of the
                                    repository (PINN_FIELDS.npz and REFERENCE_3200_COMPARISON.npz of the BGK and Shakhov
                                    calculations); EQUATION_TRAINED_FIGURE_CHECK.json records that the entries of table 8
                                    are recomputed from those arrays before drawing

variants/                           records of the two guarded variants of the refinement quoted in Appendix A.2
comparison_inputs/                  equation-trained BGK fields (401 positions) and the 3200-cell DVM fields used by
                                    facts_after_review.py
figures/                            the three figure files used in the manuscript
STATUS_NOTE.txt                     why the run records carry "paper_admission: false"
RELOCATED_PATHS.json                three text files whose local absolute paths were replaced by symbolic roots, with the
                                    SHA256 of the original and released versions

The eight-moment state after 170 accepted updates with the original envelopes (the starting state of the common-tail
replacement) is the checkpoint with SHA256 bbee1d6481b6ee3ccdcf4a0472c408368272f571f4124937368eb602f0833b95; the
five-field state is the checkpoint with SHA256 a0734fccb6cc0766bf046c8a23f8f98bf15e2e03b39d6e96ed70e1a170d09003.
Both, and the job archives listed in FACTS_STAGE1_LINEAGE.md, must be deposited alongside this folder for the scripts
stage0_five_field.py, physics_magnitudes.py and facts_*.py to be rerun; accepted_state/ is self-contained.
