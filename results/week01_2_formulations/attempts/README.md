# Retained initial attempts

The Re=400 32-cell FD run passed its relative-update stop but failed the independent steady-RHS gate (1.103e-6 > 1e-6). The Re=1000 32-cell run exhausted 200000 steps. Neither was accepted as steady. The adapter was tightened from update tolerance 1e-10 to 1e-12, and the step budget increased to 400000, preserving the same steady-RHS threshold and original physical solver. Both cases are rerun from rest; no initial attempt is deleted. Adapter provenance notes distinguish the loaded code from file-on-disk hashes sampled on completion.
