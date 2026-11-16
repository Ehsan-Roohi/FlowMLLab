# Attribution and permissions

The research paper is distributed under **Creative Commons Attribution 4.0**:
[arXiv:2605.29211v2](https://arxiv.org/abs/2605.29211v2), Ehsan Roohi (2026).
The retained numerical data and source-status documents come from the author's
[PINNs-DVM repository](https://github.com/Ehsan-Roohi/PINNs-DVM) at commit
`add5388f7da5f7eace95fff97d34d6ed58b90169` and are included with the author's
explicit authorization for this teaching extract. Attribution and exact byte
identities are recorded in `source_files.json` and `manifest.json`.

The upstream repository states that **no software licence has been assigned**.
The paper's CC BY 4.0 licence is not represented here as an upstream software
licence. No upstream solver, optimizer, training code, or checkpoint is copied
into FlowMLLab's MIT-licensed code. The NumPy teaching helpers and preparation
utility were written independently from the mathematical definitions.

The source files under `upstream/` retain their source provenance and notices;
FlowMLLab's MIT licence does not silently relicense these research materials.
The only transformed numerical deliverable is `supervised_s0_current.npz`:
name-mapped fields, common-coverage restriction, and declared linear
interpolation. The exact transformation is in `prepare_extract.py`.
