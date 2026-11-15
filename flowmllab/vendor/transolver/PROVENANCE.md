# Vendored original Transolver

Source: https://github.com/thuml/Transolver/tree/75e0f67643806a81cd1d3f6adc88dd8c02416fe7/PDE-Solving-StandardBenchmark/model

MIT license retained. Authors: Haixu Wu and colleagues.
Files: Physics_Attention.py, Transolver_Irregular_Mesh.py, Embedding.py.
Local changes: relative package imports; use PyTorch's trunc_normal_ instead of timm; positional-grid device follows input instead of hard-coded CUDA. No attention or architecture changes.
The step experiment uses Time_Input=False and unified_pos=False.
This is the original irregular-mesh Transolver, NOT GeoTransolver or Transolver++.

