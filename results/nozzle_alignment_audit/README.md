# Moving-throat alignment impact

This retained experiment quantifies why the Week 9 pre-fit coordinate audit
matters. Two matched `2×48` tanh MLPs receive the same 24 quasi-1D isentropic
training cases, architecture, optimizer settings and random seed. The only
difference is the trunk coordinate paired with each target:

- **aligned:** every Mach target retains its own physical coordinate;
- **misaligned:** every case reuses the first training nozzle's coordinate grid.

Both arrays have valid and identical shapes, so ordinary tensor-shape checks do
not expose the corrupted model.

| Blind geometry | Misaligned relative L2 | Aligned relative L2 |
|---|---:|---:|
| `x_t/L=0.365`, `A_e/A_t=2.35` | 12.46% | 1.87% |
| `x_t/L=0.435`, `A_e/A_t=3.15` | 3.88% | 1.22% |
| `x_t/L=0.495`, `A_e/A_t=3.65` | 3.84% | 1.81% |
| **Mean** | **6.72%** | **1.63%** |

Correct coordinate pairing lowers mean blind error by 75.7%. This is a
controlled quasi-1D stress test, not a DSMC result and not an accuracy claim for
the article model. Its purpose is to isolate one data-contract failure under a
matched computational budget.

The exact retained metrics are in
[`alignment_impact_metrics.json`](alignment_impact_metrics.json). Regenerate
the figure and metrics with:

```bash
python qa/build_week09_alignment_impact_figure.py
```
