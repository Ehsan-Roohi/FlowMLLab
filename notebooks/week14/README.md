# Week 14 - RANS, PINN and neural turbulence closures

- [Executed teaching notebook](W14_pyCALC_RANS_PINN_NN.ipynb)
- [Lecture in the Week 13 research-notes format](../../lectures/week14_rans_pinn_nn.pdf)
- [Editable lecture source](../../lectures/source/week14_rans_pinn_nn.md)
- [Review of the supplied notebook](../../docs/WEEK14_NOTEBOOK_REVIEW.md)
- [Relationship to the paper's actual claims](../../docs/WEEK14_PAPER_ALIGNMENT.md)
- [Validation results and limitations](../../results/week14_validation/README.md)

Research method and original solver/scripts: **Lars Davidson**. DNS:
**Myoungkyu Lee and Robert D. Moser**. New explanations, plots, diagnostic
utilities and classroom orchestration: FlowMLLab / Ehsan Roohi, AI-assisted.

## Run

Use Python 3.12 with NumPy, SciPy, pandas, matplotlib, scikit-learn, PyTorch,
nbformat, nbclient, ipykernel and pyAMG. Exact executed versions are retained
in the evidence manifest. No changes to the project's main dependencies are
required: use a separate virtual environment.

```powershell
python -m venv .venv-week14
.venv-week14/Scripts/python -m pip install -r notebooks/week14/requirements.txt
.venv-week14/Scripts/python qa/build_week14_notebook.py --execute
.venv-week14/Scripts/python qa/verify_week14.py
```

On Linux/macOS use `.venv-week14/bin/python` instead. Do not install the main
package with its different dependency pins into this isolated environment;
the notebook imports the checkout directly.

Open the notebook from this complete checkout and Run All. It trains the
small inverse PINN against a manufactured analytic control before it trains the
original-size 2-10-10-1 ReLU model for 1000 epochs and reruns the interpolation
control. It verifies hashes before reading retained CFD evidence; it does not
pretend those saved CFD outputs were just recomputed by Run All.

### Colab and dependency isolation

The Week 14 requirements describe a separate environment, not an extra to be
installed alongside the repository's older base dependency bounds. In a fresh
Colab runtime, obtain the complete checkout, run
`%pip install -r /content/FlowMLLab/notebooks/week14/requirements.txt`, restart
the runtime if NumPy or SciPy was already imported, then open this notebook and
Run All. Do not also run `pip install -e .`. A hosted Colab run has not been
validated in this revision; the saved outputs come from local Python 3.12 CPU.

To repeat complete original scripts, use:

```powershell
python qa/prepare_week14_source.py
python qa/run_week14_upstream.py baseline
python qa/run_week14_upstream.py pinn
python qa/run_week14_upstream.py nn10000
python qa/run_week14_upstream.py nn5200
python qa/run_week14_upstream.py train_ck
python qa/run_week14_upstream.py balance
python qa/run_week14_upstream.py inverse
```

The preparation script downloads a pinned public archive if absent. A changed
archive is rejected for inspection. Preparation re-extracts the selected
upstream cases, resetting only this generated `tmp/w14` run tree; do not run it
during active simulations. The inverse mode runs the original long 200,000-epoch
checkpoint-resume script. Read the validation status before claiming reproduction.

The instructor reported permission to adapt materials in the shared chat.
Exact written reuse terms are not available in this checkout; upstream code
and checkpoints are kept outside the published materials. Do not relicense
them under FlowMLLab's MIT license. This release contains attributed compact
teaching data and independently generated results, relying on the instructor's
reported permission and explicit publication request; it does not distribute
the author's solver archive or pretrained checkpoints. Downstream users must
check the original terms before further redistribution.

The assembled NN-5200 case is an explicit adaptation and did not converge.
All four CFD run gates, including failures, are retained. This is a completed
teaching/reproduction audit, not a certified numerical reproduction of every
paper result.
