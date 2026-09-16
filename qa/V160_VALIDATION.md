# v1.6.0 validation record

- Clean checkout based on public main `014cb6103493eb2f41dddbcecd6313785f57a7b5`.
- Local Python 3.12: 204 tests passed, 10 optional tests skipped, 184 subtests passed.
- Course release gate: 36 notebooks, 364 parsed code cells, 21 lecture PDFs; passed.
- Software smoke test: passed. Wheel and source distribution built; twine checks passed.
- Weeks 13, 14 and 15 rerun from this checkout with normal Jupyter permissions;
  all cells completed without errors. Hosted Colab was not tested.
- Revised W13 PDF: all 12 pages rendered and inspected; new field and loss captions
  distinguish checkpoints 55118 and 65711. Loss legend is outside the axes.
- W13 field hashes, dimensions, walls and derived psi/omega were checked. These
  numerical consistency checks do not replace matching CFD/model-residual validation.
- W15 checks include retained archive/member hashes, 130 fields, 51 geometry masks,
  106/21/3 historical train/validation/test cases, and separate DSMC DeepONet evidence.

Initial sandbox-only test failures were temporary-directory permission errors;
the normal-permission rerun passed. Core tests ran in the isolated W14 environment
(NumPy 2.5.3); GitHub release CI separately tests declared base-package dependencies.
