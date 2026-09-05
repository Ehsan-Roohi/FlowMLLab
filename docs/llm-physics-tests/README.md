# Does the physics hold?

A common numerical audit of three AI-generated rotating-hexagon simulations, curated by Ehsan Roohi.

- [Comparison page](index.html)
- [Full report](REPORT.md)
- [Raw results](results/audit.json)
- [Prompt and provenance](PROMPT.md)
- [Retained originals](originals/)

Run `node tests/audit.cjs` to reproduce the numerical audit. No dependencies required. A local static preview can be served with `python3 -m http.server 8000` from this directory.

The three settings were reported by the user and not independently verified. This is an exploratory comparison of retained code specimens, not a model leaderboard. See REPORT.md for complete conditions and limitations.

Published within FlowMLLab under its existing MIT license. Source snapshots and analysis are AI-assisted.

## GitHub Pages deployment

The page is ready in `main/docs/llm-physics-tests/`. To enable the repository's first Pages site, open Settings → Pages, choose **Deploy from a branch**, select **main** and **/docs**, then Save. The intended public URL is https://ehsan-roohi.github.io/FlowMLLab/llm-physics-tests/ after that deployment completes. At publication time, repository metadata reported `has_pages: false`; the publishing connection did not expose the administration operation needed to enable Pages.

This directory is self-contained and can also be downloaded and served by any static HTTP server. The original fragments are retained as source downloads; the comparison page supplies the standalone interface.
