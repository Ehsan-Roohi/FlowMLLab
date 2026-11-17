# Interactive GitHub Pages site

The public URL remains https://ehsan-roohi.github.io/FlowMLLab/ . The repository, article links and stars are unchanged.

`docs/index.html`, `docs/assets/site.css` and `docs/assets/cavity-explorer.js` are the presentation sources. `course/catalog.json` supplies the complete unit sequence. The build discovers notebook/PDF counts from the checkout, so the site does not maintain a separate course inventory. All existing catalogue and research sections remain available.

## Automatic publication

`.github/workflows/pages.yml` builds on pushes to `main`, checks the artifact, and deploys through GitHub Pages. Configure the repository's Pages source as **GitHub Actions**. Pull requests build without deploying. Manual dispatch can republish after recovery. If another workflow makes a commit with `GITHUB_TOKEN`, GitHub does not emit a new push workflow; that workflow should dispatch this publisher explicitly using an appropriately authorized credential. Normal maintainer pushes publish automatically.

```sh
python -m pip install -e .
# Install Poppler (pdfinfo) with the system package manager.
python qa/build_site.py --output tmp/pages-site
python qa/check_site.py tmp/pages-site
python -m http.server 8765 --directory tmp/pages-site --bind 127.0.0.1
```

The exporter hash-checks CFD inputs and the frozen selection, model weights and held-out predictions. It generates native-node field images and full-resolution centreline profiles. It performs inference with frozen weights only, never training, selecting a model or running CFD. Pressure and vorticity reuse the established matched-derivative reconstruction. Existing pressure/velocity metrics at Re=1000 and 7500 must reproduce before publication. The webpage labels training, validation and held-out cases; suspect printed Ghia samples remain visible.

The derived artifact lives under ignored `tmp/pages-site`, not in scientific result folders. It is rebuilt from tracked inputs, with `build.json` recording the Git commit and `assets/cavity/manifest.json` recording input hashes and scientific definitions. No separate hosting account, analytics service, API key or GPU is needed. If a verification gate fails, publication stops and the previous successful deployment remains available.
