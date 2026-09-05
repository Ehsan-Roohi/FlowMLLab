# One prompt, three pelican flights

A small visual coding experiment by Ehsan Roohi.

**[Watch the videos and open the live HTML animations](https://ehsan-roohi.github.io/FlowMLLab/pelican-flight/)**

| Output | Self-contained source | 12-second video |
|---|---|---|
| First retained animation | [HTML](originals/first.html) | [MP4](videos/first.mp4) |
| Medium (user-reported setting) | [HTML](originals/medium.html) | [MP4](videos/medium.mp4) |
| Light (user-reported setting) | [HTML](originals/light.html) | [MP4](videos/light.mp4) |

See [the shared prompt](PROMPT.md). The original animations use hand-authored SVG, CSS and JavaScript, with no external assets or network requests.

## Interpretation

These are three retained code specimens produced sequentially in one conversation, with earlier outputs visible in context and some tool-assisted checks during creation. They are not blind, independent, single-shot trials. The assistant did not independently verify the backend model or reasoning settings. The first animation is identified by its order rather than assigning it an unverified setting. No ranking or general model-performance claim is made.

For this publication the HTML files were copied byte-for-byte, including any existing visual or mechanical imperfections. SHA-256 hashes are in `manifest.json`. Watch for pedal attachment, chain motion, wall/ground contact, and loop continuity; appearance alone does not validate the mechanism. The same-scene similarity may reflect the shared conversation context.

## Video reproduction

The MP4s are deterministic SVG renders, not browser screen recordings. Each original animation function is evaluated at t = 0, 1/30, …, 359/30 seconds without modifying the retained HTML. CSS stroke rules are copied to the exported SVG. CairoSVG renders 960×540 frames; ffmpeg encodes H.264/yuv420p with fast-start metadata. Browser-specific rendering can differ; the live HTML remains the canonical interactive output.

With Node.js, Python (lxml and cairosvg), and ffmpeg installed:

```sh
node scripts/sample.cjs
python3 scripts/render.py
```

Inspired by [Simon Willison’s SVG pelican experiments](https://simonwillison.net/tags/pelican-riding-a-bicycle/). This adds animation and mechanical continuity; it is not his original static benchmark.
