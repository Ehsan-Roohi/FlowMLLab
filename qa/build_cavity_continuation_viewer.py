"""Package the verified 0..100 continuation preview without changing its seeds.

Install the reviewed subset once with --source /path/to/verified/media_next.
Rebuild HTML/README from the retained public bundle with no --source, or use
--check for a read-only gate. This program never runs a solver or trains a model.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/cavity_continuation"
SCHEMA = "flowmllab.cavity.continuation.preview.v1"
REYNOLDS = [100, 400, 1000, 3200, 5000, 7500, 10000]
FIELDS = ("speed", "omega", "p")
TIMES = [i / 2 for i in range(201)]
CONTROLS = {"cells": 128, "dt": .01, "method": "PIMPLE", "outer_correctors": 3,
            "pressure_correctors": 2, "convection": "central", "domain": [0, 1, 0, 1]}
SITE = "https://ehsan-roohi.github.io/FlowMLLab/"
REPO = "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/"
NAMES = {"speed": "Speed and instantaneous streamlines", "omega": "Signed vorticity", "p": "Zero-mean pressure"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _member(directory, name):
    if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Preview receipts must use plain asset filenames")
    path = Path(directory) / name
    if path.resolve().parent != Path(directory).resolve():
        raise ValueError("Preview asset redirects outside the media directory")
    return path


def admit_receipt(receipt):
    """Check the actual continuation schema and every displayed time identity."""
    if receipt.get("schema") != "flowmllab.cavity.continuation.movies.v1":
        raise ValueError("Expected a native continuation movie receipt")
    if receipt.get("physical_times") != TIMES or receipt.get("frame_interval") != .5:
        raise ValueError("This preview requires the complete common 0..100 axis at 0.5 spacing")
    if type(receipt.get("fps")) is not int or receipt["fps"] <= 0:
        raise ValueError("Invalid playback rate")
    if set(receipt.get("fields", {})) != set(FIELDS):
        raise ValueError("Expected exactly speed, signed vorticity and pressure")
    if not all(_sha(receipt.get(key)) for key in ("campaign_sha256", "render_source_sha256")):
        raise ValueError("Missing exact campaign/render provenance")
    sources = receipt.get("sources", [])
    if [s["case"]["reynolds"] for s in sources] != REYNOLDS:
        raise ValueError("Expected the seven established Reynolds numbers in order")
    for source in sources:
        case, admission = source["case"], source["admission"]
        if (case.get("cells"), case.get("dt"), case.get("outer"), case.get("pressure"), case.get("purpose")) != (128, .01, 3, 2, "continue"):
            raise ValueError("Continuation movie controls differ from the declared 128-cell baseline")
        if admission.get("admission") != "verified_checkpoint" or admission.get("case") != case:
            raise ValueError("Native continuation was not admitted from a verified checkpoint")
        prefix = admission.get("seed_prefix", {})
        if (prefix.get("status") != "sha_verified" or prefix.get("frames") != 101
                or prefix.get("trajectory_sha256") != case.get("seed_sha256")
                or prefix.get("receipt_sha256") != case.get("seed_receipt_sha256")):
            raise ValueError("Original 0..10 seed lineage is incomplete")
        seed_receipt_path = ROOT / f"results/cavity_startup/fine128/re{case['reynolds']}.json"
        if digest(seed_receipt_path) != case["seed_receipt_sha256"]:
            raise ValueError("Published seed receipt differs from the continuation lineage")
        seed = json.loads(seed_receipt_path.read_text(encoding="utf-8"))
        config = seed["config"]
        expected = {"reynolds": case["reynolds"], "cells": 128, "dt": .01,
                    "method": "pimple", "convection": "central", "outer_correctors": 3, "pressure_correctors": 2}
        if any(config.get(k) != v for k, v in expected.items()) or seed["trajectory_sha256"] != case["seed_sha256"]:
            raise ValueError("Seed physical/numerical settings disagree with the preview")
        frames = source.get("displayed_frames", [])
        if len(frames) != len(TIMES):
            raise ValueError("Native movie frame coverage is incomplete")
        later_hashes = []
        for t, frame in zip(TIMES, frames):
            if frame.get("time") != t or frame.get("step") != round(t / .01) or not _sha(frame.get("file_sha256")):
                raise ValueError("Displayed frame does not have its actual physical-time/step identity")
            if t <= 10:
                if frame["file_sha256"] != case["seed_sha256"] or frame.get("seed_index") != round(t / .1):
                    raise ValueError("Startup frame is not the declared native seed sample")
            else:
                if frame.get("seed_index") is not None or not _sha(frame.get("contract_sha256")):
                    raise ValueError("Continuation frame lacks its native contract identity")
                later_hashes.append(frame["file_sha256"])
        if len(set(later_hashes)) != len(later_hashes):
            raise ValueError("Repeated continuation payload: refusing a potentially frozen movie tail")
    files = {}
    for field in FIELDS:
        item = receipt["fields"][field]
        expected_movie = f"cavity_continuation_{field}.mp4"
        if item.get("movie") != expected_movie or not _sha(item.get("movie_sha256")):
            raise ValueError("Unexpected movie name or checksum")
        posters = [x for x in item.get("stills", []) if x.get("time") == 100.]
        if len(posters) != 1 or posters[0].get("png") != f"{field}_t100.png" or not _sha(posters[0].get("png_sha256")):
            raise ValueError("Expected one hash-identified native t=100 poster per field")
        limits = item["limits"]
        if (not all(math.isfinite(limits[k]) for k in ("data_min", "data_max", "vmin", "vmax"))
                or not limits["vmin"] <= limits["data_min"] <= limits["data_max"] <= limits["vmax"]
                or limits["vmin"] >= limits["vmax"]):
            raise ValueError("Displayed range clips or misstates retained extrema")
        files[expected_movie] = item["movie_sha256"]
        files[posters[0]["png"]] = posters[0]["png_sha256"]
    return files


def admit_video_verification(proof, receipt):
    if (proof.get("common_time_start"), proof.get("common_time_end"), proof.get("frames"), proof.get("fps")) != (0, 100, 201, receipt["fps"]):
        raise ValueError("Independent encoded-video checks do not match the native time axis")
    if proof.get("no_temporal_interpolation") is not True or proof.get("no_frozen_tail") is not True:
        raise ValueError("Missing native-time evidence")
    if [x["field"] for x in proof.get("movies", [])] != list(FIELDS):
        raise ValueError("Encoded-video coverage is incomplete")
    for row in proof["movies"]:
        field, stream = row["field"], row["stream"]
        if row["file"] != receipt["fields"][field]["movie"] or row["sha256"] != receipt["fields"][field]["movie_sha256"]:
            raise ValueError("Encoded-video proof describes a different movie")
        if (stream.get("codec_name") != "h264" or int(stream["nb_read_frames"]) != 201
                or Fraction(stream["r_frame_rate"]) != receipt["fps"]
                or abs(float(stream["duration"]) - 201 / receipt["fps"]) > 1e-5
                or int(stream["width"]) <= 0 or int(stream["height"]) <= 0):
            raise ValueError("Actual encoded frames, duration or playback rate disagree")


def _atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def install(source, bundle=BUNDLE):
    source, bundle = Path(source).resolve(), Path(bundle).resolve()
    if source == bundle or bundle.is_relative_to(source) or source.is_relative_to(bundle):
        raise ValueError("The retained scientific source and public bundle must be separate")
    receipt = json.loads((source / "movies.json").read_text(encoding="utf-8"))
    required = admit_receipt(receipt)
    required["movies.json"] = digest(source / "movies.json")
    video_proof = json.loads((source / "video_verification.json").read_text(encoding="utf-8"))
    admit_video_verification(video_proof, receipt)
    archive = json.loads((source / "local_verification.json").read_text(encoding="utf-8"))
    if not _sha(archive.get("archive_sha256")) or archive.get("verified_members", 0) < len(required):
        raise ValueError("Missing retained external archive verification")
    # Admit every selected source before writing any public asset. The full
    # diagnostic/archive payload stays external; it is not copied by this tool.
    for name, expected in required.items():
        if digest(_member(source, name)) != expected:
            raise ValueError("Selected source asset checksum mismatch: " + name)
    media = bundle / "media"
    media.mkdir(parents=True, exist_ok=True)
    for name, expected in required.items():
        destination = _member(media, name)
        if destination.exists() and digest(destination) == expected:
            continue
        temporary = media / (name + "." + uuid.uuid4().hex + ".part")
        try:
            shutil.copyfile(_member(source, name), temporary)
            if digest(temporary) != expected:
                raise ValueError("Copied asset failed byte verification: " + name)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    manifest = {"schema": SCHEMA, "physical_times": TIMES, "reynolds": REYNOLDS,
        "controls": CONTROLS,
        "assets": {name: {"sha256": expected, "bytes": (media / name).stat().st_size}
                   for name, expected in required.items()},
        "provenance": {"campaign_sha256": receipt["campaign_sha256"],
            "frozen_remote_renderer_sha256": receipt["render_source_sha256"],
            "video_verification_sha256": digest(source / "video_verification.json"),
            "retained_external_archive": {"sha256": archive["archive_sha256"], "bytes": archive["archive_bytes"],
                "verified_members": archive["verified_members"],
                "contents": "Full native-frame render provenance, exact remote scripts, diagnostic reports and additional stills; retained separately from this public preview"},
            "source_note": "The receipt identifies the exact remote rendering bytes. Current repository scripts may differ in text normalization; no byte-identical-source claim is made."},
        "encoded_video_verification": video_proof,
        "scope": "Selected preview assets only. The full render receipt includes archive members not copied into this public subset; original startup seeds, movies and lesson remain unchanged."}
    _atomic_text(media / "preview_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    build(bundle)
    return manifest


def _link(url, label):
    return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(label)}</a>'


def viewer_html(manifest, receipt):
    sections = []
    descriptions = {
        "speed": "Follow the development of the primary circulation and the smaller corner cells. Speed is scaled by the lid velocity U.",
        "omega": "Vorticity retains its sign: compare the wall layers and the changing interior circulation. Small structures at high Re remain sensitive to resolution.",
        "p": "Compare pressure patterns at the same physical time. Pressure is taken from the transient momentum solve with zero area mean, not from a steady pressure reconstruction."}
    for field in FIELDS:
        item = receipt["fields"][field]
        limits = item["limits"]
        stream = next(row["stream"] for row in manifest["encoded_video_verification"]["movies"] if row["field"] == field)
        scale = "Linear speed scale" if field == "speed" else (
            "Symmetric-log scale, linear for |ωL/U| ≤ 1" if field == "omega" else
            "Symmetric-log scale, linear for |p/(ρU²)| ≤ 0.01")
        sections.append(f'''<section class="movie" id="{field}"><h2>{NAMES[field]}</h2>
<p>{descriptions[field]}</p><figure>
<video controls preload="metadata" poster="media/{field}_t100.png" width="{int(stream['width'])}" height="{int(stream['height'])}" aria-label="{NAMES[field]}, seven Reynolds numbers, physical time 0 to 100">
<source src="media/{item['movie']}" type="video/mp4">Your browser cannot play this video. {_link('media/'+item['movie'], 'Download the MP4')}.</video>
<figcaption>The poster shows t⋆ = 100; playback begins at t⋆ = 0. {scale}. The common color limits are {limits['vmin']:.6g} to {limits['vmax']:.6g} and retain the full recorded extrema.</figcaption>
</figure></section>''')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cavity circulation from t⋆ = 0 to 100 | FlowMLLab</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f7f9fc;color:#172635;font:17px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1220px;margin:auto;padding:28px 24px 58px}}nav{{display:flex;flex-wrap:wrap;gap:12px 24px;font-size:.9rem}}a{{color:#125a93;text-underline-offset:3px}}
header{{padding:28px 0 18px}}h1{{font-size:clamp(1.9rem,4vw,3rem);line-height:1.15;letter-spacing:-.025em;margin:12px 0 18px}}h2{{font-size:1.5rem;margin:0 0 10px}}p{{max-width:1050px;margin:10px 0 16px}}
.eyebrow{{font-size:.82rem;letter-spacing:.12em;text-transform:uppercase;color:#526575}}.conditions{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px 26px;background:white;border:1px solid #dbe3eb;padding:20px 24px;margin:20px 0}}dt{{font-weight:650}}dd{{margin:2px 0 0;color:#435567}}aside{{padding:16px 22px;border-left:4px solid #326989;background:#edf3f7;margin:22px 0}}
.movie{{background:white;border:1px solid #dbe3eb;border-radius:8px;padding:24px;margin:28px 0}}figure{{margin:18px 0 0}}video{{display:block;width:100%;height:auto;background:white;border:1px solid #e0e7ee}}figcaption{{font-size:.9rem;color:#4b5c6d;margin-top:12px}}footer{{padding-top:18px;border-top:1px solid #dbe3eb;font-size:.92rem}}@media(max-width:650px){{main{{padding:18px 14px 40px}}.conditions{{grid-template-columns:1fr 1fr;padding:16px}}.movie{{padding:16px}}}}
</style></head><body><main data-time-start="0" data-time-end="100" data-frame-interval="0.5" data-frame-count="201">
<nav>{_link(SITE, 'FlowMLLab home')}{_link(SITE+'cavity-startup/', 'Original startup lesson: 0–10')}{_link(REPO+'docs/CAVITY_STARTUP_STATIONARITY.md', 'How to test stationarity')}</nav>
<header><div class="eyebrow">Week 1.2 · physical-time continuation</div><h1>How cavity circulation develops<br>from t⋆ = 0 to 100</h1>
<p>Watch the same unit-square cavity at Re = 100, 400, 1000, 3200, 5000, 7500 and 10000. The fluid starts from rest; the top lid starts impulsively and moves right at U = 1. The other walls remain stationary and all walls are impermeable. There is no lid ramp.</p></header>
<dl class="conditions"><div><dt>Physical time</dt><dd>t⋆ = tU/L, from 0 to 100</dd></div><div><dt>Grid</dt><dd>128 × 128 MAC cells</dd></div><div><dt>Timestep</dt><dd>Δt⋆ = 0.01</dd></div><div><dt>Pressure–velocity coupling</dt><dd>PIMPLE: 3 outer / 2 pressure corrections</dd></div><div><dt>Spatial convection</dt><dd>Central differencing</dd></div><div><dt>Displayed sampling</dt><dd>201 common instants, 0.5 time unit apart</dd></div></dl>
<aside><strong>A common endpoint is not a steady-state test.</strong> These are finite-resolution, two-dimensional trajectories. A flow that looks quiet may still change, and high-Re fields remain mesh-sensitive. Fixed-grid stationarity, tighter-coupling confirmation and timestep/mesh accuracy require separate evidence. No panel is labelled steady merely because its movie reaches t⋆ = 100.</aside>
<p>Black curves are instantaneous streamlines, not particle paths. Pressure has zero area mean. Every panel uses the same domain and fixed field scales for the entire interval. The limits include all retained native extrema, including the finer-cadence original startup frames; no field is clipped or smoothed. The {receipt['fps']}-frame/s playback rate changes viewing speed, not the physical timestamps.</p>
{''.join(sections)}
<section><h2>Use the movies as an experiment</h2><p>Pause at t⋆ = 10, 25, 50 and 100. Compare the primary circulation, the upper-left corner and the wall layers, then test your interpretation with signed profiles and time histories. A fixed corner probe is not a tracked vortex center. The stationarity guide provides five investigations and optional longer CPU commands; the original notebook keeps its small default solve.</p>
<p>{_link(REPO+'notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb', 'Open the original CPU notebook')} · {_link(REPO+'docs/CAVITY_STARTUP_STATIONARITY.md', 'Read the stationarity guide and exercises')}</p></section>
<footer><p>{_link(REPO+'results/cavity_continuation/README.md', 'Conditions and reproduction')} · {_link('media/preview_manifest.json', 'Published asset hashes')} · {_link('media/movies.json', 'Native-frame render provenance')}</p><p>The 0–10 seed data, startup movies, notebook exercises and lecture are retained separately. The public bundle contains selected previews; full diagnostic reports and exact remote rendering scripts remain in the hash-identified research archive.</p></footer>
</main></body></html>
'''


def readme_text(manifest, receipt):
    archive = manifest["provenance"]["retained_external_archive"]
    return f'''# Cavity circulation: physical time 0–100

[Watch the three matched-time movies]({SITE}cavity-continuation/) · [Original startup lesson, 0–10]({SITE}cavity-startup/) · [Stationarity guide](../../docs/CAVITY_STARTUP_STATIONARITY.md)

Compare speed and instantaneous streamlines, signed vorticity and zero-mean pressure at **Re = 100, 400, 1000, 3200, 5000, 7500 and 10000**. These previews extend the original startup trajectories using their native velocity and pressure states at the actual physical time.

| Shared condition | Setting |
|---|---|
| Domain and initial state | Unit square; fluid starts from rest |
| Walls | Impulsive top lid moving right at U = 1; other walls stationary; all impermeable |
| Grid and timestep | 128 × 128 MAC cells; Δt⋆ = 0.01 |
| Coupling and convection | PIMPLE, 3 outer / 2 pressure corrections; central convection |
| Physical interval | t⋆ = tU/L from 0 to 100 |
| Displayed frames | 201 actual common times at intervals of 0.5 |
| Playback | {receipt['fps']} frames/s; physical timestamps remain unchanged |
| Pressure gauge | Zero cell-area mean; pressure scaled by ρU² |

A common endpoint does not establish that every case is steady. The 128-cell high-Re fields remain provisional and mesh-sensitive. Use the frozen physical-window criteria and separate coupling, timestep and grid tests described in the guide. These two-dimensional movies do not establish three-dimensional stability.

## View and compare

- [Speed and instantaneous streamlines](media/cavity_continuation_speed.mp4) · [t⋆ = 100 poster](media/speed_t100.png).
- [Signed vorticity](media/cavity_continuation_omega.mp4) · [t⋆ = 100 poster](media/omega_t100.png).
- [Zero-mean pressure](media/cavity_continuation_p.mp4) · [t⋆ = 100 poster](media/p_t100.png).

Speed uses a linear scale. Signed vorticity uses a symmetric-log scale, linear for |ωL/U| ≤ 1; pressure is linear within |p/(ρU²)| ≤ 0.01 and symmetric-log outside. Each field has fixed cross-case and cross-time limits covering all retained native extrema, including the original finer-cadence startup samples. No temporal interpolation, repeated frozen tail, smoothing or clipping is used. Streamlines describe the velocity at an instant, not particle paths.

Pause at times 10, 25, 50 and 100, then compare physical profiles and signed probe histories. A fixed upper-left probe is not a tracked vortex center. Follow the [five stationarity investigations](../../docs/CAVITY_STARTUP_STATIONARITY.md#five-student-investigations) to distinguish a changing field, a fixed-grid candidate and numerical accuracy. The [original notebook](../../notebooks/week01_2/W1_2_Cavity_Startup_Movies.ipynb) and all its prior exercises remain unchanged.

## Evidence and reproduction

[Public asset manifest](media/preview_manifest.json) records every selected file's SHA-256 and size, the common time axis, encoded-video checks and the retained archive identity. [Full render receipt](media/movies.json) records exact native-frame identities for all seven cases. That receipt also names additional stills and PDFs kept in the external archive; they are not implied to be part of this public subset.

The separately retained research archive has SHA-256 `{archive['sha256']}` ({archive['bytes']:,} bytes; {archive['verified_members']} verified members). It contains the full diagnostic reports and exact remote rendering/analysis scripts. Its renderer hash is recorded explicitly; the currently tracked renderer is not claimed to be byte-identical after source text normalization. No live job status or future steady-state outcome is inferred from this immutable movie receipt.

From the repository root, install an already verified source bundle with:

```bash
python qa/build_cavity_continuation_viewer.py --source /path/to/verified/media_next
```

Rebuild the viewer from its retained public files, or check it without rewriting:

```bash
python qa/build_cavity_continuation_viewer.py
python qa/build_cavity_continuation_viewer.py --check
python qa/build_site.py --output tmp/pages-site
python qa/check_site.py tmp/pages-site
```

These commands package and verify retained evidence; they do not run CFD or train a network. The [original 0–10 data and lesson](../cavity_startup/README.md) remain independent of this longer preview.
'''


class _ViewerAudit(HTMLParser):
    def __init__(self):
        super().__init__(); self.videos = []; self.sources = []; self.timeline = None
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "main": self.timeline = attrs
        if tag == "video": self.videos.append(attrs)
        if tag == "source": self.sources.append(attrs)


def check_bundle(bundle=BUNDLE, *, require_readme=True):
    bundle = Path(bundle).resolve(); media = bundle / "media"
    receipt = json.loads((media / "movies.json").read_text(encoding="utf-8"))
    required = admit_receipt(receipt)
    required["movies.json"] = digest(media / "movies.json")
    manifest = json.loads((media / "preview_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("physical_times") != TIMES
            or manifest.get("reynolds") != REYNOLDS or manifest.get("controls") != CONTROLS
            or set(manifest.get("assets", {})) != set(required)):
        raise ValueError("Public continuation manifest has incompatible coverage or assets")
    if manifest["provenance"]["campaign_sha256"] != receipt["campaign_sha256"] or manifest["provenance"]["frozen_remote_renderer_sha256"] != receipt["render_source_sha256"]:
        raise ValueError("Public provenance disagrees with the unchanged remote render receipt")
    if manifest["provenance"].get("viewer_builder_sha256") != digest(__file__):
        raise ValueError("Viewer builder changed; rebuild the generated documents and receipt")
    admit_video_verification(manifest["encoded_video_verification"], receipt)
    for name, expected in required.items():
        path = _member(media, name)
        if manifest["assets"][name] != {"sha256": expected, "bytes": path.stat().st_size} or digest(path) != expected:
            raise ValueError("Public movie/poster/receipt checksum mismatch: " + name)
    if {p.name for p in media.iterdir()} != set(required) | {"preview_manifest.json"}:
        raise ValueError("Public media contains an unexpected or missing file")
    from PIL import Image
    for field in FIELDS:
        with Image.open(media / f"{field}_t100.png") as image:
            if image.mode not in ("RGB", "RGBA") or min(image.size) < 500:
                raise ValueError("Invalid continuation poster image")
            image.verify()
    page = (bundle / "index.html").read_text(encoding="utf-8")
    if page != viewer_html(manifest, receipt):
        raise ValueError("Viewer conditions or links differ from its reproducible template")
    audit = _ViewerAudit(); audit.feed(page)
    if len(audit.videos) != 3 or len(audit.sources) != 3:
        raise ValueError("Expected exactly three visible movie players")
    expected_axis = {"data-time-start": "0", "data-time-end": "100", "data-frame-interval": "0.5", "data-frame-count": "201"}
    if any(audit.timeline.get(k) != v for k, v in expected_axis.items()):
        raise ValueError("Viewer time-axis labels are inconsistent")
    for field, video, source in zip(FIELDS, audit.videos, audit.sources):
        if (source.get("src") != f"media/{receipt['fields'][field]['movie']}" or source.get("type") != "video/mp4"
                or video.get("poster") != f"media/{field}_t100.png" or "controls" not in video):
            raise ValueError("A player refers to the wrong verified movie or poster")
        if not (bundle / source["src"]).is_file() or not (bundle / video["poster"]).is_file():
            raise ValueError("A local video/poster link is missing")
    if require_readme and (bundle / "README.md").read_text(encoding="utf-8") != readme_text(manifest, receipt):
        raise ValueError("Continuation README differs from its reproducible template")
    return {"route": "cavity-continuation/", "movies": 3, "posters": 3, "frames": 201,
            "time_start": 0, "time_end": 100, "asset_bytes": sum(x["bytes"] for x in manifest["assets"].values()),
            "manifest_sha256": digest(media / "preview_manifest.json")}


def build(bundle=BUNDLE):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "media/preview_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((bundle / "media/movies.json").read_text(encoding="utf-8"))
    admit_receipt(receipt)
    manifest["provenance"]["viewer_builder_sha256"] = digest(__file__)
    _atomic_text(bundle / "media/preview_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(bundle / "index.html", viewer_html(manifest, receipt))
    _atomic_text(bundle / "README.md", readme_text(manifest, receipt))
    return check_bundle(bundle)


def export(site):
    check_bundle()
    target = Path(site) / "cavity-continuation"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUNDLE / "index.html", target / "index.html")
    shutil.copytree(BUNDLE / "media", target / "media", dirs_exist_ok=True)
    return check_bundle(target, require_readme=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", type=Path)
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.source: install(args.source)
    result = check_bundle() if args.check or args.source else build()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
