"""Give the six Week 15 comparison figures one readable footer and refresh provenance hashes.

The v1.8.2 renders carry two footer lines: the original long line, clipped at the
right margin and overlapping the ``x/H`` label of the bottom panel, and a second
short line under it. The v1.8.3 repair painted the whole footer band white and
wrote one line, which also removed the bottom panel's ``x/H`` label.

This version starts from the v1.8.2 renders (``--source`` directory, or
``git show v1.8.2:...`` when omitted), clears everything below the bottom panel's
tick labels, copies the ``x/H`` label of the panel above into the bottom panel
(the seven panels share one horizontal layout, so the copy is pixel-exact), and
writes a single footer line that fits the page. Only pixels below the bottom
panel's tick labels change; the rendered fields and their provenance inputs are
untouched. The refreshed SHA-256 values are written to ``figure_provenance.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/week15_postaudit"
PROVENANCE = RESULTS / "figure_provenance.json"
FIGURES = (
    "core_g051_Re25.png",
    "extended_g051_Re25.png",
    "core_g049_Re50.png",
    "extended_g049_Re50.png",
    "core_g049_Re100.png",
    "extended_g049_Re100.png",
)
FOOTER = (
    "Shared banded scale; true 5:1 aspect. White streamlines use each row's field. "
    "IoU: u < -0.01, 4-connectivity, components >= 8 cells. Protocols shown separately."
)
SOURCE_TAG = "v1.8.2"
AXES_COLUMNS = slice(350, 1750)  # horizontal band that holds the panel axes and their labels
DARK = 128


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not path.is_file():
        raise FileNotFoundError("DejaVu Sans is required for deterministic footers")
    return ImageFont.truetype(str(path), size=size)


def text_runs(gray: np.ndarray, start: int, stop: int) -> list[tuple[int, int]]:
    """Return [first_row, last_row] runs of rows that contain dark pixels in the axes band."""
    band = (gray[start:stop, AXES_COLUMNS] < DARK).sum(axis=1)
    runs: list[list[int]] = []
    for offset, count in enumerate(band):
        row = start + offset
        if count == 0:
            continue
        if runs and row == runs[-1][1] + 1:
            runs[-1][1] = row
        else:
            runs.append([row, row])
    return [(a, b) for a, b in runs]


def repair(source: Path, target: Path) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if image.size != (2800, 4725):
        raise ValueError(f"unexpected Week 15 figure size for {source}: {image.size}")
    gray = np.asarray(image.convert("L"))
    height = image.height
    # Runs in the lower third of the page: ..., panel N-1 ticks, panel N-1 "x/H", panel N body,
    # panel N ticks, then the footer band that also holds panel N's "x/H".
    runs = text_runs(gray, height - 1000, height)
    bodies = [run for run in runs if run[1] - run[0] > 200]
    if len(bodies) < 2:
        raise ValueError(f"could not locate the two lowest panels in {source}")
    upper_body, lower_body = bodies[-2], bodies[-1]
    after_upper = [run for run in runs if run[0] > upper_body[1] and run[1] < lower_body[0]]
    after_lower = [run for run in runs if run[0] > lower_body[1]]
    if len(after_upper) < 2 or not after_lower:
        raise ValueError(f"could not locate tick and axis labels in {source}")
    upper_ticks, upper_label = after_upper[0], after_upper[1]
    lower_ticks = after_lower[0]
    shift = lower_ticks[0] - upper_ticks[0]
    label_top, label_bottom = upper_label[0] - 6, upper_label[1] + 6
    clear_from = lower_ticks[1] + 4

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, clear_from, image.width, height), fill="white")
    label = image.crop((AXES_COLUMNS.start, label_top, AXES_COLUMNS.stop, label_bottom))
    image.paste(label, (AXES_COLUMNS.start, label_top + shift))

    footer_font = font(22)
    box = draw.textbbox((0, 0), FOOTER, font=footer_font)
    width = box[2] - box[0]
    footer_top = label_bottom + shift + 18
    if footer_top + (box[3] - box[1]) > height - 8:
        raise ValueError("footer does not fit under the bottom panel")
    draw.text(((image.width - width) / 2, footer_top), FOOTER, fill="black", font=footer_font)
    temporary = target.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", compress_level=6)
    temporary.replace(target)


def source_directory(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    directory = Path(tempfile.mkdtemp(prefix="week15_figures_"))
    for name in FIGURES:
        blob = subprocess.run(
            ["git", "show", f"{SOURCE_TAG}:results/week15_postaudit/{name}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        (directory / name).write_bytes(blob)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"directory with the matplotlib renders (default: the {SOURCE_TAG} files from git)",
    )
    args = parser.parse_args()
    source = source_directory(args.source)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    by_name = {item["figure"]: item for item in provenance}
    if set(by_name) != set(FIGURES):
        raise ValueError("figure_provenance.json does not contain the six expected figures")
    for name in FIGURES:
        repair(source / name, RESULTS / name)
        by_name[name]["sha256"] = sha256(RESULTS / name)
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for name in FIGURES:
        print(name, by_name[name]["sha256"])


if __name__ == "__main__":
    main()
