"""Replace the duplicated Week 15 bitmap footer and refresh provenance hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("DejaVu Sans is required for deterministic footers")
    return ImageFont.truetype(str(path), size=size)


def repair(path: Path) -> None:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (2800, 4725):
        raise ValueError(f"unexpected Week 15 figure size for {path}: {image.size}")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 4618, image.width, image.height), fill="white")
    footer_font = font(22)
    box = draw.textbbox((0, 0), FOOTER, font=footer_font)
    width = box[2] - box[0]
    draw.text(((image.width - width) / 2, 4655), FOOTER, fill="black", font=footer_font)
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary, format="PNG", compress_level=6)
    temporary.replace(path)


def main() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    by_name = {item["figure"]: item for item in provenance}
    if set(by_name) != set(FIGURES):
        raise ValueError("figure_provenance.json does not contain the six expected figures")
    for name in FIGURES:
        path = RESULTS / name
        repair(path)
        by_name[name]["sha256"] = sha256(path)
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for name in FIGURES:
        print(name, by_name[name]["sha256"])


if __name__ == "__main__":
    main()
