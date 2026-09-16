"""Build the FlowMLLab README hero from retained course results."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "flowmllab_hero.png"
W, H = 1800, 680


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def cover(path: Path, size: tuple[int, int], crop: tuple[float, float, float, float]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = crop
    image = image.crop((int(x0 * image.width), int(y0 * image.height), int(x1 * image.width), int(y1 * image.height)))
    scale = max(size[0] / image.width, size[1] / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = (image.width - size[0]) // 2
    top = (image.height - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


def card(canvas: Image.Image, x: int, title: str, subtitle: str, image: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    y, width, height = 286, 520, 302
    canvas.paste(image, (x, y))
    # Dark fade keeps labels readable over fields with very different color maps.
    fade = Image.new("RGBA", (width, 92), (8, 19, 38, 190))
    canvas.paste(fade, (x, y + height - 92), fade)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=18, outline=(91, 199, 229), width=3)
    draw.text((x + 22, y + height - 78), title, font=font(27, True), fill="white")
    draw.text((x + 22, y + height - 42), subtitle, font=font(19), fill=(207, 232, 244))


def main() -> None:
    canvas = Image.new("RGB", (W, H), (7, 19, 38))
    draw = ImageDraw.Draw(canvas)

    # Subtle streamlines tie the three retained experiments together.
    for offset, color in [(0, (15, 78, 111)), (24, (10, 55, 86)), (48, (12, 43, 72))]:
        points = [(0, 210 + offset), (390, 160 + offset), (760, 230 + offset), (1170, 170 + offset), (1800, 220 + offset)]
        draw.line(points, fill=color, width=3)

    draw.text((80, 52), "FlowMLLab", font=font(72, True), fill=(255, 255, 255))
    draw.text((82, 143), "From flow physics to trustworthy machine learning", font=font(34), fill=(91, 199, 229))
    draw.text(
        (82, 202),
        "Reproducible solvers  •  transparent baselines  •  blind tests  •  physical validation",
        font=font(23),
        fill=(201, 218, 231),
    )

    cavity = cover(
        ROOT / "results/pod_deeponet/pod_deeponet_ghia_validation.png",
        (520, 302),
        (0.00, 0.00, 0.31, 0.52),
    )
    cylinder = cover(
        ROOT / "results/cylinder_phase/re095_phase_stable_poster.png",
        (520, 302),
        (0.02, 0.22, 0.34, 0.86),
    )
    nozzle = cover(
        ROOT / "results/mahdavi_deeponet/nozzle_flowmllab/nozzle_back_pressure_P25_contours.png",
        (520, 302),
        (0.01, 0.13, 0.19, 0.57),
    )

    card(canvas, 80, "Cavity flow", "CFD + POD–DeepONet", cavity)
    card(canvas, 640, "Cylinder wake", "LBM + autonomous surrogate", cylinder)
    card(canvas, 1200, "Micro-nozzle", "DSMC + shock-aligned operator", nozzle)

    draw.text((80, 625), "Numerical foundations → scientific machine learning → evidence", font=font(22, True), fill=(241, 166, 73))
    draw.text((1396, 625), "MIE 690A  •  UMass Amherst", font=font(20), fill=(179, 199, 214))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)


if __name__ == "__main__":
    main()
