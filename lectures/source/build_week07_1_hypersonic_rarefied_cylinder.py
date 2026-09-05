"""Build the Week-7.1 lecture PDF with ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "week07_1_hypersonic_rarefied_cylinder.pdf"
METRICS = ROOT / "results" / "hypersonic_cylinder_week7_1" / "metrics.json"
FIELD_FIGURE = (
    ROOT / "results" / "hypersonic_cylinder_week7_1" / "mach85_baseline_audit.png"
)

PAGE = landscape((7.5 * inch, 13.333 * inch))
WIDTH, HEIGHT = PAGE
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#176B87")
TEAL = colors.HexColor("#16A6A1")
PALE = colors.HexColor("#EAF5F5")
ORANGE = colors.HexColor("#F2A541")
RED = colors.HexColor("#C94C4C")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
LIGHT = colors.HexColor("#F5F8FA")
GREEN = colors.HexColor("#2F855A")

BODY = ParagraphStyle(
    "body", fontName="Helvetica", fontSize=17, leading=22, textColor=INK,
)
SMALL = ParagraphStyle(
    "small", fontName="Helvetica", fontSize=12.5, leading=16.5, textColor=INK,
)
TINY = ParagraphStyle(
    "tiny", fontName="Helvetica", fontSize=9.5, leading=12.5, textColor=MUTED,
)
CENTER = ParagraphStyle(
    "center", parent=BODY, alignment=TA_CENTER,
)


class Deck:
    def __init__(self, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        self.canvas = canvas.Canvas(str(target), pagesize=PAGE)
        self.canvas.setTitle("Week 7.1 - Rarefied Hypersonic Cylinder")
        self.canvas.setAuthor("Ehsan Roohi")
        self.canvas.setSubject(
            "DSMC cylinder fields, Fusion-DeepONet, strong baselines, and uncertainty"
        )
        self.page = 0

    def begin(self, title: str, kicker: str = "WEEK 7.1") -> None:
        self.page += 1
        c = self.canvas
        c.setFillColor(colors.white)
        c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.rect(0, HEIGHT - 0.78 * inch, WIDTH, 0.78 * inch, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, HEIGHT - 0.84 * inch, WIDTH, 0.06 * inch, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.55 * inch, HEIGHT - 0.47 * inch, kicker)
        c.setFont("Helvetica-Bold", 25)
        c.drawString(1.75 * inch, HEIGHT - 0.52 * inch, title)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(0.55 * inch, 0.25 * inch, "FlowMLLab | MIE 690A - AI in Fluid Mechanics")
        c.drawRightString(WIDTH - 0.55 * inch, 0.25 * inch, str(self.page))

    def paragraph(
        self,
        text: str,
        x: float,
        y_top: float,
        width: float,
        height: float,
        style: ParagraphStyle = BODY,
    ) -> None:
        paragraph = Paragraph(text, style)
        _, used = paragraph.wrap(width, height)
        paragraph.drawOn(self.canvas, x, y_top - used)

    def box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill=LIGHT,
        stroke=colors.HexColor("#D6E2EA"),
        radius: float = 10,
    ) -> None:
        self.canvas.setFillColor(fill)
        self.canvas.setStrokeColor(stroke)
        self.canvas.setLineWidth(1)
        self.canvas.roundRect(x, y, width, height, radius, fill=1, stroke=1)

    def label(self, text: str, x: float, y: float, *, color=TEAL) -> None:
        c = self.canvas
        c.setFillColor(color)
        c.roundRect(x, y - 0.25 * inch, 1.15 * inch, 0.34 * inch, 7, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(x + 0.575 * inch, y - 0.14 * inch, text)

    def end(self) -> None:
        self.canvas.showPage()

    def save(self) -> None:
        self.canvas.save()


def bullet_list(items: list[str]) -> str:
    return "<br/>".join(f"<font color='#16A6A1'>&bull;</font> {item}" for item in items)


def draw_arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=TEAL) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(2.2)
    c.line(x1, y1, x2, y2)
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 10
    for offset in (2.55, -2.55):
        c.line(
            x2,
            y2,
            x2 + length * math.cos(angle + offset),
            y2 + length * math.sin(angle + offset),
        )


def title_slide(deck: Deck) -> None:
    deck.page += 1
    c = deck.canvas
    c.setFillColor(NAVY)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.circle(WIDTH - 1.25 * inch, HEIGHT - 1.15 * inch, 0.7 * inch, fill=1, stroke=0)
    c.setStrokeColor(ORANGE)
    c.setLineWidth(8)
    c.arc(WIDTH - 2.2 * inch, -0.7 * inch, WIDTH + 0.5 * inch, 2.0 * inch, 40, 115)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, HEIGHT - 0.8 * inch, "FLOWMLLAB | INCREMENTAL LECTURE 7.1")
    c.setFont("Helvetica-Bold", 38)
    c.drawString(0.75 * inch, HEIGHT - 2.15 * inch, "Rarefied hypersonic cylinder")
    c.setFillColor(colors.HexColor("#BFE9E7"))
    c.setFont("Helvetica-Bold", 28)
    c.drawString(0.75 * inch, HEIGHT - 2.75 * inch, "Operator learning, baselines, and honest UQ")
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 17)
    c.drawString(0.78 * inch, HEIGHT - 3.55 * inch, "From author-supplied DSMC fields to a leakage-controlled teaching experiment")
    c.setFillColor(colors.HexColor("#D9E2EC"))
    c.setFont("Helvetica", 12)
    c.drawString(0.78 * inch, 0.78 * inch, "Ehsan Roohi | MIE 690A - AI in Fluid Mechanics | University of Massachusetts Amherst")
    c.drawString(0.78 * inch, 0.50 * inch, "Companion: notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb")
    deck.end()


def build() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    deck = Deck(OUTPUT)
    title_slide(deck)

    deck.begin("What this increment adds")
    deck.box(0.65 * inch, 0.75 * inch, 5.85 * inch, 5.45 * inch, fill=PALE)
    deck.label("OUTCOMES", 0.95 * inch, 5.75 * inch)
    deck.paragraph(
        bullet_list([
            "Separate freestream Mach from the predicted local Mach field.",
            "Formulate a parameter-to-field operator for DSMC cylinder data.",
            "Freeze whole-case splits before model selection.",
            "Read the branch, trunk, layerwise fusion, and output head.",
            "Challenge the neural model with a structured interpolation baseline.",
            "Audit ensemble spread with empirical coverage, not visual confidence.",
        ]),
        0.95 * inch, 5.35 * inch, 5.25 * inch, 4.2 * inch,
    )
    deck.box(6.85 * inch, 3.65 * inch, 5.78 * inch, 2.55 * inch, fill=colors.HexColor("#FFF4E5"), stroke=ORANGE)
    deck.label("CLAIM", 7.15 * inch, 5.75 * inch, color=ORANGE)
    deck.paragraph(
        "The lab executes a compact, author-released DSMC derivative. The default CPU model is a <b>teaching analog</b>; it does not reproduce the paper's full-resolution accuracy or speedup.",
        7.15 * inch, 5.33 * inch, 5.15 * inch, 1.35 * inch,
    )
    deck.box(6.85 * inch, 0.75 * inch, 5.78 * inch, 2.55 * inch, fill=colors.HexColor("#FDECEC"), stroke=RED)
    deck.label("RULE", 7.15 * inch, 2.83 * inch, color=RED)
    deck.paragraph(
        "If a transparent baseline wins, retain that result. Complexity is not evidence.",
        7.15 * inch, 2.38 * inch, 5.15 * inch, 1.0 * inch,
        ParagraphStyle("rule", parent=BODY, fontName="Helvetica-Bold", fontSize=21, leading=26, textColor=RED),
    )
    deck.end()

    deck.begin("Physics: three scales, three questions")
    c = deck.canvas
    cy = 3.55 * inch
    c.setFillColor(colors.HexColor("#D9E2EC"))
    c.circle(3.0 * inch, cy, 0.65 * inch, fill=1, stroke=0)
    c.setStrokeColor(ORANGE)
    c.setLineWidth(7)
    c.arc(1.75 * inch, cy - 1.25 * inch, 4.25 * inch, cy + 1.25 * inch, 105, 150)
    draw_arrow(c, 0.75 * inch, cy, 1.75 * inch, cy, color=BLUE)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(0.75 * inch, cy + 0.3 * inch, "U_inf")
    deck.box(5.0 * inch, 4.35 * inch, 7.55 * inch, 1.55 * inch, fill=PALE)
    deck.paragraph(
        "<b>Mach number:</b> M_inf = U_inf/a_inf controls compressibility. The target M(x,y) is a different, spatial field.",
        5.3 * inch, 5.55 * inch, 6.95 * inch, 1.0 * inch,
    )
    deck.box(5.0 * inch, 2.55 * inch, 7.55 * inch, 1.55 * inch, fill=colors.HexColor("#FFF4E5"), stroke=ORANGE)
    deck.paragraph(
        "<b>Knudsen number:</b> Kn = lambda/D measures rarefaction. As lambda becomes non-negligible, continuum closure and no-slip assumptions need qualification.",
        5.3 * inch, 3.75 * inch, 6.95 * inch, 1.0 * inch,
    )
    deck.box(5.0 * inch, 0.75 * inch, 7.55 * inch, 1.55 * inch, fill=colors.HexColor("#FDECEC"), stroke=RED)
    deck.paragraph(
        "<b>Sampling scale:</b> DSMC fields estimate particle moments. A sharp shock and finite particle count make target noise spatially nonuniform.",
        5.3 * inch, 1.95 * inch, 6.95 * inch, 1.0 * inch,
    )
    deck.paragraph("cylinder", 2.35 * inch, 2.55 * inch, 1.3 * inch, 0.3 * inch, SMALL)
    deck.paragraph("shock layer", 2.1 * inch, 5.25 * inch, 1.8 * inch, 0.3 * inch, SMALL)
    deck.end()

    deck.begin("The learning object is an operator")
    deck.box(0.75 * inch, 3.45 * inch, 2.4 * inch, 1.55 * inch, fill=colors.HexColor("#FFF4E5"), stroke=ORANGE)
    deck.paragraph("<b>Branch input</b><br/>freestream M_inf", 1.05 * inch, 4.55 * inch, 1.8 * inch, 0.8 * inch, CENTER)
    draw_arrow(deck.canvas, 3.2 * inch, 4.22 * inch, 4.25 * inch, 4.22 * inch)
    deck.box(4.3 * inch, 2.8 * inch, 4.65 * inch, 2.85 * inch, fill=PALE, stroke=TEAL)
    deck.paragraph(
        "<b>G_theta: M_inf -> q(x,y)</b><br/><br/>q = [ local Mach, T/T_inf, p/p_inf ]",
        4.65 * inch, 4.95 * inch, 3.95 * inch, 1.4 * inch,
        ParagraphStyle("operator", parent=CENTER, fontName="Helvetica-Bold", fontSize=20, leading=28),
    )
    draw_arrow(deck.canvas, 9.0 * inch, 4.22 * inch, 10.0 * inch, 4.22 * inch)
    deck.box(10.05 * inch, 3.45 * inch, 2.55 * inch, 1.55 * inch, fill=colors.HexColor("#EDF2F7"))
    deck.paragraph("<b>Query</b><br/>(x,y) -> q", 10.4 * inch, 4.55 * inch, 1.85 * inch, 0.8 * inch, CENTER)
    deck.box(0.75 * inch, 0.75 * inch, 11.85 * inch, 1.45 * inch, fill=LIGHT)
    deck.paragraph(
        "A field surrogate must be judged at unseen <b>flow cases</b>, not merely unseen grid points from a case already used during training.",
        1.1 * inch, 1.77 * inch, 11.15 * inch, 0.7 * inch,
        ParagraphStyle("statement", parent=CENTER, fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=NAVY),
    )
    deck.end()

    deck.begin("Data audit: small derivative, complete provenance")
    stat_items = [
        ("20", "Mach cases"),
        ("400 x 400", "source grid per case"),
        ("50 x 50", "deterministic selection"),
        ("44,500", "finite retained points"),
        ("0.5 MB", "committed NPZ"),
    ]
    for index, (value, label) in enumerate(stat_items):
        x = 0.55 * inch + index * 2.55 * inch
        deck.box(x, 4.25 * inch, 2.22 * inch, 1.55 * inch, fill=PALE)
        deck.paragraph(
            f"<b>{value}</b><br/><font size='12'>{label}</font>", x + 0.1 * inch, 5.38 * inch, 2.02 * inch, 0.9 * inch,
            ParagraphStyle("stat", parent=CENTER, fontSize=23, leading=25, textColor=NAVY),
        )
    deck.box(0.7 * inch, 1.0 * inch, 5.75 * inch, 2.65 * inch, fill=colors.HexColor("#EDF2F7"))
    deck.label("KEPT", 1.0 * inch, 3.2 * inch)
    deck.paragraph(
        bullet_list([
            "M_inf, x, y",
            "local Mach, T/T_inf, p/p_inf",
            "case ID and original source-grid row",
            "archive and derivative SHA-256 hashes",
        ]),
        1.0 * inch, 2.8 * inch, 5.0 * inch, 1.55 * inch, SMALL,
    )
    deck.box(6.85 * inch, 1.0 * inch, 5.78 * inch, 2.65 * inch, fill=colors.HexColor("#FDECEC"), stroke=RED)
    deck.label("EXCLUDED", 7.15 * inch, 3.2 * inch, color=RED)
    deck.paragraph(
        bullet_list([
            "1.4 GB source ZIP and 4.9 GB expanded files",
            "training logs and duplicate checkpoints",
            "historical scripts with inconsistent protocols",
            "research checkpoints not needed for the lab",
        ]),
        7.15 * inch, 2.8 * inch, 5.0 * inch, 1.55 * inch, SMALL,
    )
    deck.end()

    deck.begin("Whole-case splitting prevents a false claim")
    columns = [
        ("TRAIN", "5, 6, 7, 8, 9, 10, 11, 12, 13, 14", TEAL),
        ("VALIDATION", "8.25, 8.75, 9.25, 9.75", BLUE),
        ("BLIND INTERP.", "5.5, 6.5, 7.5, 8.5, 9.5", ORANGE),
        ("BLIND EXTRA.", "15", RED),
    ]
    for index, (label, cases, color) in enumerate(columns):
        x = 0.55 * inch + index * 3.16 * inch
        deck.box(x, 3.45 * inch, 2.86 * inch, 2.25 * inch, fill=colors.white, stroke=color)
        deck.label(label, x + 0.25 * inch, 5.25 * inch, color=color)
        deck.paragraph(cases, x + 0.22 * inch, 4.65 * inch, 2.42 * inch, 0.8 * inch, CENTER)
    deck.box(0.75 * inch, 0.78 * inch, 11.85 * inch, 1.75 * inch, fill=colors.HexColor("#FFF4E5"), stroke=ORANGE)
    deck.paragraph(
        "<b>Wrong question:</b> can the network fill random points from a Mach case it has already seen?<br/><b>Scientific question:</b> can it predict a complete field at an operating condition excluded from fitting and tuning?",
        1.05 * inch, 2.15 * inch, 11.25 * inch, 1.0 * inch,
        ParagraphStyle("split", parent=BODY, fontSize=19, leading=26),
    )
    deck.end()

    deck.begin("The mandatory baseline is field interpolation")
    deck.box(0.75 * inch, 4.15 * inch, 11.85 * inch, 1.4 * inch, fill=PALE, stroke=TEAL)
    deck.paragraph(
        "q_hat(M_q) = q(M_a) + [(M_q - M_a)/(M_b - M_a)] [q(M_b) - q(M_a)]",
        1.1 * inch, 5.12 * inch, 11.15 * inch, 0.65 * inch,
        ParagraphStyle("equation", parent=CENTER, fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=NAVY),
    )
    interp = metrics["splits"]["interpolation"]["linear_case_baseline_relative_l2"]
    extra = metrics["splits"]["extrapolation"]["linear_case_baseline_relative_l2"]
    headers = ("Split", "Local Mach", "Temperature", "Pressure")
    rows = [
        ("Blind interpolation", interp["local_mach"], interp["temperature_ratio"], interp["pressure_ratio"]),
        ("Mach 15 extrapolation", extra["local_mach"], extra["temperature_ratio"], extra["pressure_ratio"]),
    ]
    x_positions = [1.0, 4.45, 7.0, 9.75]
    for x, header in zip(x_positions, headers):
        deck.canvas.setFillColor(NAVY)
        deck.canvas.setFont("Helvetica-Bold", 14)
        deck.canvas.drawString(x * inch, 3.45 * inch, header)
    for row_index, row in enumerate(rows):
        y = (2.78 - row_index * 0.7) * inch
        deck.canvas.setFillColor(LIGHT if row_index == 0 else colors.white)
        deck.canvas.rect(0.8 * inch, y - 0.18 * inch, 11.55 * inch, 0.55 * inch, fill=1, stroke=0)
        deck.canvas.setFillColor(INK)
        deck.canvas.setFont("Helvetica-Bold" if row_index == 0 else "Helvetica", 14)
        deck.canvas.drawString(x_positions[0] * inch, y, row[0])
        for x, value in zip(x_positions[1:], row[1:]):
            deck.canvas.drawString(x * inch, y, f"{100 * value:.3f}%")
    deck.paragraph(
        "Result: a neural operator must beat a sub-1% structured baseline here, or justify itself through a different deployment constraint.",
        0.95 * inch, 1.35 * inch, 11.4 * inch, 0.55 * inch,
        ParagraphStyle("baseline_result", parent=CENTER, fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=GREEN),
    )
    deck.end()

    deck.begin("Reviewed Fusion-DeepONet topology")
    c = deck.canvas
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(INK)
    c.drawString(0.7 * inch, 5.75 * inch, "M_inf")
    c.drawString(0.7 * inch, 2.65 * inch, "(x,y)")
    branch_y, trunk_y = 5.25 * inch, 2.2 * inch
    previous_branch = 1.45 * inch
    previous_trunk = 1.45 * inch
    for layer in range(4):
        x = (2.0 + layer * 1.75) * inch
        deck.box(x, branch_y - 0.42 * inch, 1.25 * inch, 0.82 * inch, fill=colors.HexColor("#FFF4E5"), stroke=ORANGE)
        deck.box(x, trunk_y - 0.42 * inch, 1.25 * inch, 0.82 * inch, fill=PALE, stroke=TEAL)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(x + 0.625 * inch, branch_y, f"B{layer + 1}: 256 tanh")
        c.drawCentredString(x + 0.625 * inch, trunk_y, f"T{layer + 1}: 256 tanh")
        draw_arrow(c, previous_branch, branch_y, x, branch_y, color=ORANGE)
        draw_arrow(c, previous_trunk, trunk_y, x, trunk_y, color=TEAL)
        draw_arrow(c, x + 0.625 * inch, branch_y - 0.45 * inch, x + 0.625 * inch, trunk_y + 0.48 * inch, color=BLUE)
        previous_branch = x + 1.25 * inch
        previous_trunk = x + 1.25 * inch
    deck.box(9.25 * inch, 3.2 * inch, 1.35 * inch, 1.15 * inch, fill=colors.HexColor("#EDF2F7"), stroke=BLUE)
    c.setFillColor(INK)
    c.drawCentredString(9.925 * inch, 3.78 * inch, "inner product")
    draw_arrow(c, previous_branch, branch_y, 9.25 * inch, 4.05 * inch, color=ORANGE)
    draw_arrow(c, previous_trunk, trunk_y, 9.25 * inch, 3.48 * inch, color=TEAL)
    deck.box(11.1 * inch, 3.2 * inch, 1.55 * inch, 1.15 * inch, fill=colors.HexColor("#FDECEC"), stroke=RED)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(11.875 * inch, 3.85 * inch, "Dense(3)")
    c.setFont("Helvetica", 10)
    c.drawCentredString(11.875 * inch, 3.55 * inch, "M, T/T_inf, p/p_inf")
    draw_arrow(c, 10.6 * inch, 3.78 * inch, 11.1 * inch, 3.78 * inch, color=RED)
    deck.paragraph(
        "Each block includes dropout in the reviewed stored model. Branch state is added to the matching trunk depth before the final branch-trunk combination.",
        1.2 * inch, 1.18 * inch, 11.0 * inch, 0.55 * inch, SMALL,
    )
    deck.end()

    deck.begin("Training protocol: every choice changes the claim")
    items = [
        ("1", "Fit scalers", "Training cases only; MinMax inputs and standardized targets."),
        ("2", "Optimize", "Adam with validation-monitored early stopping and learning-rate reduction."),
        ("3", "Weight", "Reviewed archive loss: MSE_M + MSE_T + 5 MSE_p in standardized space."),
        ("4", "Repeat", "Five independent initializations; preserve member predictions."),
        ("5", "Open blind cases", "After design is frozen; report interpolation and extrapolation separately."),
    ]
    for index, (number, title, detail) in enumerate(items):
        y = (5.65 - index * 1.02) * inch
        deck.canvas.setFillColor(TEAL if index < 4 else ORANGE)
        deck.canvas.circle(0.95 * inch, y, 0.22 * inch, fill=1, stroke=0)
        deck.canvas.setFillColor(colors.white)
        deck.canvas.setFont("Helvetica-Bold", 12)
        deck.canvas.drawCentredString(0.95 * inch, y - 4, number)
        deck.canvas.setFillColor(NAVY)
        deck.canvas.setFont("Helvetica-Bold", 16)
        deck.canvas.drawString(1.35 * inch, y + 0.06 * inch, title)
        deck.paragraph(detail, 3.5 * inch, y + 0.22 * inch, 8.85 * inch, 0.55 * inch, SMALL)
    deck.end()

    deck.begin("Deep ensembles: disagreement is not calibration")
    deck.box(0.7 * inch, 3.55 * inch, 3.0 * inch, 2.15 * inch, fill=colors.HexColor("#FFF4E5"), stroke=ORANGE)
    deck.paragraph("<b>Five fits</b><br/>different initial weights<br/>same frozen protocol", 1.05 * inch, 5.15 * inch, 2.3 * inch, 1.2 * inch, CENTER)
    draw_arrow(deck.canvas, 3.75 * inch, 4.62 * inch, 4.75 * inch, 4.62 * inch)
    deck.box(4.8 * inch, 3.55 * inch, 3.35 * inch, 2.15 * inch, fill=PALE, stroke=TEAL)
    deck.paragraph("<b>At every point</b><br/>mean mu(x,y)<br/>sample spread s(x,y)", 5.2 * inch, 5.15 * inch, 2.55 * inch, 1.2 * inch, CENTER)
    draw_arrow(deck.canvas, 8.2 * inch, 4.62 * inch, 9.2 * inch, 4.62 * inch)
    deck.box(9.25 * inch, 3.55 * inch, 3.35 * inch, 2.15 * inch, fill=colors.HexColor("#FDECEC"), stroke=RED)
    deck.paragraph("<b>Audit</b><br/>Does truth fall in<br/>mu +/- 2s?", 9.65 * inch, 5.15 * inch, 2.55 * inch, 1.2 * inch, CENTER)
    deck.box(0.85 * inch, 0.85 * inch, 11.55 * inch, 1.75 * inch, fill=LIGHT)
    deck.paragraph(
        "Low coverage means overconfidence. High coverage can also be uninformative if intervals are too wide. Report coverage <b>and</b> interval width, by target and by split.",
        1.2 * inch, 2.12 * inch, 10.85 * inch, 0.9 * inch,
        ParagraphStyle("uq", parent=CENTER, fontName="Helvetica-Bold", fontSize=19, leading=25, textColor=NAVY),
    )
    deck.end()

    deck.begin("Executed evidence: the simple baseline wins")
    splits = metrics["splits"]
    baseline = splits["interpolation"]["linear_case_baseline_relative_l2"]
    operator = splits["interpolation"]["teaching_operator_relative_l2"]
    labels = ("Local Mach", "Temperature", "Pressure")
    keys = ("local_mach", "temperature_ratio", "pressure_ratio")
    chart_x = 1.25 * inch
    chart_y = 1.3 * inch
    chart_w = 7.0 * inch
    chart_h = 4.25 * inch
    c = deck.canvas
    c.setStrokeColor(colors.HexColor("#BCCCDC"))
    for tick in range(0, 41, 10):
        y = chart_y + chart_h * tick / 40
        c.line(chart_x, y, chart_x + chart_w, y)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 10)
        c.drawRightString(chart_x - 0.12 * inch, y - 3, f"{tick}%")
    group_width = chart_w / 3
    for index, (label, key) in enumerate(zip(labels, keys)):
        center = chart_x + group_width * (index + 0.5)
        for dx, value, color, name in (
            (-0.25 * inch, baseline[key], TEAL, "field interpolation"),
            (0.25 * inch, operator[key], RED, "teaching operator"),
        ):
            height = min(chart_h, chart_h * (100 * value) / 40)
            c.setFillColor(color)
            c.rect(center + dx - 0.18 * inch, chart_y, 0.36 * inch, height, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(center, chart_y - 0.28 * inch, label)
    deck.box(8.85 * inch, 3.65 * inch, 3.65 * inch, 1.75 * inch, fill=PALE, stroke=TEAL)
    deck.paragraph(
        "<b>Baseline</b><br/>0.398% | 0.609% | 0.779%",
        9.15 * inch, 4.85 * inch, 3.05 * inch, 0.8 * inch, CENTER,
    )
    deck.box(8.85 * inch, 1.55 * inch, 3.65 * inch, 1.75 * inch, fill=colors.HexColor("#FDECEC"), stroke=RED)
    deck.paragraph(
        "<b>Teaching analog</b><br/>19.6% | 28.1% | 38.0%",
        9.15 * inch, 2.75 * inch, 3.05 * inch, 0.8 * inch, CENTER,
    )
    deck.end()

    deck.begin("Where does interpolation error live?")
    if FIELD_FIGURE.is_file():
        deck.canvas.drawImage(
            str(FIELD_FIGURE),
            0.75 * inch,
            0.75 * inch,
            11.85 * inch,
            5.15 * inch,
            preserveAspectRatio=True,
            anchor="c",
        )
    deck.end()

    deck.begin("Decision gate: when is a neural operator justified?")
    questions = [
        ("Accuracy", "Does it beat field interpolation on frozen blind cases?"),
        ("Geometry", "Must one model span grids, shapes, or boundary conditions?"),
        ("Availability", "Will deployment lack bracketing high-fidelity solutions?"),
        ("Cost", "Do many field queries amortize training and data production?"),
        ("Physics", "Are shock location, surface loads, positivity, and limits acceptable?"),
        ("Uncertainty", "Is spread calibrated on the deployment regime?"),
    ]
    for index, (title, detail) in enumerate(questions):
        row, column = divmod(index, 3)
        x = (0.7 + column * 4.18) * inch
        y = (3.65 - row * 2.5) * inch
        deck.box(x, y, 3.8 * inch, 2.05 * inch, fill=PALE if row == 0 else colors.HexColor("#FFF4E5"), stroke=TEAL if row == 0 else ORANGE)
        deck.paragraph(
            f"<b>{title}</b><br/><font size='13'>{detail}</font>",
            x + 0.25 * inch, y + 1.62 * inch, 3.3 * inch, 1.2 * inch,
            ParagraphStyle("gate", parent=CENTER, fontSize=19, leading=23),
        )
    deck.end()

    deck.begin("Lab deliverables and claim boundary")
    deck.box(0.7 * inch, 0.85 * inch, 7.1 * inch, 4.95 * inch, fill=PALE)
    deck.label("SUBMIT", 1.0 * inch, 5.35 * inch)
    deck.paragraph(
        bullet_list([
            "Baseline and teaching-operator relative-L2 errors by target and split.",
            "One physics-based interpretation of an error hot spot.",
            "Two-sigma coverage and a calibration judgment.",
            "A short explanation of random-point leakage.",
            "One deployment condition that could justify full neural training.",
        ]),
        1.0 * inch, 4.95 * inch, 6.45 * inch, 3.55 * inch,
    )
    deck.box(8.15 * inch, 3.35 * inch, 4.45 * inch, 2.45 * inch, fill=colors.HexColor("#EDF7ED"), stroke=GREEN)
    deck.label("MAY CLAIM", 8.45 * inch, 5.35 * inch, color=GREEN)
    deck.paragraph(
        "The compact DSMC derivative, frozen split, baseline, and CPU teaching analog were executed.",
        8.45 * inch, 4.9 * inch, 3.85 * inch, 1.15 * inch, SMALL,
    )
    deck.box(8.15 * inch, 0.85 * inch, 4.45 * inch, 2.15 * inch, fill=colors.HexColor("#FDECEC"), stroke=RED)
    deck.label("MAY NOT CLAIM", 8.45 * inch, 2.55 * inch, color=RED)
    deck.paragraph(
        "Published full-model accuracy, speedup, complete-field reproduction, or calibrated UQ.",
        8.45 * inch, 2.1 * inch, 3.85 * inch, 1.0 * inch, SMALL,
    )
    deck.end()

    deck.begin("References and reproducibility trail")
    references = [
        "Roohi et al. (2026), Neural Networks for Rarefied Gas Dynamics: Relaxation Problem, Polyatomic Shock Waves, and Hypersonic Cylinder Flow. Physics of Fluids 38, 057108. https://doi.org/10.1063/5.0334590",
        "Lu et al. (2021), Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence 3, 218-229. https://doi.org/10.1038/s42256-021-00302-5",
        "Lakshminarayanan, Pritzel, and Blundell (2017), Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 30.",
        "Bird (1994), Molecular Gas Dynamics and the Direct Simulation of Gas Flows. Oxford University Press.",
    ]
    for index, reference in enumerate(references):
        y = (5.72 - index * 1.08) * inch
        deck.canvas.setFillColor(TEAL)
        deck.canvas.circle(0.95 * inch, y - 0.08 * inch, 0.11 * inch, fill=1, stroke=0)
        deck.paragraph(reference, 1.25 * inch, y + 0.15 * inch, 11.0 * inch, 0.8 * inch, SMALL)
    deck.box(0.75 * inch, 0.75 * inch, 11.85 * inch, 0.95 * inch, fill=LIGHT)
    deck.paragraph(
        "Rebuild data: qa/build_hypersonic_cylinder_subset.py | Rebuild evidence: qa/run_hypersonic_cylinder_evidence.py | Source archive SHA-256 begins bda221759a37",
        1.0 * inch, 1.43 * inch, 11.35 * inch, 0.45 * inch, TINY,
    )
    deck.end()

    deck.save()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
