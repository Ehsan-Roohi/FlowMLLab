"""Exact Times New Roman lecture typography; never silently substitutes fonts.

Install licensed Times New Roman locally first (see LECTURE_TYPOGRAPHY.md).
Font binaries deliberately do not belong in this repository.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import warnings

FAMILY = "Times New Roman"
BLACK = "#000000"
_STYLES = {"regular": "Regular", "bold": "Bold", "italic": "Italic", "bolditalic": "Bold Italic"}


def font_paths() -> dict[str, Path]:
    """Resolve and verify the actual font files, rejecting fontconfig fallback."""
    from fontTools.ttLib import TTFont
    result = {}
    directory = os.environ.get("FLOWMLLAB_TNR_DIR")
    filenames = {"regular": "Times.TTF", "bold": "Timesbd.TTF", "italic": "Timesi.TTF", "bolditalic": "Timesbi.TTF"}
    for key, style in _STYLES.items():
        if directory:
            path = Path(directory) / filenames[key]
        else:
            path = Path(subprocess.check_output(
                ["fc-match", f"{FAMILY}:style={style}", "-f", "%{file}"], text=True).strip())
        if not path.is_file():
            raise RuntimeError(f"Missing Times New Roman {style}: {path}. See qa/LECTURE_TYPOGRAPHY.md")
        with TTFont(path) as font:
            family = font["name"].getDebugName(1)
            actual_style = font["name"].getDebugName(2)
            if family != FAMILY or actual_style != style:
                raise RuntimeError(f"Font substitution rejected: requested {FAMILY} {style}; got {family} {actual_style} ({path})")
            if font["OS/2"].fsType & 2:
                raise RuntimeError(f"Font does not allow document embedding: {path}")
        result[key] = path
    return result


def register_reportlab(prefix: str = "TNR") -> dict[str, str]:
    """Register verified faces under fresh names, rejecting cached-name collisions.

    Call inside the PDF build function, before creating styles, paragraphs or a
    canvas. Importing this module performs no font discovery or registration.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    names = {"regular": prefix, "bold": prefix + "B", "italic": prefix + "I", "bolditalic": prefix + "BI"}
    for key, path in font_paths().items():
        requested = TTFont(names[key], str(path))
        pdfmetrics.registerFont(requested)
        registered = pdfmetrics.getFont(names[key])
        if not getattr(registered, "_dynamicFont", False) or registered.face.name != requested.face.name:
            raise RuntimeError(
                f"ReportLab font-name collision for {names[key]!r}: an earlier font is still cached. "
                "Use a fresh TNR prefix and update every paragraph/canvas/table font name.")
    pdfmetrics.registerFontFamily(prefix, normal=names["regular"], bold=names["bold"],
                                  italic=names["italic"], boldItalic=names["bolditalic"])
    return names


def register_legacy_reportlab_aliases() -> None:
    """Deprecated compatibility bridge; prefer explicit register_reportlab names.

    Must run before creating ANY styles/paragraphs/canvas in a PDF build. It
    cannot repair already-built paragraphs or an existing PDF document's cache.
    ReportLab's public registerFont does not overwrite occupied dynamic-font
    names, so this bridge deliberately replaces its process-local name entries
    with the verified canonical font objects and checks every result. It leaves
    Courier/code fonts, colors and scientific images unchanged.
    """
    from reportlab.pdfbase import pdfmetrics
    warnings.warn("Use register_reportlab() and explicit returned names; legacy font aliases "
                  "are deprecated and must precede all PDF object construction.",
                  DeprecationWarning, stacklevel=2)
    names = register_reportlab()
    families = [
        ("TNR", "TNRB", "TNRI", "TNRBI"),
        ("FlowSerif", "FlowSerifBold", "FlowSerifItalic", "FlowSerifBoldItalic"),
        ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"),
        ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic"),
    ]
    registry = getattr(pdfmetrics, "_fonts", None)
    if not isinstance(registry, dict):
        raise RuntimeError("Unsupported ReportLab font registry; migrate to explicit TNR font names.")
    for family in families:
        for key, name in zip(_STYLES, family):
            canonical = pdfmetrics.getFont(names[key])
            registry[name] = canonical
            if pdfmetrics.getFont(name) is not canonical:
                raise RuntimeError(f"ReportLab did not resolve {name!r} to verified TNR; use explicit names.")
        pdfmetrics.registerFontFamily(family[0], normal=family[0], bold=family[1], italic=family[2], boldItalic=family[3])


def black_paragraph_styles(styles) -> None:
    """Set all supplied paragraph styles to black, retaining font sizes/layout."""
    from reportlab.lib.colors import black
    values = styles.byName.values() if hasattr(styles, "byName") else styles.values()
    for style in values:
        style.textColor = black


def configure_matplotlib() -> None:
    """TNR text and math alphabet; explicit STIX fallback for missing math glyphs."""
    import matplotlib as mpl
    from matplotlib import font_manager
    for path in font_paths().values():
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update({"font.family": FAMILY, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "mathtext.fontset": "custom",
        "mathtext.rm": FAMILY, "mathtext.it": FAMILY + ":italic", "mathtext.bf": FAMILY + ":bold",
        "mathtext.sf": FAMILY, "mathtext.tt": "DejaVu Sans Mono", "mathtext.cal": FAMILY,
        "mathtext.fallback": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})


if __name__ == "__main__":
    import json
    print(json.dumps({key: str(path) for key, path in font_paths().items()}, indent=2))
