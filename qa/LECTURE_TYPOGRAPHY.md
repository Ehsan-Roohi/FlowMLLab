# Lecture typography build contract

Use **genuine Times New Roman** for lecture text and mathematical alphabets;
use black text on white or pale backgrounds. Retain scientific colormaps and
useful color-encoded curves. Never relabel Nimbus, Liberation, Tinos, DejaVu,
or built-in PDF Times as Times New Roman.

Literal source code that depends on column alignment is an explicit monospace
exception (DejaVu Sans Mono in TeX, existing Courier in legacy ReportLab).
Keep prose and mathematical alphabets in TNR; do not destroy code alignment.

The helper fails if fontconfig resolves a substitute. Run `python qa/lecture_typography.py`
before building. Four styles are required. `FLOWMLLAB_TNR_DIR` can point to a
local folder containing `Times.TTF`, `Timesbd.TTF`, `Timesi.TTF`, `Timesbi.TTF`.
The fonts are a local build dependency, not repository content.

## Local installation provenance

The review environment uses the original Microsoft Core Fonts for the Web
`times32.exe`, downloaded from <https://downloads.sourceforge.net/corefonts/times32.exe>.
The publisher of that mirror describes these as the original unaltered files:
<https://corefonts.sourceforge.net/>. The accompanying license is available at
<https://corefonts.sourceforge.net/eula.htm>. Its installation/use grant and
document-subsetting provision cover this build; do not distribute extracted font
files in this repository. Review the license before independently installing.

Installer SHA-256: `db56595ec6ef5d3de5c24994f001f03b2a13e37cee27bc25c58f6f43e8f807ab`.
Installed locally at `/usr/local/share/fonts/truetype/msttcorefonts/`; all four
faces report family `Times New Roman`, their expected styles, and fsType 0.

## Builders

ReportLab: `from qa.lecture_typography import register_reportlab` (or
`from lecture_typography import ...` in a directly executed `qa/` script).
`register_reportlab()` supplies `TNR`, `TNRB`, `TNRI`, and `TNRBI` with correct
bold/italic mapping. Use those explicit names in **every** paragraph, canvas,
table, and inline font override. A custom fresh prefix can isolate a builder:

```python
def build_pdf(destination):
    from qa.lecture_typography import register_reportlab
    from reportlab.lib.styles import ParagraphStyle
    names = register_reportlab(prefix="CourseTNR")
    body = ParagraphStyle("body", fontName=names["regular"], textColor="#000000")
    # Create paragraphs, tables and the canvas only after registration.
```

Keep font setup inside the PDF function, never at module import or in a
notebook-only build path. This module's import does not discover fonts or import
ReportLab/Matplotlib. Existing cached font names that resolve to another face
cause `register_reportlab` to fail with an actionable error.

**Legacy alias warning:** `register_legacy_reportlab_aliases()` is deprecated.
ReportLab's `registerFont(TTFont(...))` silently leaves an occupied font-name
entry unchanged; merely registering TNR as `Helvetica` or `FlowSerif` therefore
does not guarantee TNR rendering. The compatibility helper now explicitly
replaces these process-local registry entries with verified TNR objects and
verifies resolution. This uses a guarded private ReportLab registry and cannot
repair paragraphs/canvases already constructed, so new and maintained builders
must use the explicit fresh names above. If a legacy caller remains, call the
bridge after old font registrations but **before any styles, paragraphs, tables,
or canvas are created**. Never call it midway through a PDF build.
Apply `black_paragraph_styles(styles)` after style creation; also set canvas
text, table text, and inline `<font color=...>` to black. White text on dark
panels must be redesigned with black text on pale backgrounds.

Matplotlib equations: call `configure_matplotlib()` before constructing figures.
It uses TNR mathematical alphabets and an explicit STIX fallback only for glyphs
that TNR cannot provide. Prefer searchable vector equations where feasible.

TeX: include `\input{lectures/source/lecture_typography.tex}` at the end of the
preamble and use XeLaTeX. Remove legacy `fontenc`, `inputenc`, `times`, or `newtxtext`
text-font overrides. This profile uses mathastext for TNR Latin math; existing
specialized TeX symbol fonts remain necessary for extensible delimiters/operators.
Do not claim every mathematical symbol is TNR: it is not an OpenType math font.
For an existing unicode-math document, set TNR main/sans/mono and explicit TNR
Latin/Greek math ranges, retaining its named full math font for other symbols;
do not mix mathastext into that document.

MJS/PPTX: set text `typeface`/`fontFace` to `Times New Roman`, color `#000000`, and
verify the exported PDF with `qa/audit_lecture_pdfs.py`. A PPTX font-family string
alone does not prove a renderer used or embedded the font. Use a searchable
text/vector PDF export rather than a PDF made entirely of slide screenshots.

## Verification

The read-only PDF auditor requires PyMuPDF (`python -m pip install pymupdf`).
Font identity checks use `fontTools`; ReportLab/Matplotlib or XeLaTeX are
authoring dependencies for their respective builders. Notebook-only generation
does not require the Microsoft font installation.

For the Transformer notebook execution gate, use
`python qa/build_transformer_course.py --execute --notebooks-only --output build/ci-course`.
Omit `--notebooks-only` when intentionally rebuilding lecture PDFs.

Run `python qa/audit_lecture_pdfs.py lectures --output revision_work/pdf_typography_audit.json`.
This records per-page searchable characters, text colors, fonts, missing-glyph
markers, text outside page bounds, and embedded-font inventory. It does not
replace visual review: render and inspect all changed pages and dense figures.
Raster figure labels preserve their source font and are reported separately
from the main document typography; never recolor an entire scientific image.
Inspect the **actual text-span font** for normal body text as well as bold and
italic runs. `pdffonts` listing TNR only proves that some resource is present;
it does not prove that body paragraphs use it. A heading can use TNR while the
body still uses a cached Helvetica or DejaVu font.
