"""Build the Week 15 lecture from Markdown with all repository figures."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lectures/source/week15_geometry_generalization.md"
OUTPUT = ROOT / "output/pdf/week15_geometry_generalization.pdf"
PUBLISHED = ROOT / "lectures/week15_geometry_generalization.pdf"


def main() -> None:
    pandoc = shutil.which("pandoc")
    xelatex = shutil.which("xelatex")
    if not pandoc or not xelatex:
        missing = [
            name
            for name, path in (("pandoc", pandoc), ("xelatex", xelatex))
            if not path
        ]
        raise RuntimeError("Week 15 PDF build requires: " + ", ".join(missing))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            pandoc,
            str(SOURCE),
            "-o",
            str(OUTPUT),
            "--pdf-engine",
            xelatex,
            "--resource-path",
            f"{SOURCE.parent}:{ROOT}",
            "-V",
            "geometry:margin=0.65in",
            "-V",
            "fontsize=9pt",
            "-V",
            "colorlinks=true",
            "-V",
            "linkcolor=blue",
            "-V",
            "urlcolor=blue",
        ],
        cwd=ROOT,
        check=True,
    )
    PUBLISHED.write_bytes(OUTPUT.read_bytes())
    print(OUTPUT)


if __name__ == "__main__":
    main()
