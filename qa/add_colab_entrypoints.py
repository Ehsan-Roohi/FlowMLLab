#!/usr/bin/env python3
"""Add stable one-click Colab launchers to every public notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
REPOSITORY = "Ehsan-Roohi/FlowMLLab"
LAUNCH_MARKER = "FLOWMLLAB_COLAB_LAUNCH_V1"
BOOTSTRAP_MARKER = "FLOWMLLAB_COLAB_BOOTSTRAP_V1"


def source_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def badge(relative_path: str) -> str:
    target = (
        "https://colab.research.google.com/github/"
        f"{REPOSITORY}/blob/main/{relative_path}"
    )
    return (
        f"\n<!-- {LAUNCH_MARKER} -->\n"
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({target})\n"
    )


def bootstrap(notebook_directory: str) -> str:
    return f'''# {BOOTSTRAP_MARKER}
# In Colab this cell obtains the complete repository and installs the tested package.
# In a local checkout it leaves the active environment and working directory unchanged.
from pathlib import Path as _FlowMLLabPath
import os as _flowmllab_os
import subprocess as _flowmllab_subprocess
import sys as _flowmllab_sys

if "google.colab" in _flowmllab_sys.modules or _flowmllab_os.environ.get("COLAB_RELEASE_TAG"):
    _flowmllab_root = _FlowMLLabPath("/content/FlowMLLab")
    if not (_flowmllab_root / ".git").is_dir():
        _flowmllab_subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/{REPOSITORY}.git", str(_flowmllab_root),
            ],
            check=True,
        )
    _flowmllab_subprocess.run(
        [
            _flowmllab_sys.executable, "-m", "pip", "install", "-q", "-e",
            f"{{_flowmllab_root}}[test]",
        ],
        check=True,
    )
    _flowmllab_notebook_dir = _flowmllab_root / "{notebook_directory}"
    _flowmllab_os.chdir(_flowmllab_notebook_dir)
    for _flowmllab_path in (_flowmllab_root, _flowmllab_notebook_dir):
        if str(_flowmllab_path) not in _flowmllab_sys.path:
            _flowmllab_sys.path.insert(0, str(_flowmllab_path))
    print("FlowMLLab ready:", _flowmllab_root)

'''


def title_cell_index(cells: list[dict]) -> int:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        if re.search(r"(?m)^#\s+\S", source_text(cell)):
            return index
    raise ValueError("notebook has no level-one title")


def first_code_cell_index(cells: list[dict]) -> int:
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            return index
    raise ValueError("notebook has no code cell")


def add_entrypoint(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    relative = path.relative_to(ROOT).as_posix()

    title_index = title_cell_index(cells)
    title_source = source_text(cells[title_index])
    if LAUNCH_MARKER not in title_source:
        title_lines = title_source.splitlines(keepends=True)
        insert_at = 1
        title_lines[insert_at:insert_at] = source_lines(badge(relative))
        cells[title_index]["source"] = title_lines

    code_index = first_code_cell_index(cells)
    code_source = source_text(cells[code_index])
    if BOOTSTRAP_MARKER not in code_source:
        notebook_directory = path.parent.relative_to(ROOT).as_posix()
        cells[code_index]["source"] = source_lines(
            bootstrap(notebook_directory) + code_source
        )
        cells[code_index]["execution_count"] = None
        cells[code_index]["outputs"] = []

    notebook["cells"] = cells
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Colab-ready:", relative)


def main() -> None:
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    for path in notebooks:
        add_entrypoint(path)
    print(f"Updated {len(notebooks)} notebooks.")


if __name__ == "__main__":
    main()
