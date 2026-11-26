"""Build (and optionally execute) Jupyter notebooks from the module Markdown files.

Every ```python block in module-XX-*.md becomes a code cell; prose becomes markdown cells.
A block whose first line is `# norun` is shown as text only (e.g. GPU-only or shell snippets).
Executing the notebook is the verification step: if any cell fails, the build fails.

Usage (from tutorials/data-science-ai):
    python tools/build_notebooks.py            # build + execute all modules
    python tools/build_notebooks.py 03 --no-exec
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
FENCE = re.compile(r"^```python\s*\n(.*?)^```\s*$", re.S | re.M)


def clean_md(text: str) -> str:
    text = re.sub(r"<summary>(.*?)</summary>", r"**\1**", text)
    text = text.replace("<details>", "").replace("</details>", "")
    return text.strip()


def md_to_nb(md_path: Path) -> nbformat.NotebookNode:
    src = md_path.read_text(encoding="utf-8")
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    pos = 0
    for m in FENCE.finditer(src):
        prose = clean_md(src[pos : m.start()])
        if prose:
            nb.cells.append(nbformat.v4.new_markdown_cell(prose))
        code = m.group(1).rstrip()
        if code.splitlines()[0].strip() == "# norun":
            nb.cells.append(nbformat.v4.new_markdown_cell("```python\n" + code + "\n```"))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(code))
        pos = m.end()
    tail = clean_md(src[pos:])
    if tail:
        nb.cells.append(nbformat.v4.new_markdown_cell(tail))
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*", help="module numbers, e.g. 01 02 (default: all)")
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    files = sorted(ROOT.glob("module-*.md"))
    if args.modules:
        files = [f for f in files if f.name[7:9] in args.modules]
    failed = 0
    for md in files:
        nb = md_to_nb(md)
        out = NB_DIR / f"{md.stem[:9]}.ipynb"
        n_code = sum(c.cell_type == "code" for c in nb.cells)
        if not args.no_exec:
            t0 = time.time()
            try:
                NotebookClient(nb, timeout=600, kernel_name="python3",
                               resources={"metadata": {"path": str(NB_DIR)}}).execute()
                print(f"PASS {md.name}: {n_code} code cells in {time.time() - t0:.1f}s")
            except Exception as exc:  # report and continue so all modules are checked
                failed += 1
                print(f"FAIL {md.name}: {str(exc)[-1500:]}")
        nbformat.write(nb, out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
