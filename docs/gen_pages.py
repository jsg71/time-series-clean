# docs/gen_pages.py
from __future__ import annotations
from pathlib import Path
import os
import mkdocs_gen_files

# Notebooks ON by default. Disable with: DOCS_NOTEBOOKS=0 python -m mkdocs serve
ENABLE_NOTEBOOKS = os.getenv("DOCS_NOTEBOOKS", "1").lower() in {"1", "true", "yes"}

ROOT = Path(__file__).resolve().parents[1]
SRC_BASES = [ROOT / "src", ROOT]   # supports src/ and flat layouts

IGNORE_DIRS = {
    ".git", ".github", ".gitlab", ".venv", "venv", "env",
    "build", "dist", "site", "docs", "__pycache__", "tests", "test",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
}
IGNORE_FILES = {"__main__.py", "setup.py", "conftest.py"}

def is_ignored(p: Path) -> bool:
    return any(part in IGNORE_DIRS for part in p.parts) or p.name in IGNORE_FILES

def dotted_name(py: Path, base: Path) -> str | None:
    if py.name == "__init__.py":
        rel = py.parent.relative_to(base)
    else:
        rel = py.relative_to(base).with_suffix("")
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)

# 1) API pages (one per module)
modules: list[str] = []
for BASE in SRC_BASES:
    if not BASE.exists():
        continue
    for path in sorted(BASE.rglob("*.py")):
        if is_ignored(path):
            continue
        ident = dotted_name(path, BASE)
        if not ident or ident in modules:
            continue
        modules.append(ident)
        out = Path("reference", *ident.split("."), "index.md")
        with mkdocs_gen_files.open(out, "w") as fd:
            fd.write(f"# `{ident}`\n\n::: {ident}\n")
        mkdocs_gen_files.set_edit_path(out, path)

# 1a) Nicer Reference index (grouped)
groups: dict[str, list[str]] = {}
for ident in modules:
    top = ident.split(".", 1)[0]
    groups.setdefault(top, []).append(ident)

ref_index = Path("reference", "index.md")
with mkdocs_gen_files.open(ref_index, "w") as fd:
    fd.write("# API Reference\n\n")
    for top in sorted(groups):
        fd.write(f"## {top}\n\n")
        fd.write(f"- [{top}](./{top.replace('.', '/')}/index.md)\n")
        for s in sorted(x for x in groups[top] if x != top):
            fd.write(f"  - [{s}](./{s.replace('.', '/')}/index.md)\n")
        fd.write("\n")

# 2) Notebooks → only "Open" links (no download)
nb_found: list[Path] = []
if ENABLE_NOTEBOOKS:
    nb_root = ROOT / "notebooks"
    if nb_root.exists():
        for nb in sorted(nb_root.rglob("*.ipynb")):
            if is_ignored(nb):
                continue
            rel = nb.relative_to(nb_root)
            dest = Path("notebooks") / rel
            dest = dest.with_suffix(".ipynb")
            with mkdocs_gen_files.open(dest, "wb") as fd:
                fd.write(nb.read_bytes())
            mkdocs_gen_files.set_edit_path(dest, nb)
            nb_found.append(dest)

nb_index = Path("notebooks", "index.md")
with mkdocs_gen_files.open(nb_index, "w") as fd:
    fd.write("# Notebooks\n\n")
    if nb_found:
        for dest in nb_found:
            rel = dest.relative_to("notebooks")   # e.g., 01_eda.ipynb
            stem = rel.with_suffix("")            # 01_eda
            fd.write(f"- [{stem.name}]({stem.as_posix()}/)\n")  # rendered page only
    else:
        fd.write("_No notebooks were found under `./notebooks/`._\n")

# 3) Project Structure page
def tree(path: Path, prefix: str = "", depth: int = 0, max_depth: int = 3) -> list[str]:
    if depth > max_depth:
        return []
    entries = [e for e in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
               if not is_ignored(e)]
    lines: list[str] = []
    for i, e in enumerate(entries):
        conn = "└── " if i == len(entries) - 1 else "├── "
        lines.append(f"{prefix}{conn}{e.name}")
        if e.is_dir():
            ext = "    " if i == len(entries) - 1 else "│   "
            lines.extend(tree(e, prefix + ext, depth + 1, max_depth))
    return lines

proj_page = Path("guide", "project-structure.md")
with mkdocs_gen_files.open(proj_page, "w") as fd:
    fd.write("# Project Structure\n\nAuto-generated at build time.\n\n```text\n")
    fd.write("\n".join(tree(ROOT, max_depth=3)))
    fd.write("\n```\n")

# 4) Copy README into docs so it always shows (Guide → README)
readme_src = ROOT / "README.md"
if readme_src.exists():
    readme_dst = Path("guide", "readme.md")
    with mkdocs_gen_files.open(readme_dst, "w") as fd:
        fd.write(readme_src.read_text(encoding="utf-8"))
    mkdocs_gen_files.set_edit_path(readme_dst, readme_src)

