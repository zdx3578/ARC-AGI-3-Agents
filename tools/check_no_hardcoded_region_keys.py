#!/usr/bin/env python3
"""
Fail if active_inference source code contains hardcoded region literals like "4:2".

Rule intent:
- Region targets must be discovered dynamically from runtime evidence.
- Manual overrides are allowed only through environment variables / runtime config,
  not by embedding concrete region addresses into source defaults.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "agents" / "templates" / "active_inference"
REGION_LITERAL_RE = re.compile(r"^[0-7]:[0-7]$")


def iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def find_hardcoded_region_literals(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str):
            continue
        token = node.value.strip()
        if not REGION_LITERAL_RE.fullmatch(token):
            continue
        lineno = int(getattr(node, "lineno", 0))
        findings.append((lineno, token))
    return findings


def main() -> int:
    if not TARGET_DIR.exists():
        print(f"skip: target dir not found: {TARGET_DIR}")
        return 0

    violations: list[tuple[Path, int, str]] = []
    for file_path in iter_python_files(TARGET_DIR):
        for lineno, token in find_hardcoded_region_literals(file_path):
            violations.append((file_path, lineno, token))

    if not violations:
        print("ok: no hardcoded region literals found in active_inference sources")
        return 0

    print("error: hardcoded region literals detected:")
    for file_path, lineno, token in violations:
        rel = file_path.relative_to(ROOT)
        print(f"  - {rel}:{lineno}: {token}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
