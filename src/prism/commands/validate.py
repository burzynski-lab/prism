"""Validate every analysis directory in the library.

    prism validate [--analyses-dir analyses]

Exit status 1 if any record has a problem (missing fields, unresolvable
predictor type, missing model or figure files, unimportable requirements).
"""

from __future__ import annotations

from pathlib import Path


def add_arguments(ap) -> None:
    ap.add_argument("--analyses-dir", type=Path, default=None, help="default: PRISM_ANALYSES_DIR or ./analyses")


def run(args) -> int:
    from .. import registry
    root = args.analyses_dir or registry.analyses_dir()
    problems, n = [], 0
    for path in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        n += 1
        problems += registry.validate(path)
    for p in problems:
        print(p)
    print(f"{n} analyses checked, {len(problems)} problems")
    return 1 if problems else 0
