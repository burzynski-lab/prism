"""``prism``: the single entry point of the service.

    prism serve       web application (API + front end)
    prism worker      job worker (runs queued predictions)
    prism predict     run one analysis' predictor on HepMC files from the command line
    prism validate    check every analysis record in the library (what CI runs)
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "serve": "prism.commands.serve",
    "worker": "prism.commands.worker",
    "predict": "prism.commands.predict",
    "validate": "prism.commands.validate",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="prism", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, modname in COMMANDS.items():
        mod = importlib.import_module(modname)
        p = sub.add_parser(name, help=mod.__doc__.strip().splitlines()[0], description=mod.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_arguments(p)
        p.set_defaults(run=mod.run)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.run(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
