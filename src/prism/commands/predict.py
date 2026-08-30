"""Run one analysis' predictor on HepMC files from the command line.

    prism predict --analysis <id> --hepmc events.hepmc [more ...] [--option name=value] [--out results]

The same code path as a web job: the analysis directory supplies the model
and the predictor, this command only dispatches. Writes <out>/<stem>.json
(summary: n_events, sr_efficiency +- error, analysis-specific quantities)
and <out>/<stem>.h5 (per-event probabilities and any extra arrays).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def add_arguments(ap) -> None:
    ap.add_argument("--analysis", required=True, help="analysis id (analyses/<id>/analysis.yaml)")
    ap.add_argument("--hepmc", nargs="+", type=Path, required=True, help="HepMC2/3 ASCII files, optionally gzipped")
    ap.add_argument("--out", default="results")
    ap.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                    help="per-analysis option (see the analysis record), e.g. --option selection=CR+2J")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--device", default="cpu")


def run(args) -> None:
    from .. import registry
    from ..predictors import call_run
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    a = registry.load(strict=False).get(args.analysis)
    if a is None:
        raise SystemExit(f"unknown analysis '{args.analysis}' (see analyses/)")
    if a.record.get("assets"):
        a.fetch_assets()
    predictor = a.predictor(args.device)
    options = dict(kv.split("=", 1) for kv in args.option)
    for o in a.record.get("options", []):
        options.setdefault(o["name"], o.get("default", o["choices"][0]))
    max_events = args.max_events or a.record.get("max_events", 10**9)
    for path in args.hepmc:
        summary, per_event, extras = call_run(predictor, path, max_events,
                                              progress=lambda m: print(f"  {path.name}: {m}", flush=True), options=options)
        summary["input"] = str(path)
        stem = path.name.split(".")[0]
        (out / f"{stem}.json").write_text(json.dumps(summary, indent=1))
        with h5py.File(out / f"{stem}.h5", "w") as h:
            h.create_dataset("event_probability", data=np.asarray(per_event, np.float32))
            for k, v in extras.items():
                h.create_dataset(k, data=v)
            h.attrs.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str))})
        print(f"{stem}: {summary['n_events']} events, SR efficiency {summary['sr_efficiency']:.4f} "
              f"+- {summary['sr_efficiency_err']:.4f}", flush=True)
