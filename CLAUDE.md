# CLAUDE.md

Working notes for PRISM (this repository). Read README.md first.

## What this is

The deployable service and the analysis library only. Everything about
producing a surrogate (event generation, detector simulation, tagger and
surrogate training, closure studies, the paper) lives in the training
repositories of the individual analyses and must not come here. The
emerging-jets entry was trained in https://github.com/jburzy/jet-surrogate.

## Rules

- One executable, `prism`, with subcommands (`serve`, `worker`, `predict`,
  `validate`); no standalone scripts.
- Nothing in `src/prism/` or the front end may be specific to an analysis
  or a physics scenario. Analysis code lives in `analyses/<id>/` (the
  directory is importable as a package from its `predictor.py` through
  relative imports, e.g. `from .surrogate import ...`).
- The site is data-driven: pages render from `analysis.yaml` through the
  API at runtime. Adding an analysis never touches `src/prism/static/`.
- One pixi environment. A predictor that needs a new package adds it to
  `pixi.toml` and lists it under `requirements` in its record (CI checks
  it imports).

## Deployment

CERN PaaS project `prism`, site https://prism.web.cern.ch, manifests in
`deploy/paas/`. `oc` is run by the user from lxplus (`oc apply -k
deploy/paas`). The image `ghcr.io/burzynski-lab/prism:latest` is built by
GitHub Actions on every push to `main`; an ImageStream on the cluster
re-imports it on a schedule and image-change triggers roll the deployments,
so merged pull requests go live without manual steps.

## Gotchas

- `pixi install` needs a writable cache; on shared machines set
  `PIXI_CACHE_DIR` and, if the home quota is small, `pixi config set
  --local detached-environments <dir>`.
- The HepMC reader (`prism.hepmc`) decompresses gzipped input to a
  temporary file (pyHepMC3 reads plain files only). Particle charges come
  from the PDG numbering scheme (`pdg_charge`), verified identical to
  Pythia's table for every id in the emerging-jets vocabulary and the
  example files; hidden-sector ids (49xxxxx) are neutral.
- Analysis-local predictors are loaded under the package name
  `prism_analyses.<id>` (registry.local_predictors); a helper package with
  a generic name (`surrogate/`) is fine because each analysis has its own
  namespace.
- The CalRatio entry downloads about 300 MB of models from Zenodo on
  first use into `PRISM_ASSET_DIR` (or `analyses/<id>/.assets/`).
