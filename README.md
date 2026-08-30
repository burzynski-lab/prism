# PRISM

**Physics Reinterpretation with Intelligent Surrogate Models.** A library
of preserved LHC analyses, each with a machine-learned surrogate that
predicts the analysis' signal-region efficiency for a new signal model from
generator-level events, and the web service that runs them on uploaded
HepMC files: https://prism.web.cern.ch

Upload a HepMC file of your model, choose an analysis, get back the
predicted efficiency (with its statistical uncertainty and per-event
probabilities), without detector simulation or access to the
collaboration's software.

## The library

Each analysis is one directory under `analyses/<id>/`:

| file | purpose |
|---|---|
| `analysis.yaml` | the record: title, experiment, status, signal region, inputs, model file, options, assets, validation table, figures |
| `predictor.py` | the inference code: a `Predictor` subclass registered under the type named in the record, plus any helper modules next to it |
| `README.md` | the model card: what the analysis selects, how the surrogate was trained and validated, known limits |
| model files, `figures/`, an example input | shipped in the directory, or declared under `assets` and downloaded on first use |

Entries: `emerging-jets-delphes` (a transformer surrogate of a track-based
emerging-jet tagger in an ATLAS-like Delphes simulation, the example of the
method) and `atlas-exot-2022-04-calratio` (the ATLAS CalRatio search
through the collaboration's published reinterpretation BDTs).

The service knows only the `Predictor` interface (`prism.predictors`): the
event record from the HepMC reader goes in, per-event signal-region
probabilities and a summary come out. Nothing outside `analyses/<id>/` is
specific to a model, a physics scenario, or an analysis, and the site
renders the library from the records at startup. A new analysis is a pull
request that adds one directory; start from `analyses/_template/` and see
the Contribute page of the site.

## Running

```bash
pixi install                         # one environment: service + what the predictors import
pixi run prism validate              # check every record (what CI runs on pull requests)
pixi run test                        # registry, API and worker tests
pixi run prism predict --analysis emerging-jets-delphes \
    --hepmc analyses/emerging-jets-delphes/example.hepmc.gz --out results
pixi run prism serve &  pixi run prism worker    # local service at http://localhost:8080
```

`prism predict` runs the same code path as a web job. Configuration of the
service is by environment variables (`PRISM_SERVICE_DIR`, `PRISM_ANALYSES_DIR`,
`PRISM_ASSET_DIR`, `PRISM_MAX_UPLOAD_MB`, `PRISM_MAX_EVENTS`,
`PRISM_JOB_TTL_HOURS`).

## Layout

```
src/prism/           the service: registry, predictor interface, HepMC reader, API, worker, front end
analyses/            the library (one directory per analysis, added by pull request)
tests/               registry, API, worker and plugin tests
deploy/paas/         CERN PaaS manifests (kustomize); deploy/README.md
.github/workflows/   validate the library on PRs, build the image on main, run predictions from issues
Dockerfile           the service image (ghcr.io/burzynski-lab/prism)
```

## Reinterpretation from a GitHub issue

Opening an issue with the "Reinterpretation request" template (a URL to a
HepMC file and an analysis id) runs the prediction in the service image on
GitHub Actions and posts the result as a comment.
