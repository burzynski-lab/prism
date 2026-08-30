# Deploying PRISM on CERN PaaS

One image (`Dockerfile`) runs three roles: `prism serve` (web pod),
`prism worker` (worker pods) and `prism worker --cleanup-only` (CronJob).
Jobs live on a shared PersistentVolumeClaim as `jobs.db` (SQLite, WAL) plus
one directory per job; downloaded model assets live there too
(`PRISM_ASSET_DIR`). Configuration is the `ConfigMap` (`PRISM_*` variables).

## One-time setup

1. paas.cern.ch: the project `prism`, which registers the site
   `prism.web.cern.ch` (visible under "My websites" on
   webservices.web.cern.ch). The default quota is 1 CPU of requests, which
   fits the web pod (250m) plus one worker (500m). Request more through
   Service Now to run several workers, then raise `replicas` in
   `deploy/paas/deployment-worker.yaml`.
2. Access: PaaS routes are CERN-only by default (HAProxy IP whitelist from
   the project annotation `custom-ingress.okd.cern.ch/default-route-ip-whitelist`).
   The Route overrides it with an empty
   `haproxy.router.openshift.io/ip_whitelist` (Internet). SSO, if wanted,
   is configured in the Application Portal.
3. Image: the GitHub workflow `build-image` publishes
   `ghcr.io/burzynski-lab/prism:latest` on every push to `main`. The package
   must be public (or add a pull secret to the project).

## Deploy and operate

```bash
oc login --token=... --server=https://api.paas.okd.cern.ch:443
oc project prism
oc apply -k deploy/paas
oc get pods; oc logs deploy/prism-worker -f
oc import-image prism:latest            # pull a new image now instead of waiting for the schedule
oc scale deploy/prism-worker --replicas=4
```

Local dry run of the same image:

```bash
docker build -t prism .
docker run --rm -p 8080:8080 -v $PWD/service_data:/data -e PRISM_SERVICE_DIR=/data prism serve &
docker run --rm -v $PWD/service_data:/data -e PRISM_SERVICE_DIR=/data prism worker
```

## Automatic updates

`imagestream.yaml` imports `ghcr.io/burzynski-lab/prism:latest` on a schedule
and the deployments carry image-change triggers, so every image built from
`main` rolls out by itself within about 15 minutes. Contributors only add
`analyses/<id>/`; the site renders the library from the records at startup.

## Notes

- Uploads are deleted once processed; results and logs expire after
  `PRISM_JOB_TTL_HOURS` (CronJob).
- Throughput is set by HepMC parsing (a few events per second per worker
  core). Add workers or lower `PRISM_MAX_EVENTS` for large uploads.
