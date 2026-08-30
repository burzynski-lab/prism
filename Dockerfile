# The service image: `prism serve` (web), `prism worker` (jobs), `prism worker --cleanup-only` (CronJob).
#   docker build -t prism .
#   docker run --rm -p 8080:8080 -v $PWD/service_data:/data -e PRISM_SERVICE_DIR=/data prism serve
#   docker run --rm -v $PWD:/work prism predict --analysis <id> --hepmc /work/events.hepmc --out /work/results
FROM ghcr.io/prefix-dev/pixi:0.63.2 AS build
WORKDIR /app
COPY pixi.toml pixi.lock pyproject.toml ./
COPY src ./src
COPY analyses ./analyses
RUN pixi install --locked && pixi shell-hook > /app/activate.sh

FROM debian:bookworm-slim
COPY --from=build /app /app
WORKDIR /app
ENTRYPOINT ["/bin/bash", "-c", "source /app/activate.sh && exec prism \"$@\"", "--"]
CMD ["--help"]
