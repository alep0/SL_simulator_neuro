# Docker Guide

Docker lets you run the simulator without installing any local compiler or
Python dependencies. All compilation happens inside the container.

---

## Prerequisites

- [Docker Engine](https://docs.docker.com/engine/install/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/install/) ≥ 2.20 (included in Docker Desktop)

1.- Update your package index:
sudo apt-get update

2.- Install the buildx plugin:
sudo apt-get install docker-buildx-plugin

3.- Verify it works:
docker buildx version

---

## Windows

Start Docker Desktop

## Quick start

```bash
# Build the image and start the container
docker compose up --build

# The container runs the test suite and then exits.
# To keep it alive for interactive use, override the command:
docker compose run --rm simulator bash
```

---

## Build the image manually

```bash
docker build -t neuro-sl-simulator:latest .
```

---

## Run a simulation inside Docker

```bash
docker run --rm \
    -v "$(pwd)/data:/workspace/data:ro" \
    -v "$(pwd)/results:/workspace/results" \
    -v "$(pwd)/logs:/workspace/logs" \
    neuro-sl-simulator:latest \
    bash scripts/run_simulation_analysis.sh \
        data/processed/t1/FA_RN_SI_v0-1_th-0.0/filter_kick_out/R01 \
        results/CC_Santiago/t1/R01 \
        config \
        3 t1 R01 60 raw
```

**Volume mounts**

| Host path | Container path | Purpose |
|---|---|---|
| `./data`    | `/workspace/data`    | Connectivity input data (read-only) |
| `./results` | `/workspace/results` | Output figures and matrices |
| `./logs`    | `/workspace/logs`    | Runtime log files |

---

## Run batch processing

```bash
docker run --rm \
    -v "$(pwd)/data:/workspace/data:ro" \
    -v "$(pwd)/results:/workspace/results" \
    -v "$(pwd)/logs:/workspace/logs" \
    neuro-sl-simulator:latest \
    bash scripts/run_batch_rats.sh \
        3 t1 "R01 R02 R03" /workspace CC_Santiago 60 raw
```

---

## Run the test suite

```bash
docker run --rm neuro-sl-simulator:latest \
    pytest validations/tests/ -v --tb=short
```

---

## Dockerfile overview

```
Stage 1: builder
  - ubuntu:24.04 base
  - Install GCC, OpenMP, Python 3.11, pybind11
  - Build the C++ extension (stuart_landau_simulator.so)

Stage 2: runtime
  - Copy built extension and Python source
  - Install Python runtime dependencies
  - Set WORKDIR, ENTRYPOINT
```

The multi-stage build keeps the final image small by excluding build tools.

---

## Docker Compose services

```yaml
services:
  simulator:
    build: .
    volumes:
      - ./data:/workspace/data:ro
      - ./results:/workspace/results
      - ./logs:/workspace/logs
    environment:
      - LOG_LEVEL=INFO
```

Override environment variables at runtime:

```bash
LOG_LEVEL=DEBUG docker compose run --rm simulator bash
```

---

## Updating the image

After changing source code or dependencies:

```bash
docker compose build --no-cache
```

---

## Troubleshooting

### Build fails on ARM (Apple Silicon / Raspberry Pi)

The `Dockerfile` uses `--platform linux/amd64` by default. For ARM:

```bash
DOCKER_BUILDKIT=1 docker buildx build \
    --platform linux/arm64 \
    -t neuro-sl-simulator:arm64 .
```

Note: `-march=native` is disabled automatically inside Docker to allow
cross-platform images.

---

### Permission errors on mounted volumes (Linux)

```bash
sudo chown -R $(id -u):$(id -g) results/ logs/
```

Or add `user: "${UID}:${GID}"` to `docker-compose.yml`.
