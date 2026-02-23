# =============================================================================
# Dockerfile — neuro-sl-simulator
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder) – compiles the C++ pybind11 extension
#   Stage 2 (runtime) – lean image with only runtime artifacts
#
# Build:  docker build -t neuro-sl-simulator:latest .
# Run:    docker compose up
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: build the C++ extension
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

LABEL stage="builder"

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        g++ \
        libgomp1 \
        libomp-dev \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python build dependencies
COPY config/environment.yml /tmp/environment.yml
RUN pip3 install --no-cache-dir --break-system-packages \
        "pybind11>=2.10" \
        "numpy>=1.22" \
        setuptools wheel

# Copy only the files needed for compilation
COPY setup.py .
COPY source/core/stuart_landau_simulator.cpp source/core/
COPY source/__init__.py* source/ 
COPY source/core/__init__.py* source/core/ 

# Build in-place (produces stuart_landau_simulator*.so next to setup.py)
# Note: -march=native is intentionally omitted for portability inside Docker
RUN python3 setup.py build_ext --inplace 2>&1 | tee /build/build.log


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

LABEL maintainer="Alejandro Aguado" \
      description="Stuart-Landau neural network simulator for resting-state fMRI modelling" \
      version="2.0.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LOG_DIR=/workspace/logs \
    PYTHONPATH=/workspace

# Runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgomp1 \
        bc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python runtime dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
        "numpy>=1.22" \
        "scipy>=1.8" \
        "matplotlib>=3.5" \
        "seaborn>=0.12" \
        "pytest"

# Copy built C++ extension from the builder stage
COPY --from=builder /build/stuart_landau_simulator*.so ./

# Copy project source
COPY source/       ./source/
COPY scripts/      ./scripts/
COPY validations/  ./validations/
COPY config/       ./config/

# Create required directories
RUN mkdir -p data results logs

# Make scripts executable
RUN chmod +x scripts/*.sh

# Default command: run the test suite
CMD ["python3", "-m", "pytest", "validations/tests/", "-v", "--tb=short"]
