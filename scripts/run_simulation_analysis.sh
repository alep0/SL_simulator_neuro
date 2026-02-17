#!/usr/bin/env bash
# =============================================================================
# run_simulation_analysis.sh
# =============================================================================
# Run simulation + signal analysis for a single rat.
#
# Usage:
#   chmod +x scripts/run_simulation_analysis.sh
#   ./scripts/run_simulation_analysis.sh \
#       /path/to/data \
#       /path/to/results \
#       /path/to/config \
#       <op_net>      \
#       <group>       \
#       <rat>         \
#       <tmax_seconds>
#
# Arguments:
#   $1  data_dir      Directory containing structural connectivity files.
#   $2  output_dir    Directory for output figures and matrices.
#   $3  config_dir    Directory containing JSON config files (unused at runtime
#                     but printed for traceability).
#   $4  op_net        Network mode: 2 (velocity), 3 (tau), 4 (bimodal).
#   $5  group         Experimental group label (e.g. t1, t2).
#   $6  rat           Rat identifier (e.g. R01).
#   $7  tmax          Simulation duration in seconds.
#
# Environment variables:
#   LOG_DIR    Override log directory (default: logs/).
#   LOG_LEVEL  Override console log level (default: INFO).
#   CONDA_ENV  Conda environment to activate (default: neuro_sl_env).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
SCRIPT_LOG="${LOG_DIR}/run_simulation_analysis_$(date +%Y%m%d_%H%M%S).log"

log_info()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO  $*" | tee -a "${SCRIPT_LOG}"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR $*" | tee -a "${SCRIPT_LOG}" >&2; }

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
if [ "$#" -ne 7 ]; then
    log_error "Expected 7 arguments, got $#."
    echo "Usage: $0 <data_dir> <output_dir> <config_dir> <op_net> <group> <rat> <tmax>"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
CONFIG_DIR="$3"
OP_NET="$4"
GROUP="$5"
RAT="$6"
TMAX="$7"

# Validate numeric arguments
if ! [[ "$OP_NET" =~ ^[234]$ ]]; then
    log_error "op_net must be 2, 3, or 4 (got: $OP_NET)."
    exit 1
fi

if ! [[ "$TMAX" =~ ^[0-9]+(\.[0-9]+)?$ ]] || (( $(echo "$TMAX <= 0" | bc -l) )); then
    log_error "tmax must be a positive number (got: $TMAX)."
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    log_error "data_dir does not exist: ${DATA_DIR}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-neuro_sl_env}"
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" || log_info "Conda env '${CONDA_ENV}' not found – using current Python."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "======================================================"
log_info "Stuart-Landau Simulation + Analysis"
log_info "======================================================"
log_info "data_dir   : ${DATA_DIR}"
log_info "output_dir : ${OUTPUT_DIR}"
log_info "config_dir : ${CONFIG_DIR}"
log_info "op_net     : ${OP_NET}"
log_info "group      : ${GROUP}"
log_info "rat        : ${RAT}"
log_info "tmax       : ${TMAX} s"
log_info "Log file   : ${SCRIPT_LOG}"
log_info "======================================================"

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
log_info "Validating configuration files…"
python3 validations/validate_config.py --config "${CONFIG_DIR}/simulation_config.json" \
    2>&1 | tee -a "${SCRIPT_LOG}" || { log_error "Config validation failed."; exit 1; }

# ---------------------------------------------------------------------------
# Connectivity validation
# ---------------------------------------------------------------------------
log_info "Validating connectivity data for rat ${RAT}…"
python3 validations/validate_connectivity.py \
    --data-dir "${DATA_DIR}" \
    --rat      "${RAT}" \
    --th-value "0.0" \
    --op-net   "${OP_NET}" \
    2>&1 | tee -a "${SCRIPT_LOG}" \
    || { log_error "Connectivity validation failed."; exit 1; }

# ---------------------------------------------------------------------------
# Run Python analysis pipeline
# ---------------------------------------------------------------------------
log_info "Launching analysis pipeline…"
python3 - <<EOF 2>&1 | tee -a "${SCRIPT_LOG}"
import sys, logging
import source.utils.logging_config  # configure logging

from pathlib import Path
from source.core.simulation_engine import SimulationConfig
from source.analysis.signal_analysis import AnalysisConfig, run_analysis

sim_cfg = SimulationConfig(
    tmax        = float("${TMAX}"),
    dt          = 1e-4,
    K           = 1e5,
    a           = -5.0,
    op_net      = int("${OP_NET}"),
    op_model    = 1,
    rat         = "${RAT}",
    mean_delay  = 5.8,
    sig_noise   = 1e-3,
    use_cpp     = True,
    save_data   = False,
    data_dir    = Path("${DATA_DIR}"),
    output_dir  = Path("${OUTPUT_DIR}"),
)

ana_cfg = AnalysisConfig(
    output_dir      = Path("${OUTPUT_DIR}"),
    group           = "${GROUP}",
    rat             = "${RAT}",
    lowpass_freq    = 0.5,
    highpass_freq   = 0.01,
    filter_order    = 50,
    resample_factor = 100,
    transient_steps = 15,
    mode            = "filtered",
)

run_analysis(sim_cfg, ana_cfg)
EOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log_info "======================================================"
log_info "Analysis complete for ${GROUP}/${RAT}."
log_info "Results in : ${OUTPUT_DIR}"
log_info "======================================================"
