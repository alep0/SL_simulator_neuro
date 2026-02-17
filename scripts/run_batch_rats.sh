#!/usr/bin/env bash
# =============================================================================
# run_batch_rats.sh
# =============================================================================
# Run simulation + analysis for multiple rats sequentially.
#
# Usage:
#   chmod +x scripts/run_batch_rats.sh
#   ./scripts/run_batch_rats.sh \
#       <op_net> <group> "R01 R02 R03" \
#       /path/to/project "ModelName" <tmax_seconds>
#
# Arguments:
#   $1  op_net        Network mode: 2, 3, or 4.
#   $2  group         Experimental group (e.g. t1, t2).
#   $3  rats          Space-separated list of rat IDs (quote the whole list).
#   $4  root_path     Project root directory.
#   $5  model_name    Label used for output subdirectories.
#   $6  tmax          Simulation duration (s).
#
# Environment:
#   LOG_LEVEL   Console log level (default: INFO).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
BATCH_LOG="${LOG_DIR}/batch_$(date +%Y%m%d_%H%M%S).log"

log_info()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO  $*" | tee -a "${BATCH_LOG}"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR $*" | tee -a "${BATCH_LOG}" >&2; }
log_warn()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN  $*" | tee -a "${BATCH_LOG}"; }

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
if [ "$#" -ne 6 ]; then
    log_error "Expected 6 arguments, got $#."
    echo "Usage: $0 <op_net> <group> \"<rats>\" <root_path> <model_name> <tmax>"
    exit 1
fi

OP_NET="$1"
GROUP="$2"
RATS="$3"
ROOT_PATH="$4"
MODEL_NAME="$5"
TMAX="$6"

if ! [[ "$OP_NET" =~ ^[234]$ ]]; then
    log_error "op_net must be 2, 3, or 4."
    exit 1
fi

if [ ! -d "${ROOT_PATH}" ]; then
    log_error "root_path does not exist: ${ROOT_PATH}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------
log_info "================================================================"
log_info "Batch run: op_net=${OP_NET} | group=${GROUP} | model=${MODEL_NAME}"
log_info "Rats:      ${RATS}"
log_info "Tmax:      ${TMAX} s"
log_info "Batch log: ${BATCH_LOG}"
log_info "================================================================"

PASS_COUNT=0
FAIL_COUNT=0
FAILED_RATS=""

for RAT in ${RATS}; do
    DATA_PATH="${ROOT_PATH}/data/raw/${GROUP}/RN_SI_v0-1_th-0.0/filter_kick_out/${RAT}"
    OUTPUT_PATH="${ROOT_PATH}/results/${MODEL_NAME}/${GROUP}/${RAT}"
    CONFIG_PATH="${ROOT_PATH}/config"

    log_info "--- Processing ${GROUP}/${RAT} ---"
    log_info "data_dir   : ${DATA_PATH}"
    log_info "output_dir : ${OUTPUT_PATH}"

    mkdir -p "${OUTPUT_PATH}"

    # Delegate to the single-rat script
    if bash scripts/run_simulation_analysis.sh \
            "${DATA_PATH}" \
            "${OUTPUT_PATH}" \
            "${CONFIG_PATH}" \
            "${OP_NET}" \
            "${GROUP}" \
            "${RAT}" \
            "${TMAX}" \
            2>&1 | tee -a "${BATCH_LOG}"; then
        log_info "${RAT}: DONE"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        log_warn "${RAT}: FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_RATS="${FAILED_RATS} ${RAT}"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "================================================================"
log_info "Batch complete: ${PASS_COUNT} passed, ${FAIL_COUNT} failed."
if [ "${FAIL_COUNT}" -gt 0 ]; then
    log_warn "Failed rats:${FAILED_RATS}"
fi
log_info "================================================================"

[ "${FAIL_COUNT}" -eq 0 ]   # exit 0 if all passed
