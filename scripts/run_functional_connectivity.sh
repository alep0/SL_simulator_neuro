#!/usr/bin/env bash
# =============================================================================
# run_functional_connectivity.sh
# =============================================================================
# Aggregate per-rat functional connectivity matrices into group-level plots.
#
# Usage:
#   chmod +x scripts/run_functional_connectivity.sh
#   ./scripts/run_functional_connectivity.sh \
#       /path/to/project \
#       <data_name>      \
#       <model_name>     \
#       <group>          \
#       <output_name>
#
# Arguments:
#   $1  root_path    Project root directory.
#   $2  data_name    Data sub-directory (e.g. CC_Nuredduna_2026_01_14).
#   $3  model_name   Model label (e.g. CC_Santiago).
#   $4  group        Experimental group (t1 or t2).
#   $5  output_name  Output subdirectory under results/.
# =============================================================================
set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
FC_LOG="${LOG_DIR}/functional_connectivity_$(date +%Y%m%d_%H%M%S).log"

log_info()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO  $*" | tee -a "${FC_LOG}"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR $*" | tee -a "${FC_LOG}" >&2; }

if [ "$#" -ne 5 ]; then
    log_error "Expected 5 arguments, got $#."
    echo "Usage: $0 <root_path> <data_name> <model_name> <group> <output_name>"
    exit 1
fi

ROOT_PATH="$1"
DATA_NAME="$2"
MODEL_NAME="$3"
GROUP="$4"
OUTPUT_NAME="$5"

log_info "================================================================"
log_info "Functional Connectivity Aggregation"
log_info "root_path   : ${ROOT_PATH}"
log_info "data_name   : ${DATA_NAME}"
log_info "model_name  : ${MODEL_NAME}"
log_info "group       : ${GROUP}"
log_info "output_name : ${OUTPUT_NAME}"
log_info "FC log      : ${FC_LOG}"
log_info "================================================================"

# Activate conda environment if available
CONDA_ENV="${CONDA_ENV:-neuro_sl_env}"
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" \
        || log_info "Conda env '${CONDA_ENV}' not found – using current Python."
fi

cd "${ROOT_PATH}"

python3 -m source.analysis.functional_connectivity \
    --root        "${ROOT_PATH}" \
    --data-name   "${DATA_NAME}" \
    --model-name  "${MODEL_NAME}" \
    --group       "${GROUP}" \
    --output-name "${OUTPUT_NAME}" \
    2>&1 | tee -a "${FC_LOG}"

log_info "Aggregation complete."
log_info "Results in: ${ROOT_PATH}/results/${OUTPUT_NAME}/${MODEL_NAME}/${GROUP}"
