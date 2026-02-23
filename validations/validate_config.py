"""
validate_config.py
==================
Validate simulation and analysis JSON configuration files before a run.

Usage
-----
    python validations/validate_config.py --config config/simulation_config.json
    python validations/validate_config.py --config config/analysis_config.json

Exit codes
----------
0  – all checks passed
1  – one or more validation errors detected
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source.utils.logging_config  # noqa: F401 – side-effect: configures logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas  (minimal – extend as needed)
# ---------------------------------------------------------------------------
SIMULATION_REQUIRED_KEYS: dict[str, type] = {
    "tmax":       float,
    "dt":         float,
    "K":          float,
    "a":          float,
    "op_net":     int,
    "op_model":   int,
    "rat":        str,
    "mean_vel": float,
    "sig_noise":  float,
}

ANALYSIS_REQUIRED_KEYS: dict[str, type] = {
    "lowpass_freq":    float,
    "highpass_freq":   float,
    "filter_order":    int,
    "resample_factor": int,
    "transient_steps": int,
    "cross_corr_frac": float,
    "mode":            str,
    "group":           str,
    "rat":             str,
    "output_dir":      str,
}

VALID_OP_NETS   = {2, 3, 4}
VALID_OP_MODELS = {1, 2}
VALID_MODES     = {"raw", "filtered", "bold"}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def _check_types(config: dict, schema: dict[str, type]) -> list[str]:
    errors: list[str] = []
    for key, expected_type in schema.items():
        if key not in config:
            errors.append(f"Missing required key: '{key}'")
            continue
        value = config[key]
        # Allow int where float is expected (JSON doesn't distinguish)
        if expected_type is float and isinstance(value, int):
            continue
        if not isinstance(value, expected_type):
            errors.append(
                f"Key '{key}': expected {expected_type.__name__}, "
                f"got {type(value).__name__} = {value!r}"
            )
    return errors


def validate_simulation_config(config: dict) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errors = _check_types(config, SIMULATION_REQUIRED_KEYS)

    if "op_net" in config and config["op_net"] not in VALID_OP_NETS:
        errors.append(
            f"op_net={config['op_net']} not in valid set {VALID_OP_NETS}"
        )
    if "op_model" in config and config["op_model"] not in VALID_OP_MODELS:
        errors.append(
            f"op_model={config['op_model']} not in valid set {VALID_OP_MODELS}"
        )
    if "tmax" in config and config.get("tmax", 1) <= 0:
        errors.append("tmax must be positive.")
    if "dt" in config and config.get("dt", 1) <= 0:
        errors.append("dt must be positive.")
    if "K" in config and config.get("K", 1) < 0:
        errors.append("K (coupling strength) must be non-negative.")
    if "sig_noise" in config and config.get("sig_noise", 0) < 0:
        errors.append("sig_noise must be non-negative.")
    if "mean_vel" in config and config.get("mean_vel", 0) < 0:
        errors.append("mean_vel must be non-negative.")

    # op_model=2 requires w0 and wr
    if config.get("op_model") == 2:
        for key in ("w0", "wr"):
            if key not in config:
                errors.append(f"op_model=2 requires '{key}' to be set.")

    return errors


def validate_analysis_config(config: dict) -> list[str]:
    """Return a list of error strings (empty = valid)."""
    errors = _check_types(config, ANALYSIS_REQUIRED_KEYS)

    if "mode" in config and config["mode"] not in VALID_MODES:
        errors.append(
            f"mode={config['mode']!r} not in valid set {VALID_MODES}"
        )
    if config.get("lowpass_freq", 1) <= config.get("highpass_freq", 0):
        errors.append("lowpass_freq must be greater than highpass_freq.")
    if config.get("resample_factor", 1) < 1:
        errors.append("resample_factor must be >= 1.")
    if config.get("filter_order", 1) < 1:
        errors.append("filter_order must be >= 1.")
    if not (0 < config.get("cross_corr_frac", 0.1) < 1):
        errors.append("cross_corr_frac must be in (0, 1).")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def validate_config_file(config_path: Path) -> bool:
    """
    Load and validate a JSON configuration file.

    Returns True if valid, False if errors were found.
    """
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return False

    with open(config_path, encoding="utf-8") as fh:
        try:
            config = json.load(fh)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error in %s: %s", config_path, exc)
            return False

    logger.info("Validating: %s", config_path)

    # Detect which schema to apply by the keys present
    if "tmax" in config:
        errors = validate_simulation_config(config)
        schema_name = "simulation"
    else:
        errors = validate_analysis_config(config)
        schema_name = "analysis"

    if errors:
        logger.error(
            "Validation FAILED for %s config (%s):",
            schema_name, config_path.name,
        )
        for err in errors:
            logger.error("  ✗  %s", err)
        return False

    logger.info(
        "Validation PASSED for %s config (%s). All %d required keys present.",
        schema_name, config_path.name, len(errors),
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a simulation or analysis JSON config file."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to the JSON config file"
    )
    args = parser.parse_args()

    ok = validate_config_file(args.config)
    sys.exit(0 if ok else 1)
