"""
validate_connectivity.py
========================
Check that connectivity data files for a given rat exist, can be loaded,
and pass basic sanity checks (symmetry, non-negativity, finite values).

Usage
-----
    python validations/validate_connectivity.py \\
        --data-dir /data/workspaces/neuro_sl/data/raw/t1/R01 \\
        --rat R01 \\
        --th-value 0.0 \\
        --op-net 3

Exit codes
----------
0  all checks passed
1  one or more errors detected
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source.utils.logging_config  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level checks
# ---------------------------------------------------------------------------
def check_matrix_finite(matrix: np.ndarray, name: str) -> list[str]:
    errors = []
    if not np.all(np.isfinite(matrix)):
        n_nan = np.sum(~np.isfinite(matrix))
        errors.append(f"{name}: {n_nan} non-finite values detected.")
    return errors


def check_matrix_non_negative(matrix: np.ndarray, name: str) -> list[str]:
    errors = []
    if np.any(matrix < 0):
        n_neg = np.sum(matrix < 0)
        errors.append(f"{name}: {n_neg} negative values detected.")
    return errors


def check_matrix_symmetry(matrix: np.ndarray, name: str, atol: float = 1e-6) -> list[str]:
    errors = []
    diff = np.max(np.abs(matrix - matrix.T))
    if diff > atol:
        errors.append(
            f"{name}: not symmetric (max |M - M^T| = {diff:.3e} > {atol})."
        )
    return errors


def check_matrix_shape(
    matrix: np.ndarray, expected_shape: tuple, name: str
) -> list[str]:
    errors = []
    if matrix.shape != expected_shape:
        errors.append(
            f"{name}: expected shape {expected_shape}, got {matrix.shape}."
        )
    return errors


def load_and_check(file_path: Path) -> tuple[np.ndarray | None, list[str]]:
    """Try to load *file_path* as a matrix; return (matrix, errors)."""
    if not file_path.exists():
        return None, [f"File not found: {file_path}"]
    try:
        matrix = np.loadtxt(str(file_path))
        logger.debug("Loaded %s  shape=%s", file_path.name, matrix.shape)
        return matrix, []
    except Exception as exc:
        return None, [f"Could not read {file_path}: {exc}"]


# ---------------------------------------------------------------------------
# Per-network-mode validators
# ---------------------------------------------------------------------------
def validate_op3(data_dir: Path, rat: str, th_value: str) -> list[str]:
    """Validate tau-based delay files (op_net=3)."""
    prefix = f"th-{th_value}_{rat}"
    files  = {
        "weight matrix (w)":  data_dir / f"{prefix}_w.txt",
        "tau matrix":         data_dir / f"{prefix}_tau.txt",
        "velocity matrix (v)":data_dir / f"{prefix}_v.txt",
    }
    errors: list[str] = []
    matrices: dict[str, np.ndarray] = {}

    for label, path in files.items():
        mat, errs = load_and_check(path)
        errors += errs
        if mat is not None:
            matrices[label] = mat

    if "weight matrix (w)" in matrices:
        w = matrices["weight matrix (w)"]
        errors += check_matrix_finite(w, "w")
        errors += check_matrix_non_negative(w, "w")

    if "tau matrix" in matrices:
        tau = matrices["tau matrix"]
        errors += check_matrix_finite(tau, "tau")
        errors += check_matrix_non_negative(tau, "tau")

    if "velocity matrix (v)" in matrices:
        v = matrices["velocity matrix (v)"]
        errors += check_matrix_finite(v, "v")
        n_zero = np.sum(v == 0)
        if n_zero > 0:
            logger.warning("Velocity matrix has %d zero entries.", n_zero)

    return errors


def validate_op2(data_dir: Path, rat: str, th_value: str) -> list[str]:
    """Validate velocity-based delay files (op_net=2)."""
    prefix = f"th-{th_value}_{rat}"
    files  = {
        "weight matrix (w)":   data_dir / f"{prefix}_w.txt",
        "distance matrix (d)": data_dir / f"{prefix}_d.txt",
        "velocity matrix (v)": data_dir / f"{prefix}_v.txt",
    }
    errors: list[str] = []
    matrices: dict[str, np.ndarray] = {}

    for label, path in files.items():
        mat, errs = load_and_check(path)
        errors += errs
        if mat is not None:
            matrices[label] = mat

    for name, mat in matrices.items():
        errors += check_matrix_finite(mat, name)
        errors += check_matrix_non_negative(mat, name)

    return errors


def validate_op4(data_dir: Path, rat: str, th_value: str, ending: str) -> list[str]:
    """Validate bimodal connectivity files (op_net=4)."""
    prefix = f"{rat}_th{th_value}_t"
    files  = {
        f"C1 weights ({prefix}_w{ending})": data_dir / f"{prefix}_w{ending}",
        f"C1 delays ({prefix}_m{ending})":  data_dir / f"{prefix}_m{ending}",
        f"fibre count (n)":                 data_dir / f"{prefix}_n.txt",
        f"C2 weights ({prefix}_w2)":        data_dir / f"{prefix}_w2.txt",
        f"C2 delays ({prefix}_m2)":         data_dir / f"{prefix}_m2.txt",
        f"velocity ({rat}_v)":              data_dir / f"{rat}_th{th_value}_v.txt",
    }
    errors: list[str] = []
    for label, path in files.items():
        _, errs = load_and_check(path)
        if errs:
            errors += [f"{label}: {e}" for e in errs]
    return errors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def validate_connectivity(
    data_dir: Path,
    rat: str,
    th_value: str,
    op_net: int,
    ending: str = "1.txt",
) -> bool:
    """
    Validate connectivity files for *rat* / *op_net*.

    Returns True if all checks pass, False otherwise.
    """
    logger.info(
        "Validating connectivity: data_dir=%s, rat=%s, op_net=%d",
        data_dir, rat, op_net,
    )

    if op_net == 3:
        errors = validate_op3(data_dir, rat, th_value)
    elif op_net == 2:
        errors = validate_op2(data_dir, rat, th_value)
    elif op_net == 4:
        errors = validate_op4(data_dir, rat, th_value, ending)
    else:
        logger.error("Unsupported op_net=%d", op_net)
        return False

    if errors:
        logger.error("Connectivity validation FAILED for rat=%s:", rat)
        for err in errors:
            logger.error("  ✗  %s", err)
        return False

    logger.info("Connectivity validation PASSED for rat=%s.", rat)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate connectivity data files for a given rat."
    )
    parser.add_argument("--data-dir",  required=True, type=Path)
    parser.add_argument("--rat",       required=True)
    parser.add_argument("--th-value",  default="0.0")
    parser.add_argument("--op-net",    type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--ending",    default="1.txt")
    args = parser.parse_args()

    ok = validate_connectivity(
        args.data_dir, args.rat, args.th_value, args.op_net, args.ending
    )
    sys.exit(0 if ok else 1)
