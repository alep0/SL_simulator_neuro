"""
functional_connectivity.py
==========================
Aggregate per-rat functional connectivity matrices across an experimental
group and produce group-level visualisations and histograms.

Usage (CLI)
-----------
    python -m source.analysis.functional_connectivity \\
        --root /data/workspaces/neuro_sl \\
        --data-name CC_Nuredduna_2026_01_14 \\
        --model-name CC_Santiago \\
        --group t1 \\
        --output-name CC_Functional_connectivity_filt

Usage (Python API)
------------------
    from source.analysis.functional_connectivity import aggregate_functional_connectivity
    aggregate_functional_connectivity(root, data_name, model_name, group, output_name)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)

# All rat identifiers used in the study
DEFAULT_RATS: List[str] = [
    "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09",
    "R10", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_matrix(directory: Path, filename: str) -> np.ndarray:
    """Load a whitespace-delimited matrix from *directory/filename*."""
    file_path = directory / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Expected file not found: {file_path}")
    try:
        matrix = np.loadtxt(str(file_path))
        logger.debug("Loaded %s  shape=%s", file_path.name, matrix.shape)
        return matrix
    except Exception as exc:
        raise IOError(f"Failed to read {file_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_correlation_matrix(
    matrix: np.ndarray,
    output_dir: Path,
    filename: str,
    mode: str = "correlation",
    saturate: bool = False,
) -> None:
    """
    Save a heatmap of *matrix* to *output_dir/filename*.

    Parameters
    ----------
    mode      : "correlation" (±1 scale) | "general" | "connectivity"
    saturate  : clip connectivity colour scale at 1000 (for count matrices).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    kw = {"square": True, "ax": ax}

    if mode == "general":
        sns.heatmap(matrix, cmap="coolwarm", **kw)
    elif mode == "correlation":
        sns.heatmap(matrix, cmap="coolwarm", vmin=-1, vmax=1, **kw)
    else:  # connectivity / count
        viridis = plt.cm.viridis(np.linspace(0, 1, 255))
        colors  = np.vstack(([[1, 1, 1, 1]], viridis))
        cmap    = ListedColormap(colors)
        extra   = {"vmin": 0, "vmax": 1000} if saturate else {}
        sns.heatmap(matrix, cmap=cmap, cbar=True, **extra, **kw)

    ax.set_title(filename)
    ax.set_xlabel("Oscillator")
    ax.set_ylabel("Oscillator")
    fig.tight_layout()

    save_path = output_dir / filename
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_histogram(
    matrix: np.ndarray,
    output_dir: Path,
    name: str,
    log_y: bool = True,
    threshold: float = -1.0,
) -> None:
    """
    Histogram of upper-triangle values of *matrix* exceeding *threshold*.
    """
    values = [
        matrix[i, j]
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[1])
        if matrix[i, j] > threshold
    ]

    fig, ax = plt.subplots()
    ax.hist(values, bins=50, color="burlywood", range=(-1, 1))
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Frequency (ROI pairs)")
    ax.set_title(name)

    save_path = output_dir / f"{name}_histogram.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------------
def aggregate_functional_connectivity(
    root_path: str | Path,
    data_name: str,
    model_name: str,
    group_name: str,
    output_name: str,
    rats: List[str] | None = None,
    correlation_type: str = "CC",
    signal_label: str = "trajectory_filt_resampled",
) -> np.ndarray:
    """
    Load per-rat correlation matrices, produce individual and average plots.

    Parameters
    ----------
    root_path      : Path   Project root directory.
    data_name      : str    Sub-directory under ``Figures/<correlation_type>/``.
    model_name     : str    Model label (used for path construction and filenames).
    group_name     : str    Experimental group ("t1" or "t2").
    output_name    : str    Output sub-directory name under ``results/``.
    rats           : list   Rat IDs to process (default: all 18 in the study).
    correlation_type : str  "CC" or "pearson".
    signal_label   : str    Part of the per-rat filename describing the signal type.

    Returns
    -------
    avg_matrix : np.ndarray, shape (148, 148)
        Average functional connectivity matrix across rats.
    """
    root   = Path(root_path)
    rats   = rats or DEFAULT_RATS

    output_dir = root / "results" / output_name / model_name / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Aggregating FC: model=%s, group=%s, n_rats=%d",
        model_name, group_name, len(rats),
    )

    avg_matrix = np.zeros((148, 148))
    histogram_counts: list = []

    for rat in rats:
        rat_dir = (
            root
            / "results"
            / correlation_type
            / data_name
            / model_name
            / group_name
            / rat
        )
        filename = (
            f"{group_name}_{rat}_{signal_label}"
            f"_{correlation_type}_normalised_coarse.txt"
        )

        try:
            matrix = load_matrix(rat_dir, filename)
        except (FileNotFoundError, IOError) as exc:
            logger.warning("Skipping rat %s – %s", rat, exc)
            continue

        # Individual plots
        plot_correlation_matrix(
            matrix, output_dir,
            f"{model_name}_{group_name}_{rat}_{correlation_type}_matrix.png",
            mode="correlation",
        )
        plot_histogram(
            matrix, output_dir,
            f"{model_name}_{group_name}_{rat}",
            log_y=True,
        )

        avg_matrix += matrix

        hist, edges = np.histogram(matrix, bins=30, range=(-1, 1))
        histogram_counts.append(hist)

    n_loaded = len(histogram_counts)
    if n_loaded == 0:
        logger.error("No rat data loaded for %s/%s!", model_name, group_name)
        return avg_matrix

    avg_matrix /= n_loaded

    # Group-level plots
    plot_correlation_matrix(
        avg_matrix, output_dir,
        f"{model_name}_{group_name}_avg_{correlation_type}_matrix.png",
        mode="correlation",
    )
    plot_histogram(
        avg_matrix, output_dir,
        f"{model_name}_{group_name}_avg",
        log_y=True,
    )

    # Average histogram across rats
    mean_hist = np.mean(histogram_counts, axis=0)
    centres   = (edges[:-1] + edges[1:]) / 2
    width     = edges[1] - edges[0]

    fig, ax = plt.subplots()
    ax.bar(centres, mean_hist, width=width, color="skyblue", edgecolor="black")
    ax.set_yscale("log")
    ax.set_title("Average histogram across rats")
    ax.set_xlabel(f"Average {correlation_type} correlation")
    ax.set_ylabel("Mean frequency")
    fig.tight_layout()
    fig.savefig(
        str(output_dir / f"{model_name}_{group_name}_avg_histogram_{correlation_type}.png"),
        dpi=150,
    )
    plt.close(fig)

    logger.info(
        "Aggregation complete for %s/%s (%d rats).", model_name, group_name, n_loaded
    )
    return avg_matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate functional connectivity matrices across rats."
    )
    p.add_argument("--root",         required=True, help="Project root path")
    p.add_argument("--data-name",    required=True, help="Data sub-directory name")
    p.add_argument("--model-name",   required=True, help="Model label")
    p.add_argument("--group",        required=True, help="Experimental group (t1/t2)")
    p.add_argument("--output-name",  required=True, help="Output sub-directory")
    p.add_argument("--rats",         nargs="+",     default=None,
                   help="Rat IDs to include (default: all 18)")
    p.add_argument("--correlation-type", default="CC",
                   choices=["CC", "pearson"], help="Correlation method")
    p.add_argument("--signal-label",
                   default="trajectory_filt_resampled",
                   help="Signal label used in per-rat file names")
    return p


if __name__ == "__main__":
    import source.utils.logging_config  # ensure logging is configured

    args = _build_parser().parse_args()
    aggregate_functional_connectivity(
        root_path=args.root,
        data_name=args.data_name,
        model_name=args.model_name,
        group_name=args.group,
        output_name=args.output_name,
        rats=args.rats,
        correlation_type=args.correlation_type,
        signal_label=args.signal_label,
    )
