"""
statistics.py
=============
Statistical comparison of functional connectivity distributions between
experimental groups (e.g. control t1 vs. alcohol t2).

Provides:
    * Empirical CDF plots
    * Box-plots with KS / Mann–Whitney U / Cohen's d annotations
    * Convenience function ``compare_groups`` for full analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size (pooled-SD denominator).

    Parameters
    ----------
    x, y : array-like  Sample distributions.

    Returns
    -------
    d : float
    """
    nx, ny = len(x), len(y)
    pooled_var = (
        (nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)
    ) / (nx + ny - 2)
    if pooled_var == 0:
        logger.warning("Pooled variance is zero – Cohen's d is undefined.")
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled_var))


def run_statistical_tests(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Run KS test, Mann–Whitney U test, and compute Cohen's d.

    Returns
    -------
    (ks_stat, ks_p, mw_stat, mw_p, cohens_d_val)
    """
    ks_stat, ks_p   = ks_2samp(x, y)
    mw_stat, mw_p   = mannwhitneyu(x, y, alternative="two-sided")
    d_val           = cohens_d(x, y)

    logger.info("KS:    stat=%.4f  p=%.6f", ks_stat, ks_p)
    logger.info("MWU:   stat=%.4f  p=%.6f", mw_stat, mw_p)
    logger.info("Cohen: d=%.4f",             d_val)

    return ks_stat, ks_p, mw_stat, mw_p, d_val


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_ecdf(
    x: np.ndarray,
    y: np.ndarray,
    label_x: str,
    label_y: str,
    output_dir: Path,
) -> None:
    """
    Plot two empirical CDFs and save the figure.
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    ecdf_x   = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ecdf_y   = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_sorted, ecdf_x, color="steelblue", label=label_x)
    ax.plot(y_sorted, ecdf_y, color="firebrick",  label=label_y)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("ECDF")
    ax.set_title("Empirical Cumulative Distribution Functions")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    fname = output_dir / f"{label_x}_vs_{label_y}_CDF.png"
    fig.savefig(str(fname), dpi=150)
    plt.close(fig)
    logger.info("ECDF plot saved: %s", fname)


def plot_boxplots(
    x: np.ndarray,
    y: np.ndarray,
    label_x: str,
    label_y: str,
    ks_p: float,
    mw_p: float,
    d_val: float,
    output_dir: Path,
) -> None:
    """
    Side-by-side box-plots with statistical annotation.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([x, y], labels=[label_x, label_y])
    ax.set_ylabel("Correlation")
    ax.set_title("Box-plots with Statistical Results")
    ax.grid(True)

    annotation = (
        f"KS test  p = {ks_p:.4f}\n"
        f"MWU test p = {mw_p:.4f}\n"
        f"Cohen's d = {d_val:.4f}"
    )
    ax.text(
        1.05, 0.95, annotation,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    fig.tight_layout()

    fname = output_dir / f"{label_x}_vs_{label_y}_boxplots.png"
    fig.savefig(str(fname), dpi=150)
    plt.close(fig)
    logger.info("Box-plot saved: %s", fname)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compare_groups(
    data_x: np.ndarray,
    data_y: np.ndarray,
    label_x: str,
    label_y: str,
    output_dir: Path,
) -> dict:
    """
    Run full statistical comparison between two distributions and produce figures.

    Parameters
    ----------
    data_x, data_y : np.ndarray   Flat arrays of correlation values.
    label_x        : str          Name for group X (e.g. "t1_control").
    label_y        : str          Name for group Y (e.g. "t2_alcohol").
    output_dir     : Path         Where to save figures.

    Returns
    -------
    results : dict
        Keys: ks_stat, ks_p, mw_stat, mw_p, cohens_d
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Comparing %s vs %s (n=%d, n=%d)",
                label_x, label_y, len(data_x), len(data_y))

    ks_stat, ks_p, mw_stat, mw_p, d = run_statistical_tests(data_x, data_y)

    plot_ecdf(data_x, data_y, label_x, label_y, output_dir)
    plot_boxplots(data_x, data_y, label_x, label_y, ks_p, mw_p, d, output_dir)

    return dict(ks_stat=ks_stat, ks_p=ks_p,
                mw_stat=mw_stat, mw_p=mw_p,
                cohens_d=d)


# ---------------------------------------------------------------------------
# CLI / standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import source.utils.logging_config  # noqa: F401

    parser = argparse.ArgumentParser(
        description="Statistical comparison between two FC distributions."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-x",   default="group_x")
    parser.add_argument("--label-y",   default="group_y")
    parser.add_argument("--demo",       action="store_true",
                        help="Run with synthetic random data")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if args.demo:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200) + 0.5
        results = compare_groups(x, y, args.label_x, args.label_y, out)
        print("Results:", results)
    else:
        parser.error("Supply --demo or integrate via Python API.")
