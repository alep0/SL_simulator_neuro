"""
signal_analysis.py
==================
Signal processing and functional-connectivity analysis pipeline.

Applies bandpass filtering, BOLD haemodynamic convolution, cross-correlation
functional connectivity, and power-spectrum estimation to simulated LFP
trajectories. Produces publication-quality figures and saves correlation
matrices as plain-text files.

Main entry point
----------------
    from source.analysis.signal_analysis import run_analysis
    run_analysis(config)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import butter, fftconvolve, sosfilt

from source.core.simulation_engine import SimulationConfig, run_simulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------
class AnalysisConfig:
    """
    Parameters governing the post-simulation signal-analysis step.

    Attributes
    ----------
    lowpass_freq : float
        Upper cut-off frequency for bandpass filter (Hz).
    highpass_freq : float
        Lower cut-off frequency for bandpass filter (Hz).
    filter_order : int
        Butterworth filter order.
    resample_factor : int
        Downsampling factor applied after filtering (``sl`` in original code).
    transient_steps : int
        Number of resampled time-steps to discard at the start (warm-up).
    cross_corr_frac : float
        Fraction of signal length used as the maximum cross-correlation lag.
    output_dir : Path
        Directory for saving figures and matrices.
    group : str
        Experimental group label (e.g. "t1", "t2").
    rat : str
        Rat identifier (e.g. "R01").
    mode : str
        One of ``"raw"``, ``"filtered"``, ``"bold"``.
    """

    def __init__(
        self,
        output_dir: Path,
        group: str,
        rat: str,
        lowpass_freq: float = 0.5,
        highpass_freq: float = 0.01,
        filter_order: int = 50,
        resample_factor: int = 100,
        transient_steps: int = 15,
        cross_corr_frac: float = 0.001,
        mode: str = "filtered",
    ) -> None:
        self.lowpass_freq    = lowpass_freq
        self.highpass_freq   = highpass_freq
        self.filter_order    = filter_order
        self.resample_factor = resample_factor
        self.transient_steps = transient_steps
        self.cross_corr_frac = cross_corr_frac
        self.output_dir      = Path(output_dir)
        self.group           = group
        self.rat             = rat
        self.mode            = mode

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "AnalysisConfig: mode=%s, lowpass=%.3f Hz, resample=%d",
            mode, lowpass_freq, resample_factor,
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_matrix_txt(matrix: np.ndarray, file_path: Path) -> None:
    """Write a 2-D numpy array to a whitespace-delimited text file."""
    try:
        with open(file_path, "w") as fh:
            for row in matrix:
                fh.write(" ".join(map(str, row)) + "\n")
        logger.debug("Matrix saved to %s", file_path)
    except OSError as exc:
        logger.error("Could not save matrix to %s: %s", file_path, exc)
        raise


# ---------------------------------------------------------------------------
# Signal processing functions
# ---------------------------------------------------------------------------
def bandpass_filter(
    signal: np.ndarray,
    highpass_freq: float,
    lowpass_freq: float,
    sample_rate: float,
    order: int,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    highpass_freq : float   Lower cut-off (Hz).
    lowpass_freq  : float   Upper cut-off (Hz).
    sample_rate   : float   Sampling rate (Hz).
    order         : int     Filter order.

    Returns
    -------
    filtered : np.ndarray, shape (n_samples,)
    """
    sos_hp = butter(order, highpass_freq, btype="hp", fs=sample_rate, output="sos")
    filtered = sosfilt(sos_hp, signal)
    sos_lp = butter(order, lowpass_freq,  btype="lp", fs=sample_rate, output="sos")
    return sosfilt(sos_lp, filtered)


def _periodic_hrf(t: np.ndarray, period: float = 30.0) -> np.ndarray:
    """
    Double-gamma haemodynamic response function (HRF) made periodic.

    Uses the canonical human HRF parameters from FSL.

    Parameters
    ----------
    t      : np.ndarray  Time vector (s).
    period : float       Repetition period (s).

    Returns
    -------
    hrf : np.ndarray  Periodic HRF evaluated at *t*.
    """
    t_mod = np.mod(t, period)
    a1, b1, c, a2, b2 = 6.0, 0.9, 0.35, 12.0, 0.9
    term1 = (t_mod ** (a1 - 1)) * np.exp(-t_mod / b1) / (b1 ** a1 * math.gamma(a1))
    term2 = (t_mod ** (a2 - 1)) * np.exp(-t_mod / b2) / (b2 ** a2 * math.gamma(a2))
    return term1 - c * term2


def compute_bold_signal(total_time: float, dt: float, lfp: np.ndarray) -> np.ndarray:
    """
    Convert simulated LFP to BOLD signal via the balloon-Windkessel model.

    Based on Cabral, Hugues, Sporns & Deco (2011).

    Parameters
    ----------
    total_time : float         Total duration (s).
    dt         : float         Integration time step (s).
    lfp        : np.ndarray    LFP trace, shape (n_samples,).

    Returns
    -------
    bold : np.ndarray, shape (n_samples,)
    """
    # Haemodynamic model parameters
    taus, tauf, tauo = 0.65, 0.41, 0.98
    alpha             = 0.32
    Eo                = 0.34
    vo                = 0.02
    k1, k2, k3       = 7 * Eo, 2.0, 2 * Eo - 0.2

    itaus  = 1.0 / taus
    itauf  = 1.0 / tauf
    itauo  = 1.0 / tauo
    ialpha = 1.0 / alpha

    n_t = int(round(total_time / dt))
    x   = np.zeros((n_t, 4))
    x[0, :] = [0.0, 1.0, 1.0, 1.0]

    for i in range(n_t - 1):
        x[i+1, 0] = x[i, 0] + dt * (lfp[i] - itaus * x[i, 0] - itauf * (x[i, 1] - 1))
        x[i+1, 1] = x[i, 1] + dt * x[i, 0]
        x[i+1, 2] = x[i, 2] + dt * itauo * (x[i, 1] - x[i, 2] ** ialpha)
        x[i+1, 3] = x[i, 3] + dt * itauo * (
            x[i, 1] * (1 - (1 - Eo) ** (1.0 / x[i, 1])) / Eo
            - (x[i, 2] ** ialpha) * x[i, 3] / x[i, 2]
            )

    bold = (100.0 / Eo) * vo * (
        k1 * (1 - x[:, 3])
        + k2 * (1 - x[:, 3] / x[:, 2])
        + k3 * (1 - x[:, 2])
        )
    return bold


# ---------------------------------------------------------------------------
# Cross-correlation utilities
# ---------------------------------------------------------------------------
def _cross_correlation_lags(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> np.ndarray:
    """Normalised cross-correlation for lags in [-max_lag, +max_lag]."""
    N = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(len(lags))
    for k, lag in enumerate(lags):
        if lag < 0:
            corr[k] = np.dot(x[:N + lag], y[-lag:N])
        elif lag > 0:
            corr[k] = np.dot(x[lag:N], y[:N - lag])
        else:
            corr[k] = np.dot(x, y)
    return corr


def compute_cross_correlation_matrix(
    signals: np.ndarray, frac: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the peak cross-correlation matrix for an (N × T) signal array.

    Parameters
    ----------
    signals : np.ndarray, shape (N, T)
        Normalised signal matrix.
    frac : float
        Maximum lag as a fraction of signal length.

    Returns
    -------
    lag_matrix : np.ndarray, shape (N, N)
        Lag (in samples) at which the peak cross-correlation occurs.
    cc_matrix  : np.ndarray, shape (N, N)
        Peak cross-correlation value.
    """
    N, T  = signals.shape
    max_lag = int(T * frac)
    lag_matrix = np.zeros((N, N))
    cc_matrix  = np.zeros((N, N))
    log_interval = max(1, N // 10)

    for i in range(N):
        if i % log_interval == 0:
            logger.info(
                "Cross-correlation matrix: %.1f%% (i=%d)", 100 * i / N, i
                )
        for j in range(N):
            cc = _cross_correlation_lags(signals[i], signals[j], max_lag)
            norm = np.sqrt(np.sum(signals[i] ** 2) * np.sum(signals[j] ** 2))
            if norm > 0:
                cc /= norm
            peak_idx = int(np.argmax(np.abs(cc)))
            cc_matrix[i, j]  = cc[peak_idx]
            lag_matrix[i, j] = peak_idx - max_lag

    return lag_matrix, cc_matrix


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_signals(
    signals: np.ndarray, dt_save: float, title: str, save_path: Path
) -> None:
    """Stack-plot of normalised signals."""
    n_rois, n_t = signals.shape
    time_axis = np.arange(n_t) * dt_save

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(n_rois):
        std = np.std(signals[i])
        norm_sig = signals[i] / (4 * std) if std > 0 else signals[i]
        ax.plot(time_axis, norm_sig + i + 1, "k-", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI")
    ax.set_xlim([0, time_axis[-1]])
    ax.set_ylim([0, n_rois + 1])
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Signal plot saved: %s", save_path)


def plot_power_spectrum(
    signals: np.ndarray, dt_save: float, save_path: Path
) -> None:
    """Log–log plot of mean power spectrum across all ROIs."""
    N, T = signals.shape
    freqs = fftfreq(T, dt_save)
    pos   = freqs[:T // 2]

    power = np.zeros((N, len(pos)))
    for i in range(N):
        sp = fft(signals[i])
        power[i] = np.abs(sp[:len(pos)])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pos, np.mean(power, axis=0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Average Power Spectrum")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Power spectrum saved: %s", save_path)


def plot_correlation_matrix(
    matrix: np.ndarray, title: str, save_path: Path, mode: str = "correlation"
) -> None:
    """Heatmap of a correlation or general matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    kw = {"cmap": "coolwarm", "square": True, "ax": ax}
    if mode == "correlation":
        kw.update({"vmin": -1, "vmax": 1})
    sns.heatmap(matrix, **kw)
    ax.set_title(title)
    ax.set_xlabel("Oscillator")
    ax.set_ylabel("Oscillator")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Correlation matrix plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Coarse-graining (remove invalid ROIs)
# ---------------------------------------------------------------------------
_INVALID_ROI_INDICES = frozenset([0, 1, 2, 41, 78, 79, 80, 81, 120, 157])


def coarse_grain_correlation(matrix: np.ndarray) -> np.ndarray:
    """
    Remove rows and columns corresponding to invalid ROIs.

    The list of excluded indices matches those hard-coded in the original
    analysis pipeline (ROIs 0,1,2,41,78 and their hemisphere mirror).

    Parameters
    ----------
    matrix : np.ndarray, shape (N, N)

    Returns
    -------
    reduced : np.ndarray, shape (N-10, N-10)
    """
    valid_idx = [
        i for i in range(matrix.shape[0]) if i not in _INVALID_ROI_INDICES
    ]
    tmp = matrix[np.ix_(valid_idx, valid_idx)]
    logger.debug(
        "Coarse-grained: %d → %d ROIs", matrix.shape[0], tmp.shape[0]
    )
    return tmp


# ---------------------------------------------------------------------------
# Per-signal analysis pipeline
# ---------------------------------------------------------------------------
def _analyse_signals(
    signals: np.ndarray,
    label: str,
    dt_save: float,
    transient: int,
    cfg: AnalysisConfig,
) -> None:
    """Run the full analysis on a (N × T) downsampled signal array."""
    data = signals[:, transient:]

    # Signals plot
    plot_signals(
        data, dt_save,
        title=f"Neural activity – {label}  ({data.shape[0]} ROIs)",
        save_path=cfg.output_dir / f"{label}_signals.png",
    )
    logger.info("Signal plot done: %s", label)

    # Power spectrum
    plot_power_spectrum(
        data, dt_save,
        save_path=cfg.output_dir / f"{label}_power_spectrum.png",
    )
    logger.info("Power spectrum done: %s", label)

    # Functional connectivity (cross-correlation)
    _, cc_matrix = compute_cross_correlation_matrix(data, cfg.cross_corr_frac)
    cc_matrix_nd = cc_matrix - np.eye(cc_matrix.shape[0])
    peak = np.max(np.abs(cc_matrix_nd))
    if peak > 0:
        cc_norm = cc_matrix_nd / peak
    else:
        cc_norm = cc_matrix_nd
    logger.info("Cross-correlation matrix computed: %s", label)

    # Coarse-grain
    cc_coarse = coarse_grain_correlation(cc_norm)

    # Plot and save
    tag = f"{cfg.group}_{cfg.rat}_{label}"
    matrix_path = cfg.output_dir / f"{tag}_CC_normalised_coarse.png"
    plot_correlation_matrix(cc_coarse, tag, matrix_path, mode="correlation")
    save_matrix_txt(cc_coarse, matrix_path.with_suffix(".txt"))
    logger.info("Functional connectivity saved: %s", matrix_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_analysis(
    sim_config: SimulationConfig,
    analysis_cfg: AnalysisConfig,
) -> None:
    """
    Run simulation + signal analysis in a single call.

    Parameters
    ----------
    sim_config   : SimulationConfig   Simulation parameters.
    analysis_cfg : AnalysisConfig     Analysis / filtering parameters.
    """
    t0 = time.time()
    logger.info("Analysis pipeline START – rat=%s, group=%s, mode=%s",
                analysis_cfg.rat, analysis_cfg.group, analysis_cfg.mode)

    # ------- Simulate -------
    trajectory = run_simulation(sim_config)
    logger.info("Trajectory shape: %s", trajectory.shape)

    dt      = sim_config.dt
    sl      = analysis_cfg.resample_factor
    sample_rate = 1.0 / dt

    # ------- Mode dispatch -------
    mode = analysis_cfg.mode
    rat  = analysis_cfg.rat
    grp  = analysis_cfg.group

    if mode == "raw":
        resampled = trajectory[:, ::sl]
        label = f"{grp}_{rat}_raw_resampled"
        _analyse_signals(resampled, label, dt * sl,
                         analysis_cfg.transient_steps, analysis_cfg)

    elif mode == "filtered":
        hf = analysis_cfg.highpass_freq
        lf = analysis_cfg.lowpass_freq
        fo = analysis_cfg.filter_order

        logger.info("Bandpass filtering (%.3f–%.3f Hz, order %d)…", hf, lf, fo)
        filtered = np.zeros_like(trajectory)
        for i in range(len(trajectory)):
            filtered[i] = bandpass_filter(trajectory[i], hf, lf, sample_rate, fo)
            if i % max(1, len(trajectory) // 10) == 0:
                logger.info("  Filtered node %d / %d", i, len(trajectory))

        resampled = filtered[:, ::sl]
        label = f"{grp}_{rat}_filtered_resampled"
        _analyse_signals(resampled, label, dt * sl,
                         analysis_cfg.transient_steps, analysis_cfg)

    elif mode == "bold":
        logger.info("Computing BOLD signals…")
        bold = np.zeros_like(trajectory)
        for i in range(len(trajectory)):
            bold[i] = compute_bold_signal(sim_config.tmax, dt, trajectory[i])
            if i % max(1, len(trajectory) // 10) == 0:
                logger.info("  BOLD node %d / %d", i, len(trajectory))

        resampled = bold[:, ::sl]
        label = f"{grp}_{rat}_bold_resampled"
        _analyse_signals(resampled, label, dt * sl,
                         analysis_cfg.transient_steps, analysis_cfg)

    else:
        raise ValueError(f"Unknown analysis mode: {mode!r}. Choose: raw, filtered, bold.")

    elapsed = time.time() - t0
    logger.info("Analysis pipeline DONE in %.2f s (%.2f min)", elapsed, elapsed / 60)
