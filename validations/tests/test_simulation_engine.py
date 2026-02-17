"""
test_simulation_engine.py
=========================
Unit tests for source.core.simulation_engine.

Run with:
    pytest validations/tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from source.core.simulation_engine import (
    ConnectivityProcessor,
    SimulationConfig,
)


# ---------------------------------------------------------------------------
# SimulationConfig tests
# ---------------------------------------------------------------------------
class TestSimulationConfig:
    def test_defaults_are_valid(self, tmp_path):
        cfg = SimulationConfig(data_dir=tmp_path, output_dir=tmp_path)
        assert cfg.tmax > 0
        assert cfg.dt > 0
        assert cfg.dt_save >= cfg.dt

    def test_invalid_tmax_raises(self, tmp_path):
        with pytest.raises(ValueError, match="tmax"):
            SimulationConfig(tmax=-1.0, data_dir=tmp_path, output_dir=tmp_path)

    def test_invalid_dt_raises(self, tmp_path):
        with pytest.raises(ValueError, match="dt"):
            SimulationConfig(dt=-1e-4, data_dir=tmp_path, output_dir=tmp_path)

    def test_dt_save_less_than_dt_raises(self, tmp_path):
        with pytest.raises(ValueError, match="dt_save"):
            SimulationConfig(dt=1e-3, dt_save=1e-5, data_dir=tmp_path, output_dir=tmp_path)

    def test_output_dir_is_created(self, tmp_path):
        out_dir = tmp_path / "new_subdir" / "nested"
        SimulationConfig(data_dir=tmp_path, output_dir=out_dir)
        assert out_dir.exists()


# ---------------------------------------------------------------------------
# ConnectivityProcessor tests
# ---------------------------------------------------------------------------
class TestConnectivityProcessor:
    def _random_upper_tri_matrix(self, N: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                M[i, j] = rng.random()
        return M

    def test_symmetrize_makes_matrix_symmetric(self):
        N  = 10
        C1 = self._random_upper_tri_matrix(N)
        C1_sym, *_ = ConnectivityProcessor.symmetrize(C1)
        assert np.allclose(C1_sym, C1_sym.T), "C1 must be symmetric after symmetrize()"

    def test_normalise_sums_to_one(self):
        N  = 8
        C1 = self._random_upper_tri_matrix(N) + np.eye(N) * 0  # keep off-diag only
        C1, *_ = ConnectivityProcessor.symmetrize(C1)
        C1_norm, _ = ConnectivityProcessor.normalise(C1)
        mask  = ~np.eye(N, dtype=bool)
        total = np.sum(C1_norm[mask])
        assert abs(total - 1.0) < 1e-10, f"Normalised weights sum = {total}, expected 1.0"

    def test_normalise_zero_matrix_raises(self):
        N = 5
        C1 = np.zeros((N, N))
        with pytest.raises(ValueError, match="zero total weight"):
            ConnectivityProcessor.normalise(C1)

    def test_to_delay_indices_dtype(self):
        N   = 5
        C1  = np.eye(N)
        tau = np.ones((N, N)) * 1e-3
        delays, max_hist = ConnectivityProcessor.to_delay_indices(C1, tau, dt=1e-4)
        assert delays.dtype == np.int32
        assert max_hist >= 1

    def test_to_delay_indices_zeroed_where_no_connection(self):
        N   = 4
        C1  = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=float)
        tau = np.ones((N, N)) * 5e-4
        delays, _ = ConnectivityProcessor.to_delay_indices(C1, tau, dt=1e-4)
        # Where C1 == 0, delays should be 0
        assert delays[0, 2] == 0
        assert delays[0, 3] == 0

    def test_derive_frequencies_shape(self):
        N  = 10
        C1 = np.zeros((N, N))
        f  = ConnectivityProcessor.derive_frequencies(C1, f_min=40.0, f_max=80.0)
        assert f.shape == (N,), f"Expected shape ({N},), got {f.shape}"


# ---------------------------------------------------------------------------
# Statistical sanity on cross-correlation helper
# ---------------------------------------------------------------------------
class TestCrossCorrelation:
    def test_autocorrelation_at_zero_lag(self):
        """Autocorrelation of a signal with itself should peak at lag 0."""
        from source.analysis.signal_analysis import _cross_correlation_lags

        N = 200
        rng = np.random.default_rng(1)
        x = rng.standard_normal(N)
        cc = _cross_correlation_lags(x, x, max_lag=20)
        peak_idx = int(np.argmax(np.abs(cc)))
        # Peak should be at the centre (lag = 0)
        assert peak_idx == 20, f"Expected peak at index 20 (lag=0), got {peak_idx}"


# ---------------------------------------------------------------------------
# Validation module tests
# ---------------------------------------------------------------------------
class TestConfigValidator:
    def test_valid_simulation_config_passes(self):
        from validations.validate_config import validate_simulation_config

        cfg = dict(
            tmax=60.0, dt=1e-4, K=1e5, a=-5.0,
            op_net=3, op_model=1, rat="R01",
            mean_delay=5.8, sig_noise=1e-3,
        )
        errors = validate_simulation_config(cfg)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_invalid_op_net_fails(self):
        from validations.validate_config import validate_simulation_config

        cfg = dict(
            tmax=60.0, dt=1e-4, K=1e5, a=-5.0,
            op_net=99, op_model=1, rat="R01",
            mean_delay=5.8, sig_noise=1e-3,
        )
        errors = validate_simulation_config(cfg)
        assert any("op_net" in e for e in errors)

    def test_missing_key_is_reported(self):
        from validations.validate_config import validate_simulation_config

        cfg = dict(tmax=60.0)  # missing most keys
        errors = validate_simulation_config(cfg)
        assert len(errors) > 0

    def test_valid_analysis_config_passes(self):
        from validations.validate_config import validate_analysis_config

        cfg = dict(
            lowpass_freq=0.5, highpass_freq=0.01, filter_order=50,
            resample_factor=100, transient_steps=15,
            cross_corr_frac=0.001, mode="filtered",
            group="t1", rat="R01",
            output_dir="/tmp/results",
        )
        errors = validate_analysis_config(cfg)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_invalid_mode_fails(self):
        from validations.validate_config import validate_analysis_config

        cfg = dict(
            lowpass_freq=0.5, highpass_freq=0.01, filter_order=50,
            resample_factor=100, transient_steps=15,
            cross_corr_frac=0.001, mode="INVALID",
            group="t1", rat="R01",
            output_dir="/tmp/results",
        )
        errors = validate_analysis_config(cfg)
        assert any("mode" in e for e in errors)
