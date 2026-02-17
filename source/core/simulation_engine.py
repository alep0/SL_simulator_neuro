"""
simulation_engine.py
====================
Neural network simulation pipeline for the Stuart-Landau oscillator model.

Loads structural connectivity from empirical data, optionally uses the
C++-accelerated backend, and returns the oscillator trajectory.

Supports three network modes (op_net):
    2 – Velocity-based delays (d / vel)
    3 – Tau-based delays (tau matrix)
    4 – Bimodal connectivity (two weight layers C1 / C2)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import the compiled C++ extension
# ---------------------------------------------------------------------------
try:
    import stuart_landau_simulator as _cpp_mod  # type: ignore
    CPP_AVAILABLE = True
    logger.info("C++ accelerated module loaded successfully.")
except ImportError:
    CPP_AVAILABLE = False
    logger.warning(
        "C++ module 'stuart_landau_simulator' not found. "
        "Falling back to pure-Python simulator. "
        "Build with: python setup.py build_ext --inplace"
        )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """All parameters required to run a single simulation.

    Parameters
    ----------
    K : float
        Global coupling strength.
    f : np.ndarray
        Natural oscillation frequencies (Hz) for each node.
    a : float
        Bifurcation parameter (negative ⇒ damped oscillations).
    sig_noise : float
        Amplitude of additive complex Gaussian noise.
    tmax : float
        Total simulation duration (s).
    t_prev : float
        Warm-up time not included in the output (s).
    dt : float
        Integration time step (s).
    dt_save : float
        Output sampling interval (s).  Defaults to ``dt``.
    mean_vel : float
        Mean axonal conduction velocity (m/s).  Alias for MD in original code.
    op_net : int
        Network connectivity mode: 2 (velocity), 3 (tau), 4 (bimodal).
    op_model : int
        Model variant: 1 (fixed frequencies), 2 (connectivity-derived).
    w0 : float or None
        Minimum frequency for op_model=2.
    wr : float or None
        Maximum frequency for op_model=2.
    th_value : str
        Threshold string used to build filenames (e.g. "0.0").
    rat : str
        Rat identifier (e.g. "R01").
    data_dir : Path
        Directory containing connectivity text files.
    output_dir : Path
        Directory for saving figures and results.
    save_data : bool
        Whether to save the raw trajectory as a .npy file.
    use_cpp : bool
        Attempt to use the C++ backend (ignored if CPP_AVAILABLE is False).
    ending : str
        Filename suffix used in op_net=4 connectivity files.
    """

    # Model physics
    K: float = 1e5
    f: np.ndarray = field(default_factory=lambda: 40.0 * np.ones(79 * 2))
    a: float = -5.0
    sig_noise: float = 1e-3

    # Time
    tmax: float = 60.0
    t_prev: float = 0.0
    dt: float = 1e-4
    dt_save: float = 1e-4

    # Network
    mean_vel: float = 5.8         # <--- ???
    op_net: int = 3
    op_model: int = 1
    w0: Optional[float] = None
    wr: Optional[float] = None

    # Data identifiers
    th_value: str = "0.0"
    rat: str = "R01"
    ending: str = "1.txt"

    # I/O
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    save_data: bool = False
    use_cpp: bool = True

    def __post_init__(self) -> None:
        default_base = Path("/data/workspaces/neuro_sl")    #<---
        if self.data_dir is None:
            self.data_dir = default_base / "data" / "raw"
        else:
            self.data_dir = Path(self.data_dir)

        if self.output_dir is None:
            self.output_dir = default_base / "results"
        else:
            self.output_dir = Path(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.dt_save < self.dt:
            raise ValueError("dt_save must be >= dt.")
        if self.tmax <= 0:
            raise ValueError("tmax must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        logger.debug(
            "SimulationConfig validated: rat=%s, op_net=%d, tmax=%g s",
            self.rat, self.op_net, self.tmax,
            )


# ---------------------------------------------------------------------------
# Connectivity I/O
# ---------------------------------------------------------------------------
class ConnectivityLoader:
    """Load structural connectivity files from disk."""

    @staticmethod
    def load_matrix(file_path: Path) -> np.ndarray:
        """Load a whitespace-delimited matrix from *file_path*."""
        if not file_path.exists():
            raise FileNotFoundError(f"Connectivity file not found: {file_path}")
        try:
            matrix = np.loadtxt(str(file_path))
            logger.debug("Loaded matrix %s  shape=%s", file_path.name, matrix.shape)
            return matrix
        except Exception as exc:
            raise IOError(f"Failed to read {file_path}: {exc}") from exc

    @classmethod
    def load_op4(
        cls, data_dir: Path, rat: str, th_value: str, ending: str
    ) -> Tuple[np.ndarray, ...]:
        """Bimodal connectivity (op_net=4): two weight + delay layers."""
        prefix = f"{rat}_th{th_value}_t"
        C1  = cls.load_matrix(data_dir / f"{prefix}_w{ending}")
        m1  = cls.load_matrix(data_dir / f"{prefix}_m{ending}")
        n   = cls.load_matrix(data_dir / f"{prefix}_n.txt")
        C2  = cls.load_matrix(data_dir / f"{prefix}_w2.txt")
        m2  = cls.load_matrix(data_dir / f"{prefix}_m2.txt")
        v   = cls.load_matrix(data_dir / f"{rat}_th{th_value}_v.txt")
        logger.info("Loaded op_net=4 connectivity for rat=%s", rat)
        return C1, C2, m1, m2, n, v

    @classmethod
    def load_op2(
        cls, data_dir: Path, rat: str, th_value: str,
        ending: str, mean_vel: float
    ) -> Tuple[np.ndarray, ...]:
        """Velocity-based delays (op_net=2)."""
        prefix = f"th-{th_value}_{rat}"
        C1 = cls.load_matrix(data_dir / f"{prefix}_w.txt")
        m1 = cls.load_matrix(data_dir / f"{prefix}_d.txt") / mean_vel
        v  = cls.load_matrix(data_dir / f"{prefix}_v.txt")
        logger.info("Loaded op_net=2 connectivity for rat=%s", rat)
        return C1, None, m1, None, None, v

    @classmethod
    def load_op3(
        cls, data_dir: Path, rat: str, th_value: str, ending: str
    ) -> Tuple[np.ndarray, ...]:
        """Tau-based delays (op_net=3)."""
        prefix = f"th-{th_value}_{rat}"
        C1 = cls.load_matrix(data_dir / f"{prefix}_w.txt")
        m1 = cls.load_matrix(data_dir / f"{prefix}_tau.txt")
        v  = cls.load_matrix(data_dir / f"{prefix}_v.txt")
        logger.info("Loaded op_net=3 connectivity for rat=%s", rat)
        return C1, None, m1, None, None, v


# ---------------------------------------------------------------------------
# Connectivity processing
# ---------------------------------------------------------------------------
class ConnectivityProcessor:
    """Symmetrize, normalise, and convert matrices to delay indices."""

    @staticmethod
    def symmetrize(
        C1: np.ndarray,
        m1: Optional[np.ndarray] = None,
        C2: Optional[np.ndarray] = None,
        m2: Optional[np.ndarray] = None,
        v:  Optional[np.ndarray] = None,
        n:  Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, ...]:
        """Force symmetry on all provided matrices."""
        N = C1.shape[0]
        upper = np.triu_indices(N, k=1)

        if m1 is not None:
            m1[upper[::-1]] = m1[upper]
        if n is not None:
            C1[upper] = np.round(n[upper] * C1[upper])
        C1[upper[::-1]] = C1[upper]

        if C2 is not None and m2 is not None:
            m2[upper[::-1]] = m2[upper]
            if n is not None:
                C2[upper] = np.round(n[upper] * C2[upper])
            C2[upper[::-1]] = C2[upper]

        if v is not None:
            v[upper[::-1]] = v[upper]

        return C1, m1, C2, m2, v

    @staticmethod
    def normalise(
        C1: np.ndarray, C2: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Divide weights by total off-diagonal sum."""
        N = C1.shape[0]
        mask = ~np.eye(N, dtype=bool)
        if C2 is None:
            total = np.sum(C1[mask])
            if total == 0:
                raise ValueError("C1 has zero total weight – check connectivity file.")
            return C1 / total, None
        total = np.sum(C1[mask]) + np.sum(C2[mask])
        if total == 0:
            raise ValueError("C1+C2 have zero total weight.")
        return C1 / total, C2 / total

    @staticmethod
    def to_delay_indices(
        C: np.ndarray, tau_matrix: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, int]:
        """Convert a continuous-time delay matrix to integer step indices."""
        delays = np.round(tau_matrix / dt).astype(np.int32)
        delays[C == 0] = 0
        max_history = int(np.max(delays)) + 1
        return delays, max_history

    @staticmethod
    def derive_frequencies(
        C: np.ndarray, f_min: float, f_max: float
    ) -> np.ndarray:
        """Assign linearly spaced frequencies based on node index (op_model=2)."""
        N = C.shape[0]
        n_half = N // 2
        freqs = np.zeros(N)
        for i in range(n_half):
            freqs[i]          = f_min + i
            freqs[i + n_half] = f_min + i
        logger.info(
            "Derived frequencies: min=%.1f Hz, max=%.1f Hz",
            freqs.min(), freqs.max(),
        )
        return freqs


# ---------------------------------------------------------------------------
# Pure-Python fallback simulator
# ---------------------------------------------------------------------------
class _PythonSimulator:
    """Euler-Maruyama integration of the Stuart-Landau network (pure NumPy)."""

    def __init__(self, config: SimulationConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng()
        self._N: int = 0
        self._C1: Optional[np.ndarray] = None
        self._C2: Optional[np.ndarray] = None
        self._D1: Optional[np.ndarray] = None
        self._D2: Optional[np.ndarray] = None
        self._max_history: int = 1

    def setup_network(
        self,
        C1: np.ndarray,
        Delays1: np.ndarray,
        max_history: int,
        C2: Optional[np.ndarray] = None,
        Delays2: Optional[np.ndarray] = None,
    ) -> None:
        self._N = C1.shape[0]
        self._C1 = C1
        self._C2 = C2
        self._D1 = max_history - Delays1
        self._D2 = None if Delays2 is None else max_history - Delays2
        self._max_history = max_history

    def simulate(
        self, K: float, a: float, f: np.ndarray
    ) -> np.ndarray:
        cfg = self._cfg
        N = self._N
        rng = self._rng

        iomega = 1j * 2 * np.pi * f
        kC1 = K * self._C1 * cfg.dt
        kC2 = K * self._C2 * cfg.dt if self._C2 is not None else None
        dsig = np.sqrt(cfg.dt) * cfg.sig_noise

        # Initialise history buffer
        noise_init = cfg.dt * rng.standard_normal(2 * N * self._max_history)
        Z = noise_init.view(np.complex128).reshape((N, self._max_history))

        n_save = int(cfg.tmax / cfg.dt_save)
        Zsave  = np.zeros((N, n_save), dtype=np.complex128)
        save_interval = int(cfg.dt_save / cfg.dt)
        n_steps = int((cfg.tmax + cfg.t_prev) / cfg.dt)
        log_interval = max(1, n_steps // 10)

        nt = 0
        t  = cfg.dt

        for step in range(n_steps):
            if step % log_interval == 0:
                logger.info(
                    "Python sim progress: %.1f%% (t=%.4fs)",
                    100.0 * t / cfg.tmax, t,
                )

            Znow = Z[:, -1]
            dz = Znow * (a + iomega - np.abs(Znow) ** 2) * cfg.dt

            sumz1 = np.zeros(N, dtype=np.complex128)
            for n in range(N):
                di = self._D1[n, :] - 1
                sumz1[n] = np.sum(kC1[n, :] * (Z[np.arange(N), di] - Znow[n]))

            sumz2 = np.zeros(N, dtype=np.complex128)
            if kC2 is not None:
                for n in range(N):
                    di = self._D2[n, :] - 1
                    sumz2[n] = np.sum(kC2[n, :] * (Z[np.arange(N), di] - Znow[n]))

            if cfg.mean_vel > 0:
                Z[:, :-1] = Z[:, 1:]

            noise = dsig * rng.standard_normal(2 * N).view(np.complex128)
            Z[:, -1] = Znow + dz + noise + sumz1 + sumz2

            if t > cfg.t_prev and step % save_interval == 0 and nt < n_save:
                Zsave[:, nt] = Z[:, -1]
                nt += 1

            t += cfg.dt

        logger.info("Python simulation complete. %d points saved.", nt)
        return Zsave.real


# ---------------------------------------------------------------------------
# C++ wrapper
# ---------------------------------------------------------------------------
class _CppSimulator:
    """Thin Python wrapper around the pybind11 C++ extension."""

    def __init__(self, config: SimulationConfig) -> None:
        self._cfg = config
        self._sim = None
        self._max_history: int = 1

    def setup_network(
        self,
        C1: np.ndarray,
        Delays1: np.ndarray,
        max_history: int,
        C2: Optional[np.ndarray] = None,
        Delays2: Optional[np.ndarray] = None,
    ) -> None:
        cfg = self._cfg
        N = C1.shape[0]
        self._max_history = max_history

        self._sim = _cpp_mod.StuartLandauSimulator(
            N,
            max_history,
            cfg.dt,
            cfg.dt_save,
            cfg.tmax,
            cfg.t_prev,
            cfg.sig_noise,
            cfg.mean_vel,
        )

        C1_cont = np.ascontiguousarray(C1,      dtype=np.float64)
        D1_cont = np.ascontiguousarray(max_history - Delays1, dtype=np.int32)

        if C2 is not None and Delays2 is not None:
            C2_cont = np.ascontiguousarray(C2,      dtype=np.float64)
            D2_cont = np.ascontiguousarray(max_history - Delays2, dtype=np.int32)
            self._sim.set_connectivity(C1_cont, D1_cont, C2_cont, D2_cont)
        else:
            self._sim.set_connectivity(C1_cont, D1_cont)

    def simulate(
        self, K: float, a: float, f: np.ndarray
    ) -> np.ndarray:
        if self._sim is None:
            raise RuntimeError("Call setup_network() before simulate().")
        return self._sim.simulate(K, a, f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_simulation(config: Optional[SimulationConfig] = None) -> np.ndarray:
    """
    Run a Stuart-Landau network simulation.

    Parameters
    ----------
    config : SimulationConfig, optional
        Full simulation parameters.  Defaults are used if *None*.

    Returns
    -------
    trajectory : np.ndarray, shape (N_nodes, n_time_points)
        Real-part oscillator trajectories sampled at ``config.dt_save``.

    Raises
    ------
    ValueError
        If connectivity mode (op_net) is not 2, 3, or 4.
    FileNotFoundError
        If any required connectivity file is missing.
    """
    t0 = time.time()

    if config is None:
        config = SimulationConfig()

    use_cpp = CPP_AVAILABLE and config.use_cpp
    backend = "C++ (accelerated)" if use_cpp else "Python (pure NumPy)"
    logger.info("=" * 60)
    logger.info("Stuart-Landau Simulation — START")
    logger.info("Backend: %s | rat=%s | op_net=%d | tmax=%g s",
                backend, config.rat, config.op_net, config.tmax)

    # ------------------------------------------------------------------
    # Load connectivity
    # ------------------------------------------------------------------
    op = config.op_net
    if op == 4:
        C1, C2, m1, m2, n, v = ConnectivityLoader.load_op4(
            config.data_dir, config.rat, config.th_value, config.ending
            )
    elif op == 2:
        C1, C2, m1, m2, n, v = ConnectivityLoader.load_op2(
            config.data_dir, config.rat, config.th_value,
            config.ending, config.mean_vel
            )
    elif op == 3:
        C1, C2, m1, m2, n, v = ConnectivityLoader.load_op3(
            config.data_dir, config.rat, config.th_value, config.ending
            )
    else:
        raise ValueError(
            f"Unsupported op_net={op}. Valid values: 2, 3, 4."
            )

    logger.info("Network size: %d nodes", C1.shape[0])

    # ------------------------------------------------------------------
    # Process connectivity
    # ------------------------------------------------------------------
    C1, m1, C2, m2, v = ConnectivityProcessor.symmetrize(C1, m1, C2, m2, v, n)
    C1, C2 = ConnectivityProcessor.normalise(C1, C2)

    Delays1, max_hist1 = ConnectivityProcessor.to_delay_indices(C1, m1, config.dt)
    max_history = max_hist1
    Delays2 = None

    if op == 4:
        Delays2, max_hist2 = ConnectivityProcessor.to_delay_indices(
            C2, m2, config.dt
            )
        max_history = max(max_hist1, max_hist2)

    logger.info("Max delay: %d steps (%.4f s)", max_history, max_history * config.dt)

    # ------------------------------------------------------------------
    # Instantiate simulator
    # ------------------------------------------------------------------
    if use_cpp:
        sim: _CppSimulator | _PythonSimulator = _CppSimulator(config)
    else:
        sim = _PythonSimulator(config)

    sim.setup_network(C1, Delays1, max_history, C2, Delays2)

    # ------------------------------------------------------------------
    # Determine frequencies
    # ------------------------------------------------------------------
    if config.op_model == 2:
        if config.w0 is None or config.wr is None:
            raise ValueError("op_model=2 requires w0 and wr to be set.")
        f_vec = ConnectivityProcessor.derive_frequencies(C1, config.w0, config.wr)
    else:
        f_vec = config.f

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    trajectory = sim.simulate(config.K, config.a, f_vec)

    # ------------------------------------------------------------------
    # Optionally save
    # ------------------------------------------------------------------
    if config.save_data:
        out_file = config.output_dir / "trajectory.npy"
        np.save(str(out_file), trajectory)
        logger.info("Trajectory saved to %s", out_file)

    elapsed = time.time() - t0
    logger.info("Simulation finished in %.2f s (%.2f min)", elapsed, elapsed / 60)
    logger.info("Trajectory shape: %s", trajectory.shape)
    logger.info("=" * 60)

    return trajectory
