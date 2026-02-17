# API Reference

Full reference for the Python API and Bash scripts.

---

## Python API

### `source.core.simulation_engine`

#### `SimulationConfig`

Dataclass holding all simulation parameters.

```python
from source.core.simulation_engine import SimulationConfig
import numpy as np

cfg = SimulationConfig(
    K          = 1e5,
    f          = 40.0 * np.ones(79 * 2),
    a          = -5.0,
    sig_noise  = 1e-3,
    tmax       = 60.0,
    dt         = 1e-4,
    dt_save    = 1e-4,
    mean_delay = 5.8,
    op_net     = 3,           # 2=velocity, 3=tau, 4=bimodal
    op_model   = 1,           # 1=fixed freqs, 2=connectivity-derived
    rat        = "R01",
    th_value   = "0.0",
    data_dir   = Path("data/raw/t1/R01"),
    output_dir = Path("results/t1/R01"),
    use_cpp    = True,
    save_data  = False,
)
```

**Key fields**

| Field | Type | Description |
|---|---|---|
| `K` | float | Global coupling strength |
| `f` | ndarray (N,) | Natural frequencies in Hz |
| `a` | float | Bifurcation parameter (<0: damped) |
| `tmax` | float | Simulation duration (s) |
| `dt` | float | Integration step (s) |
| `dt_save` | float | Output sampling interval (s) |
| `mean_delay` | float | Mean axonal delay (s) |
| `op_net` | int | Network mode: 2, 3, or 4 |
| `op_model` | int | Frequency model: 1 or 2 |
| `use_cpp` | bool | Use C++ back-end if available |

---

#### `run_simulation(config)`

Run a complete simulation and return the trajectory.

```python
from source.core.simulation_engine import run_simulation

trajectory = run_simulation(config)
# trajectory.shape == (N_nodes, n_time_points)
```

**Parameters**

- `config` – `SimulationConfig` or `None` (uses defaults).

**Returns**

- `trajectory` – `np.ndarray`, shape `(N, n_save)` – real part of complex oscillator state.

**Raises**

- `ValueError` – if `op_net` is not 2, 3, or 4, or if required parameters are missing.
- `FileNotFoundError` – if connectivity files are not found.

---

### `source.analysis.signal_analysis`

#### `AnalysisConfig`

```python
from source.analysis.signal_analysis import AnalysisConfig
from pathlib import Path

ana = AnalysisConfig(
    output_dir      = Path("results/t1/R01"),
    group           = "t1",
    rat             = "R01",
    lowpass_freq    = 0.5,
    highpass_freq   = 0.01,
    filter_order    = 50,
    resample_factor = 100,
    transient_steps = 15,
    cross_corr_frac = 0.001,
    mode            = "filtered",   # "raw" | "filtered" | "bold"
)
```

---

#### `run_analysis(sim_config, analysis_cfg)`

Run simulation + signal analysis in one call.

```python
from source.analysis.signal_analysis import run_analysis

run_analysis(sim_config, analysis_cfg)
```

Produces in `output_dir`:
- `<label>_signals.png` – stacked signal traces
- `<label>_power_spectrum.png` – mean log-log power spectrum
- `<label>_CC_normalised_coarse.png` – FC heatmap
- `<label>_CC_normalised_coarse.txt` – FC matrix (plain text)

---

#### `bandpass_filter(signal, highpass_freq, lowpass_freq, sample_rate, order)`

Zero-phase Butterworth bandpass filter for a 1-D signal array.

---

#### `compute_bold_signal(total_time, dt, lfp)`

Balloon-Windkessel BOLD haemodynamic model (Cabral et al. 2011).

---

#### `compute_cross_correlation_matrix(signals, frac)`

Compute the peak cross-correlation matrix for an `(N, T)` signal array.

Returns `(lag_matrix, cc_matrix)`.

---

### `source.analysis.functional_connectivity`

#### `aggregate_functional_connectivity(...)`

Load per-rat correlation matrices and produce group-level figures.

```python
from source.analysis.functional_connectivity import aggregate_functional_connectivity

avg_matrix = aggregate_functional_connectivity(
    root_path        = "/data/workspaces/neuro_sl",
    data_name        = "CC_Nuredduna_2026_01_14",
    model_name       = "CC_Santiago",
    group_name       = "t1",
    output_name      = "CC_Functional_connectivity_filt",
    rats             = ["R01", "R02", "R03"],   # None = all 18
    correlation_type = "CC",
    signal_label     = "trajectory_filt_resampled",
)
# avg_matrix.shape == (148, 148)
```

---

### `source.analysis.statistics`

#### `compare_groups(data_x, data_y, label_x, label_y, output_dir)`

Run KS, Mann–Whitney U, and Cohen's d comparisons between two distributions.

```python
from source.analysis.statistics import compare_groups
import numpy as np

results = compare_groups(
    data_x     = np.random.randn(500),
    data_y     = np.random.randn(500) + 0.3,
    label_x    = "control",
    label_y    = "alcohol",
    output_dir = Path("results/statistics"),
)
# results == {"ks_stat": ..., "ks_p": ..., "mw_stat": ..., "mw_p": ..., "cohens_d": ...}
```

---

### `source.utils.logging_config`

Import this module to configure the root logger:

```python
import source.utils.logging_config  # side-effect: sets up handlers
```

Environment variables:

| Variable | Default | Effect |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Console verbosity: DEBUG, INFO, WARNING… |
| `LOG_DIR`   | `logs/` | Directory for the rotating log file |

---

## Bash Scripts

### `scripts/run_simulation_analysis.sh`

Single-rat simulation + analysis pipeline.

```bash
./scripts/run_simulation_analysis.sh \
    <data_dir>    \   # 1: structural connectivity directory for the rat
    <output_dir>  \   # 2: results output directory
    <config_dir>  \   # 3: directory containing JSON config files
    <op_net>      \   # 4: 2, 3, or 4
    <group>       \   # 5: t1 or t2
    <rat>         \   # 6: R01 … R19
    <tmax>            # 7: simulation duration (s)
```

Logs written to `logs/run_simulation_analysis_<timestamp>.log`.

---

### `scripts/run_batch_rats.sh`

Process multiple rats sequentially.

```bash
./scripts/run_batch_rats.sh \
    <op_net>         \   # 1
    <group>          \   # 2
    "<rat1> <rat2>"  \   # 3: space-separated rat list (quote it)
    <root_path>      \   # 4: project root
    <model_name>     \   # 5: output sub-directory label
    <tmax>               # 6
```

---

### `scripts/run_functional_connectivity.sh`

Aggregate group-level FC matrices.

```bash
./scripts/run_functional_connectivity.sh \
    <root_path>   \   # 1
    <data_name>   \   # 2
    <model_name>  \   # 3
    <group>       \   # 4
    <output_name>     # 5
```

---

## Validation scripts

### `validations/validate_config.py`

```bash
python validations/validate_config.py --config config/simulation_config.json
```

Exit 0 = valid, exit 1 = errors printed to stderr.

---

### `validations/validate_connectivity.py`

```bash
python validations/validate_connectivity.py \
    --data-dir <dir> --rat <rat> --th-value 0.0 --op-net 3
```

Exit 0 = valid, exit 1 = errors.

---

## C++ Extension

The module `stuart_landau_simulator` exposes a single class:

```python
import stuart_landau_simulator as sl

sim = sl.StuartLandauSimulator(
    N           = 158,      # number of oscillators
    max_history = 1000,     # history buffer length (steps)
    dt          = 1e-4,     # integration step (s)
    dt_save     = 1e-4,     # output sampling (s)
    tmax        = 60.0,     # total time (s)
    t_prev      = 0.0,      # warm-up time (s)
    sig_noise   = 1e-3,     # noise amplitude
    mean_delay  = 5.8,      # mean delay (s)
)

sim.set_connectivity(C1, Delays1)            # single layer
sim.set_connectivity(C1, Delays1, C2, D2)   # two layers

trajectory = sim.simulate(K=1e5, a=-5.0, f=f_vec)
# trajectory.shape == (N, n_save)
```
