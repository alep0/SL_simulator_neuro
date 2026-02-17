# neuro-sl-simulator

**Stuart-Landau Neural Network Simulator for Resting-State fMRI Modelling**

A high-performance computational neuroscience toolkit that simulates resting-state brain dynamics using a delayed-coupled Stuart-Landau oscillator network fitted to empirical structural connectomes of rats. The C++ back-end (via pybind11 + OpenMP) provides ~10–50× speedup over a pure-Python baseline.

---

## Overview

The simulator models each cortical/subcortical region of interest (ROI) as a complex Stuart-Landau oscillator:

```
dZ_i/dt = Z_i (a + iω_i - |Z_i|²) + K Σ_j C_ij (Z_j(t−τ_ij) − Z_i) + noise
```

where:
- `Z_i` – complex oscillator state (real part ≈ local field potential)
- `a`   – bifurcation parameter
- `ω_i` – natural angular frequency (2π × f_i)
- `K`   – global coupling strength
- `C_ij` – normalised structural connectivity weight
- `τ_ij` – axonal conduction delay

The pipeline then:
1. Runs the ODE simulation (C++ or Python)
2. Applies bandpass filtering / BOLD haemodynamic convolution
3. Computes cross-correlation functional connectivity matrices
4. Produces group-level aggregations and statistical comparisons

---

## Project Structure

```
neuro-sl-simulator/
├── source/
│   ├── core/
│   │   ├── stuart_landau_simulator.cpp   # C++ extension (pybind11 + OpenMP)
│   │   └── simulation_engine.py          # Python simulation pipeline
│   ├── analysis/
│   │   ├── signal_analysis.py            # Filtering, BOLD, cross-correlation
│   │   ├── functional_connectivity.py    # Group-level FC aggregation
│   │   └── statistics.py                 # KS, MWU, Cohen's d
│   └── utils/
│       └── logging_config.py             # Centralised logging
├── data/
│   ├── raw/                              # Structural connectivity files
│   └── processed/                        # Intermediate data
├── config/
│   ├── simulation_config.json            # Simulation parameters
│   ├── analysis_config.json              # Analysis parameters
│   └── environment.yml                   # Conda environment
├── scripts/
│   ├── run_simulation_analysis.sh        # Single-rat pipeline
│   ├── run_batch_rats.sh                 # Multi-rat batch runner
│   └── run_functional_connectivity.sh    # Group FC aggregation
├── validations/
│   ├── validate_config.py                # JSON config validator
│   ├── validate_connectivity.py          # Data file validator
│   └── tests/
│       └── test_simulation_engine.py     # pytest unit tests
├── results/                              # Generated figures and matrices
├── logs/                                 # Runtime log files
├── docs/
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── installation.md
│   ├── api.md
│   └── Docker.md
├── setup.py                              # C++ extension build
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── LICENSE
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/neuro-sl-simulator.git
cd neuro-sl-simulator

# 2. Create conda environment and build C++ extension
conda env create -f config/environment.yml
conda activate neuro_sl_env
python setup.py build_ext --inplace

# 3. Run a single rat
./scripts/run_simulation_analysis.sh \
    data/raw/t1/RN_SI_v0-1_th-0.0/filter_kick_out/R01 \
    results/Santiago/t1/R01 \
    config \
    3 t1 R01 60
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a detailed walkthrough.

---

## Documentation

| Document | Contents |
|---|---|
| [QUICKSTART.md](docs/QUICKSTART.md) | Step-by-step guide including Docker and Conda |
| [installation.md](docs/installation.md) | Detailed installation instructions |
| [api.md](docs/api.md) | Python and Bash API reference |
| [Docker.md](docs/Docker.md) | Docker build and run instructions |

---

## Citation

If you use this software in your research, please cite the original paper (forthcoming) and acknowledge the IFISC (UIB-CSIC) computational resources.

---

## License

MIT — see [LICENSE](LICENSE).
