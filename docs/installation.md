# Installation Guide

This document covers every supported installation method for **neuro-sl-simulator**.

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.9 | 3.11 recommended |
| C++ compiler | GCC 9 / Clang 12 / MSVC 2019 | Must support C++17 |
| OpenMP | 4.5 | Included in GCC/MSVC; needs `libomp` on macOS |
| pybind11 | 2.10 | Installed automatically via pip/conda |
| CMake | — | *Not required* – uses `setuptools` |

---

## Option 1 — Conda (recommended)

Conda installs all dependencies, including the correct C++ toolchain.

```bash
# Clone the repository
git clone https://github.com/your-org/neuro-sl-simulator.git
cd neuro-sl-simulator

# Create and activate the environment
conda env create -f config/environment.yml
conda activate neuro_sl_env

# Build the C++ extension in-place
python setup.py build_ext --inplace

# Verify the build
python -c "import stuart_landau_simulator; print('C++ module OK')"
```

If the `build_ext` step fails, check that your C++ compiler supports C++17 and that OpenMP is available. On macOS, install `libomp` first:

```bash
brew install libomp
python setup.py build_ext --inplace
```

---

## Option 2 — pip + virtualenv

```bash
git clone https://github.com/your-org/neuro-sl-simulator.git
cd neuro-sl-simulator

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install pybind11 numpy scipy matplotlib seaborn
python setup.py build_ext --inplace
pip install -e .
```

---

## Option 3 — Docker (no local build required)

See [Docker.md](Docker.md) for full instructions.

```bash
docker compose up --build
```

---

## Updating the Conda environment

If `config/environment.yml` changes after a `git pull`:

```bash
conda activate neuro_sl_env
conda env update -f config/environment.yml --prune
python setup.py build_ext --inplace
```

---

## Running the tests

```bash
conda activate neuro_sl_env
pytest validations/tests/ -v --tb=short
```

Expected output: all tests pass. If the C++ module is not built, the tests that
depend on it are automatically skipped via `pytest.importorskip`.

---

## Troubleshooting

### `ImportError: No module named 'stuart_landau_simulator'`

The C++ extension has not been built or is not on `sys.path`.

```bash
python setup.py build_ext --inplace
```

The simulator will fall back to the pure-Python implementation automatically;
it is slower but produces identical results.

---

### `omp.h not found` (Linux)

```bash
sudo apt-get install libomp-dev     # Debian / Ubuntu
sudo dnf install libomp-devel       # Fedora / RHEL
```

---

### `omp.h not found` (macOS)

```bash
brew install libomp
```

---

### Windows — MSVC not found

Ensure **Microsoft C++ Build Tools** (or Visual Studio with the C++ workload) is
installed and that you are running from a **Developer Command Prompt**.

---

## Data directory layout

Place your structural connectivity files in:

```
data/raw/<group>/RN_SI_v0-1_th-0.0/filter_kick_out/<rat>/
```

For `op_net=3` (tau-based delays) the required files per rat are:

```
th-0.0_<rat>_w.txt      # weight matrix
th-0.0_<rat>_tau.txt    # conduction delay matrix (s)
th-0.0_<rat>_v.txt      # conduction velocity matrix (m/s)
```

Run the data validator before starting a simulation:

```bash
python validations/validate_connectivity.py \
    --data-dir data/raw/t1/RN_SI_v0-1_th-0.0/filter_kick_out/R01 \
    --rat R01 --th-value 0.0 --op-net 3
```
