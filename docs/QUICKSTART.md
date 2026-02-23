# Quick Start Guide

This guide walks you through a complete simulation run in under 10 minutes.

---

## 1. Clone and set up

### Option A — Conda (recommended)

```bash
git clone https://github.com/your-org/neuro-sl-simulator.git

git remote set-url origin git@github.com:alep0/SL_simulator_neuro.git
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
ssh -T git@github.com
ssh-add -l
git clone git@github.com:alep0/SL_simulator_neuro.git
cd neuro-sl-simulator

conda env create -f config/environment.yml
conda activate neuro_sl_env
python setup.py build_ext --inplace
```

### Option B — Docker (no local compiler needed)

```bash
git clone https://github.com/your-org/neuro-sl-simulator.git

git clone git@github.com:alep0/SL_simulator_neuro.git
cd neuro-sl-simulator
docker compose up --build
```

---

## 2. Place your data

Copy connectivity files for each rat into:

```
data/raw/<group>/RN_SI_v0-1_th-0.0/filter_kick_out/<rat>/
```

Minimum files for `op_net=3`:

```
th-0.0_R01_w.txt     # normalised weight matrix  (79×79)
th-0.0_R01_tau.txt   # conduction delay (s)       (79×79)
th-0.0_R01_v.txt     # velocity (m/s)             (79×79)
```

---

## 3. Validate your data

```bash
python validations/validate_connectivity.py \
    --data-dir data/processed/t1/FA_RN_SI_v0-1_th-0.0/filter_kick_out/R01 \
    --rat R01 --th-value 0.0 --op-net 3
```

```bash
python validations/validate_config.py --config config/simulation_config.json
```

---

## 4. Run a single rat

```bash
chmod +x scripts/run_simulation_analysis.sh

./scripts/run_simulation_analysis.sh \
    data/processed/t1/FA_RN_SI_v0-1_th-0.0/filter_kick_out/R01 \
    results/CC_Santiago/t1/R01 \
    config \
    3 \
    t1 \
    R01 \
    60 \
    raw
```

Arguments in order: `<data_dir> <output_dir> <config_dir> <op_net> <group> <rat> <tmax_s> <analysis_mode>`

Results appear in `results/CC_Santiago/t1/R01/`.

---

## 5. Run multiple rats in batch

```bash
chmod +x scripts/run_batch_rats.sh

./scripts/run_batch_rats.sh \
    3 \
    t1 \
    "R01 R02 R03 R04 R05" \
    "$(pwd)" \
    CC_Santiago \
    60 \
    raw
```

---

## 6. Aggregate group functional connectivity

```bash
chmod +x scripts/run_functional_connectivity.sh

./scripts/run_functional_connectivity.sh \
    "$(pwd)" \
    CC_Nuredduna_2026_01_14 \
    CC_Santiago \
    t1 \
    CC_Functional_connectivity_filt
```

Group-level figures are saved in `results/CC_Functional_connectivity_filt/`.

---

## 7. Run statistical comparisons

```python
import numpy as np
from pathlib import Path
from source.analysis.statistics import compare_groups

# Load aggregated upper-triangle correlation values
def load_upper_tri(matrix_path: Path) -> np.ndarray:
    m = np.loadtxt(str(matrix_path))
    n = m.shape[0]
    return m[np.triu_indices(n, k=1)]

data_t1 = load_upper_tri(Path("results/CC_Functional_connectivity_filt/CC_Santiago/t1/CC_Santiago_t1_avg_CC_matrix.png.txt"))
data_t2 = load_upper_tri(Path("results/CC_Functional_connectivity_filt/CC_Santiago/t2/CC_Santiago_t2_avg_CC_matrix.png.txt"))

results = compare_groups(
    data_t1, data_t2,
    label_x="t1_control",
    label_y="t2_alcohol",
    output_dir=Path("results/statistics"),
)
print(results)
```

---

## 8. Check logs

All runs write timestamped logs to `logs/`:

```bash
ls -lt logs/
tail -f logs/run_simulation_analysis_*.log
```

---

## Key configuration options

Edit `config/simulation_config.json` to change:

| Parameter | Description | Default |
|---|---|---|
| `tmax` | Simulation duration (s) | 60 |
| `K` | Global coupling strength | 1e5 |
| `a` | Bifurcation parameter | -5.0 |
| `op_net` | Network mode: 2/3/4 | 3 |
| `rat` | Rat identifier | "R01" |

Edit `config/analysis_config.json` to change:

| Parameter | Description | Default |
|---|---|---|
| `mode` | `"raw"` / `"filtered"` / `"bold"` | "filtered" |
| `lowpass_freq` | Bandpass upper cut-off (Hz) | 0.5 |
| `resample_factor` | Downsampling factor | 100 |
