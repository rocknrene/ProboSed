# ProboSed

**Probabilistic Sediment Transport & Slope Failure Modeling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open 01_stitch in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rocknrene/ProboSed/blob/main/notebooks/01_stitch.ipynb)
[![Open 02_label in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rocknrene/ProboSed/blob/main/notebooks/02_label.ipynb)

---

## Cite This Work

If you use ProboSed in your research, please cite:

> Castillo, R. (2027).
> *Slip Happens: Probabilistic Modeling of Submarine Mass Transport Deposits and Sediment Routing in Convergent Margin Basins.*
> PhD Dissertation, The Ohio State University.

---

## Mission

Deterministic slope-failure models assume a single "correct" failure condition, yet natural sedimentary systems are inherently stochastic. Variations in pore pressure, permeability architecture, grain size distribution, and seismic forcing create ensembles of possible failure pathways rather than a single deterministic outcome.

ProboSed provides a probabilistic framework for modeling submarine sediment transport and slope instability using:

- Probabilistic slope-failure ensembles
- Agent-based sediment transport
- Machine learning classification of sedimentary disturbance
- Integration of IODP core imagery and geophysical observations

The goal is to bridge sedimentology, geomechanics, and data-driven modeling to better understand submarine landslides and mass-transport deposits (MTDs).

---

## Key Features

### Probabilistic Slope Failure
Stochastic slope-stability simulations where pore pressure, shear strength, sediment density, and seismic forcing are treated as probability distributions rather than fixed values.

### Sediment Transport Agent Models
Agent-based simulations representing sediment particles moving downslope under varying forcing conditions. Applications include submarine landslides, turbidity currents, and mass-transport deposits.

### Core Disturbance Classification
Interactive tools for visual classification of IODP core imagery, including lithology identification and deformation fabric mapping. Designed to produce labeled datasets for machine learning workflows.

### Multimodal Data Integration
Supports integration of core images, grain size distributions, MAD density and porosity, vane shear measurements, and geophysical well logs.

---

## Notebooks

The two main workflows are provided as Jupyter notebooks designed to run in Google Colab.

| Notebook | Purpose |
|---|---|
| [`01_stitch.ipynb`](notebooks/01_stitch.ipynb) | Stitch raw IODP core section scans into vertical chunks |
| [`02_label.ipynb`](notebooks/02_label.ipynb) | Cut chunks into patches and classify them interactively |

All expedition-specific settings (file paths, metadata column names, classification vocabulary, patch size) are contained in a single `CONFIG` block at the top of each notebook. No other edits are needed to adapt the pipeline to a different expedition or core.

**To run:**
1. Click one of the "Open in Colab" badges above
2. Edit the `CONFIG` cell for your expedition
3. Runtime → Run all

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rocknrene/ProboSed.git
cd ProboSed
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install probosed
```

---

## Quickstart Example

Probabilistic slope-failure simulation:

```python
from slope.stability import run_ensemble, calculate_lyapunov, classify_stability
import jax.numpy as jnp

# Define a simple forward model (replace with your physics)
forward_fn = lambda q: q + 0.01 * jnp.sin(q)

# Initialize an ensemble of 1000 sediment states
initial_states = jnp.linspace(0, 1, 1000)

# Run ensemble forward 100 steps
final_states = run_ensemble(initial_states, forward_fn, n_steps=100)

# Compute Lyapunov Exponent (chaos metric)
lyapunov = calculate_lyapunov(final_states)
print(f"Lyapunov Exponent: {lyapunov:.4f}")
print(f"Stability class:   {classify_stability(float(lyapunov))}")
```

Agent-based transport simulation:

```python
from transport.agents import TransportEnsemble

ensemble = TransportEnsemble(
    n_agents           = 1000,
    slope_angle_deg    = 12.0,
    pore_pressure_mean = 0.4,
    pore_pressure_std  = 0.1,
    grain_size_mean_mm = 0.063,
)

results = ensemble.run(n_steps=500, dt=0.1)
ensemble.summary()
```

---

## Repository Structure

```
ProboSed/
├── notebooks/
│   ├── 01_stitch.ipynb       # Core image stitching pipeline
│   └── 02_label.ipynb        # Patch cutting and visual classification
│
├── core_ml/
│   ├── __init__.py
│   └── labeler.py            # PatchLabeler widget (used by 02_label.ipynb)
│
├── slope/
│   ├── __init__.py
│   └── stability.py          # Lyapunov chaos metric, stability classifier
│
├── transport/
│   ├── __init__.py
│   └── agents.py             # Agent-based sediment transport ensemble
│
├── utils/
│   ├── __init__.py
│   └── patcher.py            # Standalone patch cutting utility
│
├── __init__.py
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Roadmap

- **v0.1** — Initial probabilistic slope-failure framework ✓
- **v0.2** — Visual core description labeling pipeline (01_stitch, 02_label) ✓
- **v0.3** — Agent-based turbidity current simulation
- **v0.4** — Pore pressure evolution module
- **v1.0** — Integrated MTD simulation framework

---

## Scientific Applications

ProboSed is designed for research in:

- Submarine landslides and mass-transport deposits
- Earthquake-triggered sediment transport
- Slope stability modeling at convergent margins
- Marine sediment routing systems

Example datasets used during development include cores from the Japan Trench margin (IODP Expeditions 386 and 405).

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Development of ProboSed is part of doctoral research at The Ohio State University, School of Earth Sciences.

### Advisors

- Dr. Brendan Crowell — Co-Advisor
- Dr. Jill Leonard-Pingel — Co-Advisor

### Committee

- Dr. Cole
- Dr. Keep
- Dr. Krissek

### Field Science

- Dr. Christine Regalla — Co-Chief Scientist, IODP Expedition 405

### Data

This work uses data from the International Ocean Discovery Program (IODP), including Expeditions 405 and 386 at the Japan Trench margin.
