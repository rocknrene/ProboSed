# ProboSed
Probabilistic Sediment Transport & Slope Failure Modeling

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Cite This Work

If you use **ProboSed** in research, please cite:

Castillo, R. (2027).  
Slip Happens: Probabilistic Modeling of Submarine Mass Transport Deposits and Sediment Routing in Convergent Margin Basins.  
PhD Dissertation, The Ohio State University.

DOI: (ONE DAY I AM DREAMING)

---

# Mission

Deterministic slope-failure models assume a single “correct” failure condition, yet natural sedimentary systems are inherently stochastic. Variations in pore pressure, permeability architecture, grain size distribution, and seismic forcing create ensembles of possible slope-failure pathways rather than a single deterministic outcome.

**ProboSed** provides a probabilistic framework for modeling submarine sediment transport and slope instability using:

- probabilistic slope-failure ensembles
- agent-based sediment transport
- machine learning classification of sedimentary disturbance
- integration of IODP core data and geophysical observations

The goal is to bridge **sedimentology, geomechanics, and data-driven modeling** to better understand submarine landslides and mass-transport deposits (MTDs).

---

# Key Features

## Probabilistic Slope Failure

Implements stochastic slope-stability simulations where the following parameters are treated as probability distributions rather than fixed values:

- pore pressure
- shear strength
- sediment density
- seismic forcing

---

## Sediment Transport Agent Models

Agent-based simulations represent sediment particles moving downslope under varying forcing conditions.

Applications include:

- submarine landslides
- turbidity currents
- mass-transport deposits (MTDs)

---

## Core Disturbance Classification

Machine learning tools for analyzing sediment cores including:

- lithology prediction
- disturbance detection
- deformation classification

Designed for **IODP core imagery and physical property datasets.**

---

## Multimodal Data Integration

Supports integration of:

- core images
- grain size distributions
- MAD density and porosity
- vane shear measurements
- geophysical well logs

---

# Scientific Applications

ProboSed is designed for research in:

- submarine landslides
- mass-transport deposits (MTDs)
- earthquake-triggered sediment transport
- slope stability modeling
- marine sediment routing systems

Example datasets used during development include cores from the **Japan Trench margin (IODP Expeditions 386 and 405).**

---

# Installation

Clone the repository:

```bash
git clone https://github.com/rocknrene/probosed.git
cd probosed
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

# Quickstart Example

Example probabilistic slope-failure simulation:

```python
from probosed import slope

model = slope.ProbabilisticSlopeModel(
    slope_angle=12,
    cohesion_mean=5,
    friction_angle_mean=30,
    pore_pressure_distribution="normal"
)

results = model.run_simulation(n=10000)

model.plot_failure_probability()
```

---

# Repository Structure

```
probosed/
│
├── probosed/
│   ├── slope/
│   ├── transport/
│   ├── core_ml/
│   └── utils/
│
├── notebooks/
│
├── data_examples/
│
├── tests/
│
└── docs/
```

---

# Roadmap

v0.1 — Initial probabilistic slope-failure framework

v0.2 — Dual-head lithology/disturbance classification for sediment cores

v0.3 — Agent-based turbidity current simulation

v0.4 — Pore pressure evolution module

v1.0 — Integrated MTD simulation framework

---

# Contributing

Contributions are welcome. Please open an issue to discuss proposed changes.

---

# License

MIT License

---

# Acknowledgments

Development of ProboSed is part of doctoral research conducted at
The Ohio State University, School of Earth Sciences.

I would like to thank the mentors and collaborators who have supported
the scientific development behind this project:

Dr. Brendan Crowell — Co-Advisor  
Dr. Jill Leonard-Pingel — Co-Advisor  

Dr. Christine Regalla — Co-Chief Scientist, IODP Expedition 405

Committee Members:

Dr. Cole  
Dr. Keep

Dr. Krissek

This work is connected to research conducted using data from
the International Ocean Discovery Program (IODP), including
Expedition 405 and Expedition 386 investigations of the Japan Trench margin.
