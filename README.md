# ProboSed
Probabilistic Sediment Transport & Slope Failure Modeling

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Cite This Work

If you use **ProboSed** in research, please cite:

Castillo, R. (2026).  
Probabilistic Modeling of Submarine Mass Transport Deposits and Sediment Routing in Convergent Margin Basins.  
PhD Dissertation, The Ohio State University.

DOI: (insert DOI when available)

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
git clone https://github.com/rocknrene/probosed.git

cd probosed

Install dependencies:
pip install -r requirements.txt

Or install as a package:
pip install probosed



