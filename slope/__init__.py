"""
slope
=====
Slope stability physics for the Japan Trench frontal prism.

Modules:
  stability       -- OU ensemble simulation, Lyapunov, sensitivity, STABILITY_MAP
  toy_model       -- standalone simulation script
  sensitivity_run -- parameter sensitivity analysis script
"""
from slope.stability import (
    STABILITY_MAP,
    run_ensemble,
    calculate_lyapunov,
    failure_statistics,
    threshold_from_vcd,
    run_sensitivity,
)
