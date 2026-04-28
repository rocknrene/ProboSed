"""
transport
=========
Agent-based sediment transport model for the Japan Trench frontal prism.
 
Simulates individual sediment grain trajectories during mass transport events
using a stochastic Langevin framework consistent with the slope stability
model in slope/stability.py. All computations use pure NumPy.
 
Modules:
  agents -- SedimentAgentModel, calculate_clast_distribution, forcing_from_slope
"""
from transport.agents import (
    SedimentAgentModel,
    calculate_clast_distribution,
    forcing_from_slope,
)
