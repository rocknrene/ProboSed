import jax
import jax.numpy as jnp

class SedimentAgentModel:
    """
    Simulates individual sediment grain (agent) trajectories 
    during mass transport events.
    """
    def __init__(self, settling_velocity=0.02, drag_coeff=0.5):
        self.w_s = settling_velocity # m/s (based on grain size)
        self.C_d = drag_coeff        # Friction factor

    @jax.jit
    def move_agents(self, positions, velocities, forcing, key):
        """
        Updates positions and velocities of 10,000+ grains at once.
        v_next = v + (forcing - drag) * dt
        x_next = x + v * dt
        """
        dt = 0.05
        n_agents = positions.shape[0]
        
        # Add Brownian motion (stochastic diffusion)
        diffusion = jax.random.normal(key, (n_agents,)) * 0.01
        
        # Physics: Acceleration = Forcing - (Drag * Velocity^2)
        acceleration = forcing - (self.C_d * velocities**2)
        
        new_velocities = velocities + (acceleration + diffusion) * dt
        new_positions = positions + new_velocities * dt
        
        # Grains can't go below the bed (x=0)
        new_positions = jnp.maximum(new_positions, 0)
        
        return new_positions, new_velocities

def calculate_clast_distribution(positions):
    """
    Analyzes the ensemble to find 'Clast Clusters' 
    (The proxy for MTD thickness in core).
    """
    return jnp.histogram(positions, bins=20)
