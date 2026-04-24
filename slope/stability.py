import jax
import jax.numpy as jnp

def get_default_params():
    """
    Returns the core physical parameters derived from your 
    C0019 sediment strength observations.
    """
    return {
        "gamma": 1.0,      # Damping (how fast the bed settles)
        "alpha": 0.6,      # Forcing coupling (sensitivity to stress)
        "sigma_q": 0.45,   # Bed noise (micro-heterogeneity)
        "k_logistic": 18.0 # Steepness of the failure transition
    }

@jax.jit
def update_mobility(q, s, key, params):
    """
    The main Physics Engine (SDE Step).
    Calculates the next state of bed mobility (q) given forcing (s).
    """
    dt = 0.05
    # Stochastic component (The 'Noise' of the sediment grains)
    noise = jax.random.normal(key) * params["sigma_q"] * jnp.sqrt(dt)
    
    # Differential equation: dq = (-gamma*q + alpha*s)dt + noise
    dq = (-params["gamma"] * q + params["alpha"] * s) * dt + noise
    return q + dq

@jax.jit
def calculate_failure_probability(q, threshold_lo, threshold_hi, k):
    """
    The Logistic Gate.
    Calculates the probability that the sediment has transitioned 
    from 'Static' to 'Mass Transport Deposit' (MTD).
    """
    # Average threshold for the bi-stable state
    mid_threshold = (threshold_lo + threshold_hi) / 2.0
    return 1.0 / (1.0 + jnp.exp(-k * (q - mid_threshold)))

def run_ensemble(forcing_series, n_sims=1000, seed=42):
    """
    Runs a full Monte Carlo ensemble simulation.
    This is the core of 'ProboSed' - treating slope failure as a distribution.
    """
    params = get_default_params()
    master_key = jax.random.PRNGKey(seed)
    
    # Initialize 10,000 simulations of 'q' starting at 0 (stable)
    q_ensemble = jnp.zeros(n_sims)
    
    # This loop tracks the evolution over the time series
    results = []
    for s in forcing_series:
        master_key, subkey = jax.random.split(master_key)
        # Apply the update to all 10,000 sims at once (Vectorization!)
        q_ensemble = update_mobility(q_ensemble, s, subkey, params)
        results.append(q_ensemble)
        
    return jnp.array(results)

@jax.jit
def calculate_lyapunov(q_ensemble, epsilon=1e-5):
    '''
    Calculates the exponential rate of divergence between 
    sediment states. This is the 'Chaos Metric' for JpGU.
    '''
    # We compare the ensemble to a slightly perturbed version
    divergence = jnp.abs(q_ensemble[1:] - q_ensemble[:-1])
    # Lyapunov Exponent = Mean(log(divergence))
    return jnp.mean(jnp.log(divergence + 1e-10))

@jax.jit
def calculate_lyapunov(q_ensemble, epsilon=1e-5):
    '''
    Calculates the exponential rate of divergence between
    sediment states. This is the 'Chaos Metric' for JpGU.
    '''
    # We compare the ensemble to a slightly perturbed version
    divergence = jnp.abs(q_ensemble[1:] - q_ensemble[:-1])
    # Lyapunov Exponent = Mean(log(divergence))
    return jnp.mean(jnp.log(divergence + 1e-10))

@jax.jit
def calculate_lyapunov(q_ensemble, epsilon=1e-5):
    '''
    Calculates the exponential rate of divergence between
    sediment states. This is the 'Chaos Metric' for JpGU.
    '''
    # We compare the ensemble to a slightly perturbed version
    divergence = jnp.abs(q_ensemble[1:] - q_ensemble[:-1])
    # Lyapunov Exponent = Mean(log(divergence))
    return jnp.mean(jnp.log(divergence + 1e-10))

@jax.jit
def calculate_lyapunov(q_ensemble, epsilon=1e-5):
    '''
    Calculates the exponential rate of divergence between
    sediment states. This is the 'Chaos Metric' for JpGU.
    '''
    # We compare the ensemble to a slightly perturbed version
    divergence = jnp.abs(q_ensemble[1:] - q_ensemble[:-1])
    # Lyapunov Exponent = Mean(log(divergence))
    return jnp.mean(jnp.log(divergence + 1e-10))
