"""
agents.py
=========
Agent-based sediment transport model for the Japan Trench frontal prism.

Simulates individual sediment grain trajectories during mass transport events
using a stochastic forcing framework consistent with the slope stability model
in slope/stability.py. Each grain is treated as an independent agent subject
to gravitational forcing, drag, and stochastic diffusion representing
turbulence and grain-grain interactions.

Physical basis:
  Grain motion follows a discretized Langevin equation:
    v(t+dt) = v(t) + (F - C_d * v^2) * dt + sigma * dW
    x(t+dt) = x(t) + v(t+dt) * dt
    x(t)    = max(x(t), 0)   [grains cannot penetrate the bed]

  where:
    F      = gravitational forcing (slope angle and pore pressure dependent)
    C_d    = drag coefficient (grain-size and shape dependent)
    sigma  = diffusion coefficient (turbulence intensity)
    dW     = Wiener increment (Gaussian random noise)

Connection to ProboSed pipeline:
  The SedimentAgentModel is designed to receive MTD boundary depths from
  JCORESMiner.score_to_mtd_catalog() (core_ml/labeler.py) as physical
  constraints on the transport domain. The clast distribution output
  (calculate_clast_distribution) provides a proxy for MTD thickness
  that can be compared directly against core observations.

Physical parameter context (IODP Expeditions 386 and 405):
  - Settling velocity w_s = 0.02 m/s corresponds to fine silt
    (Stokes settling for D50 ~20 microns in seawater at 2 degrees C)
  - Drag coefficient C_d = 0.5 is appropriate for natural irregular grains
    (Dietrich 1982; intermediate between sphere and angular clast)
  - Diffusion coefficient sigma = 0.01 reflects low-turbulence
    submarine mass transport conditions

Implementation note:
  All computations use pure NumPy. JAX is not required.
  The vectorized ensemble (N=10,000 grains) runs in under 10 seconds on CPU.
  This is consistent with the OU slope stability model in slope/stability.py
  which also uses pure NumPy for portability and reproducibility.

References:
  Dietrich, W.E. (1982). Settling velocity of natural particles.
    Water Resources Research, 18(6), 1615-1626.
  Malamud, B.D., et al. (2004). Landslide inventories and their statistical
    properties. Earth Surface Processes and Landforms, 29(6), 687-711.
"""

import numpy as np


# =============================================================================
# PHYSICAL CONSTANTS
# Defined at module level for transparency and easy adjustment.
# =============================================================================

# Default settling velocity (m/s)
# Stokes settling for fine silt (D50 ~20 microns) in seawater at 2 degrees C.
# Representative of the Japan Trench frontal prism hemipelagic sediment
# (siliceous mudstone, dominant lithology at Site C0019, Exp 405).
DEFAULT_SETTLING_VELOCITY = 0.02

# Default drag coefficient (dimensionless)
# Intermediate value for natural irregular grains (Dietrich 1982).
# Sphere = 0.47; angular clast = 0.8-1.0; rounded natural grain ~ 0.5.
DEFAULT_DRAG_COEFF = 0.5

# Default diffusion coefficient (m/s per sqrt(s))
# Represents turbulence intensity during submarine mass transport.
# Low value (0.01) reflects the low-Reynolds-number conditions of
# deep-water (>6000 m) mass transport events at the Japan Trench.
DEFAULT_DIFFUSION = 0.01

# Default timestep (s)
# Small enough to resolve grain-scale dynamics without numerical instability.
# Stability criterion: dt < 1 / (2 * C_d * v_max) ~ 0.05 s for v_max ~ 10 m/s.
DEFAULT_DT = 0.05


# =============================================================================
# SEDIMENT AGENT MODEL
# =============================================================================

class SedimentAgentModel:
    """
    Simulates individual sediment grain trajectories during mass transport events.

    Each grain is an independent agent subject to gravitational forcing, drag,
    and stochastic diffusion. The ensemble of N_agents grains is propagated
    simultaneously using vectorized NumPy operations.

    The grain motion equation is a discretized Langevin equation:
      v(t+dt) = v(t) + [F - C_d * v(t)^2 + sigma * N(0,1)] * dt
      x(t+dt) = x(t) + v(t+dt) * dt
      x(t)    = max(x(t), 0)   [reflective bed boundary]

    The drag term is quadratic in velocity, appropriate for grain Reynolds
    numbers Re > 1 (turbulent drag regime). For Re << 1 (Stokes regime),
    linear drag (-C_d * v) would be more appropriate; use the Stokes variant
    for very fine clay particles (D < 2 microns).

    Parameters
    ----------
    settling_velocity : float
        Grain settling velocity in m/s. Default 0.02 m/s (fine silt).
        Increase for coarser grains: sand D50=250 microns -> ~0.03 m/s.
    drag_coeff : float
        Drag coefficient C_d (dimensionless). Default 0.5 (natural grain).
    diffusion : float
        Stochastic diffusion coefficient sigma (m/s per sqrt(s)).
        Represents turbulence intensity. Default 0.01.
    dt : float
        Timestep in seconds. Default 0.05 s.
    """

    def __init__(
        self,
        settling_velocity = DEFAULT_SETTLING_VELOCITY,
        drag_coeff        = DEFAULT_DRAG_COEFF,
        diffusion         = DEFAULT_DIFFUSION,
        dt                = DEFAULT_DT,
    ):
        self.w_s   = settling_velocity   # grain settling velocity (m/s)
        self.C_d   = drag_coeff          # drag coefficient (dimensionless)
        self.sigma = diffusion           # diffusion coefficient (m/s per sqrt(s))
        self.dt    = dt                  # timestep (s)

    def move_agents(self, positions, velocities, forcing, rng):
        """
        Update grain positions and velocities by one timestep.

        Applies the discretized Langevin equation to all N_agents grains
        simultaneously using vectorized NumPy operations.

        The bed boundary condition is reflective: grains that would move
        below x=0 are placed at x=0 with their velocity zeroed. This
        represents deposition — grains that reach the bed are considered
        deposited and no longer actively transported.

        Parameters
        ----------
        positions  : np.ndarray, shape (N_agents,)
            Current grain positions (m above bed).
        velocities : np.ndarray, shape (N_agents,)
            Current grain velocities (m/s, positive = upslope).
        forcing    : float or np.ndarray
            Gravitational forcing term (m/s^2).
            Scalar applies the same forcing to all grains.
            Array allows spatially variable or grain-specific forcing.
        rng : np.random.Generator
            NumPy random generator instance (from np.random.default_rng()).
            Passed explicitly to ensure reproducibility.

        Returns
        -------
        new_positions  : np.ndarray, shape (N_agents,)
            Updated grain positions (m above bed), clipped to >= 0.
        new_velocities : np.ndarray, shape (N_agents,)
            Updated grain velocities (m/s). Zero where grain is deposited.
        """
        n_agents = positions.shape[0]   # number of grains in the ensemble

        # Stochastic diffusion term — Brownian motion representing turbulence
        # Scaled by sqrt(dt) per the Euler-Maruyama integration scheme
        diffusion = rng.standard_normal(n_agents) * self.sigma * np.sqrt(self.dt)

        # Acceleration: gravitational forcing minus quadratic drag
        # Quadratic drag is appropriate for Re > 1 (turbulent regime)
        # Sign convention: positive velocity is upslope, negative is downslope
        acceleration = forcing - (self.C_d * velocities ** 2 * np.sign(velocities))

        # Velocity update: Euler-Maruyama step
        new_velocities = velocities + (acceleration + diffusion / self.dt) * self.dt

        # Position update: explicit Euler step
        new_positions = positions + new_velocities * self.dt

        # Reflective bed boundary: grains below x=0 are deposited
        # Velocity is zeroed at deposition to prevent re-entrainment
        deposited      = new_positions <= 0
        new_positions  = np.maximum(new_positions, 0.0)
        new_velocities = np.where(deposited, 0.0, new_velocities)

        return new_positions, new_velocities

    def run(self, n_agents=10_000, n_steps=500, forcing=0.5, seed=42):
        """
        Run a complete ensemble simulation and return the full trajectory record.

        Initializes all grains at a random height above the bed with zero
        initial velocity, then propagates the ensemble for n_steps timesteps.

        Parameters
        ----------
        n_agents : int
            Number of independent grain agents. Default 10,000.
        n_steps : int
            Number of timesteps to simulate. Default 500.
        forcing : float
            Gravitational forcing magnitude (m/s^2).
            Physically: g * sin(slope_angle) * (1 - pore_pressure_ratio).
            Default 0.5 corresponds to a 5 degree slope with moderate
            pore pressure (lambda ~ 0.4), representative of the Japan
            Trench frontal prism slope geometry.
        seed : int
            Random seed for reproducibility. Default 42.

        Returns
        -------
        dict with keys:
            positions  : np.ndarray, shape (n_steps+1, n_agents)
                Full position trajectory for every grain.
            velocities : np.ndarray, shape (n_steps+1, n_agents)
                Full velocity trajectory for every grain.
            time       : np.ndarray, shape (n_steps+1,)
                Time axis in seconds.
            n_deposited: np.ndarray, shape (n_steps+1,)
                Number of deposited grains (x=0) at each timestep.
                Proxy for deposition rate — rate of change gives
                instantaneous deposition flux.
            final_positions : np.ndarray, shape (n_agents,)
                Final grain positions at end of simulation.
        """
        rng = np.random.default_rng(seed)   # reproducible random generator

        # Initialize grain positions: uniform random between 0 and 10 m above bed
        # Represents a freshly mobilized sediment cloud
        init_positions  = rng.uniform(0, 10, n_agents)
        init_velocities = np.zeros(n_agents)   # grains start at rest

        # Storage arrays for full trajectories
        positions  = np.zeros((n_steps + 1, n_agents))
        velocities = np.zeros((n_steps + 1, n_agents))

        positions[0]  = init_positions
        velocities[0] = init_velocities

        # Forward integration — time loop, grains are vectorized
        for t in range(n_steps):
            positions[t+1], velocities[t+1] = self.move_agents(
                positions[t], velocities[t], forcing, rng
            )

        # Time axis
        time = np.arange(n_steps + 1) * self.dt   # seconds

        # Deposition count: number of grains resting at x=0 at each timestep
        n_deposited = (positions == 0).sum(axis=1)

        return {
            'positions'      : positions,
            'velocities'     : velocities,
            'time'           : time,
            'n_deposited'    : n_deposited,
            'final_positions': positions[-1],
        }

    def summary(self, results):
        """
        Print a summary of ensemble simulation results.

        Reports key statistics from the final grain position distribution,
        including the fraction deposited and the mean transport distance.
        These metrics provide a proxy for MTD runout length and deposit
        thickness that can be compared against core observations.

        Parameters
        ----------
        results : dict
            Output of run().
        """
        final = results['final_positions']
        n_dep = (final == 0).sum()
        n_tot = len(final)

        print("--- Agent Transport Ensemble Summary ---")
        print(f"  Total grains:           {n_tot:,}")
        print(f"  Deposited (x=0):        {n_dep:,}  ({100*n_dep/n_tot:.1f}%)")
        print(f"  Still in transport:     {n_tot - n_dep:,}  ({100*(n_tot-n_dep)/n_tot:.1f}%)")
        print(f"  Mean final position:    {final.mean():.3f} m above bed")
        print(f"  Max final position:     {final.max():.3f} m above bed")
        print(f"  Std final position:     {final.std():.3f} m")
        print(f"  Simulation duration:    {results['time'][-1]:.1f} s")
        print(f"  Timestep:               {self.dt} s")
        print(f"  Settling velocity:      {self.w_s} m/s")
        print(f"  Drag coefficient:       {self.C_d}")


# =============================================================================
# CLAST DISTRIBUTION ANALYSIS
# =============================================================================

def calculate_clast_distribution(positions, bins=20, range_m=None):
    """
    Compute the spatial distribution of grain positions across the ensemble.

    The histogram of final grain positions is a proxy for the thickness
    profile of the resulting MTD deposit. Grains concentrated near x=0
    represent the basal deposit; grains at higher elevations represent
    suspended or still-mobile material.

    This distribution can be compared against grain-size or clast-count
    profiles from IODP core descriptions to evaluate whether the modeled
    transport physics is consistent with observed deposit structure.

    Parameters
    ----------
    positions : np.ndarray
        Grain positions in meters above bed.
        Typically the final_positions output from SedimentAgentModel.run().
    bins : int
        Number of histogram bins. Default 20.
    range_m : tuple or None
        (min, max) range in meters for the histogram.
        If None, uses the full range of the position array.

    Returns
    -------
    counts : np.ndarray, shape (bins,)
        Number of grains in each position bin.
    bin_edges : np.ndarray, shape (bins+1,)
        Bin edge positions in meters.
    bin_centers : np.ndarray, shape (bins,)
        Bin center positions in meters (use for plotting).
    """
    if range_m is None:
        range_m = (float(positions.min()), float(positions.max()) + 1e-6)

    counts, bin_edges = np.histogram(positions, bins=bins, range=range_m)
    bin_centers       = (bin_edges[:-1] + bin_edges[1:]) / 2

    return counts, bin_edges, bin_centers


def forcing_from_slope(slope_angle_deg, pore_pressure_ratio=0.0,
                       g=9.81, rho_sed=1800, rho_water=1025):
    """
    Compute gravitational forcing from physical slope parameters.

    Converts a slope angle and pore pressure ratio to the effective
    gravitational forcing term used in SedimentAgentModel.move_agents().

    The effective forcing accounts for the reduction in normal stress
    (and hence friction) due to pore pressure:
      F = g' * sin(theta) * (1 - lambda)
    where:
      g'     = (rho_sed - rho_water) / rho_sed * g  (buoyant gravity)
      theta  = slope angle
      lambda = pore_pressure_ratio (0 = hydrostatic, 1 = lithostatic)

    Parameters
    ----------
    slope_angle_deg : float
        Slope angle in degrees. Japan Trench frontal prism: ~5-15 degrees.
    pore_pressure_ratio : float
        Lambda = pore pressure / lithostatic pressure.
        0.0 = hydrostatic (stable), 0.4 = moderate overpressure,
        1.0 = lithostatic (fully fluidized, always fails).
    g : float
        Gravitational acceleration (m/s^2). Default 9.81.
    rho_sed : float
        Bulk sediment density (kg/m^3). Default 1800 for water-saturated
        siliceous mudstone (MAD data, Exp 405 C0019).
    rho_water : float
        Seawater density (kg/m^3). Default 1025 at 6000 m depth.

    Returns
    -------
    forcing : float
        Effective gravitational forcing in m/s^2 for use in move_agents().
    """
    # buoyant gravitational acceleration
    g_prime = g * (rho_sed - rho_water) / rho_sed

    # slope angle in radians
    theta   = np.radians(slope_angle_deg)

    # effective forcing with pore pressure reduction
    forcing = g_prime * np.sin(theta) * (1.0 - pore_pressure_ratio)

    return forcing


# =============================================================================
# MAIN — DEMONSTRATION RUN
# =============================================================================

if __name__ == '__main__':

    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend
    import matplotlib.pyplot as plt

    print("=== ProboSed Agent Transport Model ===")
    print("Site C0019J, Japan Trench frontal prism\n")

    # Compute physically motivated forcing from slope parameters
    # Japan Trench frontal prism: ~8 degree slope, moderate pore pressure
    forcing = forcing_from_slope(
        slope_angle_deg    = 8.0,   # degrees — frontal prism slope geometry
        pore_pressure_ratio= 0.35,  # moderate overpressure
    )
    print(f"Gravitational forcing (8 deg slope, lambda=0.35): {forcing:.4f} m/s^2")

    # Initialize model with Japan Trench frontal prism parameters
    model = SedimentAgentModel(
        settling_velocity = 0.02,   # fine silt, D50 ~20 microns
        drag_coeff        = 0.5,    # natural irregular grains
        diffusion         = 0.01,   # low-turbulence deep-water conditions
        dt                = 0.05,   # timestep (s)
    )

    # Run ensemble
    print("Running grain transport ensemble (N=10,000, 500 steps)...")
    results = model.run(
        n_agents = 10_000,
        n_steps  = 500,
        forcing  = forcing,
        seed     = 42,
    )
    model.summary(results)

    # Compute clast distribution
    counts, edges, centers = calculate_clast_distribution(
        results['final_positions'], bins=25
    )

    # ── Figure 1: Final grain position distribution ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # Panel A: clast distribution histogram
    axes[0].set_facecolor('white')
    axes[0].bar(centers, counts, width=(centers[1]-centers[0]),
                color='#2c3e50', alpha=0.75, edgecolor='k', lw=0.5)
    axes[0].set_xlabel('Position above bed (m)', fontsize=12)
    axes[0].set_ylabel('Grain count', fontsize=12)
    axes[0].set_title(
        '(A) Final Grain Position Distribution\n'
        '(Proxy for MTD deposit thickness profile)',
        fontsize=12, fontweight='bold'
    )
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Panel B: deposition rate over time
    axes[1].set_facecolor('white')
    deposition_rate = np.diff(results['n_deposited'])   # grains deposited per step
    axes[1].plot(results['time'][1:], deposition_rate,
                 color='#cc2222', lw=1.5, alpha=0.8)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Grains deposited per timestep', fontsize=12)
    axes[1].set_title(
        '(B) Deposition Rate Over Time\n'
        '(Rate of change in deposited grain count)',
        fontsize=12, fontweight='bold'
    )
    axes[1].grid(alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('transport_summary.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("\nSaved: transport_summary.png")

    # ── Sensitivity to slope angle and pore pressure ───────────────────────
    print("\nSensitivity: fraction deposited vs slope angle and pore pressure")
    print(f"{'Slope (deg)':<14} {'Lambda':<10} {'Forcing':<12} {'Deposited %'}")
    print("-" * 50)

    for angle in [5.0, 8.0, 12.0, 15.0]:
        for lam in [0.0, 0.35, 0.6]:
            f = forcing_from_slope(angle, lam)
            r = model.run(n_agents=2000, n_steps=500, forcing=f, seed=42)
            pct_dep = 100 * (r['final_positions'] == 0).mean()
            print(f"{angle:<14.1f} {lam:<10.2f} {f:<12.4f} {pct_dep:.1f}%")
