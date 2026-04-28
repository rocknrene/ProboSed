"""
toy_model.py
============
Stochastic slope stability model for the Japan Trench frontal prism.

Simulates slope displacement trajectories under two forcing conditions:
  1. Slope-only: background tectonic noise, no fault coupling
  2. Slope + Fault: fault-coupled system with Tohoku-style megathrust impulse

Governing equations (Ornstein-Uhlenbeck processes):
  Fault slip:   ds = -k_f * s * dt + sigma_s * dW_s
  Slope:        dq = (-gamma * q + alpha * s) * dt + sigma_q * dW_q
  Mainshock:    s(t*) -> s(t*) + slip_mag  at t = mainshock_t

Sediment transport:
  Accumulated transport is the time-integrated excess displacement above
  the failure threshold: integral of max(q - threshold, 0) dt.
  This correctly captures both the frequency and duration of threshold
  exceedances, reflecting the physical dependence of MTD volume on how
  long and how far the slope exceeds the failure criterion.

Lyapunov exponent:
  Estimated from ensemble divergence relative to the ensemble mean
  trajectory. A warmup period is excluded to avoid inflation by the
  numerical transient arising from identical initial conditions (q=0).
  A positive exponent indicates sensitivity to initial conditions.

Physical parameter context (IODP Expeditions 343, 405, 386):
  - Frontal prism Vp = 1550-1750 m/s, resistivity = 0.5-1.8 Ohm-m (Exp 405)
  - Frontal prism Pc' = 17 MPa, significant overconsolidation (Exp 343)
  - Coseismic slip ~50 m at C0019 decollement, 2011 Tohoku-oki (Fulton et al. 2013)
  - Seismic strengthening of background hemipelagic sediment (Exp 386)

Usage:
  python toy_model.py
  Outputs three figures to the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PARAMETERS
# All physical parameters are defined here for transparency and reproducibility.
# Parameter choices are informed by published IODP proceedings (see module docstring).
# =============================================================================

# --- Time discretization ---
DT    = 0.01    # timestep (dimensionless simulation units)
T     = 10.0    # total simulation time (dimensionless units)
STEPS = int(T / DT)   # total number of timesteps

# --- Ensemble size ---
NPATHS = 200    # number of independent trajectory realizations
                # larger ensembles yield more stable failure probability estimates

# --- Reproducibility ---
np.random.seed(42)   # global seed for numpy legacy random calls in this script

# --- Slope dynamics ---
# gamma: slope damping coefficient (restoring force toward equilibrium)
# Reflects gravitational and sediment strength restoring forces on the slope.
# Value of 1.0 represents moderate damping; higher values (1.5-2.0) apply
# to seismically strengthened background hemipelagic sediment (Exp 386).
GAMMA     = 1.0

# sigma_q: slope noise amplitude (background tectonic variability)
# Informed by weak, poorly consolidated frontal prism material at C0019:
# Vp = 1550-1750 m/s, resistivity = 0.5-1.8 Ohm-m (Exp 405 LWD, Unit I).
SIGMA_Q   = 0.6

# THRESHOLD: failure criterion — slope fails when q(t) > THRESHOLD
# In the full model, threshold is set per-depth via threshold_from_vcd().
# Here a single value is used for the illustrative toy model.
# THRESHOLD = 1.0 corresponds to VCD score 1.5 (between scaly fabric and coherent block).
THRESHOLD = 1.0

# --- Fault slip dynamics ---
# sigma_s: fault slip noise amplitude
SIGMA_S     = 0.5

# k_f: fault slip damping coefficient (rate of post-seismic relaxation)
K_F         = 1.0

# slip_mag: mainshock slip magnitude in dimensionless scaled units
# Physical anchor: ~50 m coseismic slip documented at the C0019 decollement
# during the 2011 Mw9 Tohoku-oki earthquake (Fulton et al. 2013;
# Chester et al. 2013, Science 342:1208-1211).
# Dimensionless scaling: SLIP_MAG / 50 m = 0.06 per meter of physical slip.
SLIP_MAG    = 3.0
SLIP_PHYSICAL_M = 50.0   # meters; physical reference value — not used in simulation

# mainshock_t: time at which the mainshock impulse is applied
MAINSHOCK_T = 5.0

# --- Fault-to-slope coupling ---
# alpha: coupling coefficient controlling how strongly fault slip drives slope displacement
# Reflects coseismic stress transfer efficiency from decollement to frontal prism slope.
# Frontal prism overconsolidation (Pc' = 17 MPa; Valdez et al. 2015) provides
# physical motivation for non-zero coupling: the sediment stress history
# records prior seismic loading, implying the fault-slope system is coupled.
ALPHA = 0.5


# =============================================================================
# SIMULATION 1 — SLOPE ONLY
# Pure Ornstein-Uhlenbeck process. No fault coupling.
# Represents background tectonic noise without a megathrust event.
# =============================================================================

def run_slope_only(npaths, steps, dt, gamma, sigma_q, threshold):
    """
    Runs an ensemble of slope-only stochastic trajectories.

    Governing equation:
      dq = -gamma * q * dt + sigma_q * dW_q

    Parameters
    ----------
    npaths : int
        Number of ensemble trajectories.
    steps : int
        Number of timesteps per trajectory.
    dt : float
        Timestep.
    gamma : float
        Slope damping coefficient.
    sigma_q : float
        Slope noise amplitude.
    threshold : float
        Failure threshold.

    Returns
    -------
    trajectories : list of lists
        Full displacement time series for each path.
    p_fail : float
        Fraction of paths that exceeded the failure threshold at any point.
    """
    trajectories = []   # stores full time series for each path
    failures     = 0    # counts paths that exceeded the threshold

    for _ in range(npaths):
        q        = 0.0    # initial slope displacement
        path     = [q]    # store full trajectory
        exceeded = False  # flag: has this path crossed the threshold?

        for _ in range(steps):
            # Euler-Maruyama integration of the slope OU process
            dq = -gamma * q * dt + sigma_q * np.sqrt(dt) * np.random.randn()
            q += dq
            path.append(q)

            # Record first threshold crossing (boolean flag avoids double-counting)
            if q > threshold and not exceeded:
                exceeded = True

        trajectories.append(path)
        if exceeded:
            failures += 1

    p_fail = failures / npaths   # fraction of paths that failed
    return trajectories, p_fail


# =============================================================================
# SIMULATION 2 — SLOPE + FAULT COUPLING
# Coupled OU processes with a Tohoku-style mainshock impulse.
# =============================================================================

def run_coupled(npaths, steps, dt, gamma, sigma_q, sigma_s, k_f,
                alpha, threshold, slip_mag, mainshock_step):
    """
    Runs an ensemble of fault-coupled slope stability trajectories.

    Governing equations:
      ds = -k_f * s * dt + sigma_s * dW_s
      dq = (-gamma * q + alpha * s) * dt + sigma_q * dW_q
      s(mainshock_step) += slip_mag

    Parameters
    ----------
    npaths : int
        Number of ensemble trajectories.
    steps : int
        Number of timesteps per trajectory.
    dt : float
        Timestep.
    gamma : float
        Slope damping coefficient.
    sigma_q : float
        Slope noise amplitude.
    sigma_s : float
        Fault slip noise amplitude.
    k_f : float
        Fault slip damping coefficient.
    alpha : float
        Fault-to-slope coupling coefficient.
    threshold : float
        Failure threshold.
    slip_mag : float
        Mainshock slip magnitude (dimensionless scaled units).
    mainshock_step : int
        Timestep index at which the mainshock impulse is applied.

    Returns
    -------
    trajectories : list of lists
        Full slope displacement time series for each path.
    fault_paths : list of lists
        Full fault slip time series for each path.
    p_fail : float
        Fraction of paths where slope exceeded the failure threshold.
    transported : list of floats
        Time-integrated transported sediment for each path.
        Computed as sum of max(q - threshold, 0) * dt.
    """
    trajectories = []   # slope displacement time series per path
    fault_paths  = []   # fault slip time series per path
    transported  = []   # time-integrated excess displacement per path
    failures     = 0    # count of paths that exceeded the threshold

    for _ in range(npaths):
        q, s     = 0.0, 0.0   # initial conditions: slope and fault at rest
        path_q   = [q]         # slope trajectory
        path_s   = [s]         # fault trajectory
        exceeded = False       # first-passage flag
        mass     = 0.0         # accumulated transported sediment

        for t in range(steps):

            # Apply mainshock impulse to fault state at the designated timestep
            if t == mainshock_step:
                s += slip_mag

            # Euler-Maruyama step for fault slip OU process
            s += -k_f * s * dt + sigma_s * np.sqrt(dt) * np.random.randn()

            # Euler-Maruyama step for slope OU process with fault coupling
            # The alpha * s term is the fault-slope coupling: fault slip drives
            # slope displacement proportionally to the coupling coefficient
            q += (-gamma * q + alpha * s) * dt + sigma_q * np.sqrt(dt) * np.random.randn()

            path_q.append(q)
            path_s.append(s)

            # Record first threshold crossing
            if q > threshold and not exceeded:
                exceeded = True

            # Accumulate time-integrated sediment transport
            # Each timestep where q > threshold contributes proportionally
            # to both exceedance magnitude and duration
            if q > threshold:
                mass += (q - threshold) * dt

        trajectories.append(path_q)
        fault_paths.append(path_s)

        # Note: the transported value is stored in dimensionless model units.
        # No arbitrary scale factor is applied here; physical scaling should
        # be applied at the interpretation stage with appropriate unit conversion.
        transported.append(mass)

        if exceeded:
            failures += 1

    p_fail = failures / npaths
    return trajectories, fault_paths, p_fail, transported


# =============================================================================
# LYAPUNOV EXPONENT ESTIMATE
# =============================================================================

def estimate_lyapunov(trajectories, dt, warmup_fraction=0.01):
    """
    Estimates the maximal Lyapunov exponent from an ensemble of trajectories.

    Method:
      1. Compute the ensemble mean trajectory as the reference.
      2. Compute deviation of each path from the reference at every timestep.
      3. Track the mean log-growth rate of deviations over time.
      4. Normalize by dt to obtain units of 1/time.

    A warmup period is excluded from the Lyapunov mean to avoid inflation
    by the numerical transient at t=0: all trajectories start at q=0,
    so initial deviations are at the numerical noise floor (1e-10) and
    the first few steps show spurious explosive log-growth that does not
    reflect true dynamical sensitivity.

    Parameters
    ----------
    trajectories : list of lists
        Ensemble of displacement time series (all same length).
    dt : float
        Simulation timestep.
    warmup_fraction : float
        Fraction of timesteps to exclude as initial transient.
        Default 0.01 excludes the first 1% of steps.

    Returns
    -------
    lyapunov : float
        Estimated maximal Lyapunov exponent in units of 1/time.
        Positive -> sensitivity to initial conditions.
        Near zero -> marginal stability.
        Negative -> convergent, stable behavior.
    log_growth : np.ndarray
        Mean log-growth rate at each timestep (useful for plotting).
    """
    arr       = np.array(trajectories)           # shape: (N_paths, T_steps)
    reference = arr.mean(axis=0)                 # ensemble mean at each timestep
    delta     = arr - reference                  # deviation from ensemble mean
    delta_mag = np.abs(delta) + 1e-10            # epsilon floor prevents log(0)
    log_delta = np.log(delta_mag)                # log deviation magnitude

    # Mean log-growth rate per timestep, averaged across all paths
    log_growth = np.diff(log_delta, axis=1).mean(axis=0)   # shape: (T_steps-1,)

    # Exclude initial transient before computing the mean Lyapunov exponent
    warmup   = max(1, int(warmup_fraction * len(log_growth)))
    lyapunov = log_growth[warmup:].mean() / dt

    return lyapunov, log_growth


# =============================================================================
# MAIN — RUN SIMULATIONS AND PRODUCE FIGURES
# =============================================================================

if __name__ == '__main__':

    # Compute mainshock timestep index from physical time
    mainshock_step = int(MAINSHOCK_T / DT)
    time           = np.linspace(0, T, STEPS + 1)   # time axis for plotting

    print("Running slope-only ensemble...")
    traj_solo, p_fail_solo = run_slope_only(
        NPATHS, STEPS, DT, GAMMA, SIGMA_Q, THRESHOLD
    )

    print("Running fault-coupled ensemble...")
    traj_coupled, fault_paths, p_fail_coupled, transported = run_coupled(
        NPATHS, STEPS, DT, GAMMA, SIGMA_Q, SIGMA_S,
        K_F, ALPHA, THRESHOLD, SLIP_MAG, mainshock_step
    )

    print("Estimating Lyapunov exponent...")
    lyapunov, log_growth = estimate_lyapunov(traj_coupled, DT)

    # Print summary to console
    print(f"\n--- Simulation Results ---")
    print(f"Slope-only failure probability:    {p_fail_solo:.2f}")
    print(f"Fault-coupled failure probability: {p_fail_coupled:.2f}")
    print(f"Mean transported sediment:         {np.mean(transported):.4f} (model units)")
    print(f"Lyapunov exponent (post-warmup):   {lyapunov:.4f} per unit time")
    if lyapunov > 0:
        print("  -> Positive exponent: system exhibits sensitivity to initial conditions")
    else:
        print("  -> Non-positive exponent: system is stable / convergent")

    # =========================================================================
    # FIGURE 1 — Three-panel model summary
    # Panel A: slope-only trajectories
    # Panel B: fault-coupled trajectories
    # Panel C: transported sediment distribution
    # =========================================================================

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.patch.set_facecolor('white')                      # white figure background
    for ax in axes:
        ax.set_facecolor('white')                         # white axes background

    # Color map for trajectory visualization — 20 paths shown for clarity
    colors = plt.cm.viridis(np.linspace(0, 1, 20))

    # --- Panel A: Slope-only trajectories ---
    for path, c in zip(traj_solo[:20], colors):
        axes[0].plot(time, path, color=c, alpha=0.75, lw=1.2)   # individual trajectories

    axes[0].axhline(THRESHOLD, color='red', ls='--', lw=2,
                    label='Failure threshold')                    # failure criterion line
    axes[0].axvline(MAINSHOCK_T, color='orange', ls=':', lw=2,
                    label='Mainshock time')                       # reference time marker
    axes[0].set_xlabel('Time (simulation units)', fontsize=13)
    axes[0].set_ylabel('Slope displacement q(t)', fontsize=13)
    axes[0].set_title(
        f'(A) Slope-Only Trajectories\nP(failure) = {p_fail_solo:.2f}',
        fontsize=14, fontweight='bold'
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # --- Panel B: Fault-coupled trajectories ---
    for path, c in zip(traj_coupled[:20], colors):
        axes[1].plot(time, path, color=c, alpha=0.75, lw=1.2)   # individual trajectories

    axes[1].axhline(THRESHOLD, color='red', ls='--', lw=2,
                    label='Failure threshold')                    # failure criterion line
    axes[1].axvline(MAINSHOCK_T, color='orange', ls=':', lw=2,
                    label='Mainshock impulse')                    # mainshock time marker
    axes[1].set_xlabel('Time (simulation units)', fontsize=13)
    axes[1].set_ylabel('Slope displacement q(t)', fontsize=13)
    axes[1].set_title(
        f'(B) Fault-Coupled Trajectories\nP(failure) = {p_fail_coupled:.2f}',
        fontsize=14, fontweight='bold'
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    # --- Panel C: Transported sediment distribution ---
    transported_arr = np.array(transported)   # convert list to array for statistics

    axes[2].hist(transported_arr, bins=20, edgecolor='k',
                 color='navy', alpha=0.7)                         # frequency distribution
    axes[2].axvline(np.mean(transported_arr), color='red', ls='--', lw=2,
                    label=f'Mean = {np.mean(transported_arr):.3f}')   # mean marker
    axes[2].set_xlabel('Transported sediment (dimensionless model units)', fontsize=13)
    axes[2].set_ylabel('Count', fontsize=13)
    axes[2].set_title(
        '(C) Transported Sediment Distribution\n(Time-integrated threshold exceedance)',
        fontsize=14, fontweight='bold'
    )
    axes[2].legend(fontsize=11)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('toy_model_summary.png', dpi=180, bbox_inches='tight',
                facecolor='white')   # enforce white background on save
    plt.close()
    print("\nSaved: toy_model_summary.png")

    # =========================================================================
    # FIGURE 2 — Failure probability comparison bar chart
    # Illustrates the effect of fault coupling on slope failure probability
    # at Site C0019J, Japan Trench
    # =========================================================================

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('white')

    # Bar chart comparing slope-only vs fault-coupled failure probabilities
    bars = ax2.bar(
        ['Slope Only\n(no fault coupling)',
         'Slope + Fault\n(Tohoku impulse)'],
        [p_fail_solo, p_fail_coupled],
        color    = ['steelblue', 'tomato'],
        edgecolor= 'k',
        alpha    = 0.85,
        width    = 0.5
    )

    # Add value labels above each bar
    for bar, val in zip(bars, [p_fail_solo, p_fail_coupled]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f'{val:.2f}',
            ha='center', fontsize=14, fontweight='bold'
        )

    ax2.set_ylabel('Failure probability', fontsize=13)
    ax2.set_ylim(0, 1.15)
    ax2.set_title(
        'Effect of Fault Slip Coupling on Slope Failure\nSite C0019J, Japan Trench',
        fontsize=13, fontweight='bold'
    )
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('toy_model_failure_probability.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Saved: toy_model_failure_probability.png")

    # =========================================================================
    # FIGURE 3 — Lyapunov exponent over time
    # Shows the mean log-growth rate of trajectory perturbations.
    # The warmup period (excluded from the mean) is shaded to indicate
    # the numerical transient region.
    # =========================================================================

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor('white')
    ax3.set_facecolor('white')

    # Time axis for Lyapunov plot (one element shorter than trajectory due to diff)
    time_lyap  = np.linspace(0, T, len(log_growth))
    warmup_end = int(0.01 * len(log_growth))   # warmup cutoff index (1% of steps)

    # Shade the warmup region to indicate numerical transient exclusion zone
    ax3.axvspan(0, time_lyap[warmup_end], color='lightgray', alpha=0.5,
                label='Warmup (excluded from LE mean)')

    # Plot the raw log-growth rate time series
    ax3.plot(time_lyap, log_growth, color='#2c3e50', lw=1.5, alpha=0.8,
             label='Log-growth rate of perturbations')

    # Reference lines
    ax3.axhline(0, color='red', ls='--', lw=1.5, alpha=0.6,
                label='Zero (stability boundary)')
    ax3.axhline(lyapunov, color='navy', ls='-', lw=2,
                label=f'LE mean (post-warmup) = {lyapunov:.4f} / time unit')
    ax3.axvline(MAINSHOCK_T, color='orange', ls=':', lw=2,
                label='Mainshock')

    ax3.set_xlabel('Time (simulation units)', fontsize=13)
    ax3.set_ylabel('Mean log-growth rate of perturbations', fontsize=13)
    ax3.set_title(
        'Lyapunov Exponent Estimate\nFault-Coupled Ensemble, Site C0019J, Japan Trench',
        fontsize=13, fontweight='bold'
    )
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('toy_model_lyapunov.png', dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("Saved: toy_model_lyapunov.png")
