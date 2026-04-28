"""
stability.py
============
Slope stability physics for the Japan Trench frontal prism.

Provides:
  - STABILITY_MAP      : canonical disturbance label -> score lookup (single source of truth)
  - run_ensemble()     : vectorized Ornstein-Uhlenbeck ensemble simulation
  - calculate_lyapunov(): physically correct Lyapunov exponent estimate
  - failure_statistics(): summary statistics from an ensemble run
  - threshold_from_vcd(): maps VCD stability scores to failure thresholds
  - run_sensitivity()  : one-at-a-time parameter sensitivity analysis

Physical parameter context (IODP Expeditions 343, 405, 386):
  - Frontal prism Vp = 1550-1750 m/s, resistivity = 0.5-1.8 Ohm-m (Exp 405 LWD, Unit I)
  - Frontal prism porosity = 27-46%, Pc' = 17 MPa (JFAST Exp 343, Valdez et al. 2015)
  - Coseismic slip ~50 m at C0019 decollement, 2011 Tohoku-oki (Fulton et al. 2013)
  - Seismic strengthening of background hemipelagic sediment (Exp 386)

All functions use pure NumPy and require no JAX installation.
Vectorized ensemble simulation (N=10,000) runs in under one minute on CPU.
"""

import numpy as np


# =============================================================================
# CANONICAL STABILITY MAP
# Single source of truth for disturbance label -> stability score conversion.
# Used by both the interactive VCDLabeler and the automated JCORESMiner pathways.
# Scores follow IODP standardized disturbance classification:
#   3 = undisturbed primary fabric
#   2 = minor to moderate disturbance, fabric largely intact
#   1 = significant disturbance, fabric partially destroyed
#   0 = complete remobilization, primary fabric absent
# The two slurried variants handle spacing differences in label entry
# ('slurried / mtd' with spaces vs 'slurried/mtd' without spaces).
# =============================================================================

STABILITY_MAP = {
    'intact bedding':  3,   # undisturbed primary sedimentary fabric
    'coherent block':  2,   # deformed but coherent, fabric largely preserved
    'scaly fabric':    1,   # pervasive shearing, fabric partially destroyed
    'slurried / mtd':  0,   # complete remobilization (with spaces variant)
    'slurried/mtd':    0,   # complete remobilization (no spaces variant)
    'slurried':        0,   # fully remobilized, no primary fabric
    'chaotic':         0,   # chaotic fabric, mass transport deposit
    'sheared':         1,   # significant shearing, reduced strength
    'deformed':        1,   # deformed fabric, intermediate disturbance
    'biscuit':         1,   # drilling biscuit, partial disturbance
    'fall-in':         1,   # fall-in material, disturbed
    'brecciated':      1,   # brecciated fabric, significant disruption
    'void':            0,   # void, no material present
}


# =============================================================================
# PHYSICALLY CONSTRAINED DEFAULT PARAMETERS
# Informed by IODP Expeditions 343, 405, and 386 published proceedings.
# These are the base values used in run_ensemble() if none are supplied.
# =============================================================================

# Mainshock physical anchor:
# ~50 m of coseismic slip documented at the C0019 decollement
# during the 2011 Mw9 Tohoku-oki earthquake (Fulton et al. 2013;
# Chester et al. 2013, Science 342:1208-1211).
# SLIP_MAG = 3.0 is the dimensionless scaled equivalent used in simulation.
# Scaling factor: SLIP_MAG / SLIP_PHYSICAL_M = 3.0 / 50.0 = 0.06 per meter.
SLIP_PHYSICAL_M = 50.0   # meters, physical reference — do not use in simulation

# Slope noise physical context:
# sigma_q = 0.6 reflects weak, poorly consolidated frontal prism material.
# Supported by Vp = 1550-1750 m/s and resistivity = 0.5-1.8 Ohm-m
# in Logging Unit I at Site C0019 (Expedition 405 Scientists, 2025).

# Gamma (damping) physical context:
# gamma = 1.0 represents moderate restoring force.
# Higher values (1.5-2.0) are appropriate for seismically strengthened
# background hemipelagic sediment (Expedition 386; seismic strengthening
# effect documented in Japan Trench trench-fill basins).


# =============================================================================
# ENSEMBLE SIMULATION
# =============================================================================

def run_ensemble(
    N_paths     = 10_000,   # number of independent trajectories
    T           = 10.0,     # total simulation time (dimensionless units)
    dt          = 0.01,     # timestep
    gamma       = 1.0,      # slope damping coefficient (restoring force)
    sigma_q     = 0.6,      # slope noise amplitude; reflects weak prism material
    sigma_s     = 0.5,      # fault slip noise amplitude
    k_f         = 1.0,      # fault slip damping coefficient
    alpha       = 0.5,      # fault-to-slope coupling coefficient
    threshold   = 1.0,      # failure threshold for slope displacement
    slip_mag    = 3.0,      # mainshock slip magnitude (dimensionless scaled units)
    mainshock_t = 5.0,      # time at which mainshock impulse is applied
    seed        = 42,       # random seed for reproducibility
):
    """
    Runs a vectorized ensemble of fault-coupled slope stability trajectories.

    Governing equations (Ornstein-Uhlenbeck processes):
      Fault slip:    ds = -k_f * s * dt + sigma_s * dW_s
      Slope:         dq = (-gamma * q + alpha * s) * dt + sigma_q * dW_q
      Mainshock:     s(t*) -> s(t*) + slip_mag  at t = mainshock_t

    Failure is defined as the first passage of q(t) above the threshold:
      tau = inf{ t > 0 : q(t) > threshold }

    All N_paths are simulated simultaneously using matrix operations,
    avoiding Python loops over paths for computational efficiency.

    Parameters
    ----------
    N_paths : int
        Number of independent trajectories in the ensemble.
    T : float
        Total simulation time (dimensionless units).
    dt : float
        Timestep.
    gamma : float
        Slope damping coefficient. Higher values reflect stronger
        restoring force (e.g., seismically strengthened sediment).
    sigma_q : float
        Slope noise amplitude. Reflects background tectonic variability;
        informed by low Vp (1550-1750 m/s) frontal prism material (Exp 405).
    sigma_s : float
        Fault slip noise amplitude.
    k_f : float
        Fault slip damping coefficient.
    alpha : float
        Fault-to-slope coupling coefficient. Controls how strongly
        fault slip drives slope displacement.
    threshold : float
        Failure threshold. Set via threshold_from_vcd() from VCD scores.
    slip_mag : float
        Mainshock slip magnitude in dimensionless scaled units.
        Physical anchor: ~50 m coseismic slip at C0019 (Fulton et al. 2013).
    mainshock_t : float
        Time at which the mainshock impulse is applied.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    q_ensemble : np.ndarray, shape (T_steps+1, N_paths)
        Slope displacement at every timestep for every path.
    s_ensemble : np.ndarray, shape (T_steps+1, N_paths)
        Fault slip at every timestep for every path.
    p_fail : float
        Fraction of paths where slope exceeded threshold at any point.
    transported : np.ndarray, shape (N_paths,)
        Time-integrated transported sediment for each path.
        Computed as sum of max(q - threshold, 0) * dt.
        Captures both exceedance duration and magnitude.
    """
    rng            = np.random.default_rng(seed)   # reproducible random number generator
    steps          = int(T / dt)                    # total number of timesteps
    mainshock_step = int(mainshock_t / dt)          # timestep index of mainshock

    # Pre-generate all noise at once — faster than per-step generation
    noise_q = rng.standard_normal((steps, N_paths))   # slope noise array
    noise_s = rng.standard_normal((steps, N_paths))   # fault noise array

    # Initialize storage arrays for full trajectories
    q_ensemble = np.zeros((steps + 1, N_paths))   # slope displacement
    s_ensemble = np.zeros((steps + 1, N_paths))   # fault slip

    # Forward integration — time loop only, paths are vectorized
    for t in range(steps):

        q = q_ensemble[t]   # current slope displacement, all paths
        s = s_ensemble[t]   # current fault slip, all paths

        # Apply mainshock impulse to all paths simultaneously at designated step
        if t == mainshock_step:
            s = s + slip_mag   # additive impulse; s is a local copy here

        # Fault slip OU step — vectorized across all N_paths
        s_new = s + (-k_f * s * dt) + sigma_s * np.sqrt(dt) * noise_s[t]

        # Slope OU step with fault coupling term — vectorized across all N_paths
        q_new = q + (-gamma * q + alpha * s) * dt + sigma_q * np.sqrt(dt) * noise_q[t]

        # Store updated states
        q_ensemble[t + 1] = q_new
        s_ensemble[t + 1] = s_new

    # Compute failure statistics across full ensemble
    exceeded     = (q_ensemble > threshold)    # boolean array, True where failure occurs
    any_exceeded = exceeded.any(axis=0)        # True for each path that ever failed
    p_fail       = any_exceeded.mean()         # fraction of paths that failed

    # Time-integrated transport: captures both crossing frequency and exceedance magnitude
    # Physical interpretation: proxy for total remobilized sediment volume
    exceedance  = np.maximum(q_ensemble - threshold, 0.0)   # only positive exceedances
    transported = exceedance.sum(axis=0) * dt               # integrate over time

    return q_ensemble, s_ensemble, p_fail, transported


# =============================================================================
# LYAPUNOV EXPONENT
# =============================================================================

def calculate_lyapunov(q_ensemble, dt, warmup_fraction=0.01):
    """
    Estimates the maximal Lyapunov exponent from an ensemble of trajectories.

    The Lyapunov exponent measures the mean exponential rate at which nearby
    trajectories diverge. A positive value indicates sensitivity to initial
    conditions; near-zero indicates marginal stability; negative indicates
    convergent, stable behavior.

    Method:
      1. Compute the ensemble mean trajectory as the reference.
      2. Measure each path's deviation from the reference at every timestep.
      3. Track the mean log-growth rate of deviations across time and paths.
      4. Normalize by dt to obtain units of 1/time.

    Note on initial transient:
      All trajectories begin at q=0, so initial deviations are at the
      numerical noise floor (epsilon = 1e-10). The first few timesteps
      show spurious explosive log-growth as real stochastic differences
      develop from this near-zero baseline. A warmup period is excluded
      from the Lyapunov mean to avoid this artifact inflating the estimate.

    Parameters
    ----------
    q_ensemble : np.ndarray, shape (T_steps, N_paths)
        Slope displacement ensemble from run_ensemble().
    dt : float
        Simulation timestep.
    warmup_fraction : float
        Fraction of total timesteps to exclude as initial transient.
        Default 0.01 excludes the first 1% of steps (100 steps for T=10, dt=0.01).

    Returns
    -------
    lyapunov : float
        Estimated maximal Lyapunov exponent in units of 1/time.
        Positive -> sensitive / chaotic. Near zero -> marginal. Negative -> stable.
    log_growth : np.ndarray, shape (T_steps-1,)
        Mean log-growth rate at each timestep. Useful for plotting
        whether sensitivity increases before or after the mainshock.
    """
    # Ensemble mean trajectory at each timestep — the reference trajectory
    reference = q_ensemble.mean(axis=1, keepdims=True)   # shape: (T_steps, 1)

    # Deviation of each path from the ensemble mean
    delta = q_ensemble - reference                        # shape: (T_steps, N_paths)

    # Log magnitude of deviation — epsilon floor prevents log(0)
    log_delta = np.log(np.abs(delta) + 1e-10)

    # Mean log-growth rate per timestep, averaged over all paths
    log_growth = np.diff(log_delta, axis=0).mean(axis=1)  # shape: (T_steps-1,)

    # Exclude initial transient before computing the Lyapunov mean
    # The warmup spike is a numerical artifact of identical starting conditions,
    # not a reflection of true dynamical sensitivity
    warmup    = max(1, int(warmup_fraction * len(log_growth)))
    lyapunov  = log_growth[warmup:].mean() / dt

    return lyapunov, log_growth


# =============================================================================
# FAILURE STATISTICS
# =============================================================================

def failure_statistics(q_ensemble, transported, threshold, dt):
    """
    Computes summary statistics from a completed ensemble run.

    Parameters
    ----------
    q_ensemble : np.ndarray, shape (T_steps, N_paths)
        Slope displacement ensemble from run_ensemble().
    transported : np.ndarray, shape (N_paths,)
        Time-integrated transported sediment from run_ensemble().
    threshold : float
        Failure threshold used in the simulation.
    dt : float
        Simulation timestep.

    Returns
    -------
    dict with keys:
        p_fail              : float   fraction of paths that failed
        mean_transport      : float   mean transported sediment across paths
        std_transport       : float   standard deviation of transported sediment
        median_transport    : float   median transported sediment
        max_q               : float   maximum slope displacement across ensemble
        n_paths             : int     number of paths in ensemble
        power_law_exponent  : float   exponent of power-law fit to transport distribution
        power_law_r2        : float   R-squared of power-law fit
    """
    # Failure fraction — paths that exceeded threshold at any timestep
    exceeded = (q_ensemble > threshold).any(axis=0)
    p_fail   = exceeded.mean()

    # Power-law fit to transported sediment distribution
    # Natural landslide volume distributions commonly follow power laws
    # (e.g., Malamud et al. 2004); fitting here tests whether model output
    # is consistent with observed submarine landslide scaling relationships
    nonzero = transported[transported > 0]   # exclude zero-transport paths

    if len(nonzero) > 10:
        log_m             = np.log10(nonzero)
        counts, edges     = np.histogram(log_m, bins=20)
        midpoints         = (edges[:-1] + edges[1:]) / 2
        valid             = counts > 0   # exclude empty bins from fit

        if valid.sum() > 3:
            # Fit log-log linear model: log10(count) ~ a * log10(mass) + b
            coeffs    = np.polyfit(midpoints[valid], np.log10(counts[valid]), 1)
            residuals = np.log10(counts[valid]) - np.polyval(coeffs, midpoints[valid])
            ss_res    = (residuals ** 2).sum()
            ss_tot    = ((np.log10(counts[valid]) - np.log10(counts[valid]).mean()) ** 2).sum()
            r_sq      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            power_law_fit = (coeffs[0], r_sq)
        else:
            power_law_fit = (np.nan, np.nan)   # insufficient bins for reliable fit
    else:
        power_law_fit = (np.nan, np.nan)       # insufficient nonzero paths

    return {
        'p_fail'            : float(p_fail),
        'mean_transport'    : float(transported.mean()),
        'std_transport'     : float(transported.std()),
        'median_transport'  : float(np.median(transported)),
        'max_q'             : float(q_ensemble.max()),
        'n_paths'           : q_ensemble.shape[1],
        'power_law_exponent': power_law_fit[0],
        'power_law_r2'      : power_law_fit[1],
    }


# =============================================================================
# VCD-TO-THRESHOLD MAPPING
# =============================================================================

def threshold_from_vcd(stability_score):
    """
    Maps a VCD stability score (0-3) to a failure threshold for the slope model.

    The mapping is physically motivated by the mechanical contrast between
    fabric classes observed in IODP core material:
      Score 3 (intact bedding)  -> high threshold (theta = 2.00): stable slope
      Score 2 (coherent block)  -> moderate threshold (theta = 1.33)
      Score 1 (scaly fabric)    -> low threshold (theta = 0.67): near-failure
      Score 0 (slurried / MTD)  -> threshold = 0: failure already occurred

    The linear mapping (theta = score * 2/3) is the maximum-entropy choice
    given only the constraint that threshold increases monotonically with
    stability score. A nonlinear mapping would require additional mechanical
    justification not currently available from the core data.

    The threshold range [0, 2] was chosen so that score 3 corresponds to
    a slope requiring approximately 3.3 sigma_q displacement to fail,
    consistent with rare but physically plausible background failure rates.

    Parameters
    ----------
    stability_score : float or np.ndarray
        VCD stability score(s) in range [0, 3].

    Returns
    -------
    threshold : float or np.ndarray
        Failure threshold(s) for use in run_ensemble().
    """
    # Linear scaling: score 3 -> 2.0, score 0 -> 0.0
    # Clamp to [0, 2] to prevent negative thresholds from out-of-range scores
    threshold = np.clip(
        np.asarray(stability_score, dtype=float) * (2.0 / 3.0),
        0.0,
        2.0
    )
    return threshold


# =============================================================================
# LABELS-TO-MODEL-INPUTS PIPELINE
# =============================================================================

def labels_to_stability_inputs(labels_csv_path, backbone_df):
    """
    Converts VCDLabeler output labels to boundary conditions for the slope model.

    Connects the labeling pipeline to the physics model by reading the CSV
    produced by VCDLabeler, mapping disturbance labels to stability scores
    using the canonical STABILITY_MAP, and computing per-depth failure
    thresholds via threshold_from_vcd().

    Parameters
    ----------
    labels_csv_path : str
        Path to CSV output from VCDLabeler.
        Expected columns: filename, lithology, disturbance, author,
                          confidence, notes.
    backbone_df : pd.DataFrame
        Depth backbone with columns: Core, Section, Depth_m, Type.
        Produced by JCORESMiner.build_backbone().

    Returns
    -------
    dict with keys:
        depth_array      : np.ndarray   CSF-A depths for labeled intervals
        stability_array  : np.ndarray   0-3 stability scores per depth
        threshold_array  : np.ndarray   per-depth failure thresholds for run_ensemble()
        mtd_boundaries   : list of tuples  (top_m, bot_m) for each MTD interval
                           defined as contiguous intervals where stability < 2
    """
    import pandas as pd

    # Load label CSV and normalize column names
    df         = pd.read_csv(labels_csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map disturbance labels to stability scores using the canonical map
    # Unrecognized labels return NaN and are dropped during merge
    df['stability_score'] = (
        df['disturbance'].str.strip().str.lower().map(STABILITY_MAP)
    )

    # Extract numeric core identifier from filename for depth matching
    # Note: verify this regex captures the correct number for the filename convention in use
    # e.g., '405-C0019J-14K_section2.png' should yield 14, not 405
    df['core_num'] = df['filename'].str.extract(r'C\d{4}[A-Z]-(\d+)').astype(float)

    # Key the backbone on the same numeric core identifier
    backbone_keyed          = backbone_df.copy()
    backbone_keyed['core_num'] = (
        backbone_keyed['Core'].str.extract(r'(\d+)').astype(float)
    )

    # Merge labels with backbone depths; drop rows with missing depth or score
    merged = df.merge(
        backbone_keyed[['core_num', 'Depth_m']].drop_duplicates('core_num'),
        on='core_num', how='left'
    ).dropna(subset=['Depth_m', 'stability_score'])

    merged = merged.sort_values('Depth_m').reset_index(drop=True)

    depth_array     = merged['Depth_m'].values
    stability_array = merged['stability_score'].values
    threshold_array = threshold_from_vcd(stability_array)

    # Identify MTD intervals: contiguous depth ranges where stability < 2
    # Transition from stable (score >= 2) to disturbed (score < 2) marks MTD top;
    # transition back to stable marks MTD base
    mtd_boundaries = []
    in_mtd         = False
    mtd_top        = None

    for d, s in zip(depth_array, stability_array):
        if s < 2 and not in_mtd:
            in_mtd  = True
            mtd_top = d
        elif s >= 2 and in_mtd:
            in_mtd  = False
            mtd_boundaries.append((mtd_top, d))

    # Close any MTD that extends to the bottom of the record
    if in_mtd and mtd_top is not None:
        mtd_boundaries.append((mtd_top, depth_array[-1]))

    return {
        'depth_array'    : depth_array,
        'stability_array': stability_array,
        'threshold_array': threshold_array,
        'mtd_boundaries' : mtd_boundaries,
    }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity(param_grid, base_params, N_paths=5000):
    """
    Runs a one-at-a-time parameter sensitivity analysis.

    Each parameter in param_grid is varied across its specified values
    while all other parameters are held at their base values. This tests
    whether the qualitative result (fault coupling increases failure
    probability) is robust across physically plausible parameter ranges.

    Physical parameter ranges are informed by IODP Expeditions 343, 405, 386:
      alpha     : 0.2-0.9   (uncertainty in coseismic stress transfer efficiency)
      sigma_q   : 0.3-1.0   (range from seismically strengthened to weak remobilized)
      slip_mag  : 1.0-5.0   (scaled from ~50 m physical coseismic slip)
      gamma     : 0.5-2.0   (range from weak to seismically strengthened sediment)
      threshold : 0.33-2.0  (spans VCD scores 0 through 3 via threshold_from_vcd)

    Parameters
    ----------
    param_grid : dict
        Keys are parameter names matching run_ensemble() kwargs.
        Values are lists of values to test for that parameter.
    base_params : dict
        Base parameter values for all run_ensemble() kwargs.
        All parameters not being varied are held at these values.
    N_paths : int
        Ensemble size per run. Smaller than production for speed;
        5000 gives stable statistics for sensitivity comparisons.

    Returns
    -------
    pd.DataFrame with columns:
        parameter      : str    name of the varied parameter
        value          : float  value tested
        p_fail         : float  failure probability at that value
        mean_transport : float  mean transported sediment at that value
    """
    import pandas as pd

    results = []   # accumulate results across all parameter-value combinations
    total   = sum(len(v) for v in param_grid.values())   # total runs for progress reporting
    done    = 0

    for param_name, values in param_grid.items():
        for val in values:
            # Build parameter set: base values with one parameter overridden
            params              = base_params.copy()
            params[param_name]  = val
            params['N_paths']   = N_paths

            # Run ensemble and extract summary statistics
            _, _, p_fail, transported = run_ensemble(**params)

            results.append({
                'parameter'     : param_name,
                'value'         : val,
                'p_fail'        : round(float(p_fail), 4),
                'mean_transport': round(float(transported.mean()), 4),
            })

            done += 1
            print(f"  [{done}/{total}] {param_name} = {val:.3f}  ->  p_fail = {p_fail:.3f}")

    return pd.DataFrame(results)
