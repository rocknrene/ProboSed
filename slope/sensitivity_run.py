"""
sensitivity_run.py
==================
Parameter sensitivity analysis for the Japan Trench slope stability model.

Varies one parameter at a time across a physically motivated range while
holding all other parameters at their base values. This tests whether the
qualitative result — that fault coupling substantially increases slope
failure probability — is robust across plausible parameter choices.

Physical parameter ranges are informed by published IODP proceedings:
  - Exp 405 (JTRACK): Vp, resistivity, borehole breakouts at Site C0019
  - Exp 343 (JFAST):  consolidation, permeability, porosity at Site C0019
  - Exp 386:          shear strength and seismic strengthening, Japan Trench basins

Output:
  sensitivity_results.csv   — full results table (parameter, value, p_fail, mean_transport)
  sensitivity_analysis.png  — one-panel-per-parameter plot of P(failure) vs parameter value

Usage:
  python sensitivity_run.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the ensemble simulator and sensitivity runner from stability module
from slope.stability import run_ensemble, run_sensitivity


# =============================================================================
# BASE PARAMETERS
# These are the reference values used in the toy model.
# Each sensitivity run varies one parameter while all others remain at these values.
# Physical justifications are documented in stability.py and toy_model.py.
# =============================================================================

BASE_PARAMS = dict(
    T           = 10.0,   # total simulation time (dimensionless units)
    dt          = 0.01,   # timestep
    gamma       = 1.0,    # slope damping coefficient
    sigma_q     = 0.6,    # slope noise amplitude
    sigma_s     = 0.5,    # fault slip noise amplitude
    k_f         = 1.0,    # fault slip damping
    alpha       = 0.5,    # fault-to-slope coupling coefficient
    threshold   = 1.0,    # failure threshold
    slip_mag    = 3.0,    # mainshock impulse magnitude (dimensionless)
    mainshock_t = 5.0,    # time of mainshock impulse
    seed        = 42,     # random seed for reproducibility
)


# =============================================================================
# PARAMETER GRID
# Each key matches a run_ensemble() keyword argument.
# Ranges are physically motivated as described below.
# =============================================================================

PARAM_GRID = {

    # alpha: fault-to-slope coupling coefficient
    # Represents coseismic stress transfer efficiency from decollement to slope.
    # Range 0.2-0.9 spans from weak coupling (thick, attenuating prism) to
    # strong coupling (thin, mechanically continuous prism).
    # Overconsolidation (Pc' = 17 MPa, Exp 343) supports non-zero coupling.
    'alpha':     [0.2, 0.3, 0.5, 0.7, 0.9],

    # sigma_q: slope noise amplitude (background tectonic variability)
    # Low end (0.3-0.4): seismically strengthened background hemipelagic sediment (Exp 386)
    # Base (0.6): weak frontal prism material, Vp = 1550-1750 m/s (Exp 405, Unit I)
    # High end (0.8-1.0): highly disturbed, near-remobilized material
    'sigma_q':   [0.3, 0.4, 0.6, 0.8, 1.0],

    # slip_mag: mainshock impulse magnitude (dimensionless scaled units)
    # Physical anchor: ~50 m coseismic slip at C0019 (Fulton et al. 2013)
    # Range tests sensitivity to uncertainty in slip scaling and
    # spatial attenuation from decollement to slope surface
    'slip_mag':  [1.0, 2.0, 3.0, 4.0, 5.0],

    # gamma: slope damping coefficient (restoring force strength)
    # Low end (0.5): very weak, near-remobilized material
    # Base (1.0): typical frontal prism mudstone
    # High end (1.5-2.0): seismically strengthened sediment (Exp 386)
    'gamma':     [0.5, 0.75, 1.0, 1.5, 2.0],

    # threshold: failure criterion
    # Values correspond directly to VCD scores via threshold_from_vcd():
    #   2.00 = VCD score 3 (intact bedding)
    #   1.33 = VCD score 2 (coherent block)
    #   1.00 = base value (between score 1 and 2)
    #   0.67 = VCD score 1 (scaly fabric)
    #   0.33 = between score 0 and 1 (near-remobilized)
    # This range tests sensitivity to the VCD-to-threshold mapping choice
    'threshold': [0.33, 0.67, 1.0, 1.33, 2.0],
}


# =============================================================================
# MAIN — RUN SENSITIVITY ANALYSIS AND PRODUCE OUTPUT
# =============================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("Japan Trench Slope Stability — Sensitivity Analysis")
    print("=" * 60)
    print(f"\nBase parameters:")
    for k, v in BASE_PARAMS.items():
        print(f"  {k:15s} = {v}")
    print(f"\nEnsemble size per run: 5000 paths")
    print(f"Total runs: {sum(len(v) for v in PARAM_GRID.values())}\n")

    # Run the sensitivity analysis — one-at-a-time parameter variation
    df = run_sensitivity(PARAM_GRID, BASE_PARAMS, N_paths=5000)

    # Save results table to CSV
    df.to_csv('sensitivity_results.csv', index=False)
    print(f"\nResults saved to: sensitivity_results.csv")

    # Compute baseline failure probability for reference line on plots
    # Uses the same N_paths as sensitivity runs for fair comparison
    base_run_params         = BASE_PARAMS.copy()
    base_run_params['N_paths'] = 5000
    _, _, base_p_fail, _    = run_ensemble(**base_run_params)
    print(f"Baseline P(failure): {base_p_fail:.3f}")

    # =========================================================================
    # FIGURE — Sensitivity analysis: one panel per parameter
    # Each panel shows P(failure) vs parameter value with baseline reference
    # =========================================================================

    params     = list(PARAM_GRID.keys())    # ordered list of parameter names
    n_params   = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5))
    fig.patch.set_facecolor('white')

    # Axis labels for display (more readable than raw parameter names)
    param_labels = {
        'alpha'    : r'$\alpha$ (coupling)',
        'sigma_q'  : r'$\sigma_q$ (slope noise)',
        'slip_mag' : 'Slip magnitude',
        'gamma'    : r'$\gamma$ (damping)',
        'threshold': r'$\theta$ (failure threshold)',
    }

    for ax, param in zip(axes, params):
        ax.set_facecolor('white')

        # Subset results for this parameter
        subset = df[df['parameter'] == param].sort_values('value')

        # Plot P(failure) vs parameter value
        ax.plot(subset['value'], subset['p_fail'],
                'o-', color='navy', lw=2, ms=8,
                zorder=3)

        # Mark the base value with a vertical dashed line
        ax.axvline(BASE_PARAMS[param], color='gray', ls='--', lw=1.5,
                   alpha=0.7, label=f'Base = {BASE_PARAMS[param]}', zorder=2)

        # Mark the baseline failure probability with a horizontal dashed line
        ax.axhline(base_p_fail, color='red', ls=':', lw=1.5, alpha=0.7,
                   label=f'Baseline P = {base_p_fail:.2f}', zorder=2)

        # Annotate each data point with its P(failure) value
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['p_fail']:.2f}",
                xy=(row['value'], row['p_fail']),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=8, color='navy'
            )

        ax.set_xlabel(param_labels.get(param, param), fontsize=12)
        ax.set_ylabel('P(failure)' if param == params[0] else '', fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_title(param_labels.get(param, param),
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left' if param != 'threshold' else 'upper right')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Parameter Sensitivity Analysis — Japan Trench Slope Stability Model\n'
        'One parameter varied at a time; all others held at base values\n'
        'Site C0019J, Japan Trench (IODP Expeditions 343, 386, 405)',
        fontsize=12, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=180, bbox_inches='tight',
                facecolor='white')   # enforce white background on save
    plt.close()
    print("Saved: sensitivity_analysis.png")

    # =========================================================================
    # PRINT SUMMARY TABLE TO CONSOLE
    # =========================================================================

    print("\n--- Sensitivity Summary ---")
    print(f"{'Parameter':<15} {'Value':<10} {'P(failure)':<12} {'Mean transport':<15}")
    print("-" * 52)
    for _, row in df.iterrows():
        print(f"{row['parameter']:<15} {row['value']:<10.3f} "
              f"{row['p_fail']:<12.4f} {row['mean_transport']:<15.4f}")

    # =========================================================================
    # ROBUSTNESS CHECK
    # Identify whether the qualitative conclusion holds across all parameter values:
    # Does fault coupling always produce higher P(failure) than the slope-only case?
    # =========================================================================

    print("\n--- Robustness Check ---")
    print("Qualitative result: fault coupling raises P(failure) above slope-only baseline.")
    print(f"Slope-only baseline (from toy_model.py): ~0.38")
    print(f"Fault-coupled baseline (N=5000):          {base_p_fail:.3f}")
    print("\nParameter ranges where P(failure) exceeds slope-only baseline (0.38):")

    for param in params:
        subset  = df[df['parameter'] == param].sort_values('value')
        robust  = subset[subset['p_fail'] > 0.38]   # above slope-only baseline
        pct     = 100 * len(robust) / len(subset)
        print(f"  {param:<15}: {len(robust)}/{len(subset)} values ({pct:.0f}%)")
