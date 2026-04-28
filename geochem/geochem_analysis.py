"""
geochem_analysis.py
==================
IODP Expedition 405 — Site C0019
Porewater Geochemistry and Headspace Gas Depth Profiles

Produces three figures:
  1. C0019J_geochemistry_profiles.png
        Five-panel depth profile for the deep coring hole (93-825 mbsf).
        Captures the full sediment column from the upper prism to the
        plate boundary fault zone.

  2. C0019M_geochemistry_profiles.png
        Five-panel depth profile for the shallow coring hole (0-107 mbsf).
        Resolves the sulfate-methane transition zone (SMTZ) at ~40 mbsf,
        which is absent from the J hole record due to core recovery gaps
        at the top of that hole.

  3. C0019_combined_overview.png
        Side-by-side comparison of both holes for CH4, SO4, and Ca.
        Connects the shallow SMTZ context to the deep fault zone signals.

Integration with ProboSed pipeline:
  The plot functions accept an optional mtd_catalog argument — a DataFrame
  produced by JCORESMiner.score_to_mtd_catalog() from labeler.py. When
  supplied, MTD intervals identified from VCD observations are shaded on
  every depth profile panel, directly linking geochemical anomalies to
  sediment fabric observations. If mtd_catalog is None the figures are
  produced without MTD shading using only the hardcoded reference lines.

Usage:
  python geochem_analysis.py
  Outputs figures to OUTPUT_DIR (default: current directory).

Requirements:
  pip install pandas openpyxl matplotlib scipy

Input files (place in DATA_DIR or update paths below):
  SummarySheet-IW-Hole_Exp405_C0019J_250312.xlsx
  SummarySheet-IW-Hole_Exp405_C0019M_250312.xlsx
  SummarySheet-GC_200_160206_Exp405_C0019J.xlsm
  SummarySheet-GC_200_160206_Exp405_C0019M.xlsm

Notes on depth assignment for GC samples:
  Headspace gas samples are measured at higher spatial resolution than
  interstitial water (IW) samples. Because the IW summary sheets contain
  the authoritative CSF-A depth values per core, GC sample depths are
  estimated by linear interpolation between IW core anchors. This is
  standard practice for integrating shipboard datasets where not all
  sample types share a common depth reference file.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — safe for scripts and Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
import os


# =============================================================================
# PATHS
# Update DATA_DIR and OUTPUT_DIR if the data files live elsewhere.
# On Google Colab, set DATA_DIR to the mounted Drive path, e.g.:
#   DATA_DIR = '/content/drive/MyDrive/iodp/X405/Data & Data Tracking'
# =============================================================================
DATA_DIR   = '.'    # directory containing the .xlsx / .xlsm input files
OUTPUT_DIR = '.'    # directory where output PNGs will be written

IW_J = os.path.join(DATA_DIR, 'SummarySheet-IW-Hole_Exp405_C0019J_250312.xlsx')
IW_M = os.path.join(DATA_DIR, 'SummarySheet-IW-Hole_Exp405_C0019M_250312.xlsx')
GC_J = os.path.join(DATA_DIR, 'SummarySheet-GC_200_160206_Exp405_C0019J.xlsm')
GC_M = os.path.join(DATA_DIR, 'SummarySheet-GC_200_160206_Exp405_C0019M.xlsm')


# =============================================================================
# REFERENCE DEPTHS (mbsf)
# Geological horizons marked on every depth profile panel.
# Update values here and all figures update automatically.
# MTD boundaries from the VCD pipeline are handled separately via mtd_catalog.
# =============================================================================
FAULT_DEPTH   = 820   # plate boundary fault — constrained by JTRACK LWD logs
                      # and temperature observatory data (Fulton et al. 2013)
MTD_TOP       = 540   # approximate top of main MTD cluster, based on
                      # borehole breakout onset depth from JTRACK logging
BREAKOUT_BASE = 313   # shallowest resolved borehole breakout in C0019H
                      # (JTRACK logging; Conin et al. in review)


# =============================================================================
# COLOR PALETTE
# All colors defined here so any adjustment propagates to all figures.
# American English spelling used throughout (color not colour).
# =============================================================================
BG       = 'white'     # figure background — white for publication/dissertation
PANEL_BG = '#f7f7f7'   # axes background — light gray to distinguish panels
GRID_C   = '#dddddd'   # x-axis grid line color
TEXT_C   = '#1a1a1a'   # all text, tick labels, axis labels

# geochemical species — one color per variable, consistent across all figures
C_CH4 = '#1a7a2e'   # methane        green
C_SO4 = '#b87800'   # sulfate        amber
C_CL  = '#1a6bbf'   # chlorinity     blue
C_CA  = '#cc2222'   # calcium        red
C_ALK = '#7733cc'   # alkalinity     purple

# reference line colors
C_FAULT    = '#ff4444'   # plate boundary fault
C_MTD      = '#ffa500'   # MTD cluster / VCD-derived MTD intervals
C_BREAKOUT = '#aaaaaa'   # borehole breakout onset
C_SEAWATER = '#888888'   # seawater chlorinity reference line

# Reference line dictionary — {depth_mbsf: (legend_label, color)}
# MTD boundaries from JCORESMiner.score_to_mtd_catalog() are added at
# runtime via the mtd_catalog parameter in each plot function.
REF_LINES = {
    FAULT_DEPTH:   ('Plate boundary fault ~820 mbsf', C_FAULT),
    MTD_TOP:       ('MTD cluster top ~540 mbsf',      C_MTD),
    BREAKOUT_BASE: ('Breakout onset ~313 mbsf',       C_BREAKOUT),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_iw(path):
    """
    Load a shipboard interstitial water (IW) summary sheet.

    The CHIKYU IW summary sheets follow a fixed layout:
      - Rows 0-5 are metadata/header rows
      - Data begins at row index 6 (0-indexed)
      - Column positions (0-indexed):
          1  = core label (e.g. '2K', '5H')
          5  = depth top CSF-A (m)
          6  = depth bottom CSF-A (m)
          13 = alkalinity (mM)
          14 = chlorinity (mM)
          18 = sulfate SO4 (mM)
          19 = sodium Na (mM)
          21 = magnesium Mg (mM)
          22 = calcium Ca (mM)

    Depth is computed as the midpoint of the sampled interval:
      depth_m = (top_CSF-A + bottom_CSF-A) / 2
    This is standard practice for IODP geochemistry reports.

    Parameters
    ----------
    path : str
        Path to the IW summary sheet (.xlsx).

    Returns
    -------
    pd.DataFrame sorted by depth_m with columns:
        core, depth_m, alkalinity, chlorinity, SO4, Na, Mg, Ca, core_num
    """
    raw = pd.read_excel(path, header=None)
    d   = raw.iloc[6:].copy()          # skip the five metadata/header rows
    d.columns = range(len(d.columns)) # reset column indices to integers

    df = pd.DataFrame({
        'core':       d[1],
        'depth_m':    (d[5].astype(float) + d[6].astype(float)) / 2,   # interval midpoint
        'alkalinity': d[13].astype(float),
        'chlorinity': d[14].astype(float),
        'SO4':        d[18].astype(float),
        'Na':         d[19].astype(float),
        'Mg':         d[21].astype(float),
        'Ca':         d[22].astype(float),
    }).dropna(subset=['depth_m'])

    # extract numeric core number for GC depth interpolation (e.g. '26K' -> 26)
    df['core_num'] = df['core'].str.extract(r'^(\d+)').astype(float)

    return df.sort_values('depth_m').reset_index(drop=True)


def load_gc(path, iw_df):
    """
    Load a shipboard headspace gas (GC-FID) summary sheet and assign
    depths by interpolating from the IW core-depth anchors.

    GC sample labels follow the format '00026K-01-WR'. The core number
    (26 in this example) is parsed from the label and matched to the
    corresponding depth from the IW dataset. Linear interpolation covers
    cores not directly sampled by IW; extrapolation handles edge cases.

    The C1/C2 ratio (methane / ethane) is a standard gas source indicator:
      >1000  ->  microbial methanogenesis (expected in shallow marine sediments)
      <100   ->  possible thermogenic contribution or mixing

    Rejected samples (flagged in 'Decision of rejection') are excluded.

    Parameters
    ----------
    path  : str
        Path to the GC summary sheet (.xlsm).
    iw_df : pd.DataFrame
        Output of load_iw() for the same hole — used as depth anchor.

    Returns
    -------
    pd.DataFrame sorted by depth_m with columns:
        SampleInfo, core_num, depth_m, Methane, Ethane, C1C2
    """
    raw = pd.read_excel(path, sheet_name='Result', header=14)

    # retain only accepted, non-null methane measurements
    gc = raw[
        raw['Methane'].notna() &
        raw['Decision of rejection'].isna() &
        raw['SampleInfo'].notna()
    ].copy()

    # parse core number — handles H, K, R, X core type suffixes
    gc['core_num'] = gc['SampleInfo'].str.extract(r'^(\d+)[HKRXhkrx]').astype(float)

    # build linear depth interpolator anchored to IW core midpoints
    # extrapolation is used for GC cores outside the IW depth range
    core_map = iw_df.groupby('core_num')['depth_m'].mean()
    f_depth  = interp1d(
        core_map.index, core_map.values,
        kind='linear', fill_value='extrapolate'
    )
    gc['depth_m'] = gc['core_num'].apply(lambda x: float(f_depth(x)))

    # C1/C2 ratio — left as NaN where ethane is zero or below detection
    gc['C1C2'] = np.where(
        gc['Ethane'] > 0,
        gc['Methane'] / gc['Ethane'],
        np.nan
    )

    return gc.sort_values('depth_m').reset_index(drop=True)


# =============================================================================
# SHARED PLOTTING UTILITIES
# =============================================================================

def style_ax(ax):
    """
    Apply consistent styling to a single Axes object.

    Called on every panel immediately after creation, before data are plotted.
    Enforces the shared color palette defined at the top of this module.

    Parameters
    ----------
    ax : matplotlib Axes
    """
    ax.set_facecolor(PANEL_BG)                   # light gray panel background
    ax.tick_params(colors=TEXT_C, labelsize=8.5) # dark tick labels
    for spine in ax.spines.values():
        spine.set_color('#cccccc')               # light gray panel border
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)
    ax.grid(axis='x', color=GRID_C, lw=0.5, alpha=0.7)   # vertical grid only


def add_ref_lines(ax, ref_dict, depth_range):
    """
    Draw horizontal dashed reference lines at specified depths.

    Lines are only drawn if the depth falls within the visible depth range
    of the panel, avoiding reference lines outside the plotted interval.

    Parameters
    ----------
    ax         : matplotlib Axes
    ref_dict   : dict  {depth_mbsf: (label_str, color_str)}
    depth_range: tuple (max_depth, min_depth) — inverted axis convention
                 max_depth is the bottom of the plot, min_depth the top
    """
    for depth, (label, color) in ref_dict.items():
        # depth_range is (bottom, top) because y-axis is inverted
        if depth_range[1] <= depth <= depth_range[0]:
            ax.axhline(depth, color=color, lw=1.3, ls='--', alpha=0.8, zorder=2)


def add_mtd_shading(ax, mtd_catalog, depth_range):
    """
    Shade MTD intervals identified from VCD observations.

    Draws a semi-transparent horizontal band for each MTD interval in the
    catalog, connecting geochemical anomalies to fabric-based observations
    from the VCDLabeler / JCORESMiner pipeline.

    Only intervals within the visible depth range are drawn.

    Parameters
    ----------
    ax          : matplotlib Axes
    mtd_catalog : pd.DataFrame
        Output of JCORESMiner.score_to_mtd_catalog() from labeler.py.
        Must have columns: top_m, bottom_m, mtd_id.
    depth_range : tuple (max_depth, min_depth) — inverted axis convention
    """
    if mtd_catalog is None or mtd_catalog.empty:
        return   # nothing to draw

    for _, row in mtd_catalog.iterrows():
        top    = row['top_m']
        bottom = row['bottom_m']

        # only draw if any part of the interval is within the visible range
        if top > depth_range[0] or bottom < depth_range[1]:
            continue

        # clip to visible range
        top    = max(top,    depth_range[1])
        bottom = min(bottom, depth_range[0])

        ax.axhspan(top, bottom,
                   alpha=0.10, color=C_MTD,
                   zorder=1, label='MTD interval (VCD)')


def ref_legend(fig, ref_dict, y=0.01, mtd_catalog=None):
    """
    Add a shared color-patch legend for reference lines at the figure bottom.

    If an mtd_catalog is provided, an MTD shading swatch is added to the legend.

    Parameters
    ----------
    fig         : matplotlib Figure
    ref_dict    : dict  {depth_mbsf: (label_str, color_str)}
    y           : float  vertical position in figure coordinates
    mtd_catalog : pd.DataFrame or None
        If provided, adds an MTD interval swatch to the legend.
    """
    patches = [
        mpatches.Patch(color=c, label=l)
        for _, (l, c) in ref_dict.items()
    ]

    # add MTD swatch if catalog was provided
    if mtd_catalog is not None and not mtd_catalog.empty:
        patches.append(
            mpatches.Patch(color=C_MTD, alpha=0.3,
                           label=f'MTD intervals (n={len(mtd_catalog)}, VCD-derived)')
        )

    fig.legend(
        handles=patches, loc='lower center', ncol=len(patches),
        facecolor='#f0f0f0', edgecolor='#cccccc',
        labelcolor=TEXT_C, fontsize=8,
        bbox_to_anchor=(0.5, y)
    )


# =============================================================================
# FIGURE 1 — C0019J DEEP HOLE (93-825 mbsf)
# =============================================================================

def plot_C0019J(iw, gc, outpath, mtd_catalog=None):
    """
    Five-panel depth profile for hole C0019J (deep hole, 93-825 mbsf).

    Covers the prism column from ~93 mbsf to the plate boundary fault at
    ~825 mbsf. The key observation is the Ca enrichment at core 87K
    (~825 mbsf), interpreted as a fluid signal at the plate boundary fault.

    Geochemical signals across the five panels collectively provide
    independent validation of the VCD stability index: anomalies in SO4,
    Ca, and alkalinity co-occur with MTD intervals identified from fabric
    observations, supporting the physical interpretation that mass transport
    events create permeable pathways for fluid advection.

    Panels (left to right):
      1. Methane CH4 (log scale, ppm)
      2. Sulfate SO4 (mM)            — anomalous spike at fault
      3. Chlorinity Cl- (mM)         — seawater reference at 559 mM shown
      4. Calcium Ca2+ (mM)           — fault zone spike annotated
      5. Alkalinity (mM)             — collapse mirrors Ca spike at fault

    Parameters
    ----------
    iw          : pd.DataFrame  output of load_iw() for C0019J
    gc          : pd.DataFrame  output of load_gc() for C0019J
    outpath     : str           path for the output PNG
    mtd_catalog : pd.DataFrame or None
        Output of JCORESMiner.score_to_mtd_catalog(). If provided, MTD
        intervals are shaded on every panel and added to the legend.
    """
    depth_range = (iw['depth_m'].max() + 15, iw['depth_m'].min() - 15)

    fig, axes = plt.subplots(1, 5, figsize=(16, 14), sharey=True)
    fig.patch.set_facecolor(BG)

    for ax in axes:
        style_ax(ax)
        ax.set_ylim(*depth_range)
        ax.invert_yaxis()                            # depth increases downward
        add_ref_lines(ax, REF_LINES, depth_range)    # fault, MTD top, breakout
        add_mtd_shading(ax, mtd_catalog, depth_range) # VCD-derived MTD intervals

    # ── Panel 1: Methane ─────────────────────────────────────────────────────
    # Log scale is appropriate because methane spans four orders of magnitude
    # from near-zero at the top of the hole to ~35,000 ppm in the methanogenic
    # zone. The entire J hole record is below the SMTZ and within methanogenesis.
    ax = axes[0]
    ax.semilogx(gc['Methane'], gc['depth_m'],
                'o', color=C_CH4, ms=4.5, alpha=0.85, zorder=3)
    ax.set_xlabel('CH4 (ppm)\n[log scale]', fontsize=9)
    ax.set_title('Methane', fontsize=10, fontweight='bold')
    ax.set_xlim(1, 60000)
    ax.set_ylabel('Depth (mbsf)', fontsize=10)

    # ── Panel 2: Sulfate ─────────────────────────────────────────────────────
    # SO4 is nearly depleted throughout the J hole because the SMTZ occurs
    # very shallow (<50 mbsf), above the top of J hole recovery.
    # The anomalous SO4 spike at core 87K (~14 mM) is interpreted as
    # upward advection of SO4-bearing fluids from the subducting plate.
    ax = axes[1]
    ax.plot(iw['SO4'], iw['depth_m'],
            's-', color=C_SO4, ms=4, lw=1.3, alpha=0.9, zorder=3)
    ax.set_xlabel('SO4 (mM)', fontsize=9)
    ax.set_title('Sulfate', fontsize=10, fontweight='bold')
    ax.set_xlim(left=0)

    # ── Panel 3: Chlorinity ──────────────────────────────────────────────────
    # Chlorinity deviations from seawater (~559 mM) indicate fluid advection.
    # Freshening signals clay dehydration or gas hydrate dissociation;
    # enrichment signals evaporite dissolution or deep brine advection.
    # The vertical reference line marks the modern seawater value.
    ax = axes[2]
    ax.plot(iw['chlorinity'], iw['depth_m'],
            'D-', color=C_CL, ms=4, lw=1.3, alpha=0.9, zorder=3)
    ax.axvline(559, color=C_SEAWATER, lw=0.9, ls=':', alpha=0.7)   # seawater Cl-
    ax.set_xlabel('Cl- (mM)', fontsize=9)
    ax.set_title('Chlorinity', fontsize=10, fontweight='bold')

    # ── Panel 4: Calcium ─────────────────────────────────────────────────────
    # Ca increases gradually with depth due to carbonate diagenesis, but the
    # abrupt spike at ~825 mbsf (core 87K) at the plate boundary fault is
    # anomalous relative to the 600-800 mbsf background (~9 mM).
    # The most consistent interpretation is upward advection of Ca-enriched
    # fluids from the subducting Pacific plate where smectite-to-illite
    # reactions and carbonate dissolution generate Ca-rich pore waters.
    ax = axes[3]
    ax.plot(iw['Ca'], iw['depth_m'],
            '^-', color=C_CA, ms=4, lw=1.3, alpha=0.9, zorder=3)

    # annotate the fault zone Ca spike — the key quantitative result
    fault_row = iw[iw['depth_m'] > 820]
    if not fault_row.empty:
        ca_val = fault_row['Ca'].values[0]
        ax.annotate(
            f'Ca = {ca_val:.1f} mM\n(fault zone)',
            xy=(ca_val, fault_row['depth_m'].values[0]),
            xytext=(ca_val - 12, 800),
            color=C_CA, fontsize=7.5,
            arrowprops=dict(arrowstyle='->', color=C_CA, lw=1)
        )
    ax.set_xlabel('Ca2+ (mM)', fontsize=9)
    ax.set_title('Calcium', fontsize=10, fontweight='bold')
    ax.set_xlim(left=0)

    # ── Panel 5: Alkalinity ──────────────────────────────────────────────────
    # Alkalinity rises through the methanogenic zone due to anaerobic methane
    # oxidation (AOM) and microbial sulfate reduction, then collapses near the
    # fault as authigenic carbonate precipitation consumes dissolved inorganic
    # carbon. The collapse mirrors the Ca spike — both reflect the same
    # fluid-rock interaction at the fault zone boundary.
    ax = axes[4]
    ax.plot(iw['alkalinity'], iw['depth_m'],
            'o-', color=C_ALK, ms=4, lw=1.3, alpha=0.9, zorder=3)
    ax.set_xlabel('Alkalinity (mM)', fontsize=9)
    ax.set_title('Alkalinity', fontsize=10, fontweight='bold')
    ax.set_xlim(left=0)

    fig.suptitle(
        'IODP Exp 405 - C0019J   Pore Water and Gas Geochemistry\n'
        '(93-825 mbsf, approaching plate boundary fault)',
        color=TEXT_C, fontsize=12, fontweight='bold', y=0.985
    )

    ref_legend(fig, REF_LINES, y=0.01, mtd_catalog=mtd_catalog)
    plt.tight_layout(rect=[0, 0.06, 1, 0.975])
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {outpath}")


# =============================================================================
# FIGURE 2 — C0019M SHALLOW HOLE (0-107 mbsf)
# =============================================================================

def plot_C0019M(iw, gc, outpath, mtd_catalog=None):
    """
    Five-panel depth profile for hole C0019M (shallow hole, 0-107 mbsf).

    The M hole provides the shallow context missing from the J hole record.
    The sulfate-methane transition zone (SMTZ) is fully resolved here at
    ~40 mbsf, where SO4 reaches zero and methane rises sharply to >27,000 ppm.
    This establishes the diagenetic baseline for interpreting the J hole.

    The Ca decrease through the SMTZ (from carbonate precipitation) is the
    opposite of the Ca enrichment at the fault in C0019J, confirming that
    the deep Ca spike is a fluid advection signal rather than a diagenetic
    artifact. This cross-hole comparison is a key validation argument.

    Panels (left to right):
      1. Methane CH4 (log scale, ppm)   SMTZ transition visible
      2. Sulfate SO4 (mM)               SO4 -> 0 at ~40 mbsf
      3. Chlorinity Cl- (mM)            stable background
      4. Calcium Ca2+ (mM)              slight decrease through SMTZ
      5. Alkalinity (mM)                peak at SMTZ from AOM

    Parameters
    ----------
    iw          : pd.DataFrame  output of load_iw() for C0019M
    gc          : pd.DataFrame  output of load_gc() for C0019M
    outpath     : str           path for the output PNG
    mtd_catalog : pd.DataFrame or None
        Output of JCORESMiner.score_to_mtd_catalog(). If provided, MTD
        intervals are shaded on every panel and added to the legend.
    """
    depth_range = (iw['depth_m'].max() + 5, 0)

    fig, axes = plt.subplots(1, 5, figsize=(16, 10), sharey=True)
    fig.patch.set_facecolor(BG)

    # SMTZ reference line — specific to the M hole shallow depth range
    SMTZ     = 40
    smtz_ref = {SMTZ: ('SMTZ ~40 mbsf', '#007a5e')}

    for ax in axes:
        style_ax(ax)
        ax.set_ylim(*depth_range)
        ax.invert_yaxis()
        add_ref_lines(ax, smtz_ref, depth_range)
        add_mtd_shading(ax, mtd_catalog, depth_range)   # VCD-derived MTD intervals

    # ── Panel 1: Methane ─────────────────────────────────────────────────────
    # The jump from <200 ppm at core 4H to >27,000 ppm at core 5H marks the
    # SMTZ — below this depth all available SO4 is consumed and methanogenesis
    # dominates. clip(lower=0.1) prevents log(0) errors for near-zero values
    # in the uppermost cores where methane is essentially absent.
    ax = axes[0]
    ax.semilogx(gc['Methane'].clip(lower=0.1), gc['depth_m'],
                'o', color=C_CH4, ms=5, alpha=0.85, zorder=3)
    ax.set_xlabel('CH4 (ppm)\n[log scale]', fontsize=9)
    ax.set_title('Methane', fontsize=10, fontweight='bold')
    ax.set_xlim(0.1, 50000)
    ax.set_ylabel('Depth (mbsf)', fontsize=10)

    # ── Panel 2: Sulfate ─────────────────────────────────────────────────────
    # SO4 decreases from near-seawater values (~28 mM) at the seafloor to
    # zero at ~40 mbsf, driven by microbial sulfate reduction coupled with
    # anaerobic methane oxidation (AOM) at the SMTZ.
    ax = axes[1]
    ax.plot(iw['SO4'], iw['depth_m'],
            's-', color=C_SO4, ms=5, lw=1.5, alpha=0.9, zorder=3)
    ax.set_xlabel('SO4 (mM)', fontsize=9)
    ax.set_title('Sulfate', fontsize=10, fontweight='bold')
    ax.set_xlim(left=0)

    # ── Panel 3: Chlorinity ──────────────────────────────────────────────────
    # Chlorinity is relatively stable through the M hole — no major fresh or
    # saline fluid signals at shallow depths. This is the expected background
    # against which J hole chlorinity variations are interpreted.
    ax = axes[2]
    ax.plot(iw['chlorinity'], iw['depth_m'],
            'D-', color=C_CL, ms=5, lw=1.5, alpha=0.9, zorder=3)
    ax.axvline(559, color=C_SEAWATER, lw=0.9, ls=':', alpha=0.7)   # seawater Cl-
    ax.set_xlabel('Cl- (mM)', fontsize=9)
    ax.set_title('Chlorinity', fontsize=10, fontweight='bold')

    # ── Panel 4: Calcium ─────────────────────────────────────────────────────
    # Ca decreases slightly through the SMTZ due to authigenic carbonate
    # precipitation driven by elevated alkalinity from AOM. This is the
    # opposite of the deep Ca enrichment at the fault in C0019J, confirming
    # the fault zone signal is not a diagenetic artifact.
    ax = axes[3]
    ax.plot(iw['Ca'], iw['depth_m'],
            '^-', color=C_CA, ms=5, lw=1.5, alpha=0.9, zorder=3)
    ax.set_xlabel('Ca2+ (mM)', fontsize=9)
    ax.set_title('Calcium', fontsize=10, fontweight='bold')
    ax.set_xlim(left=0)

    # ── Panel 5: Alkalinity ──────────────────────────────────────────────────
    # Alkalinity peaks at the SMTZ — the classic AOM signature.
    # Dissolved inorganic carbon (DIC) produced by AOM raises alkalinity
    # and drives authigenic carbonate formation (MDAC).
    ax = axes[4]
    ax.plot(iw['alkalinity'], iw['depth_m'],
            'o-', color=C_ALK, ms=5, lw=1.5, alpha=0.9, zorder=3)
    ax.set_xlabel('Alkalinity (mM)', fontsize=9)
    ax.set_title('Alkalinity', fontsize=10, fontweight='bold')
    ax.set_xlim(left=0)

    fig.suptitle(
        'IODP Exp 405 - C0019M   Pore Water and Gas Geochemistry\n'
        '(0-107 mbsf) — Sulfate-Methane Transition Zone resolved at ~40 mbsf',
        color=TEXT_C, fontsize=12, fontweight='bold', y=0.985
    )

    # legend — SMTZ reference plus optional MTD catalog swatch
    smtz_patches = [mpatches.Patch(color='#007a5e', label='SMTZ ~40 mbsf')]
    if mtd_catalog is not None and not mtd_catalog.empty:
        smtz_patches.append(
            mpatches.Patch(color=C_MTD, alpha=0.3,
                           label=f'MTD intervals (n={len(mtd_catalog)}, VCD-derived)')
        )
    fig.legend(
        handles=smtz_patches, loc='lower center',
        facecolor='#f0f0f0', edgecolor='#cccccc',
        labelcolor=TEXT_C, fontsize=9,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.975])
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {outpath}")


# =============================================================================
# FIGURE 3 — COMBINED OVERVIEW (both holes, three species)
# =============================================================================

def plot_combined(iw_j, gc_j, iw_m, gc_m, outpath, mtd_catalog=None):
    """
    Combined 3-row x 2-column overview figure comparing both holes.

    Left column  = C0019M (shallow, 0-107 mbsf)  — SMTZ context
    Right column = C0019J (deep, 93-825 mbsf)    — fault zone context
    Rows: CH4 | SO4 | Ca

    The side-by-side layout allows direct comparison of:
      - Where SO4 is depleted (M hole SMTZ at ~40 mbsf)
      - Elevated background methane maintained throughout the J hole
      - Ca spike at the fault zone contrasted with stable Ca in the M hole

    The Ca contrast between the two holes is the strongest evidence that
    the deep J hole Ca enrichment is a fluid advection signal from the
    subducting plate rather than a diagenetic background trend.

    Intended as a dissertation chapter overview panel that can stand alone
    as a single figure in a paper focused on fluid-fault coupling.

    Parameters
    ----------
    iw_j, gc_j   : pd.DataFrames  C0019J data from load_iw() and load_gc()
    iw_m, gc_m   : pd.DataFrames  C0019M data from load_iw() and load_gc()
    outpath       : str            path for the output PNG
    mtd_catalog   : pd.DataFrame or None
        Output of JCORESMiner.score_to_mtd_catalog(). Applied to the J hole
        panels only (the M hole depth range rarely overlaps with MTD intervals
        identified in the deep borehole).
    """
    fig = plt.figure(figsize=(14, 16))
    fig.patch.set_facecolor(BG)

    # 3 rows, 2 columns — left = M hole (shallow), right = J hole (deep)
    gs     = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.4)
    axes_m = [fig.add_subplot(gs[r, 0]) for r in range(3)]
    axes_j = [fig.add_subplot(gs[r, 1]) for r in range(3)]

    depth_m_range = (iw_m['depth_m'].max() + 5, 0)
    depth_j_range = (iw_j['depth_m'].max() + 15, iw_j['depth_m'].min() - 15)

    SMTZ     = 40
    smtz_ref = {SMTZ: ('SMTZ ~40 mbsf', '#007a5e')}

    for ax in axes_m + axes_j:
        style_ax(ax)
        ax.invert_yaxis()

    for ax in axes_m:
        ax.set_ylim(*depth_m_range)
        add_ref_lines(ax, smtz_ref, depth_m_range)
        # MTD catalog is from the deep J hole — not applied to M hole panels

    for ax in axes_j:
        ax.set_ylim(*depth_j_range)
        add_ref_lines(ax, REF_LINES, depth_j_range)
        add_mtd_shading(ax, mtd_catalog, depth_j_range)   # VCD-derived MTD intervals

    # ── Row 0: Methane ───────────────────────────────────────────────────────
    # M hole: shows the pre-SMTZ to post-SMTZ transition in methane
    # J hole: shows the sustained high-methane methanogenic zone throughout
    axes_m[0].semilogx(gc_m['Methane'].clip(lower=0.1), gc_m['depth_m'],
                       'o', color=C_CH4, ms=5, alpha=0.85)
    axes_m[0].set_title('C0019M  Methane (log)', color=TEXT_C,
                         fontsize=10, fontweight='bold')
    axes_m[0].set_xlabel('CH4 (ppm)', fontsize=9)
    axes_m[0].set_ylabel('Depth (mbsf)', fontsize=9)

    axes_j[0].semilogx(gc_j['Methane'], gc_j['depth_m'],
                       'o', color=C_CH4, ms=4, alpha=0.85)
    axes_j[0].set_title('C0019J  Methane (log)', color=TEXT_C,
                         fontsize=10, fontweight='bold')
    axes_j[0].set_xlabel('CH4 (ppm)', fontsize=9)

    # ── Row 1: Sulfate ───────────────────────────────────────────────────────
    # M hole: gradual depletion from ~28 mM to zero at the SMTZ
    # J hole: near-zero background throughout, with anomalous spike at fault
    axes_m[1].plot(iw_m['SO4'], iw_m['depth_m'],
                   's-', color=C_SO4, ms=5, lw=1.5, alpha=0.9)
    axes_m[1].set_title('C0019M  Sulfate', color=TEXT_C,
                         fontsize=10, fontweight='bold')
    axes_m[1].set_xlabel('SO4 (mM)', fontsize=9)
    axes_m[1].set_ylabel('Depth (mbsf)', fontsize=9)
    axes_m[1].set_xlim(left=0)

    axes_j[1].plot(iw_j['SO4'], iw_j['depth_m'],
                   's-', color=C_SO4, ms=4, lw=1.3, alpha=0.9)
    axes_j[1].set_title('C0019J  Sulfate', color=TEXT_C,
                         fontsize=10, fontweight='bold')
    axes_j[1].set_xlabel('SO4 (mM)', fontsize=9)
    axes_j[1].set_xlim(left=0)

    # ── Row 2: Calcium ───────────────────────────────────────────────────────
    # M hole: slight Ca decrease through SMTZ from carbonate precipitation
    # J hole: gradual increase with depth then abrupt spike at fault zone.
    # The contrast between the two holes confirms the J hole Ca spike is a
    # fluid advection signal, not a diagenetic background trend.
    axes_m[2].plot(iw_m['Ca'], iw_m['depth_m'],
                   '^-', color=C_CA, ms=5, lw=1.5, alpha=0.9)
    axes_m[2].set_title('C0019M  Calcium', color=TEXT_C,
                         fontsize=10, fontweight='bold')
    axes_m[2].set_xlabel('Ca2+ (mM)', fontsize=9)
    axes_m[2].set_ylabel('Depth (mbsf)', fontsize=9)
    axes_m[2].set_xlim(left=0)

    axes_j[2].plot(iw_j['Ca'], iw_j['depth_m'],
                   '^-', color=C_CA, ms=4, lw=1.3, alpha=0.9)

    # annotate the fault zone Ca spike — the key result of the J hole record
    fault_row = iw_j[iw_j['depth_m'] > 820]
    if not fault_row.empty:
        ca_val = fault_row['Ca'].values[0]
        axes_j[2].annotate(
            f'Ca = {ca_val:.1f} mM\n(fault zone)',
            xy=(ca_val, fault_row['depth_m'].values[0]),
            xytext=(ca_val - 14, 800),
            color=C_CA, fontsize=7.5,
            arrowprops=dict(arrowstyle='->', color=C_CA, lw=1)
        )
    axes_j[2].set_title('C0019J  Calcium', color=TEXT_C,
                         fontsize=10, fontweight='bold')
    axes_j[2].set_xlabel('Ca2+ (mM)', fontsize=9)
    axes_j[2].set_xlim(left=0)

    # column headers distinguish the two holes at a glance
    fig.text(0.27, 0.97, 'SHALLOW - C0019M (0-107 mbsf)',
             ha='center', color='#007a5e', fontsize=11, fontweight='bold')
    fig.text(0.73, 0.97, 'DEEP - C0019J (93-825 mbsf)',
             ha='center', color=C_FAULT, fontsize=11, fontweight='bold')

    fig.suptitle(
        'IODP Exp 405 - Site C0019  Combined Geochemistry Overview',
        color=TEXT_C, fontsize=13, fontweight='bold', y=1.00
    )

    # combined legend — SMTZ, deep reference lines, and optional MTD swatch
    ref_items = {SMTZ: ('SMTZ ~40 mbsf', '#007a5e'), **REF_LINES}
    patches   = [mpatches.Patch(color=c, label=l) for _, (l, c) in ref_items.items()]
    if mtd_catalog is not None and not mtd_catalog.empty:
        patches.append(
            mpatches.Patch(color=C_MTD, alpha=0.3,
                           label=f'MTD intervals (n={len(mtd_catalog)}, VCD-derived)')
        )
    fig.legend(
        handles=patches, loc='lower center', ncol=2,
        facecolor='#f0f0f0', edgecolor='#cccccc',
        labelcolor=TEXT_C, fontsize=8,
        bbox_to_anchor=(0.5, 0.0)
    )

    plt.subplots_adjust(bottom=0.1, top=0.95)
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {outpath}")


# =============================================================================
# MAIN
# Load data, generate all three figures, print key quantitative results.
# To overlay VCD-derived MTD boundaries, pass a mtd_catalog DataFrame to
# each plot function. See labeler.py JCORESMiner.score_to_mtd_catalog().
# =============================================================================

if __name__ == '__main__':

    print("Loading data...")
    iw_j = load_iw(IW_J)
    iw_m = load_iw(IW_M)
    gc_j = load_gc(GC_J, iw_j)
    gc_m = load_gc(GC_M, iw_m)

    print(f"  C0019J — IW: {len(iw_j)} samples  |  GC: {len(gc_j)} samples")
    print(f"  C0019M — IW: {len(iw_m)} samples  |  GC: {len(gc_m)} samples")

    # Optional: load MTD catalog from the ProboSed VCD pipeline
    # Uncomment and configure when running with full pipeline output:
    # from labeler import JCORESMiner
    # miner       = JCORESMiner(pdf_path, summary_xlsx_path)
    # backbone    = miner.build_backbone()
    # vcd_df      = miner.extract(backbone)
    # mtd_catalog = JCORESMiner.score_to_mtd_catalog(vcd_df)
    mtd_catalog = None   # set to DataFrame above to enable MTD shading

    print("\nGenerating figures...")
    plot_C0019J(
        iw_j, gc_j,
        os.path.join(OUTPUT_DIR, 'C0019J_geochemistry_profiles.png'),
        mtd_catalog=mtd_catalog
    )
    plot_C0019M(
        iw_m, gc_m,
        os.path.join(OUTPUT_DIR, 'C0019M_geochemistry_profiles.png'),
        mtd_catalog=mtd_catalog
    )
    plot_combined(
        iw_j, gc_j, iw_m, gc_m,
        os.path.join(OUTPUT_DIR, 'C0019_combined_overview.png'),
        mtd_catalog=mtd_catalog
    )

    # ── Key quantitative results ──────────────────────────────────────────────
    # Record these values with the run date — they are primary observations.
    print("\nKey quantitative results:")

    fault_ca = iw_j[iw_j['depth_m'] > 820]['Ca'].values
    if len(fault_ca):
        # background Ca = mean of the 600-800 mbsf interval:
        # below the carbonate diagenesis transition but above fault influence
        bg_ca = iw_j[
            (iw_j['depth_m'] > 600) & (iw_j['depth_m'] < 800)
        ]['Ca'].mean()
        print(f"  Ca at fault zone (core 87K, ~825 mbsf): {fault_ca[0]:.1f} mM")
        print(f"  Ca background mean (600-800 mbsf):      {bg_ca:.1f} mM")
        print(f"  Ca enrichment factor at fault:          {fault_ca[0]/bg_ca:.1f}x")

    # SMTZ defined as the shallowest depth where SO4 drops below 1 mM
    smtz_depth  = iw_m[iw_m['SO4'] < 1]['depth_m'].min()
    max_ch4_idx = gc_m['Methane'].idxmax()
    print(f"  SMTZ in C0019M (SO4 < 1 mM):            {smtz_depth:.1f} mbsf")
    print(
        f"  Peak CH4 in C0019M:                      "
        f"{gc_m.loc[max_ch4_idx, 'Methane']:.0f} ppm "
        f"at {gc_m.loc[max_ch4_idx, 'depth_m']:.1f} mbsf"
    )
