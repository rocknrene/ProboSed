# ProboSed

**Probabilistic Sediment Transport and Slope Failure Modeling**
IODP Expeditions 386 and 405 — Japan Trench Frontal Prism

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Notebooks

| # | Notebook | Open |
|---|---|---|
| 01 | JCORES VCD Mining and MTD Catalog | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/01_labeling.ipynb) |
| 02 | Stochastic Slope Stability Model | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/02_toy_model.ipynb) |
| 03 | Parameter Sensitivity Analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/03_sensitivity.ipynb) |
| 04 | Porewater Geochemistry Profiles | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/04_geochem.ipynb) |
| 05 | MTD Physical Property Quantification | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/05_mtd_quantification.ipynb) |
| 06 | XRF Geochemical Lithology Profile | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/06_xrf_profile.ipynb) |
| 07 | LWD Physical Property Analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/07_lwd_analysis.ipynb) |
| 08 | Failure Probability vs Depth | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/08_failure_probability.ipynb) |
| 09 | Core Image Patch Extractor | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/09_patch_extractor.ipynb) |
| 10 | Block Size Distribution (Imagery) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/10_block_size_imagery.ipynb) |
| 11 | Block Size Distribution (Colorimetry) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/11_block_size_colorimetry.ipynb) |
| 12 | Core Image Texture Analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/12_texture_analysis.ipynb) |
| 13 | VCD Interactive Labeler | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/13_vcd_labeler.ipynb) |
| 14 | Stochastic Forcing Model | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/14_stochastic_forcing.ipynb) |
| 15 | XCT Density Integration | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/15_xct_density.ipynb) |
| 16 | Resistivity Image Panels | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/16_resistivity_panels.ipynb) |
| 17 | Structural Mask vs Stress | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/17_structural_stress.ipynb) |
| 18 | Sediment Stability Pipeline | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/18_stability_pipeline.ipynb) |
| 19 | Pipeline Summary Figures | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/19_summary_figures.ipynb) |

---

## Citation

If ProboSed is used in published research, please cite:

> Castillo, R. (2027). *Slip Happens: Probabilistic Modeling of Submarine Mass Transport Deposits and Sediment Routing in Convergent Margin Basins.* PhD Dissertation, The Ohio State University.

A `CITATION.cff` file is included for automated citation tools.

---

## Overview

Deterministic slope-failure models assume a single failure condition, yet natural sedimentary systems are inherently stochastic. Variations in pore pressure, permeability architecture, grain-size distribution, and seismic forcing produce ensembles of possible failure pathways rather than a single deterministic outcome.

ProboSed provides a probabilistic framework for quantifying submarine slope instability and sediment transport at convergent margins. The pipeline connects Visual Core Description (VCD) observations from IODP core material to a physically motivated stochastic model, validated against independent geophysical and geochemical datasets.

The framework is applied to Site C0019 of the Japan Trench frontal prism, where the 2011 Mw 9.0 Tohoku-oki earthquake produced ~50 m of coseismic slip at the plate boundary décollement (Fulton et al., 2013; Chester et al., 2013).

---

## Scientific Background

### Governing Equations

Slope displacement *q(t)* and fault slip *s(t)* are modeled as coupled Ornstein-Uhlenbeck stochastic differential equations:

```
ds = -k_f · s · dt + σ_s · dW_s
dq = (-γq + αs) · dt + σ_q · dW_q
```

Slope failure is defined as a first-passage problem:

```
τ = inf{ t > 0 : q(t) > θ(x) }
```

where the failure threshold θ(x) is derived from the VCD stability index:

```
θ(x) = clip( x · 2/3, 0, 2 )    x ∈ {0, 1, 2, 3}
```

### VCD Stability Index

Disturbance categories from IODP standardized VCD classification map to a 0–3 stability score:

| Score | Fabric class | Sedimentary interpretation | What to watch for that would make this structural instead | Failure threshold θ |
|---|---|---|---|---|
| 3 | Intact bedding | Undisturbed hemipelagic or turbidite depositional fabric | None — intact bedding is unambiguous | 2.00 |
| 2 | Coherent block | Partially remobilized; coherent blocks with preserved internal stratigraphy suggest early-stage failure or distal MTD | Tilted but unbroken bedding could be fault drag — check for slickensides or calcite veins | 1.33 |
| 1 | Scaly fabric | Pervasive shear fabric consistent with basal shear zone of MTD or active failure surface | High risk of structural misclassification — scaly fabric also diagnostic of fault gouge; needs physical property cross-check | 0.67 |
| 0 | Slurried / MTD | Complete remobilization; homogenized matrix indicates flow or liquefaction | Brecciation from coseismic slip can look identical — check for hydrothermal mineralization or sharp bounding surfaces | 0.00 |

The linear mapping is the maximum-entropy choice given only the constraint that threshold increases monotonically with stability score.

### Physical Parameter Constraints

Model parameters are informed by published IODP proceedings:

| Parameter | Value | Physical basis |
|---|---|---|
| σ_q = 0.6 | Slope noise amplitude | Frontal prism Vp = 1550–1750 m/s, resistivity = 0.5–1.8 Ω·m (Exp 405 LWD, Unit I) |
| γ = 1.0 | Slope damping | Moderate restoring force; higher values (1.5–2.0) for seismically strengthened hemipelagic sediment (Exp 386) |
| α = 0.5 | Fault-slope coupling | Pc' = 17 MPa overconsolidation supports non-zero coupling (Exp 343, Valdez et al. 2015) |
| slip_mag = 3.0 | Mainshock impulse | Scaled from ~50 m coseismic slip at C0019 décollement (Fulton et al. 2013) |

---

## Repository Structure

```
ProboSed/
├── notebooks/
│   ├── 01_labeling.ipynb           JCORES VCD mining and MTD catalog
│   ├── 02_toy_model.ipynb          Stochastic slope stability simulation
│   ├── 03_sensitivity.ipynb        Parameter sensitivity analysis
│   ├── 04_geochem.ipynb            Porewater geochemistry profiles
│   ├── 05_mtd_quantification.ipynb MTD physical property quantification
│   ├── 06_xrf_profile.ipynb        XRF geochemical lithology profile
│   ├── 07_lwd_analysis.ipynb       LWD physical property analysis
│   ├── 08_failure_probability.ipynb Failure probability vs depth
│   ├── 09_patch_extractor.ipynb    Core image patch extraction
│   ├── 10_block_size_imagery.ipynb  Block size distribution from imagery
│   ├── 11_block_size_colorimetry.ipynb Block size distribution from L* profiles
│   ├── 12_texture_analysis.ipynb   GLCM texture analysis of core images
│   ├── 13_vcd_labeler.ipynb        Interactive VCD labeling widget
│   ├── 14_stochastic_forcing.ipynb Stochastic forcing model with SRCMOD data
│   ├── 15_xct_density.ipynb        XCT density integration per MTD section
│   ├── 16_resistivity_panels.ipynb RAB resistivity image panels
│   ├── 17_structural_stress.ipynb  Structural mask vs borehole breakout stress
│   ├── 18_stability_pipeline.ipynb Sediment stability pipeline (legacy)
│   └── 19_summary_figures.ipynb    Pipeline workflow and summary figures
│
├── slope/
│   ├── __init__.py
│   ├── stability.py                Core physics — OU model, Lyapunov, sensitivity
│   ├── toy_model.py                Standalone simulation script
│   └── sensitivity_run.py          Sensitivity analysis script
│
├── core_ml/
│   ├── __init__.py
│   └── labeler.py                  JCORESMiner and VCDLabeler
│
├── geochem/
│   ├── __init__.py
│   └── geochem_analysis.py         Porewater and headspace gas depth profiles
│
├── transport/
│   ├── __init__.py
│   └── agents.py                   Agent-based sediment transport ensemble
│
├── utils/
│   ├── __init__.py
│   └── patcher.py                  Core image patch utility
│
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Notebooks

All notebooks are designed to run in Google Colab without local installation. Each notebook imports from the `slope/`, `core_ml/`, `geochem/`, `transport/`, and `utils/` modules — the `.py` files are the source of truth and the notebooks are interactive wrappers.

| Notebook | Purpose | Key outputs |
|---|---|---|
| 01_labeling | Extracts stability scores from JCORES VCD PDFs using standardized IODP disturbance terminology. Identifies MTD boundaries as contiguous intervals where stability ≤ 1. Supports all five Site C0019 holes; outputs are tagged by hole. | `C0019J_VCD_stability_log.csv`, `C0019J_MTD_catalog.csv`, `C0019J_stability_profile.png` |
| 02_toy_model | Runs the coupled OU slope stability ensemble under slope-only and fault-coupled forcing. Produces trajectory figures, failure probability comparison, and Lyapunov exponent estimate. | Three dissertation figures, failure probability, Lyapunov exponent |
| 03_sensitivity | One-at-a-time parameter sensitivity analysis across physically motivated ranges. Tests robustness of the qualitative result across α, σ_q, slip_mag, γ, and θ. | `sensitivity_results.csv`, sensitivity figure, robustness table |
| 04_geochem | Produces five-panel porewater and headspace gas depth profiles for C0019J and C0019M. Overlays MTD intervals from notebook 01 when the catalog CSV is available. | Three geochemistry figures |
| 05_mtd_quantification | Statistical comparison of resistivity, Vp, and porosity inside and outside VCD-derived MTD intervals. Validates that the stability index captures real mechanical contrasts. | `C0019J_MTD_statistics.csv`, `C0019J_MTD_properties.csv`, depth profile and box plot figures |
| 06_xrf_profile | Identifies the depth of the mafic geochemical discontinuity (SiO2 < 50 wt%) and produces the integrated four-panel physical/structural profile. | `C0019J_XRF_profile.png`, `C0019J_integrated_profile.png` |
| 07_lwd_analysis | Loads the LWD LAS file from Site C0019H and runs Mann-Whitney U comparisons of resistivity and Vp between MTD and stable matrix intervals. | Statistical comparison table |
| 08_failure_probability | Monte Carlo failure probability versus depth using vane shear strength and MAD data. Uses quantile-based depth binning to accommodate sparse sampling. | Failure probability depth profile |
| 09_patch_extractor | Extracts sliding-window image patches from ruler-free core section photographs, classified as MTD or intact using the MTD catalog. | Patch archive (MTD/intact subdirectories), `patch_manifest.csv` |
| 10_block_size_imagery | Detects block boundaries in core photographs using a combined brightness and lateral-variance signal. Fits a power law to the cumulative block size distribution. | `C0019J_block_sizes_imagery.csv`, distribution figures |
| 11_block_size_colorimetry | Block boundary detection from MSCL colorimetry L* profiles. Power law fit to test scale-invariant fragmentation. | Block size distribution, power law fit |
| 12_texture_analysis | GLCM texture metrics (contrast, correlation, energy, homogeneity), orientation statistics, and autocorrelation length from core image patches. | Texture metric depth profiles |
| 13_vcd_labeler | Interactive ipywidgets labeling interface for core image patches. Records lithology, disturbance classification, confidence, and notes to CSV. | `C0019J_VCD_labels.csv` |
| 14_stochastic_forcing | Two-component SDE model with optional SRCMOD Tohoku finite-fault slip-rate forcing. Compares response-only and forced ensemble failure probabilities. | Ensemble figures, `s_forcing.csv` |
| 15_xct_density | DICOM CT Hounsfield unit profiles per MTD section from XCT scan data. | Density profiles by MTD interval |
| 16_resistivity_panels | RAB resistivity image panel extraction for MTD-9 and MTD-10 from Hole C0019H. | Resistivity image panels |
| 17_structural_stress | Structural overprint flags versus Conin et al. JFAST/JTRACK breakout stress data. | Structural-stress comparison figure |
| 18_stability_pipeline | Early VCD lexicon parser, geotechnical data merge, and stability profile figure generation (legacy pipeline). | Stability depth profile |
| 19_summary_figures | Graphviz workflow diagram and two-panel mechanical regime depth profile. | `vcd_pipeline.png`, regime figure |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rocknrene/ProboSed.git
cd ProboSed
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install probosed
```

---

## Quickstart

### Slope stability simulation

```python
from slope.stability import run_ensemble, calculate_lyapunov, threshold_from_vcd

# Run fault-coupled ensemble
# Parameters informed by IODP Expeditions 343, 405, 386
q, s, p_fail, transported = run_ensemble(
    N_paths     = 10_000,
    gamma       = 1.0,      # slope damping
    sigma_q     = 0.6,      # slope noise (weak frontal prism material)
    alpha       = 0.5,      # fault-slope coupling
    slip_mag    = 3.0,      # mainshock impulse (~50 m coseismic slip, scaled)
    threshold   = 1.0,      # failure threshold
    mainshock_t = 5.0,
)

print(f"Failure probability: {p_fail:.3f}")

# Estimate Lyapunov exponent
lyapunov, log_growth = calculate_lyapunov(q, dt=0.01, warmup_fraction=0.01)
print(f"Lyapunov exponent:   {lyapunov:.4f} per unit time")

# Map a VCD stability score to a failure threshold
theta = threshold_from_vcd(2)   # coherent block -> theta = 1.33
print(f"Threshold (score 2): {theta:.2f}")
```

### JCORES VCD extraction

```python
from core_ml.labeler import JCORESMiner

miner    = JCORESMiner('405-C0019J_VCD.pdf', 'Summary_C0019J.xlsx')
backbone = miner.build_backbone()
vcd_df   = miner.extract(backbone)
mtds     = JCORESMiner.score_to_mtd_catalog(vcd_df, stability_threshold=1)

print(f"MTD intervals identified: {len(mtds)}")
print(mtds)
```

### Geochemistry figures

```python
from geochem.geochem_analysis import load_iw, load_gc, plot_C0019J
import pandas as pd

iw_j = load_iw('SummarySheet-IW-Hole_Exp405_C0019J_250312.xlsx')
gc_j = load_gc('SummarySheet-GC_200_160206_Exp405_C0019J.xlsm', iw_j)

# optional: overlay MTD boundaries from the VCD pipeline
mtd_catalog = pd.read_csv('C0019J_MTD_catalog.csv')

plot_C0019J(iw_j, gc_j, 'C0019J_geochemistry_profiles.png',
            mtd_catalog=mtd_catalog)
```

### Agent-based sediment transport

```python
from transport.agents import SedimentAgentModel, forcing_from_slope, calculate_clast_distribution

# compute physically motivated forcing from slope geometry
# Japan Trench frontal prism: ~8 degree slope, moderate pore pressure
forcing = forcing_from_slope(
    slope_angle_deg     = 8.0,    # degrees
    pore_pressure_ratio = 0.35,   # lambda — moderate overpressure
)

# run grain transport ensemble
model   = SedimentAgentModel(
    settling_velocity = 0.02,   # m/s — fine silt, D50 ~20 microns
    drag_coeff        = 0.5,    # natural irregular grains
)
results = model.run(n_agents=10_000, n_steps=500, forcing=forcing)
model.summary(results)

# compute clast distribution (proxy for MTD deposit thickness profile)
counts, edges, centers = calculate_clast_distribution(results['final_positions'])
```

### Core image patch extraction

```python
from utils.patcher import slice_core_image, batch_slice

# slice a single core section scan into 256×256 patches
result = slice_core_image(
    image_path    = 'C0019J_section_14K_1.tif',
    output_folder = 'patches/14K_1/',
    patch_size    = 256,
    overlap       = 0,
)
print(f"{result['n_patches']} patches written to {result['output_folder']}")

# or process a whole folder of core scans at once
batch = batch_slice(
    input_folder = 'core_scans/',
    output_root  = 'patches/',
    patch_size   = 256,
)
print(f"{batch['total_patches']} total patches from {batch['n_images']} images")
```

---

## Validation Strategy

The stability index is validated against five independent datasets, each measured by different instruments with different error structures:

| Dataset | Physical basis | Expected signature at MTD intervals |
|---|---|---|
| P-wave velocity (Vp) | Acoustic impedance — sensitive to porosity and cementation | Lower Vp in weak, disaggregated material |
| Electrical resistivity | Pore fluid connectivity | Lower resistivity in high-porosity remobilized zones |
| Porewater geochemistry | Fluid advection through permeable failure surfaces | SO4, Ca, and alkalinity anomalies co-located with MTD boundaries |
| Borehole breakouts | In-situ stress field from wellbore deformation | Stress reorientation at mechanical discontinuities |
| Lithostratigraphic column | Sedimentary facies from hand-specimen description | Chaotic or structureless intervals at modeled failure zones |

Convergence of anomalies across independent datasets at the same stratigraphic horizons provides evidence that the stability index captures real mechanical contrasts rather than artifacts of any single measurement method.

---

## Roadmap

- v0.1 — Probabilistic slope-failure framework ✓
- v0.2 — JCORES VCD extraction pipeline ✓
- v0.3 — Porewater geochemistry integration ✓
- v0.4 — Parameter sensitivity analysis ✓
- v0.5 — Agent-based sediment transport ensemble ✓
- v0.6 — Core image patch extraction utility ✓
- v0.7 — Block size distribution from core imagery ✓
- v0.8 — GLCM texture analysis pipeline ✓
- v0.9 — LWD and XRF geochemical integration ✓
- v0.10 — Interactive VCD labeling widget ✓
- v1.0 — Integrated MTD simulation framework

---

## Scientific Applications

ProboSed is designed for research in:

- Submarine landslides and mass-transport deposits
- Earthquake-triggered sediment transport at convergent margins
- Probabilistic slope stability modeling
- Marine sediment routing systems
- IODP core data integration with geophysical observations

Development uses core and logging data from the Japan Trench margin, IODP Expeditions 386 and 405, Site C0019.

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Development of ProboSed is part of doctoral research at The Ohio State University, School of Earth Sciences.

**Advisors**
Dr. Brendan Crowell — Co-Advisor
Dr. Jill Leonard-Pingel — Co-Advisor

**Dissertation Committee**
Dr. Cole · Dr. Keep · Dr. Krissek

**Field Science**
Dr. Christine Regalla — Co-Chief Scientist, IODP Expedition 405

**Data**
This work uses data from the International Ocean Discovery Program (IODP), Expeditions 386 and 405, Japan Trench margin.

---

## Selected References

Chester, F.M., et al. (2013). Structure and composition of the plate-boundary slip zone for the 2011 Tohoku-oki earthquake. *Science*, 342(6163), 1208–1211.

Fulton, P.M., et al. (2013). Low coseismic friction on the Tohoku-oki fault determined from temperature measurements. *Science*, 342(6163), 1214–1217.

Valdez, R.D., et al. (2015). Data report: permeability and consolidation behavior of sediments from the northern Japan Trench subduction zone, IODP Site C0019. *Proc. IODP*, 343/343T.

Expedition 405 Scientists (2025). Site C0019. In *Proc. IODP*, 405.

Alstott, J., Bullmore, E., & Plenz, D. (2014). powerlaw: A Python package for analysis of heavy-tailed distributions. *PLOS ONE*, 9(1), e85777.

Haralick, R.M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, 3(6), 610–621.
