# ProboSed

[![DOI](https://zenodo.org/badge/1180640235.svg)](https://doi.org/10.5281/zenodo.20669113)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Probabilistic Sediment Stability Pipeline for IODP Multi-Expedition Analysis**

ProboSed is a Python pipeline for extracting Fabric Preservation Index (FPI) scores and process flags from IODP Visual Core Description PDFs, and for running stochastic slope stability models calibrated from those scores. It supports cross-margin comparison across passive and convergent margin settings using data from IODP Expeditions 308, 386, and 405.

---

## Repository Structure

```
ProboSed/
├── notebooks/
│   ├── ProboSed_VCD_FPI_Catalog.ipynb
│   ├── ProboSed_Stochastic_Slope_Stability_Model.ipynb
│   ├── ProboSed_Parameter_Sensitivity_Analysis.ipynb
│   ├── ProboSed_Porewater_Geochemistry_Profiles.ipynb
│   ├── ProboSed_MTD_Physical_Property_Quantification.ipynb
│   ├── ProboSed_XRF_Geochemical_Lithology_Profile.ipynb
│   ├── ProboSed_LWD_Physical_Property_Analysis.ipynb
│   ├── ProboSed_Failure_Probability_vs_Depth.ipynb
│   ├── ProboSed_Core_Image_Patch_Extractor.ipynb
│   ├── ProboSed_Block_Size_Distribution_Imagery.ipynb
│   ├── ProboSed_Block_Size_Distribution.ipynb
│   ├── ProboSed_Core_Image_Texture_Analysis.ipynb
│   ├── ProboSed_VCD_Labeler.ipynb
│   ├── ProboSed_Stochastic_Forcing_Model.ipynb
│   ├── ProboSed_XCT_Density_Integration.ipynb
│   ├── ProboSed_Resistivity_Image_Panels.ipynb
│   ├── ProboSed_Structural_Mask_vs_Stress.ipynb
│   ├── ProboSed_Sediment_Stability_Pipeline.ipynb
│   └── ProboSed_Pipeline_Summary_Figures.ipynb
│
├── slope/
│   ├── __init__.py
│   ├── stability.py            Core physics -- OU model, Lyapunov, sensitivity
│   ├── toy_model.py            Standalone simulation script
│   └── sensitivity_run.py      Sensitivity analysis script
│
├── core_ml/
│   ├── __init__.py
│   └── labeler.py              JCORESMiner, StraterMiner, Exp386Miner, VCDLabeler
│                               FPI_LEXICON, PROCESS_FLAG_LEXICON
│
├── geochem/
│   ├── __init__.py
│   └── geochem_analysis.py     Porewater and headspace gas depth profiles
│
├── transport/
│   ├── __init__.py
│   └── agents.py               Agent-based sediment transport ensemble
│
├── utils/
│   ├── __init__.py
│   └── patcher.py              Core image patch extraction utility
│
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Notebooks

All notebooks are designed to run in Google Colab without local installation. Each notebook imports from the `slope/`, `core_ml/`, `geochem/`, `transport/`, and `utils/` modules -- the `.py` files are the source of truth and the notebooks are interactive wrappers.

**Run order for dissertation analysis:** 01 → 02 → 03 → remaining notebooks in any order.

| Notebook | Purpose | Key outputs |
|---|---|---|
| VCD_FPI_Catalog | Parses VCD PDFs from Expeditions 308, 386, and 405 using OCR (tesseract) and three expedition-specific miners (StraterMiner, Exp386Miner, JCORESMiner). Assigns a two-axis classification: Fabric Preservation Index (FPI 0--3) and process flag (T/G/D/F/U) to each depth interval. Builds the multi-expedition deformation facies catalog. **Run before all other notebooks.** | `fpi_catalog_all_expeditions.csv`, `fpi_catalog_exp308.csv`, `fpi_catalog_exp386.csv`, `fpi_catalog_exp405.csv`, `fpi_summary_stats.csv`, `deformation_facies_catalog.csv`, `FPI_expedition_comparison.pdf`, `FPI_depth_profiles.pdf` |
| Stochastic_Slope_Stability_Model | Runs the coupled OU slope stability ensemble under slope-only and fault-coupled forcing. Cross-margin edition: derives sigma_q from public Vp data (PANGAEA MSCL for M0081; local CSV for Exp 308) and runs separate ensemble pairs for Expeditions 308 (passive), 386 (convergent), and 405 (convergent). Produces trajectory figures, cross-margin failure probability comparison, and Lyapunov exponent estimates. | `slope_model_trajectories_crossmargin.png`, `slope_model_pfail_crossmargin.png`, `slope_model_lyapunov_crossmargin.png`, `slope_model_crossmargin_summary.csv` |
| Parameter_Sensitivity_Analysis | One-at-a-time sensitivity analysis across physically motivated parameter ranges. Cross-margin edition: runs separately for each site's BASE parameters; uses FPI-anchored threshold values; per-site slope-only baselines replace hardcoded reference. Tests robustness of qualitative result across alpha, sigma_q, slip_mag, gamma, and theta for all three margins. | `sensitivity_results_crossmargin.csv`, `sensitivity_robustness_crossmargin.csv`, per-site sensitivity figures |
| Porewater_Geochemistry_Profiles | Five-panel porewater and headspace gas depth profiles for C0019J and C0019M. Overlays deformation intervals from the FPI catalog when the catalog CSV is available. | Three geochemistry figures |
| ProboSed_MTD_Physical_Property_Quantification | Statistical comparison (Mann-Whitney U) of resistivity, Vp, and porosity inside and outside FPI-derived deformation intervals. Validates that FPI captures real mechanical contrasts. | `C0019J_MTD_statistics.csv`, `C0019J_MTD_properties.csv`, depth profile and box plot figures |
| XRF_Geochemical_Lithology_Profile | Identifies the mafic geochemical discontinuity (SiO2 < 50 wt%) at 635.94 m CSF-A. Four-panel integrated physical and structural profile. | `C0019J_XRF_profile.png`, `C0019J_integrated_profile.png` |
| LWD_Physical_Property_Analysis | Loads the C0019H LWD LAS file and runs Mann-Whitney U comparisons of resistivity and Vp between deformed and intact intervals. | Statistical comparison table |
| Failure_Probability_vs_Depth | Monte Carlo failure probability versus depth using vane shear strength and MAD data. Quantile-based depth binning for sparse sampling. | Failure probability depth profile |
| Core_Image_Patch_Extractor | Sliding-window patch extraction from ruler-free core section photographs. Classifies patches using the FPI deformation catalog. | Patch archive, `patch_manifest.csv` |
| Block_Size_Distribution_Imagery | Block boundary detection from core photographs using brightness and lateral-variance signal. Power law fit to cumulative block size distribution. | `C0019J_block_sizes_imagery.csv`, distribution figures |
| Block_Size_Distribution | Block boundary detection from MSCL colorimetry L* profiles. Power law fit. | Block size distribution, power law fit |
| Core_Image_Texture_Analysis | GLCM texture metrics (contrast, correlation, energy, homogeneity), orientation statistics, and autocorrelation length from core image patches. | Texture metric depth profiles |
| VCD_Labeler | Interactive ipywidgets labeling interface for core image patches. Records fabric type, process flag, confidence, and notes. | `C0019J_VCD_labels.csv` |
| Stochastic_Forcing_Model | Two-component SDE model with optional SRCMOD Tohoku finite-fault slip-rate forcing. Compares response-only and seismically forced ensemble failure probabilities. | Ensemble figures, `s_forcing.csv` |
| XCT_Density_Integration | DICOM CT Hounsfield unit profiles per deformation interval from XCT scan data. | Density profiles by interval |
| Resistivity_Image_Panels | RAB resistivity image panel extraction for MTD-9 and MTD-10 from Hole C0019H. | Resistivity image panels |
| Structural_Mask_vs_Stress | Structural overprint flags versus Conin et al. JFAST/JTRACK borehole breakout stress data. | Structural-stress comparison figure |
| Sediment_Stability_Pipeline | **Legacy -- retained for provenance only.** Early VCD lexicon parser and geotechnical merge pipeline using the original single-axis stability score (0--3). Superseded by ProboSed_VCD_FPI_Catalog, which implements the two-axis FPI + process flag system. Do not use as part of the active analysis pipeline. | Stability depth profile (legacy) |
| Pipeline_Summary_Figures | Graphviz workflow diagram and two-panel mechanical regime depth profile. | Workflow diagram, regime figure |

---

## FPI + Process Flag System

The two-axis classification system used throughout ProboSed:

**FPI axis (observable fabric state):**

| FPI | Label | Physical meaning |
|---|---|---|
| 3 | Intact | Primary fabric fully preserved |
| 2 | Partially preserved | Minor disruption, fabric recognizable |
| 1 | Fabric destroyed | Scaly/shear surfaces, original fabric lost |
| 0 | Structureless | No fabric, homogeneous or fluidized |

**Process flag (observable process type):** T = Tectonic | G = Gravitational | D = Depositional | F = Fluid escape | U = Undetermined

**MTD-context rule:** fault terms appearing within mass-transport context are downgraded T -> G. Faults within MTDs are soft-sediment features, not tectonic structures.

**SDE threshold mapping:** theta(FPI) = clip(FPI x 2/3, 0, 2)

---

## Expedition Coverage

| Expedition | Site | Margin | Miner | gamma | sigma_q |
|---|---|---|---|---|---|
| 308 | U1322, Ursa Basin | Passive | StraterMiner | 0.6 | Derived from local Vp CSV |
| 386 | M0081, Japan Trench | Convergent | Exp386Miner | 1.5 | Derived from PANGAEA MSCL (DOIs: PANGAEA.974882, 974883, 974885) |
| 405 | C0019J, Japan Trench | Convergent | JCORESMiner | 1.0 | 0.6 (calibrated from LWD Vp 1550--1750 m/s) |

Expedition 386 Holes M0081E and M0081F are excluded from Vp calibration due to a calibration flag in operator notes (Vp overestimated in sections 1--29, Strasser et al. 2025).

---
## Acknowledgements

This work was developed with the guidance and support of my advisors, committee members, and collaborators.

### Advisors
- Dr. Brendan Crowell (Ohio State University)  
- Dr. Jill Leonard-Pingel (Ohio State University)  

### Committee Members
- Dr. David Cole (Ohio State University)  
- Dr. Lawrence Krissek (Ohio State University)  
- Dr. Myra Keep (University of Western Australia)  
- Dr. Christine Regalla (Co-Chief Scientist, IODP Expedition 405)  

I am especially grateful to Dr. Christine Regalla for her contributions as Co-Chief Scientist of IODP Expedition 405 and for her insight into the scientific framing of this work.

### Data & Expedition Support
I acknowledge the scientific teams and data providers associated with IODP Expeditions 308, 386, and 405.

---
## Citation

If you use ProboSed, please cite:

Castillo, R. et al. (in prep). When Slip Happens: From Megathrust Rupture to Nonlinear Shallow Prism Reorganization in the 2011 Tohoku-Oki Earthquake. PhD dissertation, Ohio State University.

See also `CITATION.cff` for machine-readable citation metadata.
