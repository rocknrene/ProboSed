"""
labeler.py
==========
Visual Core Description (VCD) labeling tools for IODP core imagery.

Provides two classes:

VCDLabeler
    Interactive ipywidgets-based labeling interface for core image patches.
    Designed for use in Colab or Jupyter. Presents each image patch and
    records lithology, disturbance type, observer, confidence, and notes
    to a CSV for downstream use.

JCORESMiner
    Programmatic extraction of stability information from JCORES-format
    VCD PDFs. JCORES PDFs are machine-readable structured documents (not
    scanned handwriting), so regex extraction works reliably. Uses the
    core summary Excel file to assign accurate CSF-A depths.

Key distinction from previous version:
    The old miner was written for handwritten VCD forms and used regex
    patterns that matched form field labels ('core:', 'section:').
    JCORES PDFs use structured digital text with different conventions.
    The new miner targets JCORES-specific patterns and terminology while
    the handwritten sedimentology library is in development. 

Stability scoring:
    All disturbance label -> stability score mappings use the canonical
    STABILITY_MAP imported from stability.py. This ensures consistency
    between the interactive labeler pathway and the automated miner pathway.
    The JCORES_LEXICON (used by JCORESMiner only) remains separate because
    it maps JCORES standardized drilling disturbance codes, which are a
    different vocabulary from the VCDLabeler disturbance categories.
"""

import os
import re
import pandas as pd
import numpy as np

# Import the canonical stability map from stability.py.
# This is the single source of truth for disturbance label -> score mapping.
# Both the interactive VCDLabeler and the automated JCORESMiner pathways
# use this map to ensure consistent scoring across the pipeline.
from stability import STABILITY_MAP


# =============================================================================
# JCORES DRILLING DISTURBANCE LEXICON
# =============================================================================

# JCORES uses IODP standardized drilling disturbance codes in machine-readable PDFs.
# These are distinct from the VCDLabeler disturbance categories (which describe
# sediment fabric), and are therefore kept as a separate lexicon.
# Scores follow the same 0-3 scale as STABILITY_MAP:
#   0 = completely disrupted, primary fabric absent
#   1 = partially disrupted, some fabric preserved
#   2 = minor disturbance, fabric largely intact
#   3 = undisturbed, primary sedimentary fabric preserved
# The extract() method applies a lowest-score-wins rule: if multiple terms
# appear on a page, the most severely disturbed term governs the score.
JCORES_LEXICON = {
    # Score 0 — drilling has completely destroyed primary fabric
    'soupy':                0,   # water-saturated, completely remobilized
    'slurried':             0,   # completely disaggregated by drilling
    'flow-in':              0,   # material flowed into borehole
    'highly disturbed':     0,   # general severe disturbance code
    'completely disturbed': 0,   # general severe disturbance code
    'void':                 0,   # no material present
    # Score 1 — partially disrupted, some fabric preserved
    'moderately disturbed': 1,   # intermediate disturbance
    'biscuit':              1,   # rotary biscuiting, partial disruption
    'fall-in':              1,   # material collapsed into borehole
    'slightly disturbed':   1,   # minor to moderate disturbance
    'fractured':            1,   # fracturing without complete remobilization
    'brecciated':           1,   # brecciation, significant disruption
    # Score 2 — minor disturbance, primary fabric largely intact
    'mottled':              2,   # bioturbation or minor disturbance
    'slightly':             2,   # partial qualifier for minor disturbance
    'minor':                2,   # minor disturbance qualifier
    # Score 3 — undisturbed, primary sedimentary fabric preserved
    'undisturbed':          3,   # no drilling disturbance observed
    'undeformed':           3,   # fabric intact, no deformation
    'bioturbated':          3,   # biogenic disturbance, not drilling-induced
    'laminated':            3,   # primary lamination preserved
    'intact':               3,   # intact primary fabric
}

# Dropdown options presented in the interactive VCDLabeler widget.
# These are the user-facing labels that map to STABILITY_MAP keys.
# The disturbance options match the keys in STABILITY_MAP exactly
# (after lowercasing) to ensure consistent score assignment.
LABELER_LITHOLOGIES = [
    'Siliceous mudstone',   # dominant lithology in Japan Trench frontal prism
    'Pelagic clay',         # background hemipelagic sediment
    'Ash / Tuff',           # tephra layers, important stratigraphic markers
    'Chaotic matrix',       # mass transport deposit matrix
    'Fault gouge',          # fault zone material
    'Breccia',              # brecciated fault or slope material
    'Chert',                # siliceous chert, common in Pacific plate sequence
    'Other',                # catch-all for unlisted lithologies
]

LABELER_DISTURBANCES = [
    'Intact bedding',       # maps to stability score 3 via STABILITY_MAP
    'Coherent block',       # maps to stability score 2 via STABILITY_MAP
    'Scaly fabric',         # maps to stability score 1 via STABILITY_MAP
    'Slurried / MTD',       # maps to stability score 0 via STABILITY_MAP
    'Biscuit',              # maps to stability score 1 via STABILITY_MAP
    'Fall-in',              # maps to stability score 1 via STABILITY_MAP
    'Brecciated',           # maps to stability score 1 via STABILITY_MAP
    'Void',                 # maps to stability score 0 via STABILITY_MAP
]


# =============================================================================
# VCDLabeler — Interactive Widget for Core Image Labeling
# =============================================================================

class VCDLabeler:
    """
    Interactive labeling interface for core image patches.

    Displays each image in a folder sequentially and records:
      - Lithology
      - Disturbance type (maps to stability score in downstream analysis)
      - Observer identifier
      - Confidence (1-5)
      - Free-text notes

    Labels are appended to a CSV after each image, so progress is saved
    even if the session is interrupted.

    Disturbance labels are converted to numeric stability scores using the
    canonical STABILITY_MAP from stability.py, ensuring consistency with
    the JCORESMiner pathway.

    Usage (in Colab or Jupyter):
        labeler = VCDLabeler(
            patch_folder='/path/to/images/',
            output_csv='/path/to/labels.csv',
            authors=['Observer1', 'Observer2']
        )
        labeler.start()
    """

    def __init__(self, patch_folder, output_csv, authors):
        self.patch_folder = patch_folder   # directory containing image patches
        self.output_csv   = output_csv     # path to output CSV file
        self.authors      = authors        # list of observer identifiers

        # Collect all image files in the patch folder — supports common formats
        # including TIFF for high-resolution core line-scan imagery
        self.patch_files = sorted([
            f for f in os.listdir(patch_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        self.current_idx = 0   # index of the currently displayed image

        # Create the output CSV with headers if it does not already exist
        # Append mode is used in _save_entry so existing labels are not overwritten
        if not os.path.exists(self.output_csv):
            pd.DataFrame(columns=[
                'filename', 'lithology', 'disturbance',
                'author', 'confidence', 'notes'
            ]).to_csv(self.output_csv, index=False)

    def start(self):
        """Launch the interactive labeling widget in a Jupyter or Colab environment."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            raise ImportError(
                "ipywidgets is required for the interactive labeler. "
                "Install with: pip install ipywidgets"
            )

        # Build widget components
        self.img_w    = widgets.Image(width=500)                           # image display
        self.auth_w   = widgets.Dropdown(
            options=self.authors, description='Observer:'
        )
        self.lith_w   = widgets.Dropdown(
            options=LABELER_LITHOLOGIES, description='Lithology:'
        )
        self.dist_w   = widgets.Dropdown(
            options=LABELER_DISTURBANCES, description='Disturbance:'
        )
        self.conf_w   = widgets.IntSlider(
            value=3, min=1, max=5, description='Confidence:'
        )
        self.note_w   = widgets.Text(
            placeholder='e.g. visible shear planes, ash layer',
            description='Notes:'
        )
        self.save_btn = widgets.Button(
            description='Save & Next', button_style='success'
        )
        self.skip_btn = widgets.Button(
            description='Skip', button_style='warning'
        )
        self.prog_w   = widgets.Label(value='')   # progress indicator

        # Register button callbacks
        self.save_btn.on_click(self._save_entry)
        self.skip_btn.on_click(self._skip_entry)

        # Load the first image
        self._load_image()

        # Display the assembled widget layout
        display(widgets.VBox([
            self.prog_w,
            self.img_w,
            self.auth_w,
            self.lith_w,
            self.dist_w,
            self.conf_w,
            self.note_w,
            widgets.HBox([self.save_btn, self.skip_btn])
        ]))

    def _load_image(self):
        """Load the current image into the widget display and update progress label."""
        if self.current_idx < len(self.patch_files):
            fname = self.patch_files[self.current_idx]
            path  = os.path.join(self.patch_folder, fname)
            with open(path, 'rb') as f:
                self.img_w.value = f.read()
            self.prog_w.value = (
                f"Patch {self.current_idx + 1} / {len(self.patch_files)}: {fname}"
            )
        else:
            # All patches in this session have been labeled or skipped
            self.prog_w.value = "Labeling complete for this session."
            self.img_w.value  = b''

    def _save_entry(self, b):
        """Save the current label entry to CSV and advance to the next image."""
        if self.current_idx >= len(self.patch_files):
            return   # guard against button click after session completion

        # Build one-row DataFrame for the current label
        row = pd.DataFrame([{
            'filename':    self.patch_files[self.current_idx],
            'lithology':   self.lith_w.value,
            'disturbance': self.dist_w.value,
            'author':      self.auth_w.value,
            'confidence':  self.conf_w.value,
            'notes':       self.note_w.value,
        }])

        # Append to CSV — header=False because headers were written at init
        row.to_csv(self.output_csv, mode='a', header=False, index=False)

        self.current_idx += 1
        self._load_image()

    def _skip_entry(self, b):
        """Skip the current image without saving a label."""
        self.current_idx += 1
        self._load_image()

    @staticmethod
    def labels_to_stability(csv_path):
        """
        Load a completed labels CSV and add a numeric stability_score column.

        Uses the canonical STABILITY_MAP imported from stability.py to convert
        disturbance labels to numeric scores. This ensures the VCDLabeler
        pathway and the JCORESMiner pathway produce consistent scores.

        Unrecognized disturbance labels return NaN and should be inspected
        before downstream use — they indicate a label not covered by
        STABILITY_MAP and may indicate a data entry inconsistency.

        Disturbance labels map to stability scores as follows (see STABILITY_MAP):
          Intact bedding  -> 3   (undisturbed primary fabric)
          Coherent block  -> 2   (fabric largely intact)
          Scaly fabric    -> 1   (fabric partially destroyed)
          Slurried / MTD  -> 0   (completely remobilized)
          Biscuit         -> 1   (partial drilling disturbance)
          Fall-in         -> 1   (borehole collapse material)
          Brecciated      -> 1   (significant disruption)
          Void            -> 0   (no material)

        Parameters
        ----------
        csv_path : str
            Path to the labels CSV produced by the VCDLabeler widget.

        Returns
        -------
        pd.DataFrame
            All original columns plus a stability_score column (0-3 or NaN).
        """
        df         = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Map disturbance labels to scores using the canonical STABILITY_MAP
        # Labels are lowercased and stripped to handle minor formatting differences
        df['stability_score'] = (
            df['disturbance'].str.strip().str.lower().map(STABILITY_MAP)
        )

        # Report any unrecognized labels so they can be corrected before use
        unrecognized = df[df['stability_score'].isna()]['disturbance'].unique()
        if len(unrecognized) > 0:
            print(
                f"Warning: {len(unrecognized)} unrecognized disturbance label(s) "
                f"returned NaN stability_score:\n  {unrecognized}\n"
                f"Check spelling against STABILITY_MAP keys in stability.py."
            )

        return df


# =============================================================================
# JCORESMiner — Programmatic VCD Extraction from JCORES PDFs
# =============================================================================

class JCORESMiner:
    """
    Extracts stability information from JCORES-format VCD PDFs.

    JCORES PDFs are structured digital documents generated by the shipboard
    database system. Unlike scanned handwritten VCDs, they contain machine-
    readable text with consistent formatting, making regex extraction reliable.
    These are still compiled by expert scientists.

    The miner requires a core summary Excel file to assign accurate CSF-A
    depths. Without depth anchors, core identifiers alone are not sufficient
    to reconstruct a continuous depth profile.

    Scoring uses JCORES_LEXICON (drilling disturbance codes) rather than
    STABILITY_MAP (fabric-based VCD categories). These vocabularies are
    distinct: JCORES_LEXICON captures how drilling altered the material,
    while STABILITY_MAP captures the primary sedimentary fabric state.
    Both map to the same 0-3 scale for compatibility with threshold_from_vcd().

    Usage:
        miner = JCORESMiner(
            pdf_path='/path/to/405-C0019J_VCD.pdf',
            summary_xlsx_path='/path/to/405-C0019J_CoreSummary.xlsx'
        )
        backbone = miner.build_backbone()
        df       = miner.extract(backbone)
        df.to_csv('C0019J_stability_log.csv', index=False)
    """

    # JCORES core ID pattern: matches '405-C0019J-14K' or 'C0019J-14K'
    # Captures the core label (e.g., '14K') after the last hyphen
    CORE_PATTERN = re.compile(
        r'(?:405-)?C\d{4}[A-Z]-(\d+[A-Z])', re.IGNORECASE
    )

    # JCORES section pattern: section numbers follow tool type codes (W, H, R, F)
    # e.g., '1W' (whole round), '2H' (half round), 'CC' (core catcher)
    SECTION_PATTERN = re.compile(
        r'\b(\d{1,2}[WHRF]|CC)\b', re.IGNORECASE
    )

    # CSF-A depth pattern: JCORES prints depths as e.g. '207.45 m CSF-A'
    DEPTH_PATTERN = re.compile(
        r'(\d{1,3}\.\d{1,4})\s*m?\s*CSF-?A', re.IGNORECASE
    )

    def __init__(self, pdf_path, summary_xlsx_path):
        self.pdf_path           = pdf_path             # path to JCORES VCD PDF
        self.summary_xlsx_path  = summary_xlsx_path    # path to core summary Excel file

    def _load_summary(self):
        """
        Load and clean the core summary Excel file.

        The summary provides authoritative top and bottom depths for each core
        and the section count required to interpolate per-section depths.
        Header row is at row index 5 (0-indexed) in standard IODP core summary format.

        Returns
        -------
        pd.DataFrame with columns: core_id, top_m, bottom_m, n_sections
        """
        raw         = pd.read_excel(self.summary_xlsx_path, header=5)
        raw.columns = [c.replace('\n', ' ').strip() for c in raw.columns]

        # Drop rows without a core identifier or a valid section count
        raw = raw.dropna(subset=['Core curated'])
        raw['Number of sections'] = pd.to_numeric(
            raw['Number of sections'], errors='coerce'
        )
        raw = raw.dropna(subset=['Number of sections'])
        raw['Number of sections'] = raw['Number of sections'].astype(int)

        # Extract the core label from the full IODP-style core identifier
        # e.g., 'C0019J-14K' -> '14K'
        raw['core_id'] = raw['Core curated'].astype(str).apply(
            lambda x: x.split('-')[-1].upper() if '-' in x else x.upper()
        )

        return raw[['core_id', 'Top depth [m CSF-A]',
                    'Bottom depth [m CSF-A]', 'Number of sections']].rename(columns={
            'Top depth [m CSF-A]':    'top_m',
            'Bottom depth [m CSF-A]': 'bottom_m',
            'Number of sections':     'n_sections',
        })

    def build_backbone(self):
        """
        Build a depth backbone: one row per section per core.

        Section depths are estimated as:
          section_depth = core_top + (section_number - 1) * 1.5

        where 1.5 m is the standard IODP section length.
        The core catcher (CC) is assigned the core bottom depth.
        Section depths are capped at the core bottom to handle short cores.

        Returns
        -------
        pd.DataFrame with columns: Core, Section, Depth_m, Type
        """
        summary  = self._load_summary()
        backbone = []

        for _, row in summary.iterrows():
            core_id = row['core_id']
            top_z   = float(row['top_m'])      # core top depth in m CSF-A
            bot_z   = float(row['bottom_m'])   # core bottom depth in m CSF-A
            n_sects = int(row['n_sections'])   # number of sections in core

            for s in range(1, n_sects + 1):
                # Interpolate section depth; cap at core bottom for short cores
                depth = min(top_z + (s - 1) * 1.5, bot_z)
                backbone.append({
                    'Core':    core_id,
                    'Section': str(s),
                    'Depth_m': round(depth, 3),
                    'Type':    'Section',
                })

            # Core catcher placed at the authoritative core bottom depth
            backbone.append({
                'Core':    core_id,
                'Section': 'CC',
                'Depth_m': round(bot_z, 3),
                'Type':    'CC',
            })

        df = pd.DataFrame(backbone)
        print(
            f"Backbone built: {len(df)} section slots across {len(summary)} cores, "
            f"deepest = {df['Depth_m'].max():.2f} m CSF-A"
        )
        return df

    def extract(self, backbone):
        """
        Extract stability scores from the JCORES VCD PDF, aligned to backbone depths.

        For each PDF page:
          1. Extract core and section identifiers using JCORES regex patterns.
          2. Score stability from JCORES_LEXICON using a lowest-score-wins rule:
             if multiple disturbance terms appear on a page, the most severe governs.
          3. Match the (Core, Section) pair to the backbone to assign a CSF-A depth.
          4. Carry the last-seen core ID forward across pages where it is not restated.

        Pages that cannot be matched to a backbone entry are still recorded
        with Depth_m = NaN so they can be inspected for extraction quality.

        After page-level extraction, depths are forward-filled within each core
        for pages that did not explicitly state a section identifier (e.g.,
        cover pages, figure pages, or pages with only partial text).

        Parameters
        ----------
        backbone : pd.DataFrame
            Output of build_backbone().

        Returns
        -------
        pd.DataFrame with columns:
            PDF_Page, Core, Section, Depth_m, Stability, Disturbance, Snippet
        """
        try:
            import fitz   # PyMuPDF — required for PDF text extraction
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF extraction. "
                "Install with: pip install pymupdf"
            )

        doc = fitz.open(self.pdf_path)

        # Build a fast lookup from (Core, Section) -> Depth_m
        backbone_lookup = {
            (row['Core'], str(row['Section']).upper()): row['Depth_m']
            for _, row in backbone.iterrows()
        }

        records   = []
        last_core = None   # carry forward core ID across pages when not explicitly stated

        print(f"Extracting from {len(doc)} pages in: {self.pdf_path}")

        for i in range(len(doc)):
            page       = doc.load_page(i)
            text       = page.get_text()
            text_lower = text.lower()

            # 1. Extract core ID using JCORES pattern; carry forward if not on this page
            core_match = self.CORE_PATTERN.search(text)
            if core_match:
                last_core = core_match.group(1).upper()
            core_id = last_core

            # 2. Extract section ID from this page (may be None for non-section pages)
            sect_match = self.SECTION_PATTERN.search(text)
            section_id = sect_match.group(1).upper() if sect_match else None

            # 3. Score stability using JCORES_LEXICON, lowest-score-wins rule
            # Iterating from low to high score ensures the most severe term wins
            score             = 3               # default: undisturbed
            disturbance_found = 'Undisturbed'   # default label
            for term, val in sorted(JCORES_LEXICON.items(), key=lambda x: x[1]):
                if term in text_lower:
                    score             = val
                    disturbance_found = term
                    break   # first match is the lowest (most severe) score

            # 4. Match to backbone depth
            depth = None
            if core_id and section_id:
                depth = backbone_lookup.get((core_id, section_id))
                if depth is None and section_id != 'CC':
                    # Try stripping the tool type suffix (e.g., '1W' -> '1')
                    # to handle cases where the backbone uses numeric-only section IDs
                    section_num = re.sub(r'[A-Z]', '', section_id)
                    depth       = backbone_lookup.get((core_id, section_num))

            # Keep a short text snippet for manual QC of extraction quality
            snippet = text[:200].replace('\n', ' ').strip()

            records.append({
                'PDF_Page':    i + 1,
                'Core':        core_id,
                'Section':     section_id,
                'Depth_m':     depth,
                'Stability':   score,
                'Disturbance': disturbance_found,
                'Snippet':     snippet,
            })

        doc.close()

        df = pd.DataFrame(records)

        # Forward-fill depths within each core for pages lacking explicit section IDs
        # This handles cover pages and figure pages that belong to a core
        # but do not restate the section identifier
        df['Depth_m'] = df.groupby('Core')['Depth_m'].transform(
            lambda x: x.ffill()
        )

        matched = df['Depth_m'].notna().sum()
        print(
            f"Extraction complete: {len(df)} pages processed, "
            f"{matched} depth-matched ({matched / len(df) * 100:.0f}%)"
        )

        return df.sort_values('Depth_m').reset_index(drop=True)

    @staticmethod
    def score_to_mtd_catalog(df, stability_threshold=1):
        """
        Identify MTD intervals from a stability log DataFrame.

        An MTD interval is defined as a contiguous sequence of depth-ordered
        rows where Stability <= stability_threshold. The default threshold of 1
        captures both slurried (score 0) and scaly / sheared (score 1) intervals,
        consistent with the physical definition of a mass transport deposit as
        material that has undergone significant fabric destruction.

        Parameters
        ----------
        df : pd.DataFrame
            Output of extract(), sorted by Depth_m.
        stability_threshold : int
            Maximum stability score to include in an MTD interval.
            Default 1 captures scores 0 and 1 (slurried and scaly fabric).

        Returns
        -------
        pd.DataFrame with columns:
            mtd_id, top_m, bottom_m, thickness_m, mean_stability
        """
        # Remove rows with missing depth or stability before interval detection
        df_clean = df.dropna(subset=['Depth_m', 'Stability']).sort_values('Depth_m')

        mtds      = []     # accumulates detected MTD intervals
        in_mtd    = False  # flag: currently inside an MTD interval
        mtd_top   = None   # depth of the current MTD top
        mtd_stabs = []     # stability scores within the current MTD

        for _, row in df_clean.iterrows():
            if row['Stability'] <= stability_threshold:
                # Enter or continue an MTD interval
                if not in_mtd:
                    in_mtd    = True
                    mtd_top   = row['Depth_m']
                    mtd_stabs = []
                mtd_stabs.append(row['Stability'])
            else:
                # Close the current MTD interval on return to stable conditions
                if in_mtd:
                    mtds.append({
                        'top_m':          mtd_top,
                        'bottom_m':       row['Depth_m'],
                        'thickness_m':    round(row['Depth_m'] - mtd_top, 2),
                        'mean_stability': round(np.mean(mtd_stabs), 2),
                    })
                    in_mtd    = False
                    mtd_top   = None
                    mtd_stabs = []

        # Close any MTD interval that extends to the bottom of the record
        if in_mtd and mtd_top is not None:
            last_depth = df_clean['Depth_m'].iloc[-1]
            mtds.append({
                'top_m':          mtd_top,
                'bottom_m':       last_depth,
                'thickness_m':    round(last_depth - mtd_top, 2),
                'mean_stability': round(np.mean(mtd_stabs), 2),
            })

        result = pd.DataFrame(mtds)

        # Assign sequential MTD identifiers if any intervals were detected
        if not result.empty:
            result.insert(0, 'mtd_id',
                          [f'MTD-{i + 1}' for i in range(len(result))])

        return result
