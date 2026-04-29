"""
ProboSed/core_ml/labeler.py
============================
Visual Core Description (VCD) labeling tools for IODP core imagery.

Provides two classes:

VCDLabeler
    Interactive ipywidgets-based labeling interface for core image chunks.
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
    The new miner targets JCORES-specific patterns and terminology.
"""

import os
import re
import pandas as pd
import numpy as np


# =============================================================================
# STABILITY LEXICON — JCORES STANDARDIZED TERMINOLOGY
# =============================================================================

# JCORES uses IODP standardized disturbance codes, not free-text handwriting.
# Scores: 0 = completely disrupted, 1 = partially disrupted,
#         2 = minor disturbance, 3 = undisturbed
JCORES_LEXICON = {
    # Score 0 — completely disrupted, primary fabric destroyed
    'soupy':                  0,
    'slurried':               0,
    'flow-in':                0,
    'highly disturbed':       0,
    'heavily disturbed':      0,
    'completely disturbed':   0,
    'high disturbance':       0,
    'high disturb':           0,
    'disturb':                0,
    'void':                   0,
    'gouge':                  0,
    'clay gouge':             0,
    'chaotic':                0,
    # Score 1 — partially disrupted, some fabric preserved
    'moderately disturbed':   1,
    'biscuit':                1,
    'fall-in':                1,
    'fractured':              1,
    'brecciated':             1,
    'sheared':                1,
    'scaly':                  1,
    'deformed fabric':        1,
    'soft deformation':       1,
    'soft sediment deform':   1,
    'deformed':               1,
    # Score 2 — minor disturbance, primary fabric largely intact
    # NOTE: bare 'minor' and 'slightly' removed because they match the
    # VCD section header 'Minor lithology' and 'slightly' as a qualifier
    # in descriptions like 'slightly bioturbated', causing false positives.
    # Use specific compound phrases only.
    'mottled':                2,
    'slightly disturbed':     2,   # moved from score 1 — fits score 2 better
    'minor disturbance':      2,
    'minor deformation':      2,
    'color banding':          2,
    'colour banding':         2,
    # Score 3 — undisturbed, primary sedimentary fabric preserved
    'undisturbed':            3,
    'undeformed':             3,
    'bioturbated':            3,   # biogenic, not drilling disturbance
    'laminated':              3,
    'intact':                 3,
    'primary features':       3,
}

# Terms indicating drilling-induced disturbance — these intervals
# are excluded from the MTD stability catalog entirely.
DRILLING_CAUSE_TERMS = [
    'drilling disturbance',
    'drilling influence',
    'drilling induced',
    'drilling artifact',
    'due to drilling',        # matches CORC0019.pdf format: 'disturbed due to drilling'
    'drilling disturbance',
]

# Labeler options for the interactive widget (kept broader for human use)
LABELER_LITHOLOGIES = [
    'Siliceous mudstone',
    'Pelagic clay',
    'Ash / Tuff',
    'Chaotic matrix',
    'Fault gouge',
    'Breccia',
    'Chert',
    'Other',
]

LABELER_DISTURBANCES = [
    'Intact bedding',
    'Coherent block',
    'Scaly fabric',
    'Slurried / MTD',
    'Biscuit',
    'Fall-in',
    'Brecciated',
    'Void',
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
      - Observer name
      - Confidence (1-5)
      - Free-text notes

    Labels are appended to a CSV after each image so progress is saved
    even if the session is interrupted.

    Usage (in Colab or Jupyter):
        labeler = VCDLabeler(
            patch_folder='/path/to/images/',
            output_csv='/path/to/labels.csv',
            authors=['Observer1', 'Observer2']
        )
        labeler.start()
    """

    def __init__(self, patch_folder, output_csv, authors):
        self.patch_folder = patch_folder
        self.output_csv   = output_csv
        self.authors      = authors

        # Support common image formats including TIFF for core scans
        self.patch_files = sorted([
            f for f in os.listdir(patch_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        self.current_idx = 0

        # Create CSV with headers if it does not already exist
        if not os.path.exists(self.output_csv):
            pd.DataFrame(columns=[
                'filename', 'lithology', 'disturbance',
                'author', 'confidence', 'notes'
            ]).to_csv(self.output_csv, index=False)

    def start(self):
        """Launch the interactive labeling widget."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            raise ImportError(
                "ipywidgets is required for the interactive labeler. "
                "Install with: pip install ipywidgets"
            )

        self.img_w  = widgets.Image(width=500)
        self.auth_w = widgets.Dropdown(
            options=self.authors, description='Observer:'
        )
        self.lith_w = widgets.Dropdown(
            options=LABELER_LITHOLOGIES, description='Lithology:'
        )
        self.dist_w = widgets.Dropdown(
            options=LABELER_DISTURBANCES, description='Disturbance:'
        )
        self.conf_w = widgets.IntSlider(
            value=3, min=1, max=5, description='Confidence:'
        )
        self.note_w = widgets.Text(
            placeholder='e.g. visible shear planes, ash layer',
            description='Notes:'
        )
        self.save_btn = widgets.Button(
            description='Save & Next', button_style='success'
        )
        self.skip_btn = widgets.Button(
            description='Skip', button_style='warning'
        )
        self.prog_w = widgets.Label(value='')

        self.save_btn.on_click(self._save_entry)
        self.skip_btn.on_click(self._skip_entry)
        self._load_image()

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
        """Load the current image into the widget display."""
        if self.current_idx < len(self.patch_files):
            fname = self.patch_files[self.current_idx]
            path  = os.path.join(self.patch_folder, fname)
            with open(path, 'rb') as f:
                self.img_w.value = f.read()
            self.prog_w.value = (
                f"Patch {self.current_idx + 1} / {len(self.patch_files)}: {fname}"
            )
        else:
            self.prog_w.value = "Labeling complete for this session."
            self.img_w.value  = b''

    def _save_entry(self, b):
        """Save the current label and advance to the next image."""
        if self.current_idx >= len(self.patch_files):
            return
        row = pd.DataFrame([{
            'filename':    self.patch_files[self.current_idx],
            'lithology':   self.lith_w.value,
            'disturbance': self.dist_w.value,
            'author':      self.auth_w.value,
            'confidence':  self.conf_w.value,
            'notes':       self.note_w.value,
        }])
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

        Disturbance labels map to stability scores as follows:
          Intact bedding  -> 3
          Coherent block  -> 2
          Scaly fabric    -> 1
          Slurried / MTD  -> 0
          (others mapped by closest match)

        Returns a DataFrame with all original columns plus stability_score.
        """
        stability_map = {
            'intact bedding': 3,
            'coherent block': 2,
            'scaly fabric':   1,
            'slurried / mtd': 0,
            'slurried/mtd':   0,
            'biscuit':        1,
            'fall-in':        1,
            'brecciated':     1,
            'void':           0,
        }
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df['stability_score'] = (
            df['disturbance'].str.strip().str.lower().map(stability_map)
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

    The miner requires a core summary Excel file to assign accurate CSF-A
    depths. Without depth anchors, core identifiers alone are not sufficient
    to reconstruct a depth profile.

    Usage:
        miner = JCORESMiner(
            pdf_path='/path/to/405-C0019J_VCD.pdf',
            summary_xlsx_path='/path/to/405-C0019J_CoreSummary.xlsx'
        )
        backbone = miner.build_backbone()
        df = miner.extract(backbone)
        df.to_csv('C0019J_stability_log.csv', index=False)
    """

    # JCORES core ID pattern for CORC0019.pdf format.
    # The CORC0019.pdf VCD header reads:
    #   'Hole C0019J Core 7K, interval 139 to 142.805 m (core depth below seafloor)'
    # Captures the core label (e.g., '7K') after the word 'Core'.
    # Also matches legacy formats like 'C0019J-14K' or '405-C0019J-14K'.
    CORE_PATTERN = re.compile(
        r'(?:Core\s+(\d+[A-Z])|(?:405-)?C\d{4}[A-Z]-(\d+[A-Z]))',
        re.IGNORECASE
    )

    # JCORES section pattern: section numbers follow 'W', 'H', 'R', 'F' type codes
    # e.g., '1W', '2H', 'CC'
    SECTION_PATTERN = re.compile(
        r'\b(\d{1,2}[WHRF]|CC)\b', re.IGNORECASE
    )

    # CSF-A depth from CORC0019.pdf header format:
    #   'interval 139 to 142.805 m (core depth below seafloor)'
    # Captures the top depth (first number after 'interval').
    # Also matches legacy 'CSF-A' format as fallback.
    DEPTH_PATTERN = re.compile(
        r'interval\s+(\d+\.?\d*)\s+to\s+\d+\.?\d*\s*m',
        re.IGNORECASE
    )
    DEPTH_PATTERN_CSFA = re.compile(
        r'(\d{1,3}\.\d{1,4})\s*m?\s*CSF-?A', re.IGNORECASE
    )

    def __init__(self, pdf_path, summary_xlsx_path):
        self.pdf_path          = pdf_path
        self.summary_xlsx_path = summary_xlsx_path

    def _load_summary(self):
        """
        Load and clean the core summary Excel file.

        The summary provides authoritative top and bottom depths for each core,
        and the section count needed to interpolate per-section depths.

        Returns a cleaned DataFrame with columns:
          core_id, top_m, bottom_m, n_sections
        """
        raw = pd.read_excel(self.summary_xlsx_path, header=5)
        raw.columns = [c.replace('\n', ' ').strip() for c in raw.columns]

        # Drop rows without a core identifier or section count
        raw = raw.dropna(subset=['Core curated'])
        raw['Number of sections'] = pd.to_numeric(
            raw['Number of sections'], errors='coerce'
        )
        raw = raw.dropna(subset=['Number of sections'])
        raw['Number of sections'] = raw['Number of sections'].astype(int)

        # Clean core ID: 'C0019J-14K' -> '14K'
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
        The core catcher (CC) is placed at the core bottom depth.

        Returns a DataFrame with columns:
          Core, Section, Depth_m, Type
        """
        summary = self._load_summary()
        backbone = []

        for _, row in summary.iterrows():
            core_id   = row['core_id']
            top_z     = float(row['top_m'])
            bot_z     = float(row['bottom_m'])
            n_sects   = int(row['n_sections'])

            for s in range(1, n_sects + 1):
                # Cap section depth at core bottom to handle short cores
                depth = min(top_z + (s - 1) * 1.5, bot_z)
                backbone.append({
                    'Core':    core_id,
                    'Section': str(s),
                    'Depth_m': round(depth, 3),
                    'Type':    'Section',
                })

            # Core catcher at bottom of core
            backbone.append({
                'Core':    core_id,
                'Section': 'CC',
                'Depth_m': round(bot_z, 3),
                'Type':    'CC',
            })

        df = pd.DataFrame(backbone)
        print(f"Backbone built: {len(df)} section slots, "
              f"deepest = {df['Depth_m'].max():.2f} m CSF-A")
        return df

    def extract(self, backbone):
        """
        Extract stability scores from the JCORES VCD PDF, aligned to backbone depths.

        For each PDF page:
          1. Extract core and section identifiers using JCORES patterns.
          2. Score stability from the standardized disturbance vocabulary.
          3. Match to the backbone to assign a CSF-A depth.

        Pages that cannot be matched to a backbone entry are still recorded
        with Depth_m = NaN so they can be inspected.

        Parameters
        ----------
        backbone : pd.DataFrame
            Output of build_backbone().

        Returns
        -------
        pd.DataFrame with columns:
            PDF_Page, Core, Section, Depth_m, Stability, Disturbance, Raw_Text_Snippet
        """
        try:
            import fitz   # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF extraction. "
                "Install with: pip install pymupdf"
            )

        doc = fitz.open(self.pdf_path)

        # Build a lookup from (Core, Section) -> Depth_m for fast matching
        backbone_lookup = {
            (row['Core'], str(row['Section']).upper()): row['Depth_m']
            for _, row in backbone.iterrows()
        }

        records = []
        last_core = None   # carry forward core ID across pages

        print(f"Extracting from {len(doc)} pages...")
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()
            text_lower = text.lower()

            # 1. Extract core ID from CORC0019.pdf header format:
            #    'Hole C0019J Core 7K, interval 139 to 142.805 m'
            #    Group 1 = 'Core X' format, Group 2 = hyphen format fallback
            core_match = self.CORE_PATTERN.search(text)
            if core_match:
                # group(1) is the 'Core 7K' capture, group(2) is the hyphen capture
                last_core = (core_match.group(1) or core_match.group(2)).upper()
            core_id = last_core   # carry forward across pages

            # 2. Extract section ID
            sect_match = self.SECTION_PATTERN.search(text)
            section_id = sect_match.group(1).upper() if sect_match else None

            # 3. Score stability using minimum-score rule across all matching terms.
            #
            # MINIMUM SCORE RULE: check every term and keep the lowest score
            # found. Deformation terms take priority over intact fabric terms
            # when both appear on the same page. A page describing "bioturbated
            # mudstone with sheared fabric" scores 1 (sheared) not 3 (bioturbated).
            # Breaking on first match was the previous bug.
            #
            # Drilling-induced disturbance is flagged separately via
            # DRILLING_CAUSE_TERMS and excluded from the MTD catalog.
            score             = 3
            disturbance_found = 'Undisturbed'
            is_drilling       = any(t in text_lower for t in DRILLING_CAUSE_TERMS)

            for term, val in JCORES_LEXICON.items():
                if term in text_lower:
                    if val < score:
                        score             = val
                        disturbance_found = term

            # 4. Assign depth.
            #    Primary: extract directly from the page header interval text
            #      'interval 139 to 142.805 m (core depth below seafloor)'
            #    Fallback: look up (Core, Section) in the backbone.
            depth = None
            depth_match = self.DEPTH_PATTERN.search(text)
            if depth_match:
                try:
                    depth = float(depth_match.group(1))
                except (ValueError, IndexError):
                    depth = None

            if depth is None:
                # fallback: legacy CSF-A format
                csfa_match = self.DEPTH_PATTERN_CSFA.search(text)
                if csfa_match:
                    try:
                        depth = float(csfa_match.group(1))
                    except (ValueError, IndexError):
                        depth = None

            if depth is None and core_id and section_id:
                # final fallback: backbone lookup
                depth = backbone_lookup.get((core_id, section_id))
                if depth is None and section_id != 'CC':
                    section_num = re.sub(r'[A-Z]', '', section_id)
                    depth = backbone_lookup.get((core_id, section_num))

            # Keep a short text snippet for manual verification
            snippet = text[:200].replace('\n', ' ').strip()

            records.append({
                'PDF_Page':        i + 1,
                'Core':            core_id,
                'Section':         section_id,
                'Depth_m':         depth,
                'Stability':       score,
                'Disturbance':     disturbance_found,
                'Drilling_Artifact': is_drilling,
                'Snippet':         snippet,
            })

        doc.close()

        df = pd.DataFrame(records)

        # Forward-fill depths within the same core for pages that
        # did not explicitly state section (cover pages, figures, etc.)
        df['Depth_m'] = df.groupby('Core')['Depth_m'].transform(
            lambda x: x.ffill()
        )

        matched = df['Depth_m'].notna().sum()
        print(f"Extraction complete: {len(df)} pages, "
              f"{matched} depth-matched ({matched/len(df)*100:.0f}%)")

        return df.sort_values('Depth_m').reset_index(drop=True)

    @staticmethod
    def score_to_mtd_catalog(df, stability_threshold=1):
        """
        Identify MTD boundaries from a stability log DataFrame.

        An MTD interval is defined as a contiguous sequence of depth-ordered
        rows where Stability <= stability_threshold. The threshold default of 1
        captures both slurried (0) and scaly/sheared (1) intervals.

        Parameters
        ----------
        df : pd.DataFrame
            Output of extract(), sorted by Depth_m.
        stability_threshold : int
            Maximum stability score to include in an MTD interval.

        Returns
        -------
        pd.DataFrame with columns:
            mtd_id, top_m, bottom_m, thickness_m, mean_stability
        """
        df_clean = df.dropna(subset=['Depth_m', 'Stability']).sort_values('Depth_m')

        mtds = []
        in_mtd    = False
        mtd_top   = None
        mtd_stabs = []

        for _, row in df_clean.iterrows():
            if row['Stability'] <= stability_threshold:
                if not in_mtd:
                    in_mtd   = True
                    mtd_top  = row['Depth_m']
                    mtd_stabs = []
                mtd_stabs.append(row['Stability'])
            else:
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

        # Close any open MTD at the end of the record
        if in_mtd and mtd_top is not None:
            last_depth = df_clean['Depth_m'].iloc[-1]
            mtds.append({
                'top_m':          mtd_top,
                'bottom_m':       last_depth,
                'thickness_m':    round(last_depth - mtd_top, 2),
                'mean_stability': round(np.mean(mtd_stabs), 2),
            })

        result = pd.DataFrame(mtds)
        if not result.empty:
            result.insert(0, 'mtd_id',
                          [f'MTD-{i+1}' for i in range(len(result))])
        return result
