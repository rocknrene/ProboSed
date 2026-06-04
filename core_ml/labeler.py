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
    Programmatic extraction of fabric preservation and rheological state
    from JCORES-format VCD PDFs. JCORES PDFs are machine-readable structured
    documents (not scanned handwriting), so regex extraction works reliably.
    Uses the core summary Excel file to assign accurate CSF-A depths.
 
Key distinction from VCDLabeler:
    VCDLabeler records human observations from core images. JCORESMiner
    extracts equivalent information programmatically from the shipboard
    JCORES database text. Both produce records in the same two-axis
    observational framework.
"""
 
import os
import re
import pandas as pd
import numpy as np
 
 
# =============================================================================
# FPI LEXICON — JCORES STANDARDISED TERMINOLOGY
# =============================================================================
 
# JCORES uses IODP standardised disturbance codes in structured digital text.
# Each term is mapped to a Fabric Preservation Index (FPI) score.
#
# FPI records whether primary sedimentary fabric is intact, modified, or
# destroyed. It is directly answerable from IODP disturbance vocabulary
# without requiring a process interpretation.
#
# Scores:
#   3 — Primary fabric fully preserved. Bedding contacts sharp,
#       no crosscutting structures.
#   2 — Primary fabric partially preserved. Minor disruption but
#       bedding traceable.
#   1 — Primary fabric largely destroyed. Foliation or scaly surfaces
#       present. Bedding not traceable across section.
#   0 — No fabric. Structureless or homogeneous. Clasts in matrix
#       with no preferred orientation.
#
# Minimum-score rule: all terms on a given page are checked and the
# lowest matching score is assigned. This ensures that a page describing
# bioturbated mudstone with sheared fabric scores 1 (sheared) rather
# than 3 (bioturbated).
 
FPI_LEXICON = {
    # FPI 0 — no fabric preserved, primary structure completely destroyed
    'soupy':                   0,
    'slurried':                0,
    'flow-in':                 0,
    'highly disturbed':        0,
    'heavily disturbed':       0,
    'completely disturbed':    0,
    'high disturbance':        0,
    'high disturb':            0,
    'void':                    0,
    'gouge':                   0,
    'clay gouge':              0,
    'chaotic':                 0,
    'high distubance':         0,   # typographic variant recorded in core 60K
    # FPI 1 — fabric largely destroyed, some remnant structure present
    'moderately disturbed':    1,
    'biscuit':                 1,
    'fall-in':                 1,
    'fractured':               1,
    'brecciated':              1,
    'sheared':                 1,
    'scaly':                   1,
    'deformed fabric':         1,
    'soft-sediment deform':    1,
    'deformed':                1,
    'shear deformation':       1,
    'evidence of shear':       1,
    'some evidence of deform': 1,
    # FPI 2 — fabric partially preserved, primary structure largely intact
    # Note: bare 'minor' and 'slightly' are excluded because they match
    # section header text ('Minor lithology', 'slightly bioturbated'),
    # producing false positives. Compound phrases are used instead.
    'mottled':                 2,
    'slightly disturbed':      2,
    'minor disturbance':       2,
    'minor deformation':       2,
    # FPI 3 — fabric fully preserved, primary sedimentary structure intact
    'undisturbed':             3,
    'undeformed':              3,
    'bioturbated':             3,   # biogenic reworking, not drilling disturbance
    'laminated':               3,
    'intact':                  3,
    'primary features':        3,
}
 
# Terms identifying drilling-induced disturbance in the VCD text.
# Intervals matching these terms receive process_flag = 'drilling'.
# They are excluded from the deformation catalog but retained in the
# dataset so the depth record remains complete.
DRILLING_CAUSE_TERMS = [
    'drilling disturbance',
    'drilling influence',
    'drilling induced',
    'drilling artifact',
    'due to drilling',
]
 
# RSI term sets used by _assign_rsi() to classify rheological state
# from grain size and matrix language in the VCD text.
#
# RSI — Rheological State Index
# Records the physical character of the material at the time of
# deposition or last disturbance. Independent of FPI.
#
#   M — Mud-supported. No visible grains coarser than silt.
#       Matrix dominates. Default assignment for C0019J given the
#       mud-dominated lithology throughout the site.
#   C — Clast-bearing. Dispersed gravel- or sand-grade clasts
#       in mud matrix. Assigned only when explicit clast language
#       is present in the VCD text.
#   G — Gravel-supported or clast-supported. Matrix fills pore
#       space between grains.
#   F — Fluid-escape texture visible. Pipes, dishes, convolute
#       lamination present.
#
# Priority order for assignment: G > F > C > M.
 
RSI_GRAVEL_TERMS = [
    'clast-supported',
    'grain-supported',
    'gravel-supported',
    'framework-supported',
]
RSI_FLUID_TERMS = [
    'dish',
    'pipe',
    'fluid escape',
    'convolute',
    'dewatering',
    'water escape',
    'flame structure',
]
RSI_CLAST_TERMS = [
    'clast',
    'gravel',
    'pebble',
    'cobble',
    'boulder',
    'granule',
    'breccia',
    'fragment',
    'coarse sand',
    'extraformational',
    'lithoclast',
    'bioclast',
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
      - Disturbance type (maps to FPI score in downstream analysis)
      - Observer name
      - Confidence (1–5)
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
            self.prog_w.value = "Labelling complete for this session."
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
    def labels_to_fpi(csv_path):
        """
        Load a completed labels CSV and add a numeric FPI column.
 
        Disturbance labels map to FPI scores as follows:
          Intact bedding  -> 3
          Coherent block  -> 2
          Scaly fabric    -> 1
          Slurried / MTD  -> 0
          (others mapped by closest match)
 
        Returns a DataFrame with all original columns plus FPI.
        """
        fpi_map = {
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
        df['FPI'] = df['disturbance'].str.strip().str.lower().map(fpi_map)
        return df
 
 
# =============================================================================
# JCORESMiner — Programmatic VCD Extraction from JCORES PDFs
# =============================================================================
 
class JCORESMiner:
    """
    Extracts fabric preservation and rheological state information from
    JCORES-format VCD PDFs.
 
    JCORES PDFs are structured digital documents generated by the shipboard
    database system. Unlike scanned handwritten VCDs, they contain machine-
    readable text with consistent formatting, making regex extraction reliable.
 
    The miner requires a core summary Excel file to assign accurate CSF-A
    depths. Without depth anchors, core identifiers alone are not sufficient
    to reconstruct a depth profile.
 
    Output columns
    --------------
    PDF_Page     : int   — PDF page number (1-indexed)
    Core         : str   — core identifier (e.g. '14K')
    Hole         : str   — hole identifier (e.g. 'C0019J'), added in notebook
    Section      : str   — section identifier
    Depth_m      : float — CSF-A depth in metres
    FPI          : int   — Fabric Preservation Index (0–3)
    Disturbance  : str   — matched FPI lexicon term
    RSI          : str   — Rheological State Index (M / C / G / F)
    process_flag : str   — 'geological' / 'drilling' / 'unknown'
    Snippet      : str   — first 200 characters of page text for verification
 
    Usage:
        miner = JCORESMiner(
            pdf_path='/path/to/CORC0019.pdf',
            summary_xlsx_path='/path/to/405-C0019J_CoreSummary.xlsx'
        )
        backbone = miner.build_backbone()
        df = miner.extract(backbone)
        df.to_csv('C0019J_VCD_stability_log.csv', index=False)
    """
 
    # JCORES core ID pattern for CORC0019.pdf format.
    # Header format: 'Hole C0019J Core 7K, interval 139 to 142.805 m
    #                 (core depth below seafloor)'
    # Captures the core label (e.g. '7K') after the word 'Core'.
    # Also matches legacy formats: 'C0019J-14K' or '405-C0019J-14K'.
    CORE_PATTERN = re.compile(
        r'(?:Core\s+(\d+[A-Z])|(?:405-)?C\d{4}[A-Z]-(\d+[A-Z]))',
        re.IGNORECASE
    )
 
    # Section pattern: section numbers follow coring type codes
    # e.g. '1W', '2H', 'CC'
    SECTION_PATTERN = re.compile(
        r'\b(\d{1,2}[WHRF]|CC)\b', re.IGNORECASE
    )
 
    # CSF-A depth from CORC0019.pdf header format:
    # 'interval 139 to 142.805 m (core depth below seafloor)'
    # Captures the top depth (first number after 'interval').
    DEPTH_PATTERN = re.compile(
        r'interval\s+(\d+\.?\d*)\s+to\s+\d+\.?\d*\s*m',
        re.IGNORECASE
    )
 
    # Fallback depth pattern for legacy CSF-A format
    DEPTH_PATTERN_CSFA = re.compile(
        r'(\d{1,3}\.\d{1,4})\s*m?\s*CSF-?A', re.IGNORECASE
    )
 
    def __init__(self, pdf_path, summary_xlsx_path):
        self.pdf_path          = pdf_path
        self.summary_xlsx_path = summary_xlsx_path
 
    def _load_summary(self):
        """
        Load and clean the core summary Excel file.
 
        The summary provides authoritative top and bottom depths for each core
        and the section count needed to interpolate per-section depths.
 
        Returns a cleaned DataFrame with columns:
          core_id, top_m, bottom_m, n_sections
        """
        raw = pd.read_excel(self.summary_xlsx_path, header=5)
        raw.columns = [c.replace('\n', ' ').strip() for c in raw.columns]
 
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
          section_depth = core_top + (section_number - 1) * 1.5 m
 
        where 1.5 m is the standard IODP section length.
        The core catcher (CC) is placed at the core bottom depth.
 
        Returns a DataFrame with columns:
          Core, Section, Depth_m, Type
        """
        summary  = self._load_summary()
        backbone = []
 
        for _, row in summary.iterrows():
            core_id = row['core_id']
            top_z   = float(row['top_m'])
            bot_z   = float(row['bottom_m'])
            n_sects = int(row['n_sections'])
 
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
 
    @staticmethod
    def _assign_rsi(text_lower):
        """
        Assign Rheological State Index (RSI) from VCD page text.
 
        RSI is determined by the presence of grain size and matrix
        character terms in the VCD text. Priority order: G > F > C > M.
        RSI = M is assigned by default when no coarser material is
        described, which is appropriate for the mud-dominated C0019J site.
 
        Parameters
        ----------
        text_lower : str
            Lowercased page text.
 
        Returns
        -------
        str : one of 'M', 'C', 'G', 'F'
        """
        if any(t in text_lower for t in RSI_GRAVEL_TERMS):
            return 'G'
        if any(t in text_lower for t in RSI_FLUID_TERMS):
            return 'F'
        if any(t in text_lower for t in RSI_CLAST_TERMS):
            return 'C'
        return 'M'
 
    def extract(self, backbone):
        """
        Extract FPI and RSI from the JCORES VCD PDF, aligned to backbone depths.
 
        For each PDF page:
          1. Extract core and section identifiers using JCORES patterns.
          2. Score FPI from the standardised disturbance vocabulary using
             the minimum-score rule.
          3. Assign RSI from grain size and matrix character language.
          4. Set process_flag from DRILLING_CAUSE_TERMS.
          5. Match to the backbone to assign a CSF-A depth.
 
        Pages that cannot be matched to a backbone entry are still recorded
        with Depth_m = NaN so they can be inspected.
 
        Parameters
        ----------
        backbone : pd.DataFrame
            Output of build_backbone().
 
        Returns
        -------
        pd.DataFrame with columns:
            PDF_Page, Core, Section, Depth_m, FPI, Disturbance,
            RSI, process_flag, Snippet
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
 
        records   = []
        last_core = None   # carry forward core ID across pages
 
        print(f"Extracting from {len(doc)} pages...")
        for i in range(len(doc)):
            page       = doc.load_page(i)
            text       = page.get_text()
            text_lower = text.lower()
 
            # 1. Extract core ID from CORC0019.pdf header format
            core_match = self.CORE_PATTERN.search(text)
            if core_match:
                last_core = (core_match.group(1) or core_match.group(2)).upper()
            core_id = last_core
 
            # 2. Extract section ID
            sect_match = self.SECTION_PATTERN.search(text)
            section_id = sect_match.group(1).upper() if sect_match else None
 
            # 3. Score FPI using minimum-score rule across all matching terms.
            #    All terms in FPI_LEXICON are checked. The lowest score found
            #    is assigned, ensuring that deformation terms take priority
            #    over intact fabric terms when both appear on the same page.
            fpi               = 3
            disturbance_found = 'Undisturbed'
            for term, val in FPI_LEXICON.items():
                if term in text_lower and val < fpi:
                    fpi               = val
                    disturbance_found = term
 
            # 4. Assign RSI
            rsi = self._assign_rsi(text_lower)
 
            # 5. Set process_flag
            is_drilling  = any(t in text_lower for t in DRILLING_CAUSE_TERMS)
            if is_drilling:
                process_flag = 'drilling'
            elif fpi <= 1:
                process_flag = 'geological'
            else:
                process_flag = 'unknown'
 
            # 6. Assign depth.
            #    Primary: extract from page header interval text.
            #    Fallback 1: legacy CSF-A format.
            #    Fallback 2: backbone lookup by (Core, Section).
            depth = None
            depth_match = self.DEPTH_PATTERN.search(text)
            if depth_match:
                try:
                    depth = float(depth_match.group(1))
                except (ValueError, IndexError):
                    depth = None
 
            if depth is None:
                csfa_match = self.DEPTH_PATTERN_CSFA.search(text)
                if csfa_match:
                    try:
                        depth = float(csfa_match.group(1))
                    except (ValueError, IndexError):
                        depth = None
 
            if depth is None and core_id and section_id:
                depth = backbone_lookup.get((core_id, section_id))
                if depth is None and section_id != 'CC':
                    section_num = re.sub(r'[A-Z]', '', section_id)
                    depth = backbone_lookup.get((core_id, section_num))
 
            # Keep a short text snippet for manual verification
            snippet = text[:200].replace('\n', ' ').strip()
 
            records.append({
                'PDF_Page':     i + 1,
                'Core':         core_id,
                'Section':      section_id,
                'Depth_m':      depth,
                'FPI':          fpi,
                'Disturbance':  disturbance_found,
                'RSI':          rsi,
                'process_flag': process_flag,
                'Snippet':      snippet,
            })
 
        doc.close()
 
        df = pd.DataFrame(records)
 
        # Forward-fill depths within the same core for pages that
        # did not explicitly state a section (cover pages, figures, etc.)
        df['Depth_m'] = df.groupby('Core')['Depth_m'].transform(
            lambda x: x.ffill()
        )
 
        matched = df['Depth_m'].notna().sum()
        print(f"Extraction complete: {len(df)} pages, "
              f"{matched} depth-matched ({matched/len(df)*100:.0f}%)")
 
        return df.sort_values('Depth_m').reset_index(drop=True)
 
    @staticmethod
    def fpi_to_deformation_catalog(df, fpi_threshold=1):
        """
        Identify deformed intervals from an FPI log DataFrame.
 
        A deformed interval is defined as a contiguous sequence of
        depth-ordered rows where FPI <= fpi_threshold. The default
        threshold of 1 captures both structureless (FPI=0) and
        scaly/sheared (FPI=1) intervals.
 
        Intervals with process_flag = 'drilling' are excluded before
        catalog construction but retained in the input dataset so the
        depth record remains complete.
 
        Each interval receives a facies code combining rounded mean FPI
        and dominant RSI, e.g. DF0M (structureless mud) or DF0C
        (clast-bearing chaotic matrix).
 
        The SDE failure threshold mapping is:
            theta(FPI) = clip(FPI * 2/3, 0, 2)
 
        Parameters
        ----------
        df : pd.DataFrame
            Output of extract(), sorted by Depth_m.
        fpi_threshold : int
            Maximum FPI score to include in a deformed interval.
 
        Returns
        -------
        pd.DataFrame with columns:
            interval_id, top_m, bottom_m, thickness_m,
            mean_fpi, rsi_dominant, facies_code
        """
        df_geo   = df[df['process_flag'] != 'drilling'].copy()
        df_clean = df_geo.dropna(subset=['Depth_m', 'FPI']).sort_values('Depth_m')
 
        intervals   = []
        in_interval = False
        top         = None
        fpi_vals    = []
        rsi_vals    = []
 
        for _, row in df_clean.iterrows():
            if row['FPI'] <= fpi_threshold:
                if not in_interval:
                    in_interval = True
                    top         = row['Depth_m']
                    fpi_vals    = []
                    rsi_vals    = []
                fpi_vals.append(row['FPI'])
                rsi_vals.append(row['RSI'])
            else:
                if in_interval:
                    rsi_dom  = pd.Series(rsi_vals).mode()[0]
                    mean_fpi = round(np.mean(fpi_vals), 2)
                    intervals.append({
                        'top_m':        top,
                        'bottom_m':     row['Depth_m'],
                        'thickness_m':  round(row['Depth_m'] - top, 2),
                        'mean_fpi':     mean_fpi,
                        'rsi_dominant': rsi_dom,
                        'facies_code':  f'DF{round(mean_fpi)}{rsi_dom}',
                    })
                    in_interval = False
                    top         = None
                    fpi_vals    = []
                    rsi_vals    = []
 
        # Close any open interval at the end of the record
        if in_interval and top is not None:
            last_depth = df_clean['Depth_m'].iloc[-1]
            rsi_dom    = pd.Series(rsi_vals).mode()[0]
            mean_fpi   = round(np.mean(fpi_vals), 2)
            intervals.append({
                'top_m':        top,
                'bottom_m':     last_depth,
                'thickness_m':  round(last_depth - top, 2),
                'mean_fpi':     mean_fpi,
                'rsi_dominant': rsi_dom,
                'facies_code':  f'DF{round(mean_fpi)}{rsi_dom}',
            })
 
        result = pd.DataFrame(intervals)
        if not result.empty:
            result.insert(0, 'interval_id',
                          [f'DI-{i+1}' for i in range(len(result))])
        return result
  
