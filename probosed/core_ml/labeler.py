import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import os

class VCDLabeler:
    """
    Expert-in-the-Loop tool for digitizing handwritten Visual Core Descriptions.
    Captures lithology, disturbance, and observer confidence.
    """
    def __init__(self, patch_folder, output_csv, authors):
        self.patch_folder = patch_folder
        self.output_csv = output_csv
        self.authors = authors
        self.patch_files = [f for f in sorted(os.listdir(patch_folder)) if f.endswith('.png')]
        self.current_idx = 0
        
        # Initialize CSV if needed
        if not os.path.exists(self.output_csv):
            cols = ['filename', 'lithology', 'disturbance', 'author', 'confidence', 'notes']
            pd.DataFrame(columns=cols).to_csv(self.output_csv, index=False)

    def start(self):
        # UI Elements
        self.img_w = widgets.Image(width=400)
        self.auth_w = widgets.Dropdown(options=self.authors, description='Observer:')
        self.lith_w = widgets.Dropdown(options=['Mud', 'Silt', 'Sand', 'Clay', 'Ooze'], description='Lithology:')
        self.dist_w = widgets.Dropdown(options=['Intact', 'Disturbed', 'Biscuited', 'Slurried'], description='Disturbance:')
        self.conf_w = widgets.IntSlider(value=5, min=1, max=5, description='Confidence:')
        self.note_w = widgets.Text(placeholder='Handwriting notes...', description='Notes:')
        self.save_btn = widgets.Button(description="Save & Next", button_style='success')
        self.prog_w = widgets.Label(value="")

        self.save_btn.on_click(self._save_entry)
        self._load_image()
        
        display(widgets.VBox([
            self.img_w, self.auth_w, self.lith_w, 
            self.dist_w, self.conf_w, self.note_w, 
            self.save_btn, self.prog_w
        ]))

    def _load_image(self):
        if self.current_idx < len(self.patch_files):
            fname = self.patch_files[self.current_idx]
            path = os.path.join(self.patch_folder, fname)
            with open(path, 'rb') as f:
                self.img_w.value = f.read()
            self.prog_w.value = f"File: {fname} ({self.current_idx + 1}/{len(self.patch_files)})"
        else:
            self.prog_w.value = "🎉 All patches labeled!"

    def _save_entry(self, b):
        row = [
            self.patch_files[self.current_idx], self.lith_w.value, 
            self.dist_w.value, self.auth_w.value, 
            self.conf_w.value, self.note_w.value
        ]
        pd.DataFrame([row]).to_csv(self.output_csv, mode='a', header=False, index=False)
        self.current_idx += 1
        self._load_image()
