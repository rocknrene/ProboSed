import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import os

class VCDLabeler:
    def __init__(self, patch_folder, output_csv, authors):
        self.patch_folder = patch_folder
        self.output_csv = output_csv
        self.authors = authors
        self.patch_files = [f for f in sorted(os.listdir(patch_folder)) if f.endswith(('.png', '.jpg'))]
        self.current_idx = 0
        
        if not os.path.exists(self.output_csv):
            cols = ['filename', 'lithology', 'disturbance', 'author', 'confidence', 'notes']
            pd.DataFrame(columns=cols).to_csv(self.output_csv, index=False)

    def start(self):
        self.img_w = widgets.Image(width=400)
        self.auth_w = widgets.Dropdown(options=self.authors, description='Observer:')
        # UPDATED: Direct link to JpGU Abstract terms
        self.lith_w = widgets.Dropdown(options=['Pelagic Clay', 'Siliciclastic', 'Ash/Tuff', 'Chaotic Matrix'], description='Facies:')
        self.dist_w = widgets.Dropdown(options=['Coherent Block', 'Scaly Fabric', 'Slurried/MTD', 'Intact Bedding'], description='Dynamics:')
        self.conf_w = widgets.IntSlider(value=5, min=1, max=5, description='Confidence:')
        self.note_w = widgets.Text(placeholder='Visible shear planes...', description='Notes:')
        self.save_btn = widgets.Button(description="Save & Next", button_style='success')
        self.prog_w = widgets.Label(value="")

        self.save_btn.on_click(self._save_entry)
        self._load_image()
        
        display(widgets.VBox([self.img_w, self.auth_w, self.lith_w, self.dist_w, self.conf_w, self.note_w, self.save_btn, self.prog_w]))

    def _load_image(self):
        if self.current_idx < len(self.patch_files):
            fname = self.patch_files[self.current_idx]
            path = os.path.join(self.patch_folder, fname)
            with open(path, 'rb') as f:
                self.img_w.value = f.read()
            self.prog_w.value = f"Patch: {fname} ({self.current_idx + 1}/{len(self.patch_files)})"
        else:
            self.prog_w.value = "🎉 Data collection complete for this section!"

    def _save_entry(self, b):
        row = [self.patch_files[self.current_idx], self.lith_w.value, self.dist_w.value, self.auth_w.value, self.conf_w.value, self.note_w.value]
        pd.DataFrame([row]).to_csv(self.output_csv, mode='a', header=False, index=False)
        self.current_idx += 1
        self._load_image()