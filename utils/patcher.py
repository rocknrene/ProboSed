import cv2
import os

def slice_core_image(image_path, output_folder, patch_size=256):
    """
    Takes a high-res core scan and slices it into square patches.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found."
        
    h, w, _ = img.shape
    count = 0
    
    # Slicing logic
    for y in range(0, h - patch_size, patch_size):
        patch = img[y:y+patch_size, 0:w]
        patch_name = f"patch_{count:04d}.png"
        cv2.imwrite(os.path.join(output_folder, patch_name), patch)
        count += 1
        
    return f"Success: Created {count} patches."
