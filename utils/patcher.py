import cv2
import os

def slice_core_image(image_path, output_folder, patch_size=256):
    """
    Chops a long core image into square patches for labeling.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    # Slice vertically along the core
    count = 0
    for y in range(0, h - patch_size, patch_size):
        patch = img[y:y+patch_size, 0:w]
        patch_name = f"patch_depth_{count:04d}.png"
        cv2.imwrite(os.path.join(output_folder, patch_name), patch)
        count += 1
        
    return f"Created {count} patches in {output_folder}"
