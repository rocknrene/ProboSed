"""
patcher.py
==========
Core image patch extraction utility for ProboSed.

Slices high-resolution IODP core line-scan images into fixed-size square
patches for downstream use with the VCDLabeler interactive labeling interface
(core_ml/labeler.py) and future machine learning classification workflows.

The patcher is designed for CHIKYU-format core section scans produced by
the D/V Chikyu shipboard imaging system. These are typically high-resolution
TIFF files (~10,000 x 1000 pixels per section) stored in the IODP image
archive. The patcher extracts overlapping or non-overlapping square patches
along the core length axis (vertical in standard core orientation).

Connection to ProboSed pipeline:
  1. patcher.py      slices raw core scans -> patch PNG files
  2. 01_stitch.ipynb stitches section scans into a continuous core column
  3. core_ml/labeler.py (VCDLabeler) labels patches interactively

Usage:
  python patcher.py
  Or import slice_core_image() directly in a notebook.

Requirements:
  pip install opencv-python-headless pillow numpy
  opencv-python-headless is preferred over opencv-python for server/Colab
  environments because it has no GUI dependencies.
"""

import os
import numpy as np

# Use opencv-python-headless for server and Colab compatibility.
# This avoids Qt/GTK display dependencies that cause import errors
# in headless environments (Google Colab, HPC clusters, Docker).
try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python-headless is required for patch extraction.\n"
        "Install with: pip install opencv-python-headless"
    )


# =============================================================================
# CORE IMAGE PATCH EXTRACTION
# =============================================================================

def slice_core_image(
    image_path,
    output_folder,
    patch_size   = 256,
    overlap      = 0,
    min_coverage = 0.5,
):
    """
    Slice a high-resolution core section scan into square patches.

    Extracts patches along the core length axis (y direction in image
    coordinates, corresponding to depth in the core). Each patch is
    patch_size x patch_size pixels. The full image width is used unless
    the image is wider than patch_size, in which case patches are also
    extracted along the width axis.

    Patches at the image boundary that are smaller than patch_size are
    included if they cover at least min_coverage fraction of the patch area.
    This prevents the systematic loss of the bottom few centimeters of each
    core section that occurs when core length is not a multiple of patch_size.

    Parameters
    ----------
    image_path : str
        Path to the input core scan image.
        Supported formats: PNG, TIFF, JPEG (anything OpenCV can read).
        TIFF is preferred for lossless core imagery.
    output_folder : str
        Directory where patch PNG files will be written.
        Created automatically if it does not exist.
    patch_size : int
        Side length of each square patch in pixels. Default 256.
        Larger patches (512) retain more spatial context but produce
        fewer patches per core section.
    overlap : int
        Overlap between adjacent patches in pixels. Default 0 (no overlap).
        Overlap = patch_size // 2 doubles the number of patches and ensures
        all features appear near the center of at least one patch.
        Use overlap > 0 when training machine learning classifiers.
    min_coverage : float
        Minimum fractional coverage required to keep a boundary patch.
        Default 0.5 — boundary patches covering >= 50% of patch_size
        are kept; smaller remnants are discarded.
        Range: 0.0 (keep all) to 1.0 (only full patches).

    Returns
    -------
    dict with keys:
        n_patches     : int    total number of patches written
        output_folder : str    path to the output directory
        patch_size    : int    patch size used
        image_shape   : tuple  (height, width, channels) of the source image
        patch_files   : list   sorted list of patch filenames written

    Raises
    ------
    FileNotFoundError
        If image_path does not exist or cannot be read by OpenCV.
    ValueError
        If patch_size <= 0 or overlap >= patch_size.
    """
    # ── Input validation ─────────────────────────────────────────────────────
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Core image not found: {image_path}\n"
            f"Check the path and verify the file exists."
        )

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    if overlap >= patch_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than patch_size ({patch_size})"
        )

    # ── Load image ───────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"OpenCV could not read the image: {image_path}\n"
            f"Verify the file is a valid image (PNG, TIFF, JPEG)."
        )

    h, w = img.shape[:2]   # image height (core length) and width

    # ── Create output directory ──────────────────────────────────────────────
    os.makedirs(output_folder, exist_ok=True)

    # ── Generate patch coordinates ───────────────────────────────────────────
    # step size accounts for overlap: step = patch_size - overlap
    step      = patch_size - overlap
    min_pixels = int(min_coverage * patch_size)   # minimum patch dimension to keep

    patch_files = []   # accumulate filenames for the return dict
    count       = 0    # sequential patch index for filename

    # iterate over y (depth / core length axis) and x (width axis) positions
    for y in range(0, h, step):
        for x in range(0, w, step):

            # compute actual patch boundaries — may be smaller at image edges
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)

            # check minimum coverage: skip very small boundary remnants
            if (y_end - y) < min_pixels or (x_end - x) < min_pixels:
                continue

            # extract patch — may be smaller than patch_size at boundaries
            patch = img[y:y_end, x:x_end]

            # pad smaller boundary patches to patch_size x patch_size with zeros
            # this ensures all output patches have identical dimensions,
            # which is required by most machine learning frameworks
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded       = np.zeros(
                    (patch_size, patch_size, img.shape[2]), dtype=img.dtype
                )
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch        = padded

            # write patch to output folder
            # filename encodes position for later depth reconstruction:
            # patch_NNNN_y{y_start}_x{x_start}.png
            fname = f"patch_{count:04d}_y{y:05d}_x{x:05d}.png"
            cv2.imwrite(os.path.join(output_folder, fname), patch)
            patch_files.append(fname)
            count += 1

    print(
        f"Patch extraction complete:\n"
        f"  Source image:   {image_path}  ({h} x {w} px)\n"
        f"  Patch size:     {patch_size} x {patch_size} px\n"
        f"  Overlap:        {overlap} px\n"
        f"  Patches written:{count}\n"
        f"  Output folder:  {output_folder}"
    )

    return {
        'n_patches'    : count,
        'output_folder': output_folder,
        'patch_size'   : patch_size,
        'image_shape'  : img.shape,
        'patch_files'  : sorted(patch_files),
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_slice(
    input_folder,
    output_root,
    extensions  = ('.tif', '.tiff', '.png', '.jpg', '.jpeg'),
    patch_size  = 256,
    overlap     = 0,
    min_coverage= 0.5,
):
    """
    Slice all core images in a folder into patches.

    Each image gets its own subfolder under output_root, named after
    the image file (without extension). This preserves the source image
    identity for each patch and allows depth reconstruction after labeling.

    Parameters
    ----------
    input_folder : str
        Directory containing core scan images.
    output_root : str
        Root directory for patch output. One subfolder per image.
    extensions : tuple of str
        File extensions to process. Default covers common image formats.
        TIFF (.tif, .tiff) is preferred for lossless core scans.
    patch_size : int
        Side length of each patch in pixels. Default 256.
    overlap : int
        Overlap between patches in pixels. Default 0.
    min_coverage : float
        Minimum fractional coverage for boundary patches. Default 0.5.

    Returns
    -------
    dict with keys:
        n_images      : int   number of images processed
        total_patches : int   total patches written across all images
        results       : list  per-image result dicts from slice_core_image()
        errors        : list  (filename, error_message) for any failures
    """
    # collect all image files with the specified extensions
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in extensions
    ])

    if not image_files:
        print(f"No image files found in: {input_folder}")
        print(f"Expected extensions: {extensions}")
        return {'n_images': 0, 'total_patches': 0, 'results': [], 'errors': []}

    print(f"Found {len(image_files)} image(s) in {input_folder}")

    results       = []
    errors        = []
    total_patches = 0

    for fname in image_files:
        image_path    = os.path.join(input_folder, fname)
        # output subfolder named after the image file (without extension)
        stem          = os.path.splitext(fname)[0]
        output_folder = os.path.join(output_root, stem)

        try:
            result = slice_core_image(
                image_path    = image_path,
                output_folder = output_folder,
                patch_size    = patch_size,
                overlap       = overlap,
                min_coverage  = min_coverage,
            )
            results.append(result)
            total_patches += result['n_patches']

        except (FileNotFoundError, ValueError, Exception) as e:
            # record the error but continue processing remaining images
            errors.append((fname, str(e)))
            print(f"  ERROR: {fname} — {e}")

    print(
        f"\nBatch complete: {len(results)} images processed, "
        f"{total_patches} total patches, {len(errors)} errors"
    )

    return {
        'n_images'     : len(results),
        'total_patches': total_patches,
        'results'      : results,
        'errors'       : errors,
    }


# =============================================================================
# MAIN — DEMONSTRATION
# =============================================================================

if __name__ == '__main__':

    # Example usage — update paths to match the local data structure
    # On Google Colab, set these to the mounted Drive paths
    EXAMPLE_IMAGE  = 'example_core_section.tif'
    EXAMPLE_OUTPUT = 'patches_output'

    if os.path.exists(EXAMPLE_IMAGE):
        result = slice_core_image(
            image_path    = EXAMPLE_IMAGE,
            output_folder = EXAMPLE_OUTPUT,
            patch_size    = 256,
            overlap       = 0,
            min_coverage  = 0.5,
        )
        print(f"\nResult: {result['n_patches']} patches written to {result['output_folder']}")
    else:
        print(f"Example image not found: {EXAMPLE_IMAGE}")
        print("Update EXAMPLE_IMAGE to a valid core scan path to run the demonstration.")
        print("\nTo use in a notebook:")
        print("  from utils.patcher import slice_core_image, batch_slice")
        print("  result = slice_core_image('path/to/core.tif', 'patches/', patch_size=256)")
