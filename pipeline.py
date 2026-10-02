"""Shared image-processing pipeline and artifact writer for the project."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import preprocess_color_image
from segmentation import apply_morphology, apply_thresholds, segment_lesion


def process_image(original_bgr):
    """Return normalized, threshold, morphology, mask, and segmented image stages."""
    if original_bgr is None or original_bgr.size == 0:
        raise ValueError("The input image is empty or invalid.")
    if original_bgr.ndim != 3 or original_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR color image.")

    normalized_bgr = preprocess_color_image(original_bgr)
    normalized_gray = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2GRAY)
    thresholded = apply_thresholds(normalized_gray)[0]
    morphology = apply_morphology(thresholded)
    mask = segment_lesion(normalized_gray)

    if mask.shape != original_bgr.shape[:2]:
        mask = cv2.resize(
            mask,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    coverage = cv2.countNonZero(mask) / mask.size
    if coverage < 0.005 or coverage > 0.8:
        raise ValueError(
            f"Generated lesion mask has invalid coverage ({coverage:.1%})."
        )

    segmented_bgr = cv2.bitwise_and(original_bgr, original_bgr, mask=mask)
    return {
        "original": original_bgr,
        "normalized": normalized_bgr,
        "thresholded": thresholded,
        "morphology": morphology,
        "mask": mask,
        "segmented": segmented_bgr,
        "mask_coverage": coverage,
    }


def save_result_outputs(base_name, result, project_root):
    """Save the five processing stages, final segmentation, and comparison figure."""
    output_root = Path(project_root) / "outputs"
    stage_folders = {
        "normalized": output_root / "normalized",
        "thresholded": output_root / "thresholded",
        "morphology": output_root / "morphology",
        "masks": output_root / "masks",
        "segmented": output_root / "segmented",
        "comparison": output_root / "comparison",
    }
    for folder in stage_folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    files = {
        "normalized": ("normalized", f"{base_name}_normalized.png"),
        "thresholded": ("thresholded", f"{base_name}_thresholded.png"),
        "morphology": ("morphology", f"{base_name}_morphology.png"),
        "masks": ("mask", f"{base_name}_mask.png"),
        "segmented": ("segmented", f"{base_name}_segmented.png"),
    }
    saved_paths = {}
    for folder_name, (result_key, filename) in files.items():
        path = stage_folders[folder_name] / filename
        if not cv2.imwrite(str(path), result[result_key]):
            raise IOError(f"Could not save image to {path}")
        saved_paths[folder_name] = path

    original_rgb = cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB)
    normalized_rgb = cv2.cvtColor(result["normalized"], cv2.COLOR_BGR2RGB)
    segmented_rgb = cv2.cvtColor(result["segmented"], cv2.COLOR_BGR2RGB)
    visualizations = [
        (original_rgb, "Original Image", None),
        (normalized_rgb, "Color Normalized / Preprocessed Image", None),
        (result["thresholded"], "Thresholded Image", "gray"),
        (result["morphology"], "Morphological Processing", "gray"),
        (result["mask"], "Binary Lesion Mask", "gray"),
        (segmented_rgb, "Final Segmented Lesion", None),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (image, title, cmap) in zip(axes.flat, visualizations):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Automatic Skin Lesion Segmentation - {base_name}")
    figure.tight_layout()
    comparison_path = stage_folders["comparison"] / f"{base_name}_comparison.png"
    figure.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    saved_paths["comparison"] = comparison_path
    return saved_paths