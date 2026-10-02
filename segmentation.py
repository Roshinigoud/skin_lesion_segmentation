"""Classical image-processing segmentation for skin lesions."""

import cv2
import numpy as np


def _largest_relevant_component(mask):
    """Keep only the main connected lesion component and discard noise/background."""
    if mask is None or mask.size == 0:
        return None

    mask = np.uint8(mask)
    if not np.any(mask):
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = max(200, int(total_pixels * 0.0005))

    best_label = -1
    best_area = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue

        touches_border = (
            left <= 0
            or top <= 0
            or left + width >= mask.shape[1]
            or top + height >= mask.shape[0]
        )
        if touches_border and area < total_pixels * 0.25:
            continue

        if area > best_area:
            best_area = area
            best_label = label

    if best_label == -1:
        return None

    component = np.zeros_like(mask)
    component[labels == best_label] = 255
    return component


def apply_thresholds(gray_image):
    """Apply two Otsu threshold variations to capture dark or bright lesion regions."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # THRESH_BINARY_INV works well when lesions are darker than the surrounding skin.
    otsu_inverse = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # THRESH_BINARY helps when lesion is brighter than the skin.
    otsu_normal = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return otsu_inverse, otsu_normal


def apply_morphology(mask):
    """Clean the thresholded mask by removing small noise and filling gaps."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_largest_contour(mask):
    """Return the biggest contour in the binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)


def segment_lesion(gray_image):
    """Create a binary mask that highlights the lesion region."""
    if gray_image is None:
        raise ValueError("The grayscale image is empty. Please check preprocessing output.")

    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    best_mask = None
    best_ratio = 0.0

    for candidate in apply_thresholds(gray_image):
        cleaned = apply_morphology(candidate)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        component = _largest_relevant_component(cleaned)
        if component is None:
            continue

        white_ratio = cv2.countNonZero(component) / total_pixels
        if white_ratio < 0.005 or white_ratio > 0.8:
            continue

        if white_ratio > best_ratio:
            best_mask = component
            best_ratio = white_ratio

    if best_mask is None:
        raise ValueError(
            "Lesion mask could not be created. The thresholding and connected-component logic did not identify a valid lesion region."
        )

    # Final validation to avoid accidentally creating a mostly white or mostly black mask.
    valid_ratio = cv2.countNonZero(best_mask) / best_mask.size
    if valid_ratio < 0.005 or valid_ratio > 0.8:
        raise ValueError(
            f"Generated mask is invalid for final segmentation: lesion coverage = {valid_ratio * 100:.2f}%"
        )

    return best_mask
