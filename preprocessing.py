"""Image preprocessing for automatic skin lesion segmentation."""

from pathlib import Path

import cv2


def load_image(image_path):
    """Load an image and raise clear errors if the path or file is invalid."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input image not found. Please place an image inside dataset/input/. Path checked: {path}"
        )

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"OpenCV could not read the image: {path}. Please check the file type or image data."
        )

    return image


def resize_image(image, max_size=700):
    """Resize the image while preserving the aspect ratio."""
    height, width = image.shape[:2]
    scale = min(max_size / max(height, width), 1.0)

    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image


def improve_contrast(image):
    """Apply CLAHE in the LAB color space to improve image contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def smooth_image(image):
    """Use Gaussian blur to reduce noise and smooth the image."""
    return cv2.GaussianBlur(image, (5, 5), 0)


def preprocess_color_image(image):
    """Resize, smooth, and normalize illumination/contrast while preserving color."""
    resized = resize_image(image)
    smoothed = smooth_image(resized)
    return improve_contrast(smoothed)


def preprocess_image(image):
    """Apply the main preprocessing steps for lesion detection."""
    normalized = preprocess_color_image(image)
    grayscale = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    return grayscale
