"""Main workflow for automatic skin lesion segmentation."""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipeline import process_image, save_result_outputs


def find_input_images(input_dir):
    """Find supported images regardless of extension letter case."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def clear_directory(directory):
    """Remove stale files from a directory so each run shows the newest outputs."""
    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            for child in item.iterdir():
                if child.is_file():
                    child.unlink()


def save_image(path, image):
    """Save image data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Could not save image to: {path}")


def prepare_output_structure(project_root):
    """Create all required output folders and clear stale files."""
    output_root = project_root / "outputs"
    stage_folders = {
        "normalized": output_root / "normalized",
        "thresholded": output_root / "thresholded",
        "morphology": output_root / "morphology",
        "masks": output_root / "masks",
        "segmented": output_root / "segmented",
        "comparison": output_root / "comparison",
    }

    for folder in stage_folders.values():
        clear_directory(folder)

    dataset_output = project_root / "dataset" / "output"
    dataset_output.mkdir(parents=True, exist_ok=True)
    clear_directory(dataset_output)

    return output_root, stage_folders, dataset_output


def apply_mask_to_image(original_bgr, mask):
    """Apply a binary mask to keep only the lesion region."""
    return cv2.bitwise_and(original_bgr, original_bgr, mask=mask)


def main():
    """Process all images in dataset/input and save normalized, thresholded, mask, and segmented outputs."""
    input_dir = project_root / "dataset" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    image_files = find_input_images(input_dir)
    if not image_files:
        print("No JPG, JPEG, or PNG images were found in dataset/input/.")
        return

    output_root, stage_folders, _ = prepare_output_structure(project_root)
    print(f"Found {len(image_files)} image(s) in {input_dir}")
    used_stems = {}
    processed = 0
    failed = 0

    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original_bgr is None:
            print(f"ERROR: OpenCV could not read {image_path.name}")
            failed += 1
            continue
        try:
            result = process_image(original_bgr)
            base_name = image_path.stem
            count = used_stems.get(base_name, 0) + 1
            used_stems[base_name] = count
            if count > 1:
                base_name = f"{base_name}_{count}"
            save_result_outputs(base_name, result, project_root)
            print(f"Mask coverage: {result['mask_coverage']:.2%}")
            print(f"Saved all stages for {image_path.name}")
            processed += 1
        except Exception as error:
            print(f"ERROR processing {image_path.name}: {error}")
            failed += 1

    print(f"\nFinished: {processed} processed, {failed} failed.")
    print(f"Outputs: {output_root}")
    for name, folder in stage_folders.items():
        print(f"  {name}: {len(list(folder.glob('*')))}")


if __name__ == "__main__":
    main()
