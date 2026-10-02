"""Streamlit interface for the existing skin-lesion DIP pipeline."""

from io import BytesIO
from pathlib import Path
import hashlib
import sys
import zipfile

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import process_image as process_bgr_image  # noqa: E402
from pipeline import save_result_outputs  # noqa: E402


def decode_upload(uploaded_file):
    """Decode a multi-uploader file into the pipeline's BGR image format."""
    image = cv2.imdecode(
        np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image


def safe_output_stem(filename, used_stems):
    """Make an output stem safe and unique for files with duplicate names."""
    stem = Path(Path(filename).name).stem
    count = used_stems.get(stem, 0) + 1
    used_stems[stem] = count
    return stem if count == 1 else f"{stem}_{count}"


def encode_png(image_bgr):
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise IOError("Could not encode the segmented image as PNG.")
    return encoded.tobytes()


def create_segmentations_zip(completed):
    """Build a ZIP archive containing the successful final segmentation PNGs."""
    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in completed:
            archive.writestr(
                f"{item['output_stem']}_segmented.png",
                encode_png(item["result"]["segmented"]),
            )
    return archive_buffer.getvalue()


st.set_page_config(page_title="Automatic Skin Lesion Segmentation", layout="wide")
st.title("AUTOMATIC SKIN LESION SEGMENTATION")
st.write(
    "A Digital Image Processing based skin lesion segmentation system using color "
    "normalization, thresholding, and morphological operations."
)

st.subheader("Upload Images")
uploaded_files = st.file_uploader(
    "Choose one or more JPG, JPEG, or PNG images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more JPG, JPEG, or PNG images to begin segmentation.")
else:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    upload_key = digest.hexdigest()

    st.markdown("#### Uploaded Images")
    preview_columns = st.columns(3)
    for index, uploaded_file in enumerate(uploaded_files):
        with preview_columns[index % len(preview_columns)]:
            st.caption(uploaded_file.name)
            try:
                preview = cv2.cvtColor(decode_upload(uploaded_file), cv2.COLOR_BGR2RGB)
                st.image(preview, width="stretch")
            except ValueError as error:
                st.warning(f"Preview unavailable: {error}")

    if st.button("SEGMENT ALL IMAGES", type="primary"):
        completed = []
        failures = []
        used_stems = {}
        for uploaded_file in uploaded_files:
            try:
                result = process_bgr_image(decode_upload(uploaded_file))
                output_stem = safe_output_stem(uploaded_file.name, used_stems)
                output_paths = save_result_outputs(output_stem, result, PROJECT_ROOT)
                completed.append(
                    {
                        "name": uploaded_file.name,
                        "output_stem": output_stem,
                        "result": result,
                        "output_paths": output_paths,
                    }
                )
            except Exception as error:
                failures.append((uploaded_file.name, str(error)))
        st.session_state["batch_results"] = {
            "upload_key": upload_key,
            "completed": completed,
            "failures": failures,
        }

    stored = st.session_state.get("batch_results")
    if stored and stored["upload_key"] == upload_key:
        for filename, error in stored["failures"]:
            st.error(f"Could not process {filename}: {error}")

        if stored["completed"]:
            st.subheader("Processing Results")
            for image_index, item in enumerate(stored["completed"], start=1):
                result = item["result"]
                st.markdown(f"### Image {image_index}: {item['name']}")
                columns = st.columns(3)
                stages = [
                    (
                        "Original Image",
                        cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB),
                    ),
                    (
                        "Color Normalized / Preprocessed Image",
                        cv2.cvtColor(result["normalized"], cv2.COLOR_BGR2RGB),
                    ),
                    ("Thresholded Image", result["thresholded"]),
                    ("Morphological Processing", result["morphology"]),
                    ("Binary Lesion Mask", result["mask"]),
                ]
                for stage_index, (label, image) in enumerate(stages):
                    columns[stage_index % 3].image(
                        image, caption=label, width="stretch"
                    )

                st.markdown("#### Final Segmented Lesion")
                segmented_rgb = cv2.cvtColor(
                    result["segmented"], cv2.COLOR_BGR2RGB
                )
                st.image(
                    segmented_rgb,
                    caption=f"Lesion mask coverage: {result['mask_coverage']:.1%}",
                    width="stretch",
                )
                segmented_png = encode_png(result["segmented"])
                st.download_button(
                    f"Download {item['name']} Segmentation",
                    data=segmented_png,
                    file_name=f"{item['output_stem']}_segmented.png",
                    mime="image/png",
                    key=f"download-{item['output_stem']}-{image_index}",
                )
                relative_output = Path(item["output_paths"]["segmented"]).relative_to(
                    PROJECT_ROOT
                )
                st.caption(f"Saved output: {relative_output.as_posix()}")

            st.download_button(
                "Download All Segmented Images (ZIP)",
                data=create_segmentations_zip(stored["completed"]),
                file_name="skin_lesion_segmentations.zip",
                mime="application/zip",
                key="download-all-segmentations",
            )
            st.success(
                f"Processed {len(stored['completed'])} of {len(uploaded_files)} uploaded image(s)."
            )