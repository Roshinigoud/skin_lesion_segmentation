# Automatic Skin Lesion Segmentation

A classical Digital Image Processing (DIP) project that segments skin lesions using color normalization, Otsu thresholding, morphological cleanup, connected-component selection, and a binary lesion mask. The same processing pipeline powers the batch script and Streamlit application; it does not use a machine-learning model.

## DIP Pipeline

Each image passes through these stages:

1. Read the original color image as OpenCV BGR.
2. Resize when needed, smooth, and apply CLAHE color/contrast normalization in LAB.
3. Convert the normalized image to grayscale and generate an inverse Otsu threshold.
4. Apply morphological opening and closing.
5. Select and validate the relevant connected lesion component. Mask background is 0 and lesion foreground is 255.
6. Apply the mask to the original image with `cv2.bitwise_and`.

This is an educational segmentation method. Its results depend on image quality and lesion/background contrast and are not a medical diagnosis.

## Project Structure

```text
Automatic-Skin-Lesion-Segmentation/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .streamlit/config.toml
|-- dataset/input/             # Optional local batch inputs
|-- outputs/
|   |-- normalized/
|   |-- thresholded/
|   |-- morphology/
|   |-- masks/
|   |-- segmented/
|   `-- comparison/
`-- src/
    |-- main.py                 # Batch runner
    |-- pipeline.py             # Shared pipeline and output writer
    |-- preprocessing.py
    `-- segmentation.py
```

## Installation

From the project root, create and activate a virtual environment if desired, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run Batch Processing

Place JPG, JPEG, or PNG images in `dataset/input/`, then run from the project root:

```bash
python src/main.py
```

The batch runner processes every supported input automatically and writes one file per image/stage to the matching `outputs/` folder. Each comparison image contains the original, normalized, thresholded, morphological, mask, and final segmented views, with the filename in its title. Batch runs refresh the generated output folders; input files are not changed.

## Run the Streamlit App

From the project root:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`) in Chrome or another modern browser. Upload one or more JPG, JPEG, or PNG files, inspect the previews, then select **SEGMENT ALL IMAGES**. Results are displayed per image. Each final image has its own PNG download, and the app also offers a ZIP containing all successful segmentations.

The app saves generated stages in the project `outputs/` folders. On hosted Streamlit instances the local filesystem is ephemeral; download results before the instance restarts if they need to be retained.

## Streamlit Community Cloud Deployment

1. Push this project to a GitHub repository, including `app.py`, `src/`, `requirements.txt`, and `.streamlit/config.toml`.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Deploy and open the shareable `*.streamlit.app` URL in Chrome on desktop or phone.

The app uses paths relative to its own project directory and does not depend on the local OneDrive path. Do not publish private or unlicensed input images to a public repository. Streamlit Cloud's local output storage is temporary; use downloads to retain results after a session.

## Dependencies

The direct dependencies are listed in `requirements.txt`: headless OpenCV, NumPy, Matplotlib, and Streamlit. Pillow is installed as a Streamlit dependency and is not directly used by this code.