# VisionInspect AI 🔍

A computer vision mini project for surface inspection and defect detection using **classical OpenCV techniques—without deep learning**.

Developed for **FCV Lab (CSE 3176), Manipal Institute of Technology**.

## Overview

VisionInspect AI processes uploaded images, dataset samples, and webcam snapshots to identify candidate surface defects. A Streamlit interface displays inspection results, highlighted regions, and intermediate processing stages.

The project explores how image enhancement, segmentation, contours, and feature extraction can support industrial visual inspection.

## Features

- Image upload and webcam snapshot inspection
- Separate live webcam demo
- PASS / DEFECTIVE verdict based on configurable thresholds
- Candidate-region contours and bounding boxes
- Six-stage processing visualisation
- Adjustable sensitivity, smoothing, gamma, and area thresholds
- Region features including elongation, Hu moments, ORB keypoints, and edge density
- Shape comparison for metal nuts using reference contours
- Downloadable processing-stage montages

## Processing Pipeline

1. **Preprocessing:** resize images and convert to grayscale.
2. **Enhancement:** apply gamma correction and CLAHE.
3. **Filtering:** use bilateral, median, and Gaussian filters.
4. **Segmentation:** generate object masks and candidate regions using Otsu and adaptive thresholding, with optional colour and texture cues.
5. **Feature extraction:** calculate contour geometry, Hu moments, ORB keypoints, and edge statistics.
6. **Decision:** compare candidate-region area and optional shape distance against configured thresholds.

## Technologies

Python · OpenCV · NumPy · Pandas · Streamlit

## Project Files

The application files are inside `Visioninspect-ai-main/`.

| File / Folder | Purpose |
|---|---|
| `pipeline.py` | Image processing and inspection logic |
| `app.py` | Streamlit application |
| `live_demo.py` | Live webcam inspection |
| `evaluate.py` | Parameter calibration and evaluation |
| `configs/` | Category-specific settings |
| `data/` | Local datasets and sample images |
| `results/` | Saved outputs and evaluation results |
| `report_material/` | Supporting report and presentation material |

## Run Locally — Windows

Open a terminal in the folder containing `app.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the application:

http://localhost:8501

Open webcam snapshot mode directly:

http://localhost:8501/?category=webcam

For the separate live webcam window:

```powershell
.\.venv\Scripts\python.exe live_demo.py
```

Press `q` to quit, `s` to save a frame, or `m` to toggle processing stages.

## Dataset

The project supports the **tile** and **metal_nut** categories from MVTec AD.

Download these separately from the official dataset page:

https://www.mvtec.com/company/research/datasets/mvtec-ad

Follow the dataset's licence and attribution requirements. Extract the categories into the application's `data/` directory:

```text
data/
├── tile/
│   ├── train/
│   ├── test/
│   └── ground_truth/
└── metal_nut/
    ├── train/
    ├── test/
    └── ground_truth/
```

Dataset sample mode requires these files. Image uploads and webcam capture can be used without downloading MVTec AD.

## Limitations

- This is an educational prototype, not a production quality-control system.
- Settings depend on the object category, lighting, background, and camera position.
- Normal texture, colour patterns, shadows, and reflections can produce false positives.
- Small or low-contrast defects may be missed.
- The displayed confidence is a heuristic threshold score, not a calibrated probability.
- Changes to the pipeline require fresh evaluation; earlier saved metrics may not describe the modified version.

