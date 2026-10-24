# Problem statement and objectives - raw material

The synopsis problem statement (docs/FCV_LAB_SYNOPSIS.docx) can be used as is. Points to keep consistent:

## Problem (from the synopsis)
Manual inspection is slow, inconsistent and error-prone; automated systems need expensive hardware.
Build a real-time, low-cost inspection system with a standard webcam and OpenCV that handles varying
lighting, noise and accurate defect localisation, and outputs interpretable results with severity.

## Objectives (as built - each one can be ticked off in the report)
1. Read images / webcam video and normalise them (Lab 1). ✔
2. Correct lighting with gamma and CLAHE (Lab 2). ✔
3. Remove noise while keeping edges (bilateral, median, Gaussian) and detect edges (Canny, Sobel) (Lab 3). ✔
4. Segment the object (Otsu) and candidate defects (adaptive threshold, colour, morphology, contours) (Lab 4). ✔
5. Describe each defect with features (ORB, Hu moments, elongation, edge density) (Lab 5). ✔
6. Give a PASS/DEFECTIVE verdict with defect boxes, confidence and severity. ✔
7. Measure accuracy, precision, recall (+ F1) and FPS on public data (MVTec AD) with a fair tuning/test split. ✔
8. Provide a GUI (Streamlit) and a real-time webcam demo. ✔

## System requirements (the Lab 11 report format asks for these)
- Hardware: any laptop (tested on a MacBook, Apple Silicon), built-in or USB webcam; no GPU.
- Software: Python 3 (tested 3.14), OpenCV 5.0 (opencv-python), NumPy, pandas, Streamlit; VS Code as the IDE.
- Input: images (PNG/JPG) or live webcam; plain, darker background recommended for own objects.
- Performance target: real time (> 15 FPS); achieved ≈ 40-50 FPS on 400-px-wide images.

## Scope check against the synopsis
| Synopsis promise | Status |
|---|---|
| Scratches, dents, cracks | ✔ cracks (tile 11/12), dents (card photo); fine scratches mostly missed (metal_nut 3/16) |
| Missing components, printing errors | Not evaluated: no such data in tile/metal_nut. Closest: bent/flipped nut detected by shape (30/33). Mention as future work or drop from the claims |
| CLAHE + gamma, Gaussian/median/bilateral | ✔ |
| Adaptive threshold + contours | ✔ |
| ORB, Hu moments, shape descriptors | ✔ (ORB and Hu as features; Hu shape matching also used in the decision) |
| Original, intermediate outputs, regions, confidence, annotations | ✔ (six-stage view, defect table) |
| Public datasets + self-collected webcam images | ✔ MVTec AD tile/metal_nut + own photos (only 2 per object) |
| Precision, recall, accuracy, FPS | ✔ (+ F1, confusion matrix) |
