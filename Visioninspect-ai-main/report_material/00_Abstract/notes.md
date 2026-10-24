# Abstract - raw material

Start from the synopsis abstract (docs/FCV_LAB_SYNOPSIS.docx); it already matches the build. Facts to add / update:

- Datasets used: MVTec AD categories **tile** (textured surface) and **metal_nut** (machined part), plus our own photos of a textured card, a white object and a coin.
- Pipeline actually built: gamma + CLAHE → bilateral + median, Canny, Gaussian → Otsu object mask, adaptive threshold (dark + bright spots), HSV colour cue, morphology, contours → ORB, Hu moments, minAreaRect elongation, edge density → area-threshold decision (+ Hu-moment outline check for metal nut).
- Output: PASS/DEFECTIVE, defect boxes and contours, confidence, severity (% of object), crack vs. blob type, six-stage view, Streamlit GUI + live webcam window.
- Key numbers (held-out test split, calibrated on a separate tuning split):
  - tile: accuracy 0.765, precision 0.848, recall 0.650, F1 0.736 (119 images)
  - metal_nut: accuracy 0.806, precision 0.953, recall 0.641, F1 0.766 (129 images)
  - speed: about 19-26 ms per image (≈ 40-50 FPS) on a laptop CPU
- One honest sentence: subtle low-contrast defects (fine scratches, glue strips) remain hard for classical thresholding.

Note: the synopsis says "accurate visual defect detection"; with recall ≈ 0.65, "explainable, real-time defect detection with good precision" is a fairer wording.
