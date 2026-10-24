# Methodology - plain-language notes

## Pipeline (one image)
```
Image ─► 1 Resize to width 400, grayscale            (Lab 1)
      ─► 2 Gamma (LUT) + CLAHE                         (Lab 2)  even out lighting
      ─► 3 Bilateral + median, unsharp, Canny, Sobel   (Lab 3)  remove noise, keep edges
      ─► 4 Object mask: Otsu + morphology (+ margin)   (Lab 4)  inspect only the part
      ─► 5 Gaussian blur (sigma) to hide fine texture  (Lab 3)
      ─► 6 Candidate defects = OR of:                  (Lab 4)
             adaptive threshold, dark spots
             adaptive threshold, bright spots
             HSV saturation differs from the surface   (colour cue)
             Canny edge density differs (texture cue, optional)
      ─► 7 Morphological close, findContours, drop blobs < 30 px
      ─► 8 Features per region: elongation, Hu moment, ORB keypoints,
           edge density, contrast, gradient            (Lab 5)
      ─► 9 Decision: largest region (% of object) ≥ area threshold → DEFECTIVE
           (metal_nut also: Hu-moment outline distance ≥ shape threshold → DEFECTIVE)
      ─► 10 Draw red boxes, yellow contours, PASS/DEFECTIVE banner (Lab 1)
```

## Why each step (one line each)
- **Resize to 400 px**: same scale for every image, and fast enough for real time (~20 ms per image).
- **Gamma**: fixes images that are too dark or too bright overall.
- **CLAHE**: equalises contrast locally (8x8 tiles), so a shadow on one side does not hide a defect; the clip limit stops noise from being boosted too much.
- **Bilateral filter**: smooths flat areas but keeps edges; a defect edge is exactly what we want to keep.
- **Median blur**: removes single bright/dark pixels (sensor noise).
- **Unsharp mask + Canny**: sharper edges give a cleaner edge map; edge density is used as a feature.
- **Gaussian blur (sigma)**: the surface texture (tile speckles, metal grain) is small; blurring removes it so only bigger spots remain. Sigma was calibrated.
- **Otsu object mask**: we only want to inspect the part, not the table. Otsu picks the threshold automatically from the histogram. The mask is eroded by a margin because object edges and shadows are not defects.
- **Background flattening**: background pixels are set to the object's median grey so they do not disturb the adaptive threshold near the object edge.
- **Adaptive threshold**: compares each pixel with the Gaussian-weighted mean of its 51x51 neighbourhood. Works under uneven lighting, unlike one global threshold. Run twice to find dark spots and bright spots.
- **HSV saturation cue**: ink, marker and oil change the colour; a grey/white surface has low, even saturation.
- **Morphological close + contours**: joins broken pieces of one defect, then splits the mask into separate regions.
- **Area rule**: a good surface still gives small candidate spots; only a region bigger than the area threshold (% of object) counts. This keeps the decision simple and explainable.
- **Shape check (metal_nut)**: bent and flipped nuts have a normal surface but a different outline; Hu moments are invariant to position, scale and rotation, so `cv2.matchShapes` can compare outlines of nuts lying at any angle.
- **Confidence**: 0.5 + 0.5 × min(1, |score − threshold| / threshold): 0.5 at the threshold (unsure), 1.0 far from it.
- **Severity**: defect area as % of the object area.
- **Crack vs. blob**: minAreaRect elongation ≥ 3 → "linear: crack/scratch", otherwise "blob: stain/dent".

## Calibration and evaluation (evaluate.py)
1. Images: MVTec test/good = good (0); every other test folder = defective (1). Random train/good images are added so both classes have about the same size (tile 84 + 84, metal_nut 93 + 93).
2. Stratified split per folder with a fixed seed (42): 30% tuning, 70% held-out test.
3. Grid search on the **tuning split only**: smoothing sigma, sensitivity C, colour threshold (or off), texture threshold (or off). For each combination the area threshold (and shape threshold) with the best balanced accuracy is chosen.
4. The chosen settings are saved in `configs/<category>.json` and evaluated **once** on the held-out split.
5. Five good training nuts (not in the evaluation set) are the shape references.
6. Speed = mean pipeline time per image on a MacBook (Apple Silicon), features included.

## Own photos and webcam
- Card, object, coin: object mask = Otsu on brightness AND Otsu on the Lab b channel (warm cream/gold objects on a bluish desk), object assumed in the centre of the frame.
- Settings for the own photos were set by eye on one clean and one defective photo per object; no accuracy is claimed for them.
- Live webcam: verdict = majority of the last 5 frames to stop flicker.

## Design choices that changed during development (see results/improvement_log.md)
- Baseline (adaptive threshold only) was too weak on tile: the speckle texture produced as many blobs as real defects.
- Adding the colour cue (tile) and the Hu shape check (metal_nut) gave the largest gains.
- Fixing the metal_nut mask (only the real centre hole is removed) raised precision to 0.95.
