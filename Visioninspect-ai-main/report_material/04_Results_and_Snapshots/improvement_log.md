# Improvement log

All numbers are on the **held-out test split** (70%, seed 42). Every setting was chosen on the
30% tuning split only; held-out images were never used for tuning.
tile: 119 held-out images, metal_nut: 129 held-out images.

| Version | Change | Category | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| v1 baseline | Pipeline as specified: gamma + CLAHE, bilateral + median, Gaussian (sigma 1-3), adaptive threshold (dark + bright), close, contours, largest-blob area rule. Grid C 3-25. | tile | 0.538 | 1.000 | 0.083 | 0.154 |
| | | metal_nut | 0.612 | 0.889 | 0.250 | 0.390 |
| v2 | **tile:** added HSV colour cue (saturation differs from the surface median) and optional Canny edge-density (texture) cue, both blurred with sigma 6; C grid extended to 40 because low C turns the tile speckles into one big blob. Calibration chose colour on, texture off. **metal_nut:** added Hu-moment shape check (`cv2.matchShapes` against 5 good training nuts that are not in the evaluation set) and the colour cue. | tile | 0.765 | 0.848 | 0.650 | 0.736 |
| | | metal_nut | 0.744 | 0.830 | 0.609 | 0.703 |
| v3 | **metal_nut:** fixed the object mask. Otsu marked the dark textured ring inside the nut as background, so many "holes" were cut out of the inspected area. Now only the largest hole (the real centre hole) is removed. (tile unchanged: it uses the full-image mask.) | metal_nut | 0.806 | 0.953 | 0.641 | 0.766 |

## Tried and not kept (tuning split only)
- tile: larger Gaussian sigma (4-8), CLAHE off, block 101 for the adaptive threshold: best tuning balanced accuracy 0.65, so local thresholding alone cannot separate the speckle texture from the defects.
- tile: texture cue at thresholds 6-20: it catches glue_strip and rough well but adds false alarms on good tiles; the OR rule with one area threshold did not improve tuning accuracy, so calibration switched it off.
- metal_nut: C above 40 (i.e. adaptive-threshold cue effectively off) gave the same tuning accuracy as C = 40. On the nut the spot cue adds almost nothing; colour + shape do the work.

## What still fails (honest limits)
- metal_nut **scratch** (3/16): thin, low-contrast scratches look like the nut's normal machining marks.
- tile **glue_strip / gray_stroke / rough**: soft, low-contrast changes in texture; they are caught only when the colour shifts.
- Small tuning set (49 / 57 images), so the chosen thresholds are somewhat noisy.
