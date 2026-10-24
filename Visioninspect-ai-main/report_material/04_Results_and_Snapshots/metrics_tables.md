# Final metrics (held-out test split)

Calibrated on a 30% tuning split, tested once on the remaining 70% (stratified per folder, seed 42).
Classes balanced by adding random train/good images to test/good.

| Category | Test images | Accuracy | Precision | Recall | F1 | Balanced acc. | TP | TN | FP | FN | ms/image | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tile | 119 | 0.765 | 0.848 | 0.650 | 0.736 | 0.766 | 39 | 52 | 7 | 21 | 25.9 | 39 |
| metal_nut | 129 | 0.806 | 0.953 | 0.641 | 0.766 | 0.805 | 41 | 63 | 2 | 23 | 19.4 | 51 |

## tile: per-folder correct

| Folder | n | correct | accuracy |
|---|---|---|---|
| test/crack | 12 | 11 | 0.92 |
| test/glue_strip | 13 | 5 | 0.38 |
| test/good | 23 | 20 | 0.87 |
| test/gray_stroke | 11 | 4 | 0.36 |
| test/oil | 13 | 13 | 1.00 |
| test/rough | 11 | 6 | 0.55 |
| train/good | 36 | 32 | 0.89 |

Calibrated settings: `{"smooth": 2.0, "C": 40, "use_color": true, "sat_delta": 6, "use_texture": false, "area_thr": 0.6617}`  (tuning balanced accuracy 0.876)

## metal_nut: per-folder correct

| Folder | n | correct | accuracy |
|---|---|---|---|
| test/bent | 17 | 14 | 0.82 |
| test/color | 15 | 8 | 0.53 |
| test/flip | 16 | 16 | 1.00 |
| test/good | 15 | 14 | 0.93 |
| test/scratch | 16 | 3 | 0.19 |
| train/good | 50 | 49 | 0.98 |

Calibrated settings: `{"smooth": 2.5, "C": 40, "use_color": true, "sat_delta": 25, "area_thr": 1.4856, "shape_thr": 0.00067}`  (tuning balanced accuracy 0.792)

## Baseline vs final (see improvement_log.md)

| Category | Version | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| tile | baseline (adaptive threshold only) | 0.538 | 1.000 | 0.083 | 0.154 |
| tile | final (+ HSV colour cue) | 0.765 | 0.848 | 0.650 | 0.736 |
| metal_nut | baseline | 0.612 | 0.889 | 0.250 | 0.390 |
| metal_nut | final (+ Hu shape check, colour cue, mask fix) | 0.806 | 0.953 | 0.641 | 0.766 |

## Own photos (visual demonstration only, 1 clean + 1 defective photo per object)

| Object | Photo | Expected | Verdict | Largest defect % | Confidence |
|---|---|---|---|---|---|
| card | good_01.jpeg | PASS | PASS | 0.0 | 1.0 |
| card | defect_01.jpeg | DEFECTIVE | DEFECTIVE | 2.16 | 1.0 |
| coin | good_01.jpeg | PASS | PASS | 0.0 | 1.0 |
| coin | defect_01.jpeg | DEFECTIVE | DEFECTIVE | 1.86 | 0.93 |
| object | good_01.jpeg | PASS | PASS | 0.0 | 1.0 |
| object | defect_01.jpeg | DEFECTIVE | DEFECTIVE | 6.67 | 1.0 |

Coin settings were hand-tuned on its two photos (C=40, colour threshold 30, sigma 3); with the generic webcam profile the clean coin is wrongly flagged because of its embossed design.
