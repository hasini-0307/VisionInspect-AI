# Conclusion, limitations, future work - raw points

## Conclusion points
- A complete classical pipeline (Labs 1-5) inspects a part in about 20-26 ms (≈ 40-50 FPS) on a laptop, so real-time use with a normal webcam is possible.
- Held-out results: tile accuracy 0.765 / F1 0.736; metal_nut accuracy 0.806 / precision 0.953 / F1 0.766.
- Big, high-contrast or coloured defects (cracks, oil, colour spots, bent/flipped nuts, pen and marker marks, dents) are found reliably and are fully explainable (every decision comes from a visible region and a threshold).
- Simple cues added on top of the adaptive threshold gave the biggest improvements: colour (HSV) for tile and Hu-moment shape matching for metal_nut.

## Limitations (honest)
- **Subtle defects are missed**: metal_nut scratches 3/16, tile glue_strip 5/13, gray_stroke 4/11, rough 6/11. They look like the normal surface texture to a threshold-based method.
- **Recall around 0.65**: about one in three defective parts is passed as good on the held-out split.
- **Texture-heavy or embossed parts** (tile speckles, coin design) create many false candidate regions; the coin only works with settings hand-tuned on its two photos.
- **Settings depend on the product and lighting**: every new object needs its own calibration.
- **Small calibration sets** (49 and 57 tuning images), so the thresholds are somewhat noisy.
- **Own photos**: only one clean and one defective photo per object, so they show that it works, not how often.
- The object mask assumes a plain background with a different brightness/colour from the part.
- Shape check needs a few good reference parts and a clean outline.

## Future work
- Golden-template comparison: align each part to a good reference (ORB matching + homography, Lab 7) and subtract, to catch subtle scratches.
- Texture descriptors from Lab 5 (LBP histograms, HOG) per patch, compared with good patches.
- Learn the decision from many features (elongation, Hu, ORB, edge density, contrast) with a simple classifier such as an SVM, instead of one area threshold.
- Controlled lighting (diffuse ring light, fixed camera height) for the webcam setup.
- Pixel-level evaluation with the MVTec ground-truth masks (IoU), not only image-level accuracy.
- More own-object photos (e.g. 30 good + 30 defective) to report real accuracy for the webcam use case.
- Conveyor demo: trigger inspection when a part enters the frame; log results to a CSV.
