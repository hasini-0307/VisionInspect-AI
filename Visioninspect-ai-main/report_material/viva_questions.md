# Likely viva questions - short answers

**1. Why CLAHE and not plain histogram equalisation?**
Plain equalisation uses one histogram for the whole image, so a bright or shadowed area dominates. CLAHE equalises
small 8x8 tiles separately, so contrast improves everywhere even with uneven light. The clip limit (≈2) stops it
from boosting noise too much. (equalizeHist is still available as an option.)

**2. Why gamma correction?**
It fixes overall brightness with a power curve (out = in^(1/γ)) using a lookup table, which is very fast. A too-dark
webcam frame becomes usable before thresholding.

**3. Why a bilateral filter (and median)?**
We need to remove noise without blurring defect edges. A bilateral filter averages only pixels that are close in
position *and* similar in brightness, so it smooths flat areas but stops at edges. The 3x3 median then removes
single noisy pixels. A plain Gaussian blur would also blur the edges of cracks.

**4. Then why do you also use a Gaussian blur?**
On purpose, after edge detection: the surface texture (tile speckles, metal grain) is small, and a Gaussian with
sigma ≈ 2-2.5 removes it so only larger spots survive thresholding. Sigma was calibrated on the tuning split.

**5. Why adaptive thresholding instead of one global (Otsu) threshold for defects?**
Lighting changes across the part, so one global value would mark a whole dark side as defective. The adaptive
threshold compares each pixel with the Gaussian-weighted mean of its own 51x51 neighbourhood: "is this spot darker
(or brighter) than what is around it by more than C?" We run it twice, for dark spots and for bright spots
(on 255 − image). Otsu is used where a global split works well: separating the object from the background.

**6. What do ORB and Hu moments do here?**
ORB (oriented FAST keypoints + rotated BRIEF) finds corner-like points; we count keypoints inside each defect, and
real defects (cracks, broken edges) usually have more. Hu moments are 7 numbers describing a shape that do not
change when it moves, rotates or scales. We report the first one per defect, and for the metal nut we compare the
whole outline with good nuts using `cv2.matchShapes` (based on Hu moments). That is how bent and flipped nuts
are found (30 of 33) even when the surface looks normal.

**7. How are crack and blob told apart, and what is severity / confidence?**
`cv2.minAreaRect` gives the tightest rotated box; elongation = long side / short side. ≥ 3 → "linear: crack/scratch",
otherwise "blob: stain/dent". Severity = defect area as % of the object. Image confidence =
0.5 + 0.5·min(1, |score − threshold| / threshold): 0.5 right at the threshold, 1.0 far away from it.

**8. How were the thresholds chosen? Is it not overfitting?**
`evaluate.py` splits the images per folder with a fixed seed: 30% tuning, 70% held-out test. A grid search over
sigma, C and the colour/texture thresholds runs on the tuning images only; for each combination the area threshold
with the best balanced accuracy is picked. The final setting is then tested once on the held-out images, which were
never used for tuning. Classes were balanced with extra good images from train/good.

**9. Why no deep learning?**
The project goal and the lab rules ask for the Lab 1-5 image-processing techniques. Classical methods need no GPU and
no large labelled dataset, run at ≈ 40-50 FPS on a laptop, and every decision is explainable (we can show the region
and the threshold). The trade-off is lower recall on subtle defects than state-of-the-art deep models.

**10. Why are some defects missed?**
Fine scratches on the metal nut look like its normal machining marks, and glue strips / gray strokes / rough patches
on the tile change the texture only slightly. A threshold on brightness, colour or area cannot separate them without
also flagging good parts, so we chose fewer false alarms (precision 0.85-0.95) over catching everything
(recall ≈ 0.65). Future fix: compare against an aligned good template, or LBP/HOG texture features.

## Extra questions worth preparing
- *Why resize to 400 px?* Same scale for every image and fast (about 20 ms per image).
- *Why erode the object mask (margin)?* Object edges, rims and shadows create strong gradients that are not defects.
- *Why the HSV colour cue?* Ink, marker and oil change the saturation; on tile it was the single most useful cue (oil 13/13).
- *What is balanced accuracy?* (recall on defective + recall on good) / 2; fair when classes differ in size.
- *What happens with a new product?* Collect a few good and defective images, run the same calibration, save a new config.
- *Why is the coin a hard case?* Its embossed emblem and letters are strong dark/bright spots, exactly what the adaptive threshold looks for.
