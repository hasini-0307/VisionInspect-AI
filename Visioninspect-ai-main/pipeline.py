"""
VisionInspect AI - core inspection pipeline (classical OpenCV only, no deep learning).

Stages (and the FCV lab each one comes from):
  1. Read, resize to width 400, grayscale                      (Lab 1)
  2. Gamma correction (LUT) + CLAHE / histogram equalisation   (Lab 2)
  3. Bilateral + median filter, unsharp mask, Canny, Sobel,
     Gaussian smoothing to hide fine texture                   (Lab 3)
  4. Object mask (Otsu + morphology), adaptive threshold for
     dark and bright spots, HSV colour cue, contours           (Lab 4)
  5. Per-defect features: minAreaRect elongation, Hu moments,
     ORB keypoints, edge density, contrast                     (Lab 5)
  6. Decision: largest defect area (% of object) vs threshold, then
     draw boxes / contours / PASS-DEFECTIVE banner             (Lab 1 drawing)
"""
import json
import os
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "configs")
WIDTH = 400

# Settings shared by every category. Each category below overrides some of them,
# and configs/<category>.json (written by evaluate.py) overrides those again.
BASE = dict(
    gamma=1.0,          # Lab 2: >1 brightens, <1 darkens
    enhance="clahe",    # Lab 2: "clahe", "histeq" or "none"
    clahe_clip=2.0,
    smooth=2.0,         # Lab 3: Gaussian sigma that hides fine surface texture
    block=51,           # Lab 4: adaptive threshold neighbourhood (odd number)
    C=10,               # Lab 4: sensitivity - how much darker/brighter than the neighbourhood
    min_area=30,        # blobs smaller than this (pixels) are ignored as noise
    area_thr=1.0,       # FAIL if the largest blob covers >= this % of the object
    mask_mode="full",   # "full" = whole image minus a border, "otsu" = segment the object
    margin=10,          # pixels ignored next to the image border / object edge
    fill_holes=True,    # fill holes inside the Otsu object (False keeps e.g. the nut's centre hole out)
    use_color=False,    # HSV saturation cue for coloured marks (pen, marker, oil tint)
    sat_delta=60,       # how far a pixel's saturation may differ from the surface's median
    use_texture=False,  # edge-density cue for patches that are too smooth / too rough
    tex_delta=8,        # allowed difference in edge density (% of pixels that are Canny edges)
    region_sigma=2.0,   # Gaussian sigma used to blur the colour and texture maps
    use_shape=False,    # Lab 5: compare the object outline with good reference outlines (Hu moments)
    shape_thr=0.0,      # FAIL if cv2.matchShapes distance to the closest reference >= this
    shape_refs=[],      # reference outlines (lists of points), filled in by evaluate.py
)

DEFAULTS = {
    "tile": dict(mask_mode="full", margin=10, use_color=True, use_texture=True, region_sigma=6.0),
    "metal_nut": dict(mask_mode="otsu", margin=6, fill_holes=False, use_color=True, use_shape=True),
    # Own phone photos: hand-set by eye on a couple of photos (not calibrated)
    "card": dict(mask_mode="otsu_color", margin=15, use_color=True),
    "coin": dict(mask_mode="otsu_color", margin=12, use_color=True, smooth=3.0, C=40, sat_delta=30),
    "object": dict(mask_mode="otsu_color", margin=16, use_color=True),
    # Live webcam: plain, light-coloured object on a darker background
    "webcam": dict(mask_mode="otsu", margin=16, use_color=True),
}
CATEGORIES = list(DEFAULTS)


def load_config(category):
    """Defaults for the category, overridden by configs/<category>.json if it exists."""
    cfg = dict(BASE)
    cfg.update(DEFAULTS.get(category, {}))
    path = os.path.join(CONFIG_DIR, f"{category}.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f).get("params", {}))
    return cfg


# ---------------------------------------------------------------- Lab 1
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_width(img, width=WIDTH):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- Lab 2
def gamma_correct(gray, gamma):
    if abs(gamma - 1.0) < 1e-3:
        return gray
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(gray, table)


def enhance(gray, cfg):
    out = gamma_correct(gray, cfg["gamma"])
    if cfg["enhance"] == "clahe":
        out = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(8, 8)).apply(out)
    elif cfg["enhance"] == "histeq":
        out = cv2.equalizeHist(out)
    return out


# ---------------------------------------------------------------- Lab 4 (object mask)
def otsu(channel, centre_rule=False):
    """Otsu threshold, flipped if needed so that the object is white.
    Border rule: the image border is background. Centre rule: the object is in the middle of the frame."""
    blur = cv2.GaussianBlur(channel, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = th.shape
    if centre_rule:
        flip = th[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5].mean() < 127
    else:
        flip = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]]).mean() > 127
    return 255 - th if flip else th


def object_mask(gray, cfg, color=None):
    """Returns (object, inspect): the whole object, and the part we inspect (object minus a margin).
    Modes: "full" whole image; "otsu" Otsu on brightness; "otsu_color" brightness AND Lab b channel
    (warm cream/gold objects on a bluish-grey desk, robust to an unevenly lit desk)."""
    h, w = gray.shape
    m = cfg["margin"]
    if cfg["mask_mode"] == "full":
        obj = np.full((h, w), 255, np.uint8)
        inspect = np.zeros((h, w), np.uint8)
        inspect[m:h - m, m:w - m] = 255
        return obj, inspect

    if cfg["mask_mode"] == "otsu_color" and color is not None:
        b = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)[:, :, 2]
        th = cv2.bitwise_and(otsu(gray, centre_rule=True), otsu(b, centre_rule=True))
    else:
        th = otsu(gray)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # keep only the largest object
    contours, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    obj = np.zeros_like(th)
    if contours:
        big = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        cv2.drawContours(obj, contours, big, 255, cv2.FILLED)
        if not cfg["fill_holes"]:
            # cut the largest hole back out (the centre hole of the nut); smaller "holes"
            # are just dark texture that Otsu mislabelled, so they stay filled
            holes = [i for i, hr in enumerate(hier[0]) if hr[3] == big]
            if holes:
                hole = max(holes, key=lambda i: cv2.contourArea(contours[i]))
                cv2.drawContours(obj, contours, hole, 0, cv2.FILLED)
    inspect = cv2.erode(obj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1)))
    return obj, inspect


# ---------------------------------------------------------------- main pipeline
def inspect_image(img_bgr, cfg, with_features=True):
    """Run the full pipeline on a BGR image. Returns a dict with verdict, score, defects and stage images."""
    t0 = time.perf_counter()

    # Lab 1: resize + grayscale
    color = resize_width(img_bgr)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # Lab 2: gamma + CLAHE
    enh = enhance(gray, cfg)

    # Lab 3: edge-preserving denoise, unsharp mask, Canny, Sobel
    den = cv2.medianBlur(cv2.bilateralFilter(enh, 7, 50, 50), 3)
    sharp = cv2.addWeighted(den, 1.5, cv2.GaussianBlur(den, (0, 0), 3), -0.5, 0)
    edges = cv2.Canny(sharp, 50, 150)

    # Lab 4: object mask
    obj, insp = object_mask(gray, cfg, color)
    obj_area = max(int(cv2.countNonZero(insp)), 1)

    # Flatten the background to the object's median grey so it does not disturb
    # the local mean of the adaptive threshold near the object edge.
    work = den.copy()
    if cv2.countNonZero(obj):
        work[obj == 0] = int(np.median(den[obj > 0]))
    # Lab 3: Gaussian smoothing hides fine texture; only larger spots survive
    smooth = cv2.GaussianBlur(work, (0, 0), cfg["smooth"])

    # Lab 4: adaptive threshold twice - dark spots and bright spots
    block, C = int(cfg["block"]) | 1, float(cfg["C"])
    dark = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)
    bright = cv2.adaptiveThreshold(255 - smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)
    cand = cv2.bitwise_or(dark, bright)

    # Lab 4 (colour segmentation): ink / oil changes the saturation of a uniform surface
    has_obj = cv2.countNonZero(insp) > 0
    if cfg["use_color"] and has_obj:
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        sat = cv2.GaussianBlur(hsv[:, :, 1].astype(np.float32), (0, 0), cfg["region_sigma"])
        dev = np.abs(sat - np.median(sat[insp > 0]))
        colour = ((dev > cfg["sat_delta"]) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
        cand = cv2.bitwise_or(cand, colour)

    # Texture cue: % of Canny edge pixels around each point; a glue patch is too smooth,
    # a rough patch too busy compared with the rest of the surface
    if cfg["use_texture"] and has_obj:
        density = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), cfg["region_sigma"]) * (100.0 / 255)
        dev = np.abs(density - np.median(density[insp > 0]))
        cand = cv2.bitwise_or(cand, (dev > cfg["tex_delta"]).astype(np.uint8) * 255)

    cand = cv2.bitwise_and(cand, insp)
    clean = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        if cv2.contourArea(c) < cfg["min_area"] * 0.25:   # quick reject before counting pixels
            continue
        region = np.zeros_like(clean)
        cv2.drawContours(region, [c], -1, 255, cv2.FILLED)
        region = cv2.bitwise_and(region, insp)
        area = cv2.countNonZero(cv2.bitwise_and(clean, region))
        if area >= cfg["min_area"]:
            blobs.append((c, area))
    blobs.sort(key=lambda b: -b[1])

    score = 100.0 * blobs[0][1] / obj_area if blobs else 0.0
    thr = float(cfg["area_thr"])
    shape_dist = shape_distance(obj, cfg)

    # Decision: each cue divided by its threshold; >= 1 means that cue says "defective".
    # With only the area cue, confidence = 0.5 + 0.5 * min(1, |score - thr| / thr).
    ratio = score / thr if thr > 0 else 0.0
    if shape_dist is not None:
        ratio = max(ratio, shape_dist / cfg["shape_thr"])
    defective = ratio >= 1.0
    confidence = 0.5 + 0.5 * min(1.0, abs(ratio - 1.0))

    # Lab 5: features for each blob
    keypoints = []
    defects = []
    if with_features:
        keypoints = cv2.ORB_create(nfeatures=500).detect(enh, insp)
        gx = cv2.Sobel(den, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(den, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        for i, (c, area) in enumerate(blobs[:20]):
            defects.append(defect_features(i + 1, c, area, obj_area, thr, den, edges, grad, keypoints))

    ms = (time.perf_counter() - t0) * 1000.0
    shape_bad = shape_dist is not None and shape_dist >= cfg["shape_thr"]
    annotated = annotate(color, blobs, obj_area, thr, defective, confidence, score,
                         outline_bad=obj if shape_bad else None)
    return dict(
        defective=bool(defective),
        verdict="DEFECTIVE" if defective else "PASS",
        score=score,
        shape_dist=shape_dist,
        shape_bad=bool(shape_bad),
        confidence=confidence,
        n_blobs=len(blobs),
        n_defects=sum(1 for _, a in blobs if 100.0 * a / obj_area >= thr),
        defects=defects,
        ms=ms,
        annotated=annotated,
        stages=dict(color=color, enhanced=enh, edges=edges, smooth=smooth,
                    candidates=cand, clean=clean, object=obj, inspect=insp,
                    blobs=blobs, keypoints=keypoints),
    )


def outer_contour(obj):
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def shape_distance(obj, cfg):
    """Lab 5: Hu-moment shape distance (cv2.matchShapes) between the object's outline and the
    closest good reference outline. A bent or flipped part has a different outline. None if unused."""
    if not (cfg["use_shape"] and cfg["shape_refs"] and cfg["shape_thr"] > 0):
        return None
    c = outer_contour(obj)
    if c is None:
        return None
    refs = [np.array(r, np.int32).reshape(-1, 1, 2) for r in cfg["shape_refs"]]
    return min(cv2.matchShapes(c, r, cv2.CONTOURS_MATCH_I1, 0) for r in refs)


def blob_area(contour, shape):
    """Pixel count of a filled contour (contourArea under-counts thin cracks)."""
    x, y, w, h = cv2.boundingRect(contour)
    roi = np.zeros((h, w), np.uint8)
    cv2.drawContours(roi, [contour - [x, y]], -1, 255, cv2.FILLED)
    return int(cv2.countNonZero(roi))


# ---------------------------------------------------------------- Lab 5 features
def defect_features(idx, c, area, obj_area, thr, den, edges, grad, keypoints):
    region = np.zeros(den.shape, np.uint8)
    cv2.drawContours(region, [c], -1, 255, cv2.FILLED)
    ring = cv2.subtract(cv2.dilate(region, np.ones((9, 9), np.uint8)), region)

    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    elong = max(rw, rh) / max(min(rw, rh), 1.0)
    hu = cv2.HuMoments(cv2.moments(region, binaryImage=True)).flatten()
    hu1 = -np.sign(hu[0]) * np.log10(abs(hu[0])) if hu[0] != 0 else 0.0

    grown = cv2.dilate(region, np.ones((7, 7), np.uint8))
    n_kp = sum(1 for k in keypoints
               if grown[min(int(k.pt[1]), den.shape[0] - 1), min(int(k.pt[0]), den.shape[1] - 1)])
    edge_density = cv2.countNonZero(cv2.bitwise_and(edges, region)) / max(area, 1)
    contrast = abs(cv2.mean(den, region)[0] - cv2.mean(den, ring)[0]) if cv2.countNonZero(ring) else 0.0
    gradient = cv2.mean(grad, region)[0]
    severity = 100.0 * area / obj_area

    # bigger, higher-contrast, edgier blobs -> more confident it is a real defect
    conf = (0.4 * min(1.0, severity / thr if thr > 0 else 1.0)
            + 0.3 * min(1.0, contrast / 40.0)
            + 0.3 * min(1.0, edge_density / 0.3))
    x, y, w, h = cv2.boundingRect(c)
    return dict(
        id=idx,
        type="linear: crack/scratch" if elong >= 3 else "blob: stain/dent",
        area_px=int(area),
        severity_pct=round(severity, 3),
        elongation=round(elong, 2),
        hu1_log=round(float(hu1), 3),
        orb_keypoints=int(n_kp),
        edge_density=round(edge_density, 3),
        contrast=round(contrast, 1),
        gradient=round(gradient, 1),
        confidence=round(conf, 2),
        bbox=[int(x), int(y), int(w), int(h)],
    )


# ---------------------------------------------------------------- drawing (Lab 1)
def annotate(color, blobs, obj_area, thr, defective, confidence, score, outline_bad=None):
    out = color.copy()
    if outline_bad is not None:                                         # magenta: outline does not match
        c = outer_contour(outline_bad)
        if c is not None:
            cv2.drawContours(out, [c], -1, (255, 0, 255), 2)
            cv2.putText(out, "shape mismatch", (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 0, 255), 2, cv2.LINE_AA)
    for i, (c, area) in enumerate(blobs):                               # blobs are sorted largest first
        if i < 10:
            cv2.drawContours(out, [c], -1, (0, 255, 255), 1)          # yellow: the 10 largest candidates
        if 100.0 * area / obj_area >= thr:                              # red box: big enough to fail
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)
    bg = (0, 0, 200) if defective else (0, 150, 0)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), bg, cv2.FILLED)
    text = f"{'DEFECTIVE' if defective else 'PASS'}  conf {confidence:.2f}  defect {score:.2f}%"
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


STAGE_TITLES = [
    "1 Input (Lab 1: read/resize)",
    "2 Gamma + CLAHE (Lab 2)",
    "3 Bilateral+median+Canny (Lab 3)",
    "4 Adaptive thr + mask (Lab 4)",
    "5 Contours + ORB (Lab 4/5)",
    "6 Result",
]


def stage_images(res):
    """The six stage images (BGR) in display order, matching STAGE_TITLES."""
    s = res["stages"]
    to_bgr = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    thr_view = to_bgr(s["candidates"])
    outline, _ = cv2.findContours(s["inspect"], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thr_view, outline, -1, (255, 128, 0), 1)       # blue: inspected area

    blob_view = to_bgr(s["enhanced"] // 2)
    cv2.drawContours(blob_view, [c for c, _ in s["blobs"]], -1, (0, 255, 255), cv2.FILLED)
    blob_view = cv2.drawKeypoints(blob_view, s["keypoints"], None, (0, 255, 0))

    return [s["color"], to_bgr(s["enhanced"]), to_bgr(s["edges"]), thr_view, blob_view, res["annotated"]]


def montage(res, titles=STAGE_TITLES):
    """2x3 grid of all pipeline stages with captions, for the report."""
    panels = []
    for img, title in zip(stage_images(res), titles):
        bar = np.zeros((26, img.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, img]))
    return np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
