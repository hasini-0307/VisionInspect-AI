"""
VisionInspect AI - Streamlit front end.

    python3 -m streamlit run app.py
"""
import glob
import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P

DATA = os.path.join(P.ROOT, "data")
RESULTS = os.path.join(P.ROOT, "results")

CATEGORY_LABELS = {
    "tile": "Tile (MVTec AD)",
    "metal_nut": "Metal nut (MVTec AD)",
    "card": "Textured card (own photos)",
    "object": "White object (own photos)",
    "coin": "Coin (own photos - hard case)",
    "webcam": "Webcam (plain object)",
}
INPUTS = {
    "tile": ["Dataset sample", "Upload image", "Webcam snapshot"],
    "metal_nut": ["Dataset sample", "Upload image", "Webcam snapshot"],
    "card": ["My own photos", "Upload image", "Webcam snapshot"],
    "object": ["My own photos", "Upload image", "Webcam snapshot"],
    "coin": ["My own photos", "Upload image", "Webcam snapshot"],
    "webcam": ["Webcam snapshot", "Upload image", "My own photos"],
}
STAGE_CAPTIONS = [
    ("1. Input", "Lab 1 - imread, resize to width 400 (INTER_AREA)"),
    ("2. Enhanced", "Lab 2 - gamma correction (LUT) + CLAHE"),
    ("3. Filtered + edges", "Lab 3 - bilateral + median, unsharp mask, Canny"),
    ("4. Segmentation", "Lab 4 - adaptive threshold (dark + bright) + colour cue, inside the Otsu object mask (blue)"),
    ("5. Defect regions", "Lab 4/5 - morphology, contours (yellow), ORB keypoints (green)"),
    ("6. Result", "Lab 1 drawing - red boxes, yellow contours, verdict banner"),
]

st.set_page_config(page_title="VisionInspect AI", page_icon="🔍", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.verdict {border-radius: 10px; padding: 14px 20px; color: white; font-size: 1.6rem; font-weight: 700;
          display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;}
.verdict small {font-size: 0.95rem; font-weight: 400; opacity: 0.9;}
.pass {background: #1e8e3e;} .fail {background: #c5221f;}
.gt {font-size: 0.9rem; padding: 4px 0;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def warm_up():
    """The first OpenCV call is slow (~60 ms); run it once so the shown timing is realistic."""
    P.inspect_image(np.full((300, 300, 3), 128, np.uint8), P.load_config("tile"))
    return True


warm_up()


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data
def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def ground_truth_overlay(img, path):
    """Draw the MVTec ground-truth defect outline in green (dataset samples only)."""
    folder, name = os.path.basename(os.path.dirname(path)), os.path.splitext(os.path.basename(path))[0]
    category_root = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    mask_path = os.path.join(category_root, "ground_truth", folder, f"{name}_mask.png")
    if not os.path.exists(mask_path):
        return img
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), max(2, img.shape[1] // 200))
    return out


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("VisionInspect AI")
    # a link like ?category=tile&folder=crack opens that view directly (handy demo bookmarks)
    qp = st.query_params
    cats = list(CATEGORY_LABELS)
    cat = st.selectbox("Category", cats, format_func=CATEGORY_LABELS.get,
                       index=cats.index(qp["category"]) if qp.get("category") in cats else 0)
    source = st.radio("Input", INPUTS[cat])
    cfg = P.load_config(cat)

    st.subheader("Settings")
    if st.button("Reset to calibrated values", width="stretch"):
        for k in [k for k in st.session_state if k.endswith(f"_{cat}")]:
            del st.session_state[k]
    cfg["C"] = st.slider("Sensitivity C (higher = less sensitive)", 3, 60, int(cfg["C"]), key=f"C_{cat}",
                         help="Adaptive threshold constant: how much darker/brighter than its neighbourhood a pixel must be.")
    cfg["area_thr"] = st.slider("Area threshold (% of object)", 0.05, 10.0, float(round(cfg["area_thr"], 2)), 0.05,
                                key=f"thr_{cat}", help="DEFECTIVE if the largest defect covers at least this % of the object.")
    cfg["min_area"] = st.slider("Min defect size (pixels)", 5, 300, int(cfg["min_area"]), key=f"min_{cat}")
    cfg["gamma"] = st.slider("Gamma", 0.4, 2.5, float(cfg["gamma"]), 0.05, key=f"gamma_{cat}",
                             help=">1 brightens dark images, <1 darkens bright ones.")
    cfg["clahe_clip"] = st.slider("CLAHE clip limit", 0.5, 4.0, float(cfg["clahe_clip"]), 0.1, key=f"clip_{cat}")
    cfg["smooth"] = st.slider("Smoothing sigma", 0.5, 6.0, float(cfg["smooth"]), 0.25, key=f"smooth_{cat}",
                              help="Gaussian blur that hides fine surface texture before thresholding.")
    use_mask = st.checkbox("Use object mask (Otsu)", value=cfg["mask_mode"] != "full", key=f"mask_{cat}",
                           help="Only inspect the object, not the background.")
    if not use_mask:
        cfg["mask_mode"] = "full"
        cfg["use_shape"] = False          # the outline check needs the object mask
    cfg["use_color"] = st.checkbox("Colour cue (HSV saturation)", value=cfg["use_color"], key=f"color_{cat}",
                                   help="Flags ink, marker or oil that changes the surface colour.")
    st.caption("Calibrated values come from configs/<category>.json (tile, metal_nut) "
               "or hand-set defaults in pipeline.py (own photos, webcam).")

# ------------------------------------------------------------------ input
st.title("VisionInspect AI")
st.caption("Industrial quality inspection with classical OpenCV - no deep learning.  "
           "FCV Lab (CSE 3176) mini project.")

img, label, path = None, None, None
if source == "Dataset sample":
    root = os.path.join(DATA, cat, "test")
    folders = sorted(os.listdir(root), key=lambda f: (f != "good", f))
    c1, c2, c3 = st.columns([2, 2, 1])
    folder = c1.selectbox("Folder (ground truth)", folders, key=f"folder_{cat}",
                          index=folders.index(qp["folder"]) if qp.get("folder") in folders else 0)
    files = sorted(glob.glob(os.path.join(root, folder, "*.png")))
    key = f"file_{cat}_{folder}"
    if c3.button("Random image", width="stretch"):
        st.session_state[key] = random.choice(files)
    name = c2.selectbox("Image", files, format_func=os.path.basename, key=key)
    path, label = name, folder
elif source == "My own photos":
    objects = [cat] if cat in ("card", "coin", "object") else ["card", "object", "coin"]
    files = sorted(f for o in objects for f in glob.glob(os.path.join(DATA, "samples", "webcam", o, "*.*"))
                   if not os.path.basename(f).startswith("."))
    wanted = [f for f in files if qp.get("photo") and os.path.basename(f).startswith(qp["photo"])]
    path = st.selectbox("Photo", files, key=f"photo_{cat}", index=files.index(wanted[0]) if wanted else 0,
                        format_func=lambda f: os.path.relpath(f, os.path.join(DATA, "samples", "webcam")))
    label = "defect" if "defect" in os.path.basename(path) else "good"
elif source == "Upload image":
    up = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg", "bmp"])
    if up is not None:
        img = decode(up.getvalue())
else:
    shot = st.camera_input("Take a snapshot (allow camera access in the browser)")
    if shot is not None:
        img = decode(shot.getvalue())

if path is not None:
    img = decode(read_bytes(path))

if img is None:
    st.info("Choose or capture an image to inspect.")
    st.stop()

# ------------------------------------------------------------------ run the pipeline
res = P.inspect_image(img, cfg)
fps = 1000.0 / res["ms"] if res["ms"] > 0 else 0.0

css = "fail" if res["defective"] else "pass"
reasons = []
if res["n_defects"]:
    reasons.append(f"surface defect {res['score']:.2f}% &ge; {cfg['area_thr']:.2f}%")
if res["shape_bad"]:
    reasons.append("outline does not match a good part")
detail = "reason: " + " and ".join(reasons) if reasons else \
    f"largest defect {res['score']:.2f}% &lt; threshold {cfg['area_thr']:.2f}%"
st.markdown(f"<div class='verdict {css}'><span>{res['verdict']}</span>"
            f"<small>confidence {res['confidence']:.2f} &nbsp;|&nbsp; {detail}</small></div>",
            unsafe_allow_html=True)
if label is not None:
    expected_bad = label not in ("good",)
    ok = expected_bad == res["defective"]
    st.markdown(f"<div class='gt'>Ground truth: <b>{'DEFECTIVE' if expected_bad else 'GOOD'}</b> ({label}) - "
                f"prediction is {'✅ correct' if ok else '❌ wrong'}</div>", unsafe_allow_html=True)

m = st.columns(5)
m[0].metric("Confidence", f"{res['confidence']:.2f}")
m[1].metric("Largest defect", f"{res['score']:.2f}%")
m[2].metric("Defects over threshold", res["n_defects"],
             help=f"{res['n_blobs']} candidate regions in total; only those covering at least "
                  f"{cfg['area_thr']:.2f}% of the object count as defects.")
m[3].metric("Time", f"{res['ms']:.0f} ms")
m[4].metric("Speed", f"{fps:.0f} FPS")
if res["shape_dist"] is not None:
    st.caption(f"Shape check (Hu moments, cv2.matchShapes): distance {res['shape_dist']:.5f} "
               f"vs threshold {cfg['shape_thr']:.5f} -> {'MISMATCH (bent / flipped?)' if res['shape_bad'] else 'outline OK'}")

c1, c2 = st.columns(2)
shown = ground_truth_overlay(img, path) if source == "Dataset sample" else img
c1.image(rgb(P.resize_width(shown, 600)), width="stretch",
         caption="Input" + (" - green outline = MVTec ground-truth defect" if source == "Dataset sample" else ""))
c2.image(rgb(res["annotated"]), width="stretch",
         caption="Result - red box = defect over the threshold, yellow = 10 largest candidate regions")

# ------------------------------------------------------------------ stages
st.subheader("Processing stages")
stages = P.stage_images(res)
for row in range(2):
    cols = st.columns(3)
    for i in range(3):
        k = row * 3 + i
        title, caption = STAGE_CAPTIONS[k]
        cols[i].markdown(f"**{title}**")
        cols[i].image(rgb(stages[k]), width="stretch", caption=caption)

ok, buf = cv2.imencode(".jpg", P.montage(res))
st.download_button("Download stage montage (for the report)", buf.tobytes(), "montage.jpg", "image/jpeg")

# ------------------------------------------------------------------ defect table
st.subheader("Candidate regions and their Lab 5 features (largest 20)")
if res["defects"]:
    df = pd.DataFrame(res["defects"]).drop(columns=["bbox"])
    df["fails"] = df.severity_pct >= cfg["area_thr"]
    st.dataframe(df, hide_index=True, column_config={
        "severity_pct": st.column_config.NumberColumn("severity %", help="defect area as % of the object"),
        "elongation": st.column_config.NumberColumn(help="long side / short side of minAreaRect; >= 3 means linear"),
        "hu1_log": st.column_config.NumberColumn("Hu 1 (log)", help="first Hu moment, -log10 scale"),
        "orb_keypoints": st.column_config.NumberColumn("ORB kp", help="ORB keypoints inside the region"),
        "edge_density": st.column_config.NumberColumn(help="share of Canny edge pixels inside the region"),
        "contrast": st.column_config.NumberColumn(help="mean grey difference to a ring around the region"),
        "gradient": st.column_config.NumberColumn(help="mean Sobel gradient magnitude"),
        "fails": st.column_config.CheckboxColumn("over threshold"),
    })
else:
    st.write("No defect regions found.")

# ------------------------------------------------------------------ saved evaluation
with st.expander("Saved evaluation metrics"):
    mpath = os.path.join(RESULTS, f"{cat}_metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            met = json.load(f)
        t = met["test"]
        st.markdown(f"Held-out test split: **{t['n']} images** (calibrated on {met['split']['tuning']} separate "
                    f"tuning images, seed {met['split']['seed']}).")
        cols = st.columns(5)
        for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "fps"]):
            col.metric(k.upper() if k in ("f1", "fps") else k.capitalize(), f"{t[k]:.3f}" if k != "fps" else f"{t[k]:.0f}")
        a, b = st.columns(2)
        a.markdown("**Confusion matrix**")
        a.dataframe(pd.DataFrame([[t["TN"], t["FP"]], [t["FN"], t["TP"]]],
                                 index=["true good", "true defective"], columns=["pred good", "pred defective"]))
        b.markdown("**Per folder**")
        pf = pd.DataFrame(met["per_folder"]).T
        pf["accuracy"] = (pf.correct / pf.n).round(2)
        b.dataframe(pf)
        st.markdown("**Calibrated settings**")
        st.json({k: v for k, v in met["params"].items() if k != "shape_refs"})
    elif os.path.exists(os.path.join(RESULTS, "webcam_samples", "summary.json")):
        st.write("Own photos are visual demonstrations only (one clean and one defective photo per object), "
                 "not an accuracy figure. Settings were set by eye.")
        st.dataframe(pd.read_json(os.path.join(RESULTS, "webcam_samples", "summary.json")), hide_index=True)
    else:
        st.write("No saved metrics yet - run `python3 evaluate.py --category tile`.")

with st.expander("How it works"):
    st.markdown("""
1. **Lab 1** - read the image, resize to width 400, convert to grayscale.
2. **Lab 2** - gamma correction (lookup table) and CLAHE even out the lighting.
3. **Lab 3** - bilateral + median filter remove noise but keep edges; Canny finds edges; a Gaussian blur hides fine texture.
4. **Lab 4** - Otsu threshold + morphology finds the object; adaptive threshold finds spots darker or brighter than their
   neighbourhood; an HSV saturation check finds coloured marks; contours give the defect regions.
5. **Lab 5** - each region is described with minAreaRect elongation (crack vs. blob), Hu moments, ORB keypoints and
   edge density. For the metal nut, Hu-moment shape matching compares the outline with good nuts (bent / flipped parts).
6. **Decision** - if the largest region covers at least the area threshold (% of the object), the part is DEFECTIVE.
   Thresholds were calibrated on a tuning split and tested on separate held-out images.
""")
