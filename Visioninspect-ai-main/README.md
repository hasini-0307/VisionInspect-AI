# VisionInspect AI

Real-time industrial quality inspection and defect detection with **classical OpenCV only** (no deep learning).
Input: MVTec AD images (tile, metal_nut), our own phone photos, or a webcam.
Output: PASS / DEFECTIVE verdict, defect regions, confidence and severity.

FCV Lab (CSE 3176) mini project, Manipal Institute of Technology.

## Files

| File | What it does |
|---|---|
| `pipeline.py` | The inspection pipeline (Lab 1-5 functions), settings per category, drawing, stage montage |
| `evaluate.py` | Calibrates the settings on a tuning split and measures accuracy on held-out images |
| `app.py` | Streamlit GUI (the main demo) |
| `live_demo.py` | Real-time webcam window |
| `configs/` | Calibrated settings for tile and metal_nut (written by `evaluate.py`) |
| `results/` | Metrics, predictions, example montages, own-photo results, improvement log |
| `report_material/` | Raw material for the report and PPT (not the report itself) |
| `data/`, `docs/` | Datasets and course documents (never modified) |

## Setup (once)

```bash
git clone https://github.com/sriyaM26/visioninspect-ai.git
cd visioninspect-ai
python3 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Every time you open a new terminal: `cd visioninspect-ai && source .venv/bin/activate`

### Get the MVTec AD data (not in this repo - too large, and licensed by MVTec)
1. Go to https://www.mvtec.com/company/research/datasets/mvtec-ad and download the **tile** and **metal_nut**
   categories (free, non-commercial licence CC BY-NC-SA 4.0).
2. Extract them so the folders look like this:
```
data/tile/train/good/...       data/tile/test/<crack, good, ...>/...       data/tile/ground_truth/...
data/metal_nut/train/good/...  data/metal_nut/test/<bent, good, ...>/...   data/metal_nut/ground_truth/...
```
Our own photos are already included in `data/samples/webcam/`. The calibrated settings (`configs/`) and all
results are in the repo, so the app works for own photos and webcam even before the download;
the dataset is only needed for the tile / metal_nut samples and for re-running `evaluate.py`.

## Run

```bash
python3 -m streamlit run app.py            # GUI, opens http://localhost:8501
python3 live_demo.py                       # webcam window (q quit, s save, m stages)
python3 live_demo.py --camera 1            # if camera 0 is the wrong one
python3 evaluate.py --category tile        # re-calibrate + evaluate (about 2-3 min)
python3 evaluate.py --category metal_nut   # (about 1-2 min)
```

Handy app links (open the app first):
- `http://localhost:8501/?category=tile&folder=crack`
- `http://localhost:8501/?category=metal_nut&folder=bent`
- `http://localhost:8501/?category=object&photo=defect`
- `http://localhost:8501/?category=card&photo=good`

### macOS camera permission (do this before the demo)
System Settings → Privacy & Security → Camera → switch ON **Visual Studio Code** (and/or Terminal).
Quit VS Code completely (Cmd+Q), reopen it, then run `python3 live_demo.py`.
The app's "Webcam snapshot" uses the **browser's** camera permission instead (click Allow in Chrome/Safari).

## Results (held-out test split, never used for tuning)

| Category | Images | Accuracy | Precision | Recall | F1 | Speed |
|---|---|---|---|---|---|---|
| tile | 119 | 0.765 | 0.848 | 0.650 | 0.736 | ~26 ms (≈39 FPS) |
| metal_nut | 129 | 0.806 | 0.953 | 0.641 | 0.766 | ~19 ms (≈51 FPS) |

Own photos (card, object, coin; 1 clean + 1 defective each): all 6 verdicts correct, but these are
visual demonstrations, not an accuracy figure. The coin only works with settings hand-tuned on its two photos.
Details: `results/improvement_log.md`.

## Demo script (about 5 minutes)

1. **Start** (before the evaluators arrive): `python3 -m streamlit run app.py`, open the tile/crack link above.
   Keep `results/webcam_samples/` open in Finder as a backup.
2. **Tile, crack** - "This is an MVTec AD tile. The green outline is the official ground truth; our red box
   finds the crack. Below are the six stages: Lab 1 resize, Lab 2 gamma + CLAHE, Lab 3 bilateral/median + Canny,
   Lab 4 adaptive threshold + contours, Lab 5 ORB keypoints and features."
3. **Tile, good** - "A good tile passes. The yellow outlines are small candidate regions from the tile's own
   texture; they are below the area threshold, so it is PASS."
4. **Move the Sensitivity C slider down** - "Lower C makes it more sensitive; texture starts to count as defects.
   That is why we calibrated C and the area threshold on a tuning split."
5. **Metal nut, bent** - "The surface looks fine, but the outline is different. We compare the outline with good
   nuts using Hu moments (cv2.matchShapes), so it is flagged as a shape mismatch."
6. **Defect table** - "Each region gets Lab 5 features: elongation from minAreaRect decides crack vs. blob,
   Hu moment, ORB keypoint count, edge density, contrast."
7. **Saved evaluation metrics expander** - show accuracy / precision / recall / F1, confusion matrix, FPS.
8. **Own photos** - White object (pen mark found, clean one passes), Textured card (dent + marker).
9. **Live** - `python3 live_demo.py`: show a clean white object (PASS), then draw a pen mark or put a coloured
   sticker on it (DEFECTIVE). Press `m` to show the stages, `s` to save a frame.

**Webcam tips:** plain light object, centred, on a darker plain background, steady light, no strong shadows.
If it gives false alarms, raise *Sensitivity C* or *Area threshold* in the app (Webcam category) or
use the "Webcam snapshot" input in the app instead of `live_demo.py`.

**Backup plan if the webcam fails:** stay in the Streamlit app. Use *White object (own photos)* and
*Textured card (own photos)*: these are real webcam-style photos. Or use *Upload image* with the saved
`results/webcam_samples/*_annotated.jpg` / `*_montage.jpg`. Say: "The live camera needs macOS permission;
these are photos of the same objects taken earlier, processed by the same pipeline."

## Honest limits
- Thin, low-contrast scratches on the metal nut are mostly missed (3/16), because they look like normal machining marks.
- Soft tile defects (glue strip, gray stroke, rough) are caught only when they change the colour.
- The coin's embossed design looks like defects to a threshold-based method. It is kept out of the main demo.
- Thresholds depend on the lighting and the object; a new object needs new settings.
