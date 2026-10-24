"""
Calibrate and evaluate VisionInspect AI on one MVTec AD category.

  python3 evaluate.py --category tile
  python3 evaluate.py --category metal_nut

1. Collect images: test/good -> label 0, every other test folder -> label 1,
   plus random train/good images so the two classes are roughly balanced.
2. Stratified split (per folder, fixed seed): 30% tuning, 70% held-out test.
3. Grid-search smoothing sigma and sensitivity C on the TUNING split only;
   for each pair pick the area threshold with the best balanced accuracy.
4. Evaluate the chosen settings once on the held-out split and save everything.
"""
import argparse
import glob
import itertools
import json
import os
import random
import shutil
import time

import cv2
import numpy as np
import pandas as pd

import pipeline as P

SEED = 42
TUNE_FRACTION = 0.30
N_SHAPE_REFS = 5
# Values tried during calibration. None = switch that cue off.
# C goes above 25 because on textured surfaces low C turns the texture itself into blobs.
GRID = {
    "tile": dict(smooth=[2.0, 3.0], C=[10, 15, 20, 25, 30, 40],
                 sat_delta=[None, 4, 6, 8, 12], tex_delta=[None, 6, 8, 10]),
    "metal_nut": dict(smooth=[1.5, 2.0, 2.5, 3.0], C=[10, 15, 20, 25, 30, 40],
                      sat_delta=[None, 15, 25, 40]),
}
SWITCH = dict(sat_delta="use_color", tex_delta="use_texture")


def find_dataset(category, data_dir):
    direct = os.path.join(data_dir, category)
    if os.path.isdir(os.path.join(direct, "test")):
        return direct
    for path in glob.glob(os.path.join(data_dir, "**", category), recursive=True):
        if os.path.isdir(os.path.join(path, "test")):
            return path
    raise SystemExit(f"Could not find '{category}' with a test/ folder under {data_dir}/")


def collect(root):
    """List of (path, folder, label)."""
    items = []
    for folder in sorted(os.listdir(os.path.join(root, "test"))):
        files = sorted(glob.glob(os.path.join(root, "test", folder, "*.png")))
        label = 0 if folder == "good" else 1
        items += [(f, "test/" + folder, label) for f in files]
    n_good = sum(1 for i in items if i[2] == 0)
    n_bad = len(items) - n_good
    train = sorted(glob.glob(os.path.join(root, "train", "good", "*.png")))
    extra = random.Random(SEED).sample(train, min(len(train), max(0, n_bad - n_good)))
    items += [(f, "train/good", 0) for f in extra]
    return items


def stratified_split(items):
    rng = random.Random(SEED)
    tune, test = [], []
    for folder in sorted({i[1] for i in items}):
        group = [i for i in items if i[1] == folder]
        rng.shuffle(group)
        k = max(1, round(len(group) * TUNE_FRACTION))
        tune += group[:k]
        test += group[k:]
    return tune, test


def reference_outlines(root, used, cfg):
    """Outlines of a few train/good parts that are NOT in the evaluation set (for the shape cue)."""
    train = [p for p in sorted(glob.glob(os.path.join(root, "train", "good", "*.png"))) if p not in used]
    refs = []
    for p in random.Random(SEED + 1).sample(train, N_SHAPE_REFS):
        gray = cv2.cvtColor(P.resize_width(P.load_image(p)), cv2.COLOR_BGR2GRAY)
        obj, _ = P.object_mask(gray, cfg)
        c = cv2.approxPolyDP(P.outer_contour(obj), 1.0, True)
        refs.append(c.reshape(-1, 2).tolist())
    return refs


def candidates(values):
    u = np.unique(values)
    return [float(t) for t in list((u[:-1] + u[1:]) / 2) + [u[-1] + 1e-3] if t > 0]


def balanced_acc(pred, y):
    tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    return (tpr + tnr) / 2


def best_threshold(scores, labels, shapes=None):
    """Thresholds (between observed values) with the best balanced accuracy.
    With shapes: rule is score >= area_thr OR shape >= shape_thr, both searched together."""
    s, y = np.asarray(scores), np.asarray(labels)
    best = (-1.0, None, None)
    shape_cands = [None] if shapes is None else candidates(shapes)
    for t2 in shape_cands:
        shape_pred = np.zeros(len(y), bool) if t2 is None else np.asarray(shapes) >= t2
        for t in candidates(s):
            bacc = balanced_acc((s >= t) | shape_pred, y)
            if bacc > best[0]:
                best = (bacc, t, t2)
    return best


def metrics(labels, preds):
    y = np.asarray(labels)
    p = np.asarray(preds)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(
        n=len(y), accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        balanced_accuracy=(rec + (tn / (tn + fp) if tn + fp else 0.0)) / 2,
        TP=tp, TN=tn, FP=fp, FN=fn,
    )


def grid_params(category):
    """Every combination of the grid, as a dict of pipeline settings."""
    grid = GRID[category]
    for values in itertools.product(*grid.values()):
        params = {}
        for key, v in zip(grid, values):
            if key in SWITCH:
                params[SWITCH[key]] = v is not None
                if v is None:
                    continue
            params[key] = v
        yield params


def calibrate(category, tune_imgs, tune_labels, base):
    best = None
    for params in grid_params(category):
        cfg = dict(base, **params, shape_thr=1.0)   # shape_thr only so the distance gets computed
        res = [P.inspect_image(im, cfg, with_features=False) for im in tune_imgs]
        shapes = [r["shape_dist"] for r in res] if base["use_shape"] else None
        bacc, thr, shape_thr = best_threshold([r["score"] for r in res], tune_labels, shapes)
        if thr is not None and (best is None or bacc > best[0]):
            params = dict(params, area_thr=round(thr, 4))
            if base["use_shape"]:
                params["shape_thr"] = round(shape_thr, 6) if shape_thr else 0.0
            best = (bacc, params)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=["tile", "metal_nut"])
    ap.add_argument("--data-dir", default=os.path.join(P.ROOT, "data"))
    args = ap.parse_args()
    cat = args.category

    root = find_dataset(cat, args.data_dir)
    items = collect(root)
    tune, test = stratified_split(items)
    print(f"[{cat}] {len(items)} images ({sum(i[2] for i in items)} defective) "
          f"-> tuning {len(tune)}, held-out test {len(test)}")

    # --- calibrate on the tuning split only (config file is ignored so we start from defaults)
    cfg_path = os.path.join(P.CONFIG_DIR, f"{cat}.json")
    if os.path.exists(cfg_path):
        os.remove(cfg_path)
    base = P.load_config(cat)
    if base["use_shape"]:
        base["shape_refs"] = reference_outlines(root, {p for p, _, _ in items}, base)
    tune_imgs = [P.resize_width(P.load_image(p)) for p, _, _ in tune]
    t0 = time.time()
    tune_bacc, params = calibrate(cat, tune_imgs, [l for _, _, l in tune], base)
    print(f"Calibrated in {time.time() - t0:.0f}s: {params}  (tuning balanced acc {tune_bacc:.3f})")

    saved = dict(params)
    if base["use_shape"]:
        saved["shape_refs"] = base["shape_refs"]
    os.makedirs(P.CONFIG_DIR, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(dict(category=cat, calibrated_on=f"{len(tune)} tuning images", seed=SEED,
                       tuning_balanced_accuracy=round(tune_bacc, 4), params=saved), f, indent=2)
    cfg = P.load_config(cat)

    # --- evaluate once on the held-out split
    rows, results = [], []
    for path, folder, label in test:
        img = P.load_image(path)
        res = P.inspect_image(img, cfg)
        results.append(res)
        rows.append(dict(path=os.path.relpath(path, P.ROOT), folder=folder, label=label,
                         pred=int(res["defective"]), score=round(res["score"], 4),
                         shape_dist=None if res["shape_dist"] is None else round(res["shape_dist"], 6),
                         confidence=round(res["confidence"], 3), n_blobs=res["n_blobs"],
                         ms=round(res["ms"], 2)))
    df = pd.DataFrame(rows)
    df["correct"] = df.label == df.pred
    m = metrics(df.label, df.pred)
    m["mean_ms"] = float(df.ms.mean())
    m["fps"] = 1000.0 / m["mean_ms"]
    per_folder = df.groupby("folder").agg(n=("correct", "size"), correct=("correct", "sum"))
    per_folder["acc"] = (per_folder.correct / per_folder.n).round(3)

    print("\nHeld-out test results")
    print(pd.Series({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}).to_string())
    print("\nConfusion matrix (rows = true, cols = predicted)")
    print(pd.DataFrame([[m["TN"], m["FP"]], [m["FN"], m["TP"]]],
                       index=["true good", "true defective"], columns=["pred good", "pred defective"]).to_string())
    print("\nPer-folder correct")
    print(per_folder.to_string())

    # --- save
    os.makedirs(os.path.join(P.ROOT, "results"), exist_ok=True)
    out = dict(category=cat, params=params, split=dict(tuning=len(tune), test=len(test), seed=SEED),
               tuning_balanced_accuracy=tune_bacc, test=m,
               per_folder={k: dict(n=int(v.n), correct=int(v.correct)) for k, v in per_folder.iterrows()})
    with open(os.path.join(P.ROOT, "results", f"{cat}_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    df.to_csv(os.path.join(P.ROOT, "results", f"{cat}_predictions.csv"), index=False)

    ex_dir = os.path.join(P.ROOT, "results", f"{cat}_examples")
    shutil.rmtree(ex_dir, ignore_errors=True)
    os.makedirs(ex_dir)
    kinds = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
    for kind, (lab, pred) in kinds.items():
        idx = df.index[(df.label == lab) & (df.pred == pred)].tolist()
        for i in random.Random(SEED).sample(idx, min(3, len(idx))):
            name = f"{kind}_{df.folder[i].replace('/', '-')}_{os.path.basename(df.path[i])}"
            cv2.imwrite(os.path.join(ex_dir, name.replace(".png", ".jpg")), P.montage(results[i]))
    print(f"\nSaved configs/{cat}.json, results/{cat}_metrics.json, results/{cat}_predictions.csv, results/{cat}_examples/")


if __name__ == "__main__":
    main()
