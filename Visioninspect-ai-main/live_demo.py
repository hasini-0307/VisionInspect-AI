"""
VisionInspect AI - real-time webcam inspection (OpenCV window).

    python3 live_demo.py                      # webcam profile, camera 0
    python3 live_demo.py --category card      # use another profile
    python3 live_demo.py --camera 1           # try another camera

Keys:  q = quit   s = save frame   m = show / hide the processing stages
"""
import argparse
import collections
import os
import sys
import time

import cv2

import pipeline as P

SAVE_DIR = os.path.join(P.ROOT, "results", "live")

CAMERA_HELP = """
Could not open the camera (index {index}).

On macOS:
  1. System Settings -> Privacy & Security -> Camera
  2. Turn ON the app you are running this from (Terminal, iTerm or Visual Studio Code)
  3. Quit that app completely (Cmd+Q) and open it again, then re-run this script
Other things to try:
  - another camera index:  python3 live_demo.py --camera 1
  - close other apps that use the camera (Zoom, FaceTime, Photo Booth, browser tabs)
  - backup plan: use the Streamlit app with saved images (python3 -m streamlit run app.py)
"""


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(10):                      # the first frames are often black while the camera starts
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 2:
            return cap
        time.sleep(0.1)
    cap.release()
    return None


def save(frame, res, tag):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(SAVE_DIR, f"{stamp}_{tag}")
    cv2.imwrite(base + "_raw.jpg", frame)
    cv2.imwrite(base + "_annotated.jpg", res["annotated"])
    cv2.imwrite(base + "_montage.jpg", P.montage(res))
    print(f"Saved {base}_*.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--category", default="webcam", choices=P.CATEGORIES)
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q)")
    ap.add_argument("--no-window", action="store_true", help="no GUI window (for a quick camera test)")
    args = ap.parse_args()

    cfg = P.load_config(args.category)
    cap = open_camera(args.camera)
    if cap is None:
        print(CAMERA_HELP.format(index=args.camera))
        sys.exit(1)
    print(f"Camera {args.camera} open, profile '{args.category}'. Keys: q = quit, s = save, m = stages")

    votes = collections.deque(maxlen=5)      # verdict = majority of the last 5 frames (less flicker)
    fps = 0.0
    show_stages = False
    n = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
        res = P.inspect_image(frame, cfg)
        votes.append(res["defective"])
        stable_bad = sum(votes) > len(votes) / 2

        view = res["annotated"].copy()
        # replace the banner with the stable (majority) verdict
        colour = (0, 0, 200) if stable_bad else (0, 150, 0)
        cv2.rectangle(view, (0, 0), (view.shape[1], 30), colour, cv2.FILLED)
        cv2.putText(view, f"{'DEFECTIVE' if stable_bad else 'PASS'}  defect {res['score']:.2f}%",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        cv2.putText(view, f"FPS {fps:4.1f}  ({res['ms']:.0f} ms)", (8, view.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        n += 1
        if args.no_window:
            if n == 1 or n % 10 == 0:
                print(f"frame {n}: {res['verdict']} (stable {'DEFECTIVE' if stable_bad else 'PASS'}), "
                      f"defect {res['score']:.2f}%, {res['ms']:.0f} ms, {fps:.1f} FPS")
        else:
            cv2.imshow("VisionInspect AI - live (q quit, s save, m stages)", P.resize_width(view, 800))
            if show_stages:
                cv2.imshow("Processing stages", P.resize_width(P.montage(res), 1100))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save(frame, res, "defective" if stable_bad else "pass")
            if key == ord("m"):
                show_stages = not show_stages
                if not show_stages:
                    cv2.destroyWindow("Processing stages")
        if args.frames and n >= args.frames:
            if args.no_window:
                save(frame, res, "test")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
