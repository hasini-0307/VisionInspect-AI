# Lab 1-5 functions used in VisionInspect AI

Line numbers refer to the final code (`pipeline.py` unless another file is named).

## Lab 1 - Introduction to OpenCV (read, display, write, resize, draw, video)
| Function | Where | What it does in our project |
|---|---|---|
| `cv2.imread` | pipeline.py:77, app.py:90 | read dataset images / MVTec ground-truth masks |
| `cv2.imdecode` | app.py:80 | read uploaded files and webcam snapshots in the GUI |
| `cv2.resize` (INTER_AREA) | pipeline.py:85 | every image resized to width 400 (fast, same scale for all) |
| `cv2.cvtColor` BGR→GRAY / HSV / Lab / RGB | pipeline.py:164, 195, 132; app.py:70 | grayscale for processing, HSV for the colour cue, Lab for the object mask, RGB for display |
| `cv2.rectangle` | pipeline.py:345, 347 | red box around each defect, verdict banner background |
| `cv2.putText` | pipeline.py:349, 338 | PASS / DEFECTIVE text, "shape mismatch" label, stage captions |
| `cv2.drawContours` | pipeline.py:342, 337 | yellow defect outlines, magenta object outline |
| `cv2.VideoCapture` | live_demo.py:37 | real-time webcam input |
| `cv2.imshow` / `cv2.waitKey` | live_demo.py:107-110 | live window, keys q / s / m |
| `cv2.imwrite` | evaluate.py:242, live_demo.py:55-57 | save montages and saved frames |

## Lab 2 - Image enhancement
| Function | Where | What it does |
|---|---|---|
| Gamma (power-law) via `cv2.LUT` | pipeline.py:90-93 | brightens or darkens the image with a lookup table (gamma slider) |
| `cv2.createCLAHE` (clip 2.0, 8x8 tiles) | pipeline.py:99 | local contrast equalisation; evens out uneven lighting |
| `cv2.equalizeHist` | pipeline.py:101 | global histogram equalisation (option `enhance="histeq"`) |

## Lab 3 - Image filtering
| Function | Where | What it does |
|---|---|---|
| `cv2.bilateralFilter(7, 50, 50)` | pipeline.py:170 | removes noise but keeps edges sharp |
| `cv2.medianBlur(3)` | pipeline.py:170 | removes salt-and-pepper specks |
| Unsharp masking: `cv2.GaussianBlur` + `cv2.addWeighted(1.5, -0.5)` | pipeline.py:171 | sharpens edges before edge detection |
| `cv2.Canny(50, 150)` | pipeline.py:172 | edge map; used for edge density (feature and texture cue) |
| `cv2.Sobel` + `cv2.magnitude` | pipeline.py:238-240 | gradient strength inside each defect ("gradient" feature) |
| `cv2.GaussianBlur` (sigma "smooth") | pipeline.py:184 | hides fine surface texture before thresholding (calibrated) |
| `cv2.GaussianBlur` (sigma 6) | pipeline.py:196, 204 | large-scale blur of the colour and edge-density maps |

## Lab 4 - Segmentation
| Function | Where | What it does |
|---|---|---|
| `cv2.threshold` + `THRESH_OTSU` | pipeline.py:110 | separates the object from the background automatically |
| Colour segmentation (Lab b channel AND brightness) | pipeline.py:132-133 | object mask for our own photos (warm object, bluish desk) |
| `cv2.adaptiveThreshold` (GAUSSIAN_C, block 51, C) ×2 | pipeline.py:188-189 | dark spots and bright spots compared with their neighbourhood |
| Colour segmentation in HSV (saturation) | pipeline.py:195-199 | finds ink, marker, oil (colour different from the surface) |
| `cv2.morphologyEx` OPEN / CLOSE, `cv2.erode`, `cv2.dilate` | pipeline.py:137-138, 153, 209, 295 | clean the masks, join broken pieces, keep a margin from edges |
| `cv2.bitwise_and` / `cv2.bitwise_or` | pipeline.py:190-208 | combine cues; keep only pixels inside the object |
| `cv2.findContours`, `cv2.contourArea`, `cv2.boundingRect` | pipeline.py:141, 211, 214, 285 | turn the binary mask into separate defect regions |

## Lab 5 - Feature extraction
| Function | Where | What it does |
|---|---|---|
| `cv2.ORB_create().detect` (oriented FAST + rotated BRIEF) | pipeline.py:237 | keypoints; count inside each defect ("ORB kp") |
| `cv2.drawKeypoints` | pipeline.py:374 | shows ORB keypoints in stage 5 |
| `cv2.minAreaRect` | pipeline.py:297 | elongation = long / short side; ≥ 3 → crack/scratch, else stain/dent |
| `cv2.moments` + `cv2.HuMoments` | pipeline.py:299 | first Hu moment of each defect (shape descriptor) |
| `cv2.matchShapes` (Hu moments) | pipeline.py:280 | compares the nut outline with good nuts → finds bent / flipped nuts |
| `cv2.approxPolyDP` | evaluate.py:87 | simplifies the reference outlines stored in the config |
| Edge density, contrast (ring), Sobel gradient | pipeline.py:305-307 | texture / strength features of each defect |

## Note on the lab manual
The manual's Lab 5 names HOG, SIFT, LBP, FAST and Harris, and "shape descriptors". We use ORB (built on
the FAST detector, free and fast) and Hu moments (a classic shape descriptor). The manual's Lab 4 lists
thresholding and colour segmentation, which we use; morphology and contours are standard companions.
