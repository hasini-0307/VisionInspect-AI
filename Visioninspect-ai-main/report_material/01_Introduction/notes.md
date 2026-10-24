# Introduction - raw points

- Visual quality inspection: checking every product for scratches, cracks, dents, stains, missing or deformed parts before it ships.
- Manual inspection is slow, tiring and inconsistent; inspectors miss small defects after hours of work.
- Industrial machine-vision systems exist but need special cameras, lighting and paid software, so they are too expensive for small factories and colleges.
- Deep learning (CNNs, anomaly detection networks) gives high accuracy but needs GPUs, many labelled images and is hard to explain ("black box").
- Classical computer vision (enhancement, filtering, segmentation, features) is fast, needs no training data beyond a few good samples, runs on a normal laptop + webcam, and every decision can be explained by showing the region and the threshold.
- MVTec AD (Bergmann et al., CVPR 2019 / IJCV 2021) is the standard benchmark for industrial anomaly detection: 15 categories, real defects, pixel-accurate ground-truth masks. We use two categories with different challenges:
  - **tile**: strong random texture, defects = crack, glue strip, gray stroke, oil, rough.
  - **metal_nut**: shiny machined object, defects = bent, colour, flip, scratch.
- Own photos (card, white object, coin) show the webcam use case with a phone/webcam-quality camera.
- Project goal in one line: a low-cost, real-time, explainable inspection tool built only from the techniques of FCV Labs 1-5.
- Report structure: problem statement → methodology → results → lab functions → conclusion.
