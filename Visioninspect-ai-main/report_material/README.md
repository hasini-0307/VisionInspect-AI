# Report material (raw - not the report)

Organised by the chapters of the report template (docs/FCV Lab-Mini Project-Report Template).

| Folder | Contents |
|---|---|
| 00_Abstract | key facts and numbers to update the synopsis abstract |
| 01_Introduction | background points |
| 02_Problem_Statement_and_Objectives | objectives as built, system requirements, scope check against the synopsis |
| 03_Methodology | `methodology_notes.md` (pipeline, why each step, calibration), `stage_montages/` (6-stage figures) |
| 04_Results_and_Snapshots | `metrics_tables.md`, `confusion_matrix_*.png`, `improvement_log.md`, `ui_screenshots/` (Streamlit GUI), TP/TN/FP/FN example montages, `own_photos/` |
| 05_Lab1-5_Functions_Utilized | `function_mapping.md` (function → file:line → purpose) |
| 06_Conclusion_Limitations_Future_Work | conclusion points, honest limitations, future work |
| 07_References | MVTec AD, OpenCV, method papers, lab-manual textbooks |
| viva_questions.md | 10 likely viva questions with short answers (+ extras) |

Example montage file names: `TP_test-crack_004.jpg` = true positive, folder test/crack, image 004.
TP = defective found, TN = good passed, FP = good wrongly flagged, FN = defect missed.

The Lab 11 page of the lab manual also lists "UI Design (include screenshots)", "Image Enhancement/filtering
techniques" and "VS code for functional design" for the final report: use `ui_screenshots/`,
`05_Lab1-5_Functions_Utilized/` and the code files for those.
