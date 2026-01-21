from __future__ import annotations
from pathlib import Path
import csv
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

IN_DIR = Path("assets/synthetic_images")
RUNS_DIR = Path("runs")

def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(gray)

def score_pair(clean_gray: np.ndarray, post_gray: np.ndarray):
    s, diff = ssim(clean_gray, post_gray, full=True)
    return 1.0 - float(s), diff

def priority(score: float) -> str:
    if score >= 0.35: return "P1"
    if score >= 0.20: return "P2"
    if score >= 0.10: return "P3"
    return "P4"

def estimated_water_liters_saved(p: str) -> float:
    if p == "P4": return 25.0
    if p == "P3": return 10.0
    return 0.0

def diff_to_heatmap(diff: np.ndarray) -> np.ndarray:
    anomaly = (1.0 - diff)
    anomaly = (anomaly * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(anomaly, cv2.COLORMAP_JET)

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for clean_path in sorted(IN_DIR.glob("*_clean.jpg")):
        post_path = Path(str(clean_path).replace("_clean.jpg", "_post.jpg"))
        if post_path.exists():
            pairs.append((clean_path, post_path))

    if not pairs:
        raise SystemExit("No *_clean.jpg / *_post.jpg pairs found in assets/synthetic_images")

    rows = []
    demo_overlay_path = out_dir / "demo_overlay.png"

    for idx, (cpath, ppath) in enumerate(pairs, start=1):
        clean_bgr = cv2.imread(str(cpath))
        post_bgr = cv2.imread(str(ppath))

        cgray = preprocess(clean_bgr)
        pgray = preprocess(post_bgr)

        score, diff = score_pair(cgray, pgray)
        p = priority(score)
        saved = estimated_water_liters_saved(p)

        if idx == 1:
            heat = diff_to_heatmap(diff)
            post_resized = cv2.resize(post_bgr, (heat.shape[1], heat.shape[0]))
            overlay = cv2.addWeighted(post_resized, 0.6, heat, 0.4, 0)
            cv2.imwrite(str(demo_overlay_path), overlay)

        stem = cpath.stem  # coach01_camA_01_clean
        parts = stem.split("_")
        coach_id, cam_id, img_id = parts[0], parts[1], parts[2]

        rows.append([coach_id, cam_id, img_id, round(score, 4), p, saved])

    report_path = out_dir / "cleaning_schedule.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["coach_id", "camera_id", "image_id", "anomaly_score", "priority", "estimated_water_liters_saved"])
        w.writerows(rows)

    print(f"✅ Report: {report_path}")
    print(f"✅ Demo overlay: {demo_overlay_path}")
    print("Now copy demo_overlay.png to docs/demo.png")

if __name__ == "__main__":
    main()