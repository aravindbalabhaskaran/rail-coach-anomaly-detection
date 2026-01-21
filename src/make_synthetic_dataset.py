from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import random

OUT_DIR = Path("assets/synthetic_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def draw_simple_interior(w=640, h=360) -> np.ndarray:
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    cv2.rectangle(img, (0, int(h*0.70)), (w, h), (200, 200, 200), -1)  # floor
    for x in range(60, w, 140):  # seat blocks
        cv2.rectangle(img, (x, int(h*0.45)), (x+80, int(h*0.70)), (180, 180, 190), -1)
        cv2.rectangle(img, (x+10, int(h*0.50)), (x+70, int(h*0.68)), (160, 160, 170), 2)
    cv2.line(img, (0, int(h*0.25)), (w, int(h*0.25)), (170, 170, 170), 2)  # window line
    noise = np.random.normal(0, 4, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def add_anomalies(img: np.ndarray, severity: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    n = {0: 0, 1: 6, 2: 14, 3: 26}[severity]
    for _ in range(n):
        x = random.randint(0, w-1)
        y = random.randint(int(h*0.30), h-1)
        r = random.randint(6, 30)
        color = random.randint(20, 90)
        cv2.circle(out, (x, y), r, (color, color, color), -1)
    if severity >= 2:
        x1 = random.randint(50, w-50)
        y1 = random.randint(int(h*0.60), h-20)
        x2 = x1 + random.randint(-80, 80)
        y2 = y1 + random.randint(10, 60)
        cv2.line(out, (x1, y1), (x2, y2), (60, 60, 60), thickness=12)
    return out

def main():
    random.seed(7)
    np.random.seed(7)
    for i in range(1, 11):
        base = draw_simple_interior()
        severity = random.choice([0, 1, 1, 2, 2, 3])
        clean = base
        post = add_anomalies(base, severity)
        cv2.imwrite(str(OUT_DIR / f"coach01_camA_{i:02d}_clean.jpg"), clean)
        cv2.imwrite(str(OUT_DIR / f"coach01_camA_{i:02d}_post.jpg"), post)
    print(f"✅ Synthetic dataset created in: {OUT_DIR}")

if __name__ == "__main__":
    main()