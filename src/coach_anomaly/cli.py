from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

from .ingest import list_images
from .preprocess import preprocess_bgr
from .detect import get_detector
from .report import write_outputs
from .utils import timestamp_run_dir, ensure_dir

def main():
    ap = argparse.ArgumentParser(description="Coach anomaly detection pipeline (PyTorch + CV).")
    ap.add_argument("--input", type=str, required=True, help="Input folder with images.")
    ap.add_argument("--out", type=str, default="runs", help="Output base folder.")
    ap.add_argument("--method", type=str, default="baseline", help="baseline | pretrained-fasterrcnn")
    ap.add_argument("--score-thresh", type=float, default=0.6, help="Score threshold for detector.")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda")
    args = ap.parse_args()

    input_dir = Path(args.input)
    out_base = ensure_dir(Path(args.out))
    run_dir = timestamp_run_dir(out_base)

    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    detector = get_detector(args.method, args.device, args.score_thresh)

    results = []
    for p in tqdm(images, desc="Processing images"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        pre = preprocess_bgr(img)
        dets = detector(pre)
        results.append({"path": p, "image_bgr": pre, "detections": dets})

    outputs = write_outputs(run_dir, results)
    print("Done.")
    print(f"Report: {outputs['report_html']}")
    print(f"Summary: {outputs['summary_json']}")

if __name__ == "__main__":
    main()
