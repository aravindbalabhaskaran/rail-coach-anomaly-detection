from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
import torch
import torchvision

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float
    label: str

def _baseline_detect(img_bgr: np.ndarray) -> List[Detection]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    dets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.001 * (h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        score = min(0.99, float(area / (h * w)) * 10.0)
        dets.append(Detection((x, y, x + bw, y + bh), score, "anomaly_blob"))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:15]

class TorchDetector:
    def __init__(self, device: str = "cpu", score_thresh: float = 0.6):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.score_thresh = score_thresh
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.to(self.device).eval()
        self.labels = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

    @torch.no_grad()
    def __call__(self, img_bgr: np.ndarray) -> List[Detection]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        out = self.model([tensor.to(self.device)])[0]

        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()

        dets: List[Detection] = []
        for box, sc, lb in zip(boxes, scores, labels):
            if float(sc) < self.score_thresh:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            name = self.labels[int(lb)] if int(lb) < len(self.labels) else f"class_{int(lb)}"
            dets.append(Detection((x1, y1, x2, y2), float(sc), name))
        return dets

def get_detector(method: str, device: str, score_thresh: float):
    if method == "pretrained-fasterrcnn":
        return TorchDetector(device=device, score_thresh=score_thresh)
    if method == "baseline":
        return _baseline_detect
    raise ValueError(f"Unknown detect method: {method}")
