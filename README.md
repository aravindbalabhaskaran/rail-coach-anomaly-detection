# Rail Coach Anomaly Detection (PyTorch + OpenCV)

Post-journey image analysis pipeline to flag cleanliness & safety anomalies in rail coaches
(e.g., trash, spills, loose objects) using Computer Vision and PyTorch.

## Problem Statement
Manual inspection of train coaches after journeys is time-consuming and inconsistent.
This project automates post-journey inspection by analyzing camera images and highlighting
potential anomalies that require maintenance attention.

## Why this matters
- **Maintenance automation:** reduces repetitive manual checks and speeds up turnaround time
- **Safety & comfort:** detects hazards (spills/objects) before next departure
- **Operational efficiency:** prioritizes cleaning/repair teams using a structured report

## Solution Overview (Pipeline)
1. **Ingest** post-journey images (multiple angles/zoom)
2. **Preprocess** images (denoise + contrast enhancement + sharpening)
3. **Detect anomalies**
   - v0: baseline CV detector (fast, no training)
   - v1: PyTorch detector (pretrained / fine-tuned)
4. **Generate report** (annotated images + HTML + JSON)
5. **Optional notification** to maintenance (email integration)

## Quickstart (How to run)
```bash
# 1) Create env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 2) Install
pip install -r requirements.txt

# 3) Run (example)
python -m coach_anomaly.cli --input assets/sample_images --out runs
