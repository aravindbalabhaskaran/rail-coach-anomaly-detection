# 🚆 Rail Coach Anomaly Detection for Sustainable Cleaning

This project demonstrates a **privacy-safe, reproducible computer vision pipeline** for detecting cleanliness and maintenance anomalies in rail coaches using **existing cabin camera infrastructure**.

The system is designed to support **condition-based cleaning**, reducing unnecessary water usage, energy consumption, and labor by prioritizing only coaches that actually require attention.

> ⚠️ **Note on data privacy:**  
> Real rail-coach interior images cannot be shared due to access and privacy constraints.  
> This repository therefore includes a **synthetic but realistic demo dataset** that fully replicates the intended real-world pipeline.

---

## 🧠 Problem Statement

Rail coaches are typically cleaned on a **fixed schedule**, regardless of their actual condition.  
This leads to:
- unnecessary water and energy consumption  
- avoidable labor costs  
- inefficient maintenance planning  

This project addresses the problem by answering one question:

> **“Can post-journey images from existing cameras be used to automatically decide *whether* and *how urgently* a coach needs cleaning?”**

---

## 💡 Solution Overview

The pipeline:
1. Compares **post-journey images** against a clean reference
2. Computes an **anomaly score** using image similarity
3. Converts scores into **cleaning priorities**
4. Estimates **resource savings** from skipped or reduced cleaning
5. Produces a **visual heatmap** for fast human inspection

All of this is done **without requiring new hardware**.

---

## 🌱 Sustainability & Real-World Impact

By enabling **condition-based cleaning**, this approach can:
- Reduce unnecessary water usage
- Lower energy consumption of cleaning equipment
- Optimize labor allocation
- Improve coach turnaround time

The system converts raw computer-vision outputs into **actionable operational decisions**, bridging AI and sustainability.

---

## 🚀 1-Minute Demo (Privacy-Safe)

This repository includes a **fully reproducible demo** using synthetic images  
(no real train or passenger data required).

### Run the demo
```bash
python src/make_synthetic_dataset.py
python src/demo_run.py