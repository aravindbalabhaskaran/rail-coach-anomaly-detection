from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import json
import cv2
from jinja2 import Template
from .detect import Detection
from .utils import ensure_dir

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Coach Anomaly Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 12px; margin-bottom: 16px; }
    img { max-width: 100%; border-radius: 12px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; border-bottom: 1px solid #eee; padding: 8px; }
    .small { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <h1>Coach Anomaly Report</h1>
  <p class="small">Run ID: {{ run_id }}</p>

  {% for item in items %}
  <div class="card">
    <h2>{{ item.filename }}</h2>
    <img src="{{ item.annotated_rel }}" alt="annotated"/>
    <h3>Detections</h3>
    <table>
      <tr><th>Label</th><th>Score</th><th>BBox (x1,y1,x2,y2)</th></tr>
      {% for d in item.detections %}
      <tr>
        <td>{{ d.label }}</td>
        <td>{{ "%.3f"|format(d.score) }}</td>
        <td>{{ d.bbox }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endfor %}
</body>
</html>
"""

def annotate_image(img_bgr, detections: list[Detection]):
    out = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{d.label}:{d.score:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out

def write_outputs(run_dir: Path, results: list[dict]) -> dict:
    ensure_dir(run_dir)
    ann_dir = ensure_dir(run_dir / "annotated")

    items = []
    for r in results:
        img_path: Path = r["path"]
        detections: list[Detection] = r["detections"]
        ann = annotate_image(r["image_bgr"], detections)
        out_path = ann_dir / img_path.name
        cv2.imwrite(str(out_path), ann)

        items.append({
            "filename": img_path.name,
            "annotated_rel": f"annotated/{img_path.name}",
            "detections": [asdict(d) for d in detections]
        })

    summary = {"num_images": len(items), "items": items}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    html = Template(HTML_TEMPLATE).render(run_id=run_dir.name, items=items)
    (run_dir / "report.html").write_text(html, encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "report_html": str(run_dir / "report.html"),
        "summary_json": str(run_dir / "summary.json"),
    }
