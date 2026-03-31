import json
import numpy as np
import cv2

with open("outputs/reid_metrics.json") as f:
    metrics = json.load(f)

W, H = 600, 400
canvas = np.ones((H, W, 3), dtype=np.uint8) * 245

metrics_display = {
    "CMC@1\n(tuned)": 50.0,
    "CMC@3":          90.0,
    "CMC@5":          90.0,
    "mAP\n(tuned)":   56.8,
    "CMC@1\n(orig)":  62.5,
    "mAP\n(orig)":    62.6,
}

colors = [
    (40,  160, 80),
    (40,  160, 80),
    (40,  160, 80),
    (40,  160, 80),
    (160, 130, 40),
    (160, 130, 40),
]

bar_w   = 60
spacing = 30
x_start = 60
max_val = 100
chart_h = 280
base_y  = 340

for i, ((label, val), color) in enumerate(zip(metrics_display.items(), colors)):
    bh = int(val / max_val * chart_h)
    bx = x_start + i * (bar_w + spacing)
    by = base_y - bh

    cv2.rectangle(canvas, (bx, by), (bx + bar_w, base_y), color, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bar_w, base_y), (80,80,80), 1)

    val_label = f"{val:.1f}%"
    (vw, vh), _ = cv2.getTextSize(val_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(canvas, val_label, (bx + (bar_w - vw)//2, by - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 1)

    for j, part in enumerate(label.split("\n")):
        cv2.putText(canvas, part,
                    (bx + 4, base_y + 18 + j * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,60,60), 1)

# Axes
cv2.line(canvas, (45, 50), (45, base_y), (80,80,80), 2)
cv2.line(canvas, (45, base_y), (W-20, base_y), (80,80,80), 2)

# Y axis labels
for pct in [0, 25, 50, 75, 100]:
    y = base_y - int(pct / 100 * chart_h)
    cv2.line(canvas, (42, y), (48, y), (80,80,80), 1)
    cv2.putText(canvas, f"{pct}%", (8, y+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1)

# Title
cv2.putText(canvas, "ReID Evaluation Metrics — Project V1.2",
            (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 1)

# Legend
cv2.rectangle(canvas, (350, 355), (370, 370), (40,160,80), -1)
cv2.putText(canvas, "Tuned (visual)", (375, 368),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,40,40), 1)
cv2.rectangle(canvas, (350, 375), (370, 390), (160,130,40), -1)
cv2.putText(canvas, "Original", (375, 388),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40,40,40), 1)

cv2.imwrite("outputs/reid_metrics_chart.png", canvas)
print("Metrics chart saved → outputs/reid_metrics_chart.png")