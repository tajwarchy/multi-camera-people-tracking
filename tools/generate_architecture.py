import cv2
import numpy as np

W, H = 1200, 500
canvas = np.ones((H, W, 3), dtype=np.uint8) * 245

def box(img, x, y, w, h, color, label, sublabel=None):
    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (80,80,80), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, label, (x + (w-tw)//2, y + h//2 - (6 if sublabel else 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    if sublabel:
        (sw, sh), _ = cv2.getTextSize(sublabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, sublabel, (x + (w-sw)//2, y + h//2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)

def arrow(img, x1, y1, x2, y2):
    cv2.arrowedLine(img, (x1,y1), (x2,y2), (80,80,80), 2, tipLength=0.15)

# Colors
CAM_C  = (180, 100, 40)
DET_C  = (40,  140, 80)
TRK_C  = (40,  100, 180)
EMB_C  = (160, 60,  160)
REID_C = (180, 130, 20)
BEV_C  = (40,  160, 160)
OUT_C  = (80,  80,  80)

# Row 1 — per camera (3 columns)
cam_xs = [30, 430, 830]
for i, cx in enumerate(cam_xs):
    box(canvas, cx,  30,  160, 50, CAM_C, f"Camera {i+1}", "EPFL Lab .avi")
    arrow(canvas, cx+80, 80, cx+80, 120)
    box(canvas, cx,  120, 160, 50, DET_C, "YOLOv8m", "Person Detection")
    arrow(canvas, cx+80, 170, cx+80, 210)
    box(canvas, cx,  210, 160, 50, TRK_C, "StrongSort", "Local Track IDs")
    arrow(canvas, cx+80, 260, cx+80, 300)
    box(canvas, cx,  300, 160, 50, EMB_C, "OSNet x1_0", "512-d Embeddings")

# Arrows converging to ReID
arrow(canvas, 110,  350, 560, 390)
arrow(canvas, 510,  350, 580, 390)
arrow(canvas, 910,  350, 610, 390)

# ReID Associator
box(canvas, 480, 390, 240, 55, REID_C, "ReID Associator", "Cosine Sim + EMA")
arrow(canvas, 600, 445, 420, 445)
arrow(canvas, 600, 445, 780, 445)

# Outputs
box(canvas, 270, 425, 150, 55, BEV_C, "BEV Projector", "Homography")
box(canvas, 780, 425, 150, 55, OUT_C, "Global ID Map", "JSON Export")
arrow(canvas, 345, 480, 345, 500)
box(canvas, 270, 460, 150, 35, BEV_C, "BEV Animation", "MP4 Export")

# Title
cv2.putText(canvas, "Project V1.2 — Multi-Camera People Tracking System Architecture",
            (180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40,40,40), 1)

cv2.imwrite("outputs/system_architecture.png", canvas)
print("Architecture diagram saved → outputs/system_architecture.png")