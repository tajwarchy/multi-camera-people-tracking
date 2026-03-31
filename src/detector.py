import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple


class Detector:
    def __init__(self, cfg: dict):
        self.device = cfg["device"]
        model_path = cfg["detector"]["model"]
        self.conf = cfg["detector"]["confidence"]
        self.iou = cfg["detector"]["iou_threshold"]
        self.classes = cfg["detector"]["classes"]
        self.resolution = cfg["resolution"]

        self.model = YOLO(model_path)
        print(f"[Detector] Loaded {model_path} on {self.device}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single BGR frame.
        Returns: np.ndarray of shape (N, 5) — [x1, y1, x2, y2, conf]
                 Empty array of shape (0, 5) if no detections.
        """
        h, w = frame.shape[:2]
        scale = self.resolution / max(h, w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

        results = self.model.predict(
            source=resized,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()   # (N, 4)
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)  # (N, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)        # (N, 1) class id
        dets = np.concatenate([xyxy, conf, cls], axis=1)    # (N, 6)


        # Scale boxes back to original frame size
        dets[:, :4] /= scale
        return dets.astype(np.float32)