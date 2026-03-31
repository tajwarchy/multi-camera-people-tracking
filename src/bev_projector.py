import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class BEVProjector:
    def __init__(self, cfg: dict):
        h_dir         = Path(cfg["paths"]["homography_dir"])
        floor_path    = cfg["bev"]["floor_plan_path"]
        self.canvas_w = cfg["bev"]["canvas_width"]   # 800
        self.canvas_h = cfg["bev"]["canvas_height"]  # 600

        # Load homographies for cam0, cam1, cam2 only
        self.homographies: Dict[str, np.ndarray] = {}
        cam_map = {"cam01": "cam0", "cam02": "cam1", "cam03": "cam2"}
        for cam_id, file_key in cam_map.items():
            p = h_dir / f"{file_key}_H.npy"
            if p.exists():
                self.homographies[cam_id] = np.load(str(p))
                print(f"[BEVProjector] Loaded homography for {cam_id} ← {file_key}_H.npy")
            else:
                print(f"[BEVProjector] WARNING: No homography found for {cam_id}")

        # Load floor plan
        self.floor_plan = cv2.imread(floor_path)
        if self.floor_plan is None:
            raise FileNotFoundError(f"Floor plan not found: {floor_path}")
        self.floor_plan = cv2.resize(self.floor_plan, (self.canvas_w, self.canvas_h))

        # Percentile bounds from probe_scale.py
        self.x_min =  10.0
        self.x_max = 400.0
        self.y_min = -280.0
        self.y_max = 295.0
        self.margin = 30

        x_span = self.x_max - self.x_min
        y_span = self.y_max - self.y_min

        # Uniform scale preserving aspect ratio
        self.scale = min(
            (self.canvas_w - 2 * self.margin) / x_span,
            (self.canvas_h - 2 * self.margin) / y_span
        )

        # Center data on canvas
        self.offset_x = (self.canvas_w - x_span * self.scale) / 2
        self.offset_y = (self.canvas_h - y_span * self.scale) / 2

        print(f"[BEVProjector] scale={self.scale:.4f} | "
              f"offset=({self.offset_x:.1f}, {self.offset_y:.1f})")

    def project_foot_point(self, cam_id: str,
                           foot_x: float, foot_y: float) -> Optional[Tuple[int, int]]:
        """
        Project a foot point from camera image coords to BEV canvas pixel coords.
        Returns (canvas_x, canvas_y) or None if cam has no homography.
        """
        H = self.homographies.get(cam_id)
        if H is None:
            return None

        pt = np.array([[[foot_x, foot_y]]], dtype=np.float64)
        tv = cv2.perspectiveTransform(pt, H).reshape(2)

        cx = int((tv[0] - self.x_min) * self.scale + self.offset_x)
        cy = int((self.canvas_h - self.offset_y) - (tv[1] - self.y_min) * self.scale)

        cx = max(0, min(self.canvas_w - 1, cx))
        cy = max(0, min(self.canvas_h - 1, cy))
        return cx, cy

    def project_tracks(self, cam_id: str,
                       tracks: np.ndarray) -> Dict[int, Tuple[int, int]]:
        """
        Project all tracks for a camera to BEV canvas coords.
        Args:
            tracks: (M, 5) — [x1, y1, x2, y2, track_id]
        Returns:
            { track_id: (canvas_x, canvas_y) }
        """
        result = {}
        for t in tracks:
            x1, y1, x2, y2, tid = t[0], t[1], t[2], t[3], int(t[4])
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            pos = self.project_foot_point(cam_id, foot_x, foot_y)
            if pos is not None:
                result[tid] = pos
        return result