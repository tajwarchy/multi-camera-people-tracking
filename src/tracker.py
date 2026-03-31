import numpy as np
import torch
from pathlib import Path


class Tracker:
    def __init__(self, cfg: dict):
        self.device = cfg["device"]
        self.fallback = cfg["fallback_device"]
        t_cfg = cfg["tracker"]

        # Import here to catch boxmot issues early
        from boxmot import StrongSort
        self.tracker = StrongSort(
            reid_weights=Path("checkpoints/osnet_x1_0_market.pth"),
            device=torch.device(self.device),
            half=False,
        )
        self.tracker.max_age = t_cfg["max_age"]
        self.tracker.min_hits = t_cfg["min_hits"]
        print(f"[Tracker] StrongSORT initialized on {self.device}")

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Update tracker with current frame detections.
        Args:
            dets: (N, 5) array — [x1, y1, x2, y2, conf]
            frame: BGR frame (used internally by StrongSORT for ReID)
        Returns:
            tracks: (M, 5) array — [x1, y1, x2, y2, track_id]
                    Empty array of shape (0, 5) if no active tracks.
        """
        if dets.shape[0] == 0:
            return np.zeros((0, 5), dtype=np.float32)

        try:
            tracks = self.tracker.update(dets, frame)
        except Exception as e:
            print(f"[Tracker] MPS error, falling back to CPU: {e}")
            import torch
            self.tracker.device = torch.device(self.fallback)
            tracks = self.tracker.update(dets, frame)

        if tracks is None or len(tracks) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        # tracks shape from boxmot: (M, 7+) — [x1,y1,x2,y2,id,conf,cls,...]
        out = np.zeros((len(tracks), 5), dtype=np.float32)
        out[:, :4] = tracks[:, :4]   # bbox
        out[:, 4] = tracks[:, 4]     # track_id  (col 4 is still id in v16)
        return out