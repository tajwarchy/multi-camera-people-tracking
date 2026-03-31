import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


def get_global_color(gid: int) -> Tuple[int, int, int]:
    """Deterministic BGR color from global ID."""
    np.random.seed(gid * 31 + 7)
    return tuple(np.random.randint(60, 230, 3).tolist())


class BEVAnimator:
    def __init__(self, cfg: dict, projector):
        self.projector   = projector
        self.tail_len    = cfg["bev"]["trajectory_tail_length"]  # 30
        self.canvas_w    = cfg["bev"]["canvas_width"]
        self.canvas_h    = cfg["bev"]["canvas_height"]

        # { global_id: deque of (cx, cy) positions }
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.tail_len)
        )

    def update(self,
               frame_records: Dict[str, dict],
               enriched: bool = True) -> np.ndarray:
        """
        Build one BEV canvas frame from all camera records at this timestep.
        Args:
            frame_records: { cam_id: single enriched record for this frame }
        Returns:
            canvas: BGR image (canvas_h, canvas_w, 3)
        """
        canvas = self.projector.floor_plan.copy()

        # Collect global_id → canvas position for this frame
        # If multiple cameras see same global ID, average their positions
        gid_positions: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        for cam_id, record in frame_records.items():
            tracks     = record["tracks"]       # (M, 5)
            global_ids = record.get("global_ids", [])

            if len(tracks) == 0:
                continue

            positions = self.projector.project_tracks(cam_id, tracks)

            for i, t in enumerate(tracks):
                local_tid = int(t[4])
                gid = global_ids[i] if i < len(global_ids) else -1
                if gid == -1:
                    continue
                if local_tid in positions:
                    gid_positions[gid].append(positions[local_tid])

        # Average positions per global ID and update trajectories
        frame_gid_pos: Dict[int, Tuple[int, int]] = {}
        for gid, pos_list in gid_positions.items():
            cx = int(np.mean([p[0] for p in pos_list]))
            cy = int(np.mean([p[1] for p in pos_list]))
            self.trajectories[gid].append((cx, cy))
            frame_gid_pos[gid] = (cx, cy)

        # Draw trajectory tails
        for gid, traj in self.trajectories.items():
            if len(traj) < 2:
                continue
            color = get_global_color(gid)
            pts = list(traj)
            for j in range(1, len(pts)):
                alpha = j / len(pts)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(canvas, pts[j - 1], pts[j], faded, 2)

        # Draw current position dots + global ID label
        for gid, (cx, cy) in frame_gid_pos.items():
            color = get_global_color(gid)
            cv2.circle(canvas, (cx, cy), 10, color, -1)
            cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 2)
            label = f"G:{gid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(canvas, (cx + 12, cy - th - 4),
                          (cx + 12 + tw + 4, cy + 4), color, -1)
            cv2.putText(canvas, label, (cx + 14, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return canvas