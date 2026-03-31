import cv2
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List
from src.config_loader import load_yaml


GLOBAL_CFG = "configs/global_config.yaml"
CAM_CFGS   = ["configs/cam01.yaml", "configs/cam02.yaml", "configs/cam03.yaml"]
CAM_IDS    = ["cam01", "cam02", "cam03"]

# Sample every Nth frame for gallery to avoid redundant near-identical crops
GALLERY_SAMPLE_INTERVAL = 10
# One query crop per (global_id, cam_id) combination
QUERY_PER_GID_CAM = 1


def extract_crop(frame: np.ndarray, bbox: np.ndarray,
                 target_h: int = 256, target_w: int = 128) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (target_w, target_h))


def main():
    global_cfg = load_yaml(GLOBAL_CFG)
    query_dir  = Path(global_cfg["eval"]["query_dir"])
    gallery_dir = Path(global_cfg["eval"]["gallery_dir"])
    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Load frame paths per camera
    from src.pipeline_single_cam import load_frames
    frame_paths = {}
    for cam_cfg_path in CAM_CFGS:
        cam_cfg = load_yaml(cam_cfg_path)
        cam_id  = cam_cfg["cam_id"]
        frame_paths[cam_id] = load_frames(
            cam_cfg["source"],
            cam_cfg["start_frame"],
            cam_cfg["end_frame"]
        )

    # Load enriched records
    emb_dir = Path(global_cfg["paths"]["embedding_store_dir"])
    stores  = {}
    for cam_id in CAM_IDS:
        with open(emb_dir / f"{cam_id}_enriched.pkl", "rb") as f:
            stores[cam_id] = pickle.load(f)

    # Track which (gid, cam_id) pairs have had a query saved
    query_saved: Dict[tuple, bool] = {}
    gallery_counts: Dict[int, int] = {}

    query_total   = 0
    gallery_total = 0

    for cam_id in CAM_IDS:
        records = stores[cam_id]
        fps_list = frame_paths[cam_id]

        for frame_idx, record in enumerate(records):
            tracks     = record["tracks"]
            global_ids = record.get("global_ids", [])

            if len(tracks) == 0:
                continue

            frame = cv2.imread(fps_list[frame_idx])
            if frame is None:
                continue

            for i, t in enumerate(tracks):
                gid = global_ids[i] if i < len(global_ids) else -1
                if gid == -1:
                    continue

                crop = extract_crop(frame, t[:4])
                if crop is None:
                    continue

                # Filename encodes gid and cam for evaluation
                # Format: gid_{gid:04d}_cam_{cam_id}_frame_{frame_idx:06d}.png
                fname = f"gid_{gid:04d}_cam_{cam_id}_frame_{frame_idx:06d}.png"

                # Save as query if not yet saved for this (gid, cam_id)
                key = (gid, cam_id)
                if key not in query_saved:
                    cv2.imwrite(str(query_dir / fname), crop)
                    query_saved[key] = True
                    query_total += 1

                # Save as gallery at sampled intervals
                elif frame_idx % GALLERY_SAMPLE_INTERVAL == 0:
                    cv2.imwrite(str(gallery_dir / fname), crop)
                    gallery_counts[gid] = gallery_counts.get(gid, 0) + 1
                    gallery_total += 1

    print(f"\n[ReID Eval] Query crops   : {query_total}")
    print(f"[ReID Eval] Gallery crops : {gallery_total}")
    print(f"[ReID Eval] Query dir     : {query_dir}")
    print(f"[ReID Eval] Gallery dir   : {gallery_dir}")
    print(f"\nGallery counts per global ID:")
    for gid, cnt in sorted(gallery_counts.items()):
        print(f"  G:{gid} → {cnt} gallery crops")


if __name__ == "__main__":
    main()