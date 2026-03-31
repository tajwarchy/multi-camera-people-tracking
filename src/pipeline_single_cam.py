import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

from src.config_loader import load_camera_config
from src.detector import Detector
from src.tracker import Tracker


COLORS = {}

def get_color(track_id: int):
    if track_id not in COLORS:
        np.random.seed(track_id * 7)
        COLORS[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
    return COLORS[track_id]


def load_frames(frames_dir: str, start: int, end: int) -> List[np.ndarray]:
    p = Path(frames_dir)
    files = sorted(p.glob("frame_*.png"))[start:end + 1]
    if not files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    print(f"[Pipeline] Loading {len(files)} frames from {frames_dir}")
    return [str(f) for f in files]


def annotate_frame(frame: np.ndarray, tracks: np.ndarray, cam_id: str) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
        color = get_color(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cam_id} | ID:{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def run_single_cam(global_cfg_path: str, cam_cfg_path: str) -> List[dict]:
    """
    Run detection + tracking on a single camera.
    Returns list of per-frame track records:
      [{ frame_id, cam_id, tracks: np.ndarray(M,5) }, ...]
    """
    cfg = load_camera_config(global_cfg_path, cam_cfg_path)
    cam_id = cfg["cam"]["cam_id"]
    out_dir = Path(cfg["output"]["per_camera_dir"]) / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(cfg)
    tracker = Tracker(cfg)

    frame_paths = load_frames(
        cfg["cam"]["source"],
        cfg["cam"]["start_frame"],
        cfg["cam"]["end_frame"]
    )

    # Video writer setup
    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    fps = cfg["output"]["fps"]
    fourcc = cv2.VideoWriter_fourcc(*cfg["output"]["fourcc"])
    vid_path = out_dir / f"{cam_id}_tracked.mp4"
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))

    all_records = []

    for frame_id, fpath in enumerate(tqdm(frame_paths, desc=f"[{cam_id}]")):
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"[WARNING] Could not read frame: {fpath}")
            continue

        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame)

        record = {
            "frame_id": frame_id,
            "cam_id": cam_id,
            "tracks": tracks.copy()
        }
        all_records.append(record)

        annotated = annotate_frame(frame, tracks, cam_id)
        writer.write(annotated)

    writer.release()
    print(f"[{cam_id}] Saved annotated video → {vid_path}")
    print(f"[{cam_id}] Total frame records: {len(all_records)}")
    return all_records


if __name__ == "__main__":
    # Test on cam01 first
    records = run_single_cam(
        global_cfg_path="configs/global_config.yaml",
        cam_cfg_path="configs/cam01.yaml"
    )
    # Quick sanity print — frames with at least one track
    tracked = [r for r in records if len(r["tracks"]) > 0]
    print(f"\nFrames with active tracks: {len(tracked)} / {len(records)}")
    if tracked:
        sample = tracked[len(tracked) // 2]
        print(f"Sample frame {sample['frame_id']}: {len(sample['tracks'])} tracks")
        print(sample["tracks"])