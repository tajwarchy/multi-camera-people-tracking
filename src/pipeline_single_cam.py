import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

from src.config_loader import load_camera_config
from src.detector import Detector
from src.tracker import Tracker
from src.embedder import Embedder


COLORS = {}

def get_color(track_id: int):
    if track_id not in COLORS:
        np.random.seed(track_id * 7)
        COLORS[track_id] = tuple(np.random.randint(50, 255, 3).tolist())
    return COLORS[track_id]


def load_frames(frames_dir: str, start: int, end: int) -> List[str]:
    p = Path(frames_dir)
    files = sorted(p.glob("frame_*.png"))[start:end + 1]
    if not files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    print(f"[Pipeline] Found {len(files)} frames in {frames_dir}")
    return [str(f) for f in files]


def annotate_frame(frame: np.ndarray, tracks: np.ndarray,
                   cam_id: str, global_ids: List[int] = None) -> np.ndarray:
    out = frame.copy()
    for i, t in enumerate(tracks):
        x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
        gid = global_ids[i] if global_ids is not None else -1
        color = get_color(tid)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Top label: local ID
        local_label = f"{cam_id}|L:{tid}"
        (tw, th), _ = cv2.getTextSize(local_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, local_label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bottom label: global ID (white box)
        if gid != -1:
            global_label = f"G:{gid}"
            (gw, gh), _ = cv2.getTextSize(global_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y2), (x1 + gw + 4, y2 + gh + 6), (255, 255, 255), -1)
            cv2.putText(out, global_label, (x1 + 2, y2 + gh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def run_single_cam(global_cfg_path: str, cam_cfg_path: str) -> List[dict]:
    """
    Run detection + tracking + embedding extraction on a single camera.
    Returns list of per-frame records:
      [{ frame_id, cam_id, tracks: (M,5), embeddings: (M, dim) }, ...]
    Saves embedding store as outputs/embeddings/{cam_id}_embeddings.pkl
    """
    cfg = load_camera_config(global_cfg_path, cam_cfg_path)
    cam_id = cfg["cam"]["cam_id"]
    out_dir = Path(cfg["output"]["per_camera_dir"]) / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(cfg["paths"]["embedding_store_dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(cfg)
    tracker  = Tracker(cfg)
    embedder = Embedder(cfg)

    frame_paths = load_frames(
        cfg["cam"]["source"],
        cfg["cam"]["start_frame"],
        cfg["cam"]["end_frame"]
    )

    # Video writer
    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    fps    = cfg["output"]["fps"]
    fourcc = cv2.VideoWriter_fourcc(*cfg["output"]["fourcc"])
    vid_path = out_dir / f"{cam_id}_tracked.mp4"
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))

    all_records = []

    for frame_id, fpath in enumerate(tqdm(frame_paths, desc=f"[{cam_id}]")):
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"[WARNING] Could not read: {fpath}")
            continue

        dets       = detector.detect(frame)
        tracks     = tracker.update(dets, frame)
        embeddings = embedder.extract(frame, tracks)

        record = {
            "frame_id"  : frame_id,
            "cam_id"    : cam_id,
            "tracks"    : tracks.copy(),       # (M, 5)
            "embeddings": embeddings.copy()    # (M, 512)
        }
        all_records.append(record)

        annotated = annotate_frame(frame, tracks, cam_id)
        writer.write(annotated)

    writer.release()

    # Save embedding store
    store_path = emb_dir / f"{cam_id}_embeddings.pkl"
    with open(store_path, "wb") as f:
        pickle.dump(all_records, f)
    print(f"[{cam_id}] Embedding store saved → {store_path}")
    print(f"[{cam_id}] Annotated video saved → {vid_path}")

    tracked = [r for r in all_records if len(r["tracks"]) > 0]
    print(f"[{cam_id}] Frames with active tracks: {len(tracked)} / {len(all_records)}")
    return all_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", default="cam01", help="cam01 | cam02 | cam03")
    args = parser.parse_args()

    run_single_cam(
        global_cfg_path="configs/global_config.yaml",
        cam_cfg_path=f"configs/{args.cam}.yaml"
    )